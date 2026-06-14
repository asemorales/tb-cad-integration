"""Evaluate a trained checkpoint on the TBX11K test split.

Loads a Lightning checkpoint, runs inference over `data/<root>/test/`, and
prints the headline metrics: per-class sensitivity / specificity / AUROC,
macro averages, the sklearn classification report, and the confusion matrix.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --config experiments/configs/default.yaml --ckpt experiments/results/best.ckpt
    uv run python scripts/evaluate.py --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from tb_classifier.config import load_config
from tb_classifier.data import build_test_loader
from tb_classifier.training import TBLitModule
from tb_classifier.utils.metrics import CLASS_NAMES


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/configs/default.yaml",
        help="Path to the YAML run config (used for data + model shape).",
    )
    parser.add_argument(
        "--ckpt",
        default="experiments/results/best.ckpt",
        help="Path to the .ckpt file to evaluate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on: 'auto' | 'cpu' | 'cuda' | 'mps'.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _pick_device(args.device)
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    print(f"config: {args.config}")
    print(f"ckpt:   {ckpt_path}")
    print(f"device: {device}")

    test_loader = build_test_loader(cfg.data)
    print(
        f"test set: {len(test_loader.dataset)} images, "
        f"class counts: {test_loader.dataset.class_counts()}"
    )

    module = TBLitModule.load_from_checkpoint(
        str(ckpt_path),
        model_cfg=cfg.model,
        training_cfg=cfg.training,
        map_location=device,
    )
    module.eval().to(device)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in test_loader:
            logits = module(batch["image"].to(device, non_blocking=True))
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"])

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    n_classes = len(CLASS_NAMES)
    labels_onehot = np.eye(n_classes)[labels]

    print("\n=== Per-class ===")
    print(f"{'class':<14} {'sens':>8} {'spec':>8} {'auroc':>8} {'support':>8}")
    sens_all, spec_all, auc_all = [], [], []
    for i, name in enumerate(CLASS_NAMES):
        cls_pred = preds == i
        cls_true = labels == i
        tp = int((cls_pred & cls_true).sum())
        fp = int((cls_pred & ~cls_true).sum())
        fn = int((~cls_pred & cls_true).sum())
        tn = int((~cls_pred & ~cls_true).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        auc = roc_auc_score(labels_onehot[:, i], probs[:, i])
        sens_all.append(sens)
        spec_all.append(spec)
        auc_all.append(auc)
        print(f"{name:<14} {sens:8.4f} {spec:8.4f} {auc:8.4f} {int(cls_true.sum()):8d}")

    macro_auroc = roc_auc_score(labels_onehot, probs, multi_class="ovr", average="macro")
    print(
        f"{'macro':<14} {np.mean(sens_all):8.4f} {np.mean(spec_all):8.4f} "
        f"{macro_auroc:8.4f} {len(labels):8d}"
    )
    print(f"\naccuracy: {(preds == labels).mean():.4f}")

    print("\n=== sklearn classification report ===")
    print(classification_report(labels, preds, target_names=list(CLASS_NAMES), digits=4))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    header = " " * 14 + "  ".join(f"{n:>10s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{c:>10d}" for c in cm[i])
        print(f"{name:<14}{row}")


if __name__ == "__main__":
    main()
