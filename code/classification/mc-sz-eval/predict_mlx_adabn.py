"""AdaBN: adapt BatchNorm statistics to the target domain, then predict (no retraining).

Standard unsupervised domain adaptation (Li et al., 2016, "Revisiting Batch
Normalization For Practical Domain Adaptation"). We keep all learned weights frozen
and only recompute each BatchNorm layer's running mean/variance from the UNLABELED
target images (a label-free, backprop-free forward pass), then run inference. This
re-centers the internal feature distribution to the target scanner/site and can move
AUC (unlike thresholding). It is transductive: BN stats are estimated over the target
test images themselves, which is the accepted AdaBN protocol since no labels are used.

Models with no BatchNorm (e.g. convnext = LayerNorm) are unaffected; their output
equals the zero-shot prediction by construction.

Writes <output>/predictions.csv in the standard schema (label,p_healthy,p_sick_non_tb,p_tb)
so the existing metrics/summary scripts consume it unchanged. Mirrors predict_mlx.py
exactly except for the BN-adaptation pass.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mlx.modes.image_classification.data import (
    load_standard_classification_directory,
    resolve_evaluation_dir,
)
from mlx.modes.image_classification.utils import load_checkpoint_bundle


def _slug(name: str) -> str:
    return name.replace("-", "_")


def adapt_batchnorm(model, loader, device) -> int:
    """Recompute BN running stats over the (unlabeled) target loader. Returns #BN layers."""
    bn_layers = [m for m in model.modules()
                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        return 0
    model.eval()                 # dropout off; we only want BN to update
    for m in bn_layers:
        m.reset_running_stats()  # clear source-domain stats
        m.momentum = None        # cumulative moving average over all target batches
        m.train()                # enable running-stat updates for BN only
    with torch.no_grad():
        for images, _ in loader:
            model(images.to(device))
    model.eval()
    return len(bn_layers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True, help="external root containing test/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = {
        "model": args.model, "model_path": args.weights,
        "dataset_path": args.dataset, "device": args.device,
        "batch_size": args.batch_size,
    }
    model, metadata = load_checkpoint_bundle(config)
    model = model.to(args.device)
    classes = metadata["classes"]

    eval_dir = resolve_evaluation_dir(args.dataset)
    dataset = load_standard_classification_directory(
        eval_dir, label_names=classes,
        input_size=metadata["input_size"], colored=metadata["colored"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    n_bn = adapt_batchnorm(model, loader, args.device)

    rows = []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(args.device)), dim=1).cpu().numpy()
            for prob, label in zip(probs, labels.tolist()):
                rows.append((int(label), *(float(v) for v in prob)))

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.output) / "predictions.csv"
    header = ["label", *(f"p_{_slug(name)}" for name in classes)]
    with open(out_csv, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(header)
        for row in rows:
            w.writerow([row[0], *(f"{v:.6f}" for v in row[1:])])
    print(f"{args.model:>26}  AdaBN({n_bn} BN layers)  wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
