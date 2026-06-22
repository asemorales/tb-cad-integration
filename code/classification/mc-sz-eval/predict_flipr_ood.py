"""FlipR external-validation predictor: zero-shot, AdaBN, or lung-crop, in one path.

FlipR lives in a separate framework (the `tb_classifier` Lightning package), so the
mlx AdaBN/crop sweeps cannot load it. This script runs the same corrections natively
on the FlipR model:

  * default      -- plain forward pass (zero-shot, or lung-crop if --dataset points at
                    a pre-cropped set, matching how the mlx models did crop).
  * --adabn      -- recompute every BatchNorm layer's running stats on the unlabeled
                    target images, then predict. FlipR's truncated ResN18 backbone has
                    15 BatchNorm layers, so AdaBN applies exactly as for the resnet
                    baselines (Li et al., 2016).

Checkpoint loading is robust to both forms the repo has used: a real Lightning
checkpoint ({"state_dict": {"model.backbone..."}}) and the bare state_dict
({"backbone...": tensor}) that `experiments/results/best.ckpt` currently holds.

Writes <output>/predictions.csv with the shared schema
(label,p_healthy,p_sick_non_tb,p_tb) so mc_sz_metrics.py consumes it unchanged.

Run inside the FlipR venv from the flipr package directory:
    cd code/classification/flipr
    .venv/bin/python ../mc-sz-eval/predict_flipr_ood.py \
        --config experiments/configs/default.yaml \
        --ckpt experiments/results/best.ckpt \
        --dataset /abs/path/to/external/mc --output results/external_adabn/mc --adabn
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from tb_classifier.config import load_config
from tb_classifier.data import build_test_loader
from tb_classifier.models import build_classifier
from tb_classifier.utils.metrics import CLASS_NAMES


def _slug(name: str) -> str:
    return name.replace("-", "_")


def _load_state_dict(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load weights from either a Lightning checkpoint or a bare state_dict."""
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    # Lightning wraps the classifier as `self.model`, so keys are `model.backbone...`.
    # The bare state_dict already starts at `backbone...`. Strip the wrapper if present.
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")


def adapt_batchnorm(model, loader, device) -> int:
    """Recompute BN running stats over the (unlabeled) target loader. Returns #BN layers."""
    bn_layers = [m for m in model.modules()
                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        return 0
    model.eval()
    for m in bn_layers:
        m.reset_running_stats()  # clear source-domain stats
        m.momentum = None        # cumulative moving average over all target batches
        m.train()                # enable running-stat updates for BN only
    with torch.no_grad():
        for batch in loader:
            model(batch["image"].to(device))
    model.eval()
    return len(bn_layers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/default.yaml")
    parser.add_argument("--ckpt", default="experiments/results/best.ckpt")
    parser.add_argument("--dataset", default=None, help="override dataset root (absolute path)")
    parser.add_argument("--output", default="results", help="directory to write predictions.csv")
    parser.add_argument("--adabn", action="store_true", help="adapt BatchNorm stats to target")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.data.root = args.dataset
    device = torch.device(args.device)

    model = build_classifier(cfg.model).to(device)
    _load_state_dict(model, args.ckpt)
    model.eval()

    test_loader = build_test_loader(cfg.data)

    tag = "zero-shot"
    if args.adabn:
        n_bn = adapt_batchnorm(model, test_loader, device)
        tag = f"AdaBN({n_bn} BN layers)"

    rows: list[tuple] = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["image"].to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for prob, label in zip(probs, batch["label"].tolist()):
                rows.append((int(label), *(float(value) for value in prob)))

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.output) / "predictions.csv"
    header = ["label", *(f"p_{_slug(name)}" for name in CLASS_NAMES)]
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[0], *(f"{value:.6f}" for value in row[1:])])

    print(f"{'flipr':>26}  {tag}  wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
