"""Dump per-image P(tb) for the FlipR classifier onto the test split.

Loads the trained Lightning checkpoint and runs the same val/test preprocessing
the model was trained with (``tb_classifier.data.build_test_loader``: longest-max
resize to 512, pad to square, ImageNet normalisation). Writes per-image softmax
to ``<output>/predictions.csv`` with the common schema shared with the mlx
models:
    label,p_healthy,p_sick_non_tb,p_tb

Run inside the FlipR venv from the flipr package directory:
    cd code/classification/flipr
    .venv/bin/python ../who-eval/predict_flipr.py \
        --config experiments/configs/default.yaml \
        --ckpt experiments/results/best.ckpt \
        --output results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from tb_classifier.config import load_config
from tb_classifier.data import build_test_loader
from tb_classifier.training import TBLitModule
from tb_classifier.utils.metrics import CLASS_NAMES


def _slug(name: str) -> str:
    return name.replace("-", "_")


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/default.yaml")
    parser.add_argument("--ckpt", default="experiments/results/best.ckpt")
    parser.add_argument("--output", default="results", help="directory to write predictions.csv")
    parser.add_argument("--dataset", default=None, help="override dataset root (absolute path)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.data.root = args.dataset
    device = _pick_device(args.device)

    test_loader = build_test_loader(cfg.data)
    module = TBLitModule.load_from_checkpoint(
        args.ckpt,
        model_cfg=cfg.model,
        training_cfg=cfg.training,
        map_location=device,
    )
    module.eval().to(device)

    rows: list[tuple] = []
    with torch.inference_mode():
        for batch in test_loader:
            logits = module(batch["image"].to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for prob, label in zip(probs, batch["label"].tolist()):
                rows.append((int(label), *(float(value) for value in prob)))

    out_csv = Path(args.output) / "predictions.csv"
    header = ["label", *(f"p_{_slug(name)}" for name in CLASS_NAMES)]
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[0], *(f"{value:.6f}" for value in row[1:])])

    print(f"{'flipr':>26}  wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
