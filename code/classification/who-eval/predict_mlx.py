"""Dump per-image P(tb) for one mlx classifier onto the test split.

Reuses the exact mlx benchmark loading path (``load_checkpoint_bundle`` +
``load_standard_classification_directory``), so the preprocessing matches how
each model was trained and benchmarked. The only addition over ``--action
benchmark`` is that per-image softmax probabilities are written to disk, which
the WHO operating-point metrics need and the standard benchmark does not save.

Writes ``<output>/predictions.csv`` with the common schema:
    label,p_healthy,p_sick_non_tb,p_tb

Run inside the mlx venv (it imports the mlx package):
    code/_mlx/.venv/bin/python predict_mlx.py \
        --model resnet18 \
        --weights ../resnet18/results/resnet18.pth \
        --dataset ../../../dataset/classification \
        --output ../resnet18/results
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="mlx model name, e.g. resnet18")
    parser.add_argument("--weights", required=True, help="path to the .pth checkpoint")
    parser.add_argument("--dataset", required=True, help="classification dataset root (with test/)")
    parser.add_argument("--output", required=True, help="directory to write predictions.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = {
        "model": args.model,
        "model_path": args.weights,
        "dataset_path": args.dataset,
        "device": args.device,
        "batch_size": args.batch_size,
    }

    model, metadata = load_checkpoint_bundle(config)
    model = model.to(args.device).eval()
    classes = metadata["classes"]

    eval_dir = resolve_evaluation_dir(args.dataset)
    dataset = load_standard_classification_directory(
        eval_dir,
        label_names=classes,
        input_size=metadata["input_size"],
        colored=metadata["colored"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    rows: list[tuple] = []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(args.device)), dim=1).cpu().numpy()
            for prob, label in zip(probs, labels.tolist()):
                rows.append((int(label), *(float(value) for value in prob)))

    out_csv = Path(args.output) / "predictions.csv"
    header = ["label", *(f"p_{_slug(name)}" for name in classes)]
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[0], *(f"{value:.6f}" for value in row[1:])])

    print(f"{args.model:>26}  wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
