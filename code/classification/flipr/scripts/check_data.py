"""Smoke test for the TB dataloaders, driven by a YAML config.

Usage:
    uv run python scripts/check_data.py
    uv run python scripts/check_data.py --config experiments/configs/default.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter

from tb_classifier.config import load_config
from tb_classifier.data import TBDataset, build_dataloaders, build_test_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/configs/default.yaml",
        help="Path to the YAML run config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"loaded config: {args.config}  (name={cfg.name})")
    print(f"  data: {cfg.data}")

    train_ds = TBDataset(cfg.data.root, split="train")
    val_ds = TBDataset(cfg.data.root, split="val")
    test_ds = TBDataset(cfg.data.root, split="test")
    print(f"  train records: {len(train_ds):>5}   class counts: {train_ds.class_counts()}")
    print(f"  val   records: {len(val_ds):>5}   class counts: {val_ds.class_counts()}")
    print(f"  test  records: {len(test_ds):>5}   class counts: {test_ds.class_counts()}")

    train_loader, val_loader = build_dataloaders(cfg.data)
    test_loader = build_test_loader(cfg.data)

    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        batch = next(iter(loader))
        img = batch["image"]
        label_counts = Counter(batch["label"].tolist())
        print(f"\n[{split}] batch:")
        print(f"  image tensor: shape={tuple(img.shape)} dtype={img.dtype} "
              f"min={img.min():.3f} max={img.max():.3f}")
        print(f"  labels: {batch['label'].tolist()}  histogram={dict(label_counts)}")
        print(f"  sample file: {batch['file_name'][0]}")


if __name__ == "__main__":
    main()
