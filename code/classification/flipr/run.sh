#!/usr/bin/env bash
# Reproduce FlipR (image classification). Standalone PyTorch Lightning module.
# Reads the dataset from ../../../dataset/classification via experiments/configs/default.yaml.
# Trained weights are not distributed, so this retrains from scratch (seed 42, 100 epochs).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

uv run python scripts/train.py --config experiments/configs/default.yaml

echo
echo "Training done. Evaluate the best checkpoint written under experiments/results/:"
echo "  uv run python scripts/evaluate.py --config experiments/configs/default.yaml --ckpt experiments/results/<run>/checkpoints/best.ckpt"
