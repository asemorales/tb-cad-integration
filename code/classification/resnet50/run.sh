#!/usr/bin/env bash
# Reproduce ResNet-50 (image classification): train, then benchmark on the held-out test split.
# Captured from notebooks/3_class_TB_Image_Classification.ipynb (seed 42, 100 epochs, 512x512).
# Runs on the shared mlx engine in code/_mlx.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX="$HERE/../../_mlx"
DATASET="$HERE/../../../dataset/classification"
OUT="$HERE/results"
MODEL="resnet50"

cd "$MLX"
python -m mlx --mode image_classification --action train \
  --dataset "$DATASET" --output "$OUT" --model "$MODEL" \
  --epochs 100 --batch-size 16 --height 512 --width 512 \
  --device cuda --seed 42

python -m mlx --mode image_classification --action benchmark \
  --dataset "$DATASET" --model "$MODEL" \
  --model-path "$OUT/$MODEL.pth" --output "$OUT" \
  --batch-size 16 --device cuda
