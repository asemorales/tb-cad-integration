#!/usr/bin/env bash
# Reproduce DraxNet-YOLO26 (object detection, Ultralytics via mlx).
# Only epochs=100 is confirmed (from results/results.csv). batch/imgsz/lr reflect
# mlx + ultralytics defaults and were not separately logged for the paper run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX="$HERE/../../_mlx"
DATA_YAML="$HERE/../../../dataset/object-detection/dataset.yaml"
OUT="$HERE/results"
MODEL="draxnet-yolo26"

cd "$MLX"
python -m mlx --mode object_detection --action train \
  --dataset "$DATA_YAML" --model "$MODEL" --output "$OUT" \
  --run-name "$MODEL" --epochs 100 --batch-size 16 \
  --height 512 --width 512 --device cuda --seed 42
