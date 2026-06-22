#!/usr/bin/env bash
# Reproduce YOLOv8n (object detection, Ultralytics via mlx).
# Trains from scratch (random init, no COCO pretraining). M2 FIX: matched to the
# committed yolo26 / draxnet-yolo26 runs at 256x256, seed 0, so the four-detector
# YOLO cohort shares one resolution and seed (removes the resolution+seed confound).
# 100 epochs, batch 16. The bare model name resolves to yolov8.yaml at nano scale
# (Ultralytics "Assuming scale='n'"), matching the yolo26 convention.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX="$HERE/../../_mlx"
DATA_YAML="$HERE/../../../dataset/object-detection/dataset.yaml"
OUT="$HERE"
MODEL="yolov8n"

cd "$MLX"
# Ultralytics saves to <project>/<name>; project=$HERE and name=results lands the
# run artifacts directly in yolov8n/results/ (flat, matching the yolo26 layout).
python -m mlx --mode object_detection --action train \
  --dataset "$DATA_YAML" --model "$MODEL" --output "$OUT" \
  --run-name results --epochs 100 --batch-size 16 \
  --height 256 --width 256 --device cuda --seed 0 --cache
