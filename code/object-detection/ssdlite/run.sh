#!/usr/bin/env bash
# Train + evaluate SSDLite320-MobileNetV3-Large (non-YOLO lightweight detector).
# Reuses the shared mlx venv (torchvision is already installed there). torchmetrics
# + pycocotools are needed for COCO mAP; install once if missing (see README).
#
# Resumable: re-running picks up from results/weights/last.pt.
#
# Usage (detached):  nohup ./run.sh > train_ssdlite.log 2>&1 &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$HERE/../../.."

source "$REPO/code/_mlx/.venv/bin/activate"
export TMPDIR="${TMPDIR:-/home/ase/.cache/pip-tmp}"

cd "$HERE"
python train.py --epochs 100 --batch-size 16 --seed 42 --device cuda --output "$HERE/results"

# Final metrics + complexity/benchmark row (params, MACs, CPU latency, disk).
python complexity.py
echo "SSDLite done. Best checkpoint: results/weights/best.pt"
