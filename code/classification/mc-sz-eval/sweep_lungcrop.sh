#!/usr/bin/env bash
# Lung-crop OOD sweep: run each classifier on the pre-cropped external sets
# (.tmp/external/<src>_lungcrop, staged once by crop_lungs.py) using the UNCHANGED
# predict_mlx.py path, so preprocessing is identical to the zero-shot sweep and the
# only difference is the lung-field crop. CPU. Excludes lighttbnet + FlipR.
#
#   nohup bash code/classification/mc-sz-eval/sweep_lungcrop.sh > code/classification/mc-sz-eval/sweep_lungcrop.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$CLS/../.." && pwd)"
WHO="$CLS/who-eval"
MLX_DIR="$REPO/code/_mlx"
PY="$MLX_DIR/.venv/bin/python"
EXT="$REPO/.tmp/external"
# shellcheck disable=SC1091
source "$MLX_DIR/.venv/bin/activate"

MODELS=(
  "efficientnet-b0|efficientnet_b0"
  "mobilenetv3-large|mobilenet_v3_large"
  "mobilenetv3-small|mobilenet_v3_small"
  "densenet121|densenet121"
  "resnet18|resnet18"
  "resnet50|resnet50"
  "convnext-tiny|convnext_tiny"
  "draxnet|draxnet"
  "drax-mobilenetv3-large|drax_mobilenet_v3_large"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r folder model <<<"$entry"
  wpath="$CLS/$folder/results/$model.pth"
  [ -f "$wpath" ] || { echo "SKIP $folder (no weights)"; continue; }
  for src in mc sz; do
    ds="$EXT/${src}_lungcrop"
    [ -d "$ds/test" ] || { echo "SKIP $folder/$src (no crops at $ds)"; continue; }
    out="$CLS/$folder/results/external_lungcrop/$src"
    mkdir -p "$out"
    PYTHONPATH="$MLX_DIR" "$PY" "$WHO/predict_mlx.py" \
      --model "$model" --weights "$wpath" \
      --dataset "$ds" --output "$out" --device cpu --batch-size 16 \
      && "$PY" "$HERE/mc_sz_metrics.py" "$out" "$src" \
      || echo "  FAILED $folder/$src"
  done
done
echo "lung-crop sweep done."
