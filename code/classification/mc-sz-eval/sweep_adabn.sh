#!/usr/bin/env bash
# AdaBN OOD sweep: recompute BatchNorm stats on the unlabeled target set, then predict.
# CPU (GPU VRAM held by paused lighttbnet). Excludes lighttbnet (still training) and
# FlipR (separate framework). Writes results/external_adabn/<src>/predictions.csv and
# per-(model,src) mc_sz_metrics.csv, then a combined adabn summary via the rescored
# summarizer pointed at the adabn outputs.
#
#   nohup bash code/classification/mc-sz-eval/sweep_adabn.sh > code/classification/mc-sz-eval/sweep_adabn.log 2>&1 &
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
    out="$CLS/$folder/results/external_adabn/$src"
    PYTHONPATH="$MLX_DIR" "$PY" "$HERE/predict_mlx_adabn.py" \
      --model "$model" --weights "$wpath" \
      --dataset "$EXT/$src" --output "$out" --device cpu --batch-size 16 \
      && "$PY" "$HERE/mc_sz_metrics.py" "$out" "$src" \
      || echo "  FAILED $folder/$src"
  done
done
echo "AdaBN sweep done."
