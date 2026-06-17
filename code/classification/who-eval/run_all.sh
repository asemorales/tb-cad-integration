#!/usr/bin/env bash
# M2 (WHO 2025 CAD operating-point) evaluation for all nine classifiers.
#
# For each model: dump per-image P(tb) on the test split (predictions.csv), then
# compute the WHO indices (who_metrics.csv): AUC, pAUC over 40-60% specificity,
# specificity at 90% sensitivity. Finally build the summary table + ROC figure.
#
# The eight torchvision/Drax models run through the mlx package (its venv); FlipR
# is a standalone PyTorch Lightning package (its own venv). Both write the same
# predictions.csv schema so who_metrics.py treats them identically.
#
#   bash code/classification/who-eval/run_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$CLS/../.." && pwd)"
DATA="$REPO/dataset/classification"

MLX_DIR="$REPO/code/_mlx"
MLX_PY="$MLX_DIR/.venv/bin/python"
FLIPR_DIR="$CLS/flipr"
FLIPR_PY="$FLIPR_DIR/.venv/bin/python"

DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-16}"

# folder|mlx_model_name|weight_filename for the eight mlx classifiers.
MLX_MODELS=(
  "efficientnet-b0|efficientnet_b0|efficientnet_b0.pth"
  "mobilenetv3-large|mobilenet_v3_large|mobilenet_v3_large.pth"
  "densenet121|densenet121|densenet121.pth"
  "resnet18|resnet18|resnet18.pth"
  "resnet50|resnet50|resnet50.pth"
  "convnext-tiny|convnext_tiny|convnext_tiny.pth"
  "draxnet|draxnet|draxnet.pth"
  "drax-mobilenetv3-large|drax_mobilenet_v3_large|drax_mobilenet_v3_large.pth"
)

echo "### mlx classifiers"
for entry in "${MLX_MODELS[@]}"; do
  IFS='|' read -r folder model weight <<<"$entry"
  results="$CLS/$folder/results"
  PYTHONPATH="$MLX_DIR" "$MLX_PY" "$HERE/predict_mlx.py" \
    --model "$model" \
    --weights "$results/$weight" \
    --dataset "$DATA" \
    --output "$results" \
    --device "$DEVICE" --batch-size "$BATCH"
  "$MLX_PY" "$HERE/who_metrics.py" "$results"
done

echo "### FlipR"
( cd "$FLIPR_DIR" && "$FLIPR_PY" "$HERE/predict_flipr.py" \
    --config experiments/configs/default.yaml \
    --ckpt experiments/results/best.ckpt \
    --dataset "$DATA" \
    --output "$FLIPR_DIR/results" \
    --device "$DEVICE" )
"$MLX_PY" "$HERE/who_metrics.py" "$FLIPR_DIR/results"

echo "### summary"
( cd "$HERE" && "$MLX_PY" summarize.py )
