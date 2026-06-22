#!/usr/bin/env bash
# Montgomery (MC) + Shenzhen (SZ) out-of-distribution inference for every classifier.
#
# Zero-shot: each model trained on the 3-class TBX11K-derived set is run on the two
# external binary CXR sets. Score = P(tb) from the 3-class softmax, normal pooled as
# negative (the same binary-TB framing as the WHO eval). For each (model, source) we
# dump predictions.csv (reusing who-eval's predict_mlx.py / predict_flipr.py
# unchanged) then compute binary ACC + AUC (mc_sz_metrics.py). Finally summarize.
#
#   EXTERNAL=/path/to/external bash code/classification/mc-sz-eval/run_all.sh
# EXTERNAL must contain mc/test/{healthy,tb} and sz/test/{healthy,tb} (see stage_mc_sz.py).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLS="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$CLS/../.." && pwd)"
WHO="$CLS/who-eval"

EXTERNAL="${EXTERNAL:-$REPO/.tmp/external}"
MLX_DIR="$REPO/code/_mlx"
MLX_PY="$MLX_DIR/.venv/bin/python"
FLIPR_DIR="$CLS/flipr"
FLIPR_PY="$FLIPR_DIR/.venv/bin/python"
DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-16}"

SOURCES=()
for s in mc sz; do
  if [ -n "$(ls -A "$EXTERNAL/$s/test/tb" 2>/dev/null)" ]; then
    SOURCES+=("$s")
  else
    echo "  NOTE: $s not staged (no $EXTERNAL/$s/test/tb), skipping it."
  fi
done
if [ "${#SOURCES[@]}" -eq 0 ]; then
  echo "No external sources staged; nothing to sweep."
  exit 0
fi
echo "Sweeping sources: ${SOURCES[*]}"

# folder|mlx_model_name|weight_filename
MLX_MODELS=(
  "efficientnet-b0|efficientnet_b0|efficientnet_b0.pth"
  "mobilenetv3-large|mobilenet_v3_large|mobilenet_v3_large.pth"
  "mobilenetv3-small|mobilenet_v3_small|mobilenet_v3_small.pth"
  "densenet121|densenet121|densenet121.pth"
  "resnet18|resnet18|resnet18.pth"
  "resnet50|resnet50|resnet50.pth"
  "convnext-tiny|convnext_tiny|convnext_tiny.pth"
  "draxnet|draxnet|draxnet.pth"
  "drax-mobilenetv3-large|drax_mobilenet_v3_large|drax_mobilenet_v3_large.pth"
  "lighttbnet|lighttbnet|lighttbnet.pth"
)

echo "### mlx classifiers (MC + SZ)"
for entry in "${MLX_MODELS[@]}"; do
  IFS='|' read -r folder model weight <<<"$entry"
  case " ${EXCLUDE:-} " in *" $folder "*) echo "  EXCLUDE $folder (skipped)"; continue;; esac
  wpath="$CLS/$folder/results/$weight"
  if [ ! -f "$wpath" ]; then
    echo "  SKIP $folder: weight not found ($wpath)"
    continue
  fi
  for src in "${SOURCES[@]}"; do
    out="$CLS/$folder/results/external/$src"
    mkdir -p "$out"
    if PYTHONPATH="$MLX_DIR" "$MLX_PY" "$WHO/predict_mlx.py" \
        --model "$model" --weights "$wpath" \
        --dataset "$EXTERNAL/$src" --output "$out" \
        --device "$DEVICE" --batch-size "$BATCH"; then
      "$MLX_PY" "$HERE/mc_sz_metrics.py" "$out" "$src" || echo "  metrics FAILED: $folder/$src"
    else
      echo "  predict FAILED: $folder/$src"
    fi
  done
done

echo "### FlipR (MC + SZ)"
if [ -f "$FLIPR_DIR/experiments/results/best.ckpt" ]; then
  for src in "${SOURCES[@]}"; do
    out="$FLIPR_DIR/results/external/$src"
    mkdir -p "$out"
    if ( cd "$FLIPR_DIR" && "$FLIPR_PY" "$WHO/predict_flipr.py" \
        --config experiments/configs/default.yaml \
        --ckpt experiments/results/best.ckpt \
        --dataset "$EXTERNAL/$src" --output "$out" --device "$DEVICE" ); then
      "$MLX_PY" "$HERE/mc_sz_metrics.py" "$out" "$src" || echo "  metrics FAILED: flipr/$src"
    else
      echo "  predict FAILED: flipr/$src"
    fi
  done
else
  echo "  SKIP FlipR: checkpoint not found"
fi

echo "### summary"
"$MLX_PY" "$HERE/summarize_mc_sz.py"
