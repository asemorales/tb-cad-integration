#!/usr/bin/env bash
# LightTBNet, N=4 residual blocks (Capellan-Martin et al., ISBI 2023, arXiv:2309.02140).
# Retrained under THIS paper's unified baseline recipe (not LightTBNet's native
# 256px/grayscale/binary/focal-loss setup), so it is an apples-to-apples member of
# the 3-class classifier cohort: seed 42, 100 epochs, 512x512, batch 16, Adam
# lr 1e-3, unweighted cross-entropy, best-by-val-loss. This is the fair head-to-head
# the manuscript's "would require retraining under this paper's protocol" caveat calls for.
# Architecture is N=4 (paper's best val-AUC / low-compute config, ~1.5M params).
# NOTE: micro-batch 8 x grad-accum 2 = EFFECTIVE BATCH 16, identical optimizer
# batch to the rest of the cohort (fair comparison). LightTBNet has no early
# downsampling stem, so at 512x512 its first block runs full-resolution convs that
# exceed the 4 GB GPU at a true batch 16 (peak ~3.9 GB+); micro-batch 8 (peak ~3.1
# GB) with 2-step gradient accumulation reproduces the batch-16 gradient update.
# Everything else matches the cohort (512px, seed 42, Adam lr 1e-3, unweighted CE,
# 100 epochs, best-by-val-loss). Caveat: BatchNorm stats are computed over the
# size-8 micro-batch rather than the full 16 (a minor, disclosed difference).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX="$HERE/../../_mlx"
DATASET="$HERE/../../../dataset/classification"
OUT="$HERE/results"
MODEL="lighttbnet"

cd "$MLX"
python -m mlx --mode image_classification --action train \
  --dataset "$DATASET" --output "$OUT" --model "$MODEL" \
  --epochs 100 --batch-size 8 --grad-accum-steps 2 --height 512 --width 512 \
  --device cuda --seed 42

python -m mlx --mode image_classification --action benchmark \
  --dataset "$DATASET" --model "$MODEL" \
  --model-path "$OUT/$MODEL.pth" --output "$OUT" \
  --batch-size 16 --device cuda
