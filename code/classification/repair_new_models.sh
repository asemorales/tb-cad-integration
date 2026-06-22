#!/usr/bin/env bash
# Repair job: the overnight queue's lighttbnet training hit a CUDA OOM startup race
# (it began before the just-finished detector released GPU memory). This waits until
# the queue's GPU work is fully done, retrains any new-model checkpoint that is
# missing, runs its MC/SZ inference, and re-summarizes. Idempotent: skips models
# that already have a checkpoint and sources that aren't staged.
#
#   nohup bash code/classification/repair_new_models.sh > code/classification/repair_new_models.log 2>&1 &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MLX_DIR="$REPO/code/_mlx"
CLS="$REPO/code/classification"
EXTERNAL="$REPO/.tmp/external"
export TMPDIR=/home/ase/.cache/pip-tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# shellcheck disable=SC1091
source "$MLX_DIR/.venv/bin/activate"

log() { echo "[$(date '+%F %T')] $*"; }

# 1. Wait until the overnight queue has fully finished (no GPU contention).
log "waiting for overnight_queue.sh to finish..."
while pgrep -f "classification/overnight_queue.sh" >/dev/null 2>&1; do sleep 120; done
log "overnight queue gone."

# 2. Wait until the GPU is actually free (no process holding /dev/nvidia0).
log "waiting for GPU to be free..."
while fuser /dev/nvidia0 2>/dev/null | grep -q .; do sleep 30; done
log "GPU free."

MODELS=(lighttbnet mobilenetv3-small)

# 3. Retrain any new model missing a checkpoint.
for M in "${MODELS[@]}"; do
  if ls "$CLS/$M/results/"*.pth >/dev/null 2>&1; then
    log "$M already has a checkpoint, skipping training."
  else
    log "training $M ..."
    bash "$CLS/$M/run.sh" && log "$M trained." || log "$M training FAILED."
    while fuser /dev/nvidia0 2>/dev/null | grep -q .; do sleep 15; done
  fi
done

# 4. MC/SZ inference for the two new models on whatever sources are staged.
declare -A WEIGHT=( [lighttbnet]=lighttbnet.pth [mobilenetv3-small]=mobilenet_v3_small.pth )
declare -A MNAME=( [lighttbnet]=lighttbnet [mobilenetv3-small]=mobilenet_v3_small )
for M in "${MODELS[@]}"; do
  wpath="$CLS/$M/results/${WEIGHT[$M]}"
  [ -f "$wpath" ] || { log "no weight for $M, skipping inference."; continue; }
  for src in mc sz; do
    [ -n "$(ls -A "$EXTERNAL/$src/test/tb" 2>/dev/null)" ] || { log "$src not staged, skip $M/$src"; continue; }
    out="$CLS/$M/results/external/$src"
    mkdir -p "$out"
    if PYTHONPATH="$MLX_DIR" python "$CLS/who-eval/predict_mlx.py" \
        --model "${MNAME[$M]}" --weights "$wpath" \
        --dataset "$EXTERNAL/$src" --output "$out" --device cuda --batch-size 16; then
      python "$CLS/mc-sz-eval/mc_sz_metrics.py" "$out" "$src" || log "metrics FAILED $M/$src"
    else
      log "predict FAILED $M/$src"
    fi
  done
done

# 5. Re-summarize so the final table includes the repaired models.
log "re-summarizing..."
python "$CLS/mc-sz-eval/summarize_mc_sz.py"
log "repair complete."
