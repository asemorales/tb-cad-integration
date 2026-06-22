#!/usr/bin/env bash
# Pause lighttbnet training (true OS SIGSTOP, zero progress lost), run the MC+SZ
# sweep for all OTHER classifiers on CPU (GPU VRAM is still held by the paused
# lighttbnet, so the sweep must be CPU; SIGSTOP frees the CPU it was using), then
# SIGCONT to resume lighttbnet exactly where it left off.
#
#   nohup bash code/classification/pause_sweep_resume.sh > code/classification/pause_sweep_resume.log 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MLX_DIR="$REPO/code/_mlx"
CLS="$REPO/code/classification"
EXTERNAL="$REPO/.tmp/external"
export TMPDIR=/home/ase/.cache/pip-tmp
# shellcheck disable=SC1091
source "$MLX_DIR/.venv/bin/activate"
log() { echo "[$(date '+%F %T')] $*"; }

MAIN=$(pgrep -f "model lighttbnet" | head -1)
if [ -z "$MAIN" ]; then log "lighttbnet not running; sweeping anyway (will include nothing to pause)."; PIDS=""; else
  PIDS="$MAIN $(pgrep -P "$MAIN" | tr '\n' ' ')"
fi

resume() {
  if [ -n "$PIDS" ]; then
    kill -CONT $PIDS 2>/dev/null && log "RESUMED lighttbnet ($PIDS)."
  fi
}
trap resume EXIT INT TERM

if [ -n "$PIDS" ]; then
  log "PAUSING lighttbnet (SIGSTOP $PIDS), epoch=$(tail -1 "$CLS"/lighttbnet/results/training.csv 2>/dev/null|cut -d, -f1)"
  kill -STOP $PIDS 2>/dev/null
fi

log "running CPU sweep for all classifiers except lighttbnet (MC + SZ)..."
EXCLUDE="lighttbnet" EXTERNAL="$EXTERNAL" DEVICE="cpu" BATCH="16" bash "$CLS/mc-sz-eval/run_all.sh"

log "CPU sweep done; resuming lighttbnet."
resume
trap - EXIT INT TERM
log "pause/sweep/resume complete. Partial summary: $CLS/mc-sz-eval/mc_sz_summary.csv"
