#!/usr/bin/env bash
# Orchestrate the no-retrain OOD recovery experiments end to end:
#   1. wait for crop_lungs.py to finish staging lung crops
#   2. run the lung-crop inference sweep
#   3. wait for both the AdaBN and lung-crop sweeps to be idle
#   4. emit the unified comparison table (compare_conditions.py)
#
#   nohup bash code/classification/mc-sz-eval/finalize_recovery.sh > code/classification/mc-sz-eval/finalize_recovery.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="$REPO/code/_mlx/.venv/bin/python"
# shellcheck disable=SC1091
source "$REPO/code/_mlx/.venv/bin/activate"
log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for crop_lungs.py ..."
while pgrep -f crop_lungs.py >/dev/null 2>&1; do sleep 30; done
log "crops staged. launching lung-crop sweep."
bash "$HERE/sweep_lungcrop.sh" > "$HERE/sweep_lungcrop.log" 2>&1

log "waiting for AdaBN sweep to finish ..."
while pgrep -f sweep_adabn.sh >/dev/null 2>&1 || pgrep -f predict_mlx_adabn.py >/dev/null 2>&1; do sleep 30; done

log "building unified recovery comparison ..."
"$PY" "$HERE/compare_conditions.py"
log "finalize_recovery done. Table: $HERE/mc_sz_recovery.csv"
