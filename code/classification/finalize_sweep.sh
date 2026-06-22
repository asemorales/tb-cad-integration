#!/usr/bin/env bash
# Finalize: run the MC/SZ sweep over ALL classifiers once it is safe to do so.
# Supersedes the overnight queue's Phase 3/4 (kill that queue before launching this).
# Waits for: (1) lighttbnet training to finish (so its checkpoint is the final
# best-of-100, and the GPU is free), and (2) the Shenzhen download to complete.
# Then stages MC/SZ and runs the full sweep + summary. No GPU contention.
#
#   nohup bash code/classification/finalize_sweep.sh > code/classification/finalize_sweep.log 2>&1 &
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

# 1. Wait for lighttbnet training to finish (frees GPU, finalizes its checkpoint).
log "waiting for lighttbnet training to finish..."
while pgrep -f "model lighttbnet" >/dev/null 2>&1; do sleep 120; done
log "lighttbnet training done (checkpoint: $(ls "$CLS"/lighttbnet/results/*.pth 2>/dev/null || echo MISSING))."

# 2. Wait for the Shenzhen download to complete (Montgomery is already done).
SZ_ZIP="$EXTERNAL/ChinaSet_AllFiles.zip"; SZ_BYTES=3770205534
WAITED=0
while [ "$(stat -c%s "$SZ_ZIP" 2>/dev/null || echo 0)" != "$SZ_BYTES" ]; do
  if [ "$WAITED" -ge 64800 ]; then
    log "WARNING: SZ still incomplete after 18h ($(stat -c%s "$SZ_ZIP" 2>/dev/null||echo 0)/$SZ_BYTES); proceeding with whatever is staged (likely MC only)."
    break
  fi
  sleep 120; WAITED=$((WAITED + 120))
done

# 3. Stage whatever is byte-complete (idempotent; skips already-staged sets).
log "staging external data..."
python "$CLS/mc-sz-eval/stage_mc_sz.py" "$EXTERNAL" || log "staging had issues (continuing)."

# 4. Ensure GPU is free, then run the full sweep + summary.
while fuser /dev/nvidia0 2>/dev/null | grep -q .; do sleep 30; done
log "running full MC/SZ sweep over all classifiers..."
EXTERNAL="$EXTERNAL" DEVICE="${DEVICE:-cuda}" BATCH="${BATCH:-16}" bash "$CLS/mc-sz-eval/run_all.sh"
log "finalize complete. Summary: $CLS/mc-sz-eval/mc_sz_summary.csv"
