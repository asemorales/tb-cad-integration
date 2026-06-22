#!/usr/bin/env bash
# Overnight job chain (autonomous). Order:
#   1. Wait for the detector retrain to release the 4 GB GPU.
#   2. Train LightTBNet (N=4) then MobileNetV3-small on the 3-class set (GPU, sequential).
#   3. Wait for the MC/SZ downloads, then stage them as binary test dirs.
#   4. Run the MC/SZ out-of-distribution inference sweep over every classifier.
#
# Launch detached:
#   nohup bash code/classification/overnight_queue.sh > code/classification/overnight_queue.log 2>&1 &
set -uo pipefail   # NOT -e: a single model failure must not abort the overnight chain.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MLX_DIR="$REPO/code/_mlx"
CLS="$REPO/code/classification"
DET="$REPO/code/object-detection"
EXTERNAL="$REPO/.tmp/external"
export TMPDIR=/home/ase/.cache/pip-tmp

log() { echo "[$(date '+%F %T')] $*"; }

# shellcheck disable=SC1091
source "$MLX_DIR/.venv/bin/activate"

# ---- 1. wait for the detector retrain (GPU) -------------------------------------
log "Phase 1: waiting for detector retrain to finish (frees the GPU)..."
WAITED=0
while :; do
  det_done=0
  [ -f "$DET/yolo11n/results/weights/best.pt" ] && det_done=1
  running=0
  pgrep -f "mlx --mode object_detection" >/dev/null 2>&1 && running=1
  if [ "$det_done" = 1 ] && [ "$running" = 0 ]; then
    log "Detector retrain complete (yolo11n best.pt present, no detector process)."
    break
  fi
  if [ "$WAITED" -ge 64800 ]; then   # 18h safety cap
    log "WARNING: detector wait hit 18h cap; proceeding anyway."
    break
  fi
  sleep 120; WAITED=$((WAITED + 120))
done

# ---- 2. train the two new classifiers (GPU, sequential) ------------------------
log "Phase 2: training new classifiers..."
for M in lighttbnet mobilenetv3-small; do
  if ls "$CLS/$M/results/"*.pth >/dev/null 2>&1; then
    log "  $M already has a checkpoint, skipping training."
    continue
  fi
  log "  training $M ..."
  bash "$CLS/$M/run.sh" && log "  $M done." || log "  $M FAILED (see above)."
done

# ---- 3. wait for downloads + stage whatever MC/SZ is complete ------------------
log "Phase 3: ensuring MC/SZ external data is staged (degrades to whatever's ready)..."
MC_ZIP="$EXTERNAL/NLM-MontgomeryCXRSet.zip"
SZ_ZIP="$EXTERNAL/ChinaSet_AllFiles.zip"
MC_BYTES=616853875
SZ_BYTES=3770205534
sz_of() { stat -c%s "$1" 2>/dev/null || echo 0; }
staged() { [ -n "$(ls -A "$EXTERNAL/$1/test/tb" 2>/dev/null)" ]; }

WAITED=0
while :; do
  # Stage any complete-but-not-yet-staged set.
  if [ "$(sz_of "$MC_ZIP")" = "$MC_BYTES" ] && ! staged mc; then
    log "  Montgomery zip complete; staging..."; python "$CLS/mc-sz-eval/stage_mc_sz.py" "$EXTERNAL" || true
  fi
  if [ "$(sz_of "$SZ_ZIP")" = "$SZ_BYTES" ] && ! staged sz; then
    log "  Shenzhen zip complete; staging..."; python "$CLS/mc-sz-eval/stage_mc_sz.py" "$EXTERNAL" || true
  fi
  # Proceed once both are staged, or once at least one is staged and the 12h cap is hit.
  if staged mc && staged sz; then
    log "  both MC and SZ staged."; break
  fi
  if [ "$WAITED" -ge 64800 ]; then
    if staged mc || staged sz; then
      log "  18h cap: proceeding with partial data (mc=$(staged mc && echo yes || echo no), sz=$(staged sz && echo yes || echo no)); the other can be swept later."
    else
      log "  12h cap and NO external data staged (mc=$(sz_of "$MC_ZIP")/$MC_BYTES sz=$(sz_of "$SZ_ZIP")/$SZ_BYTES). Skipping sweep."
      log "Overnight queue finished early (no external data)."
      exit 0
    fi
    break
  fi
  sleep 120; WAITED=$((WAITED + 120))
done

# ---- 4. MC/SZ inference sweep (run_all.sh skips any source not staged) ----------
log "Phase 4: running MC/SZ inference sweep (ACC + AUC) over all classifiers..."
EXTERNAL="$EXTERNAL" DEVICE="${DEVICE:-cuda}" BATCH="${BATCH:-16}" bash "$CLS/mc-sz-eval/run_all.sh"

log "Overnight queue complete. Summary: $CLS/mc-sz-eval/mc_sz_summary.csv"
