#!/usr/bin/env bash
# Set up and train NanoDet-Plus-m (320) on the TB detection set.
#
# NanoDet lives in its own repo with its own dependency stack, so it gets its own
# venv (do NOT reuse code/_mlx/.venv: NanoDet pins older torch/pytorch-lightning
# and would break the shared engine). This script:
#   1. creates a NanoDet venv,
#   2. clones the upstream repo,
#   3. converts the YOLO labels to COCO JSON,
#   4. trains from scratch (100 epochs, seed handled by NanoDet config).
#
# Higher dependency risk than SSDLite: NanoDet + recent torch/lightning versions
# drift. If install fails, pin torch==1.13 / pytorch-lightning==1.9 in the venv
# (see README). Usage:  nohup ./run.sh > train_nanodet.log 2>&1 &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv-nanodet"
SRC="$HERE/nanodet-src"

# 1. venv
if [ ! -d "$VENV" ]; then
  python3.11 -m venv "$VENV"
fi
source "$VENV/bin/activate"
export TMPDIR="${TMPDIR:-/home/ase/.cache/pip-tmp}"
pip install --upgrade pip

# 2. clone + install NanoDet (CUDA 11.8 wheels match the 580/sm_61 driver)
if [ ! -d "$SRC" ]; then
  git clone https://github.com/RangiLyu/nanodet.git "$SRC"
fi
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
pip install pytorch-lightning pycocotools opencv-python-headless tensorboard pyaml termcolor
pip install -e "$SRC"

# 3. YOLO -> COCO
python "$HERE/yolo_to_coco.py" --root "$HERE/../../../dataset/object-detection" --out "$HERE/annotations"

# 4. train from scratch
cd "$SRC"
python tools/train.py "$HERE/config/nanodet_tb.yml"

echo "NanoDet training launched/finished. Best checkpoint under workspace/nanodet_tb/."
echo "Params/MACs:   python $SRC/tools/flops.py --config $HERE/config/nanodet_tb.yml --input_shape 3,320,320"
echo "Eval mAP:      python $SRC/tools/test.py --task val --config $HERE/config/nanodet_tb.yml --model <best.ckpt>"
