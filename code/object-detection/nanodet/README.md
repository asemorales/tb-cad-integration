# NanoDet-Plus-m (320) (non-YOLO lightweight detector)

The lightest non-YOLO detector in the comparison: an anchor-free, FCOS-style
detector (ShuffleNetV2 backbone, GhostPAN neck, generalized-focal-loss head),
~1.2M params. It is the "even lighter than YOLO26" point and a second non-YOLO
family alongside SSDLite.

## What this is and is not

Same disclosure as SSDLite: a representative cross-family baseline, reported with
its settings, NOT a controlled head-to-head. Different framework, resolution
(320), optimizer, and label assignment. Never present a gap as a tuned
comparison.

## Why a separate venv

NanoDet pins an older torch / pytorch-lightning stack and would break
`code/_mlx/.venv` (which the YOLO and SSDLite runs depend on). `run.sh` builds an
isolated `.venv-nanodet`. This is the higher-friction model: upstream NanoDet and
recent torch/lightning versions drift, so installs can fail. If they do, pin
known-good versions in the venv:

```bash
source .venv-nanodet/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu118 torch==1.13.1 torchvision==0.14.1
pip install pytorch-lightning==1.9.5
pip install -e nanodet-src
```

(torch 1.13 cu118 still includes sm_61 / Pascal, so it drives the 1050 Ti.)

## Files

| File | Role |
| --- | --- |
| `yolo_to_coco.py` | YOLO labels -> COCO JSON (train/val), backgrounds kept as negatives |
| `config/nanodet_tb.yml` | NanoDet-Plus-m config retargeted to nc=2 + this dataset |
| `run.sh` | venv + clone + convert + train (resumable via NanoDet's own checkpointing) |

Edit the absolute paths in `config/nanodet_tb.yml` (`data.*.img_path`,
`ann_path`, `save_dir`) if the checkout location differs.

## Run

```bash
cd code/object-detection/nanodet && nohup ./run.sh > train_nanodet.log 2>&1 &
```

## Re-eval (params, MACs, metrics)

NanoDet ships its own tools, used so the numbers match upstream conventions:

```bash
source .venv-nanodet/bin/activate
# params + MACs at 320 (FLOPs reported; MACs = FLOPs/2 to match our convention)
python nanodet-src/tools/flops.py --config config/nanodet_tb.yml --input_shape 3,320,320
# COCO mAP@0.5 and mAP@0.5:0.95 on val
python nanodet-src/tools/test.py --task val --config config/nanodet_tb.yml \
    --model workspace/nanodet_tb/model_best/model_best.ckpt
```

For the CPU latency / peak-RSS / disk row in the manuscript's on-device table,
mirror `code/object-detection/ssdlite/complexity.py`: build the NanoDet model from
the config on CPU, time a single 320x320 forward at 1 and 6 threads, read VmHWM,
and read the `.ckpt` size. Record `input_res=320`.
