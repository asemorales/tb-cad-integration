# SSDLite320-MobileNetV3-Large (non-YOLO lightweight detector)

A cross-family lightweight detector added so the comparison is not YOLO-only.
SSDLite is a different paradigm from the YOLO cohort: an SSD multi-scale anchor
head on a MobileNetV3 backbone with depthwise-separable convolutions. Footprint
is in YOLO26's class (~2.2M params, vs yolo26 2.57M), with lower mult-adds
because SSDLite is deliberately MAC-frugal.

## What this is and is not

- It IS a representative non-YOLO lightweight baseline, reported alongside the
  YOLO detectors with its settings fully disclosed.
- It is NOT a controlled head-to-head with the YOLO cohort. It uses a different
  framework (torchvision), optimizer (SGD + cosine), anchor scheme, native input
  resolution (320, not 512), and mAP tooling (torchmetrics/pycocotools). Present
  any gap as "different family, disclosed settings," never as a tuned comparison.

## Files

| File | Role |
| --- | --- |
| `data.py` | YOLO-format -> torchvision target adapter (handles background images) |
| `transforms.py` | to-tensor + train-time horizontal flip |
| `train.py` | from-scratch training, 100 epochs, seed 42, best-by-val-mAP50 |
| `complexity.py` | params, MACs, CPU latency/RSS, disk -> `results/ondevice_benchmark_ssdlite.csv` |
| `config.yaml` | the recorded settings |
| `run.sh` | train then benchmark, resumable |

## Environment

Reuses `code/_mlx/.venv` (torch 2.7.1+cu118, torchvision 0.22.1). The COCO mAP
metric needs two extra packages installed into that venv once:

```bash
source code/_mlx/.venv/bin/activate
pip install torchmetrics pycocotools
```

## Run

```bash
cd code/object-detection/ssdlite && ./run.sh        # or: nohup ./run.sh > train_ssdlite.log 2>&1 &
```

Outputs land in `results/`: `results.csv` (epoch, train_loss, val_map50,
val_map), `weights/best.pt`, `weights/last.pt`, and
`ondevice_benchmark_ssdlite.csv`.

## Resolution caveat

SSDLite320's anchors are designed for 320x320 and its internal transform resizes
to that; training/benchmarking at 512 would fight the anchor design. We therefore
run it at native 320 and disclose this. The complexity row records `input_res=320`
so it is never silently compared against the 512 YOLO rows.
