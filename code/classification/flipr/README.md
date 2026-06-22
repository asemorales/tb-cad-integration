# tb-classifier

Read the paper here : <insert paper>

A custom 3-class chest X-ray classifier for TB detection, built for ALIVE.
The architecture is a **3-stage-ablated ResNet18** with a custom **FlipR** block inserted after `layer2`, 
a learned lateral asymmetry gate motivated by the unilateral nature of many TB findings
on CXR. Trained from scratch.

Classes: `healthy` · `sick-non-tb` · `tb`. 

## Best Epoch Results

These are the recorded results from the `test` set of TBX11K : 

| Class       | Sensitivity | Specificity | AUROC  | Support |
| ----------- | :---------: | :---------: | :----: | :-----: |
| healthy     |   0.9965    |   0.9942    | 0.9993 |   570   |
| sick-non-tb |   0.9947    |   0.9884    | 0.9987 |   570   |
| tb          |   0.9417    |   1.0000    | 0.9996 |   120   |
| **macro**   | **0.9776**  | **0.9942**  | **0.9992** | 1260 |

Accuracy 0.9905 · macro F1 0.9850

## Reproduction

Each run is defined by a YAML config. The config used for this experiment (and the only config left present) is located at `experiments/configs/default.yaml`

### 1. Install

```bash
uv sync
```

### 2. Data

Clone your TBX11K locally. Some data parsing was done to lead to the final layout. 

```
data/tbx11k/
  train/{healthy,sick-non-tb,tb}/*.png
  val/  {healthy,sick-non-tb,tb}/*.png
  test/ {healthy,sick-non-tb,tb}/*.png
```

Source: https://alive-research.s3.ap-southeast-1.amazonaws.com/tbx11k-bounding-box-detection.zip

### 3. Train

```bash
uv run python scripts/train.py --config experiments/configs/default.yaml 
```

To resume from a checkpoint:

```bash
uv run python scripts/train.py --config experiments/configs/default.yaml --resume path/to/last.ckpt
```

### 4. Evaluate

```bash
uv run python scripts/evaluate.py --ckpt <path to your checkpoint>=
```

Prints the per-class table (sensitivity / specificity / AUROC / support), the macro
row, accuracy, the sklearn classification report, and the confusion matrix.


### FlipRBlock

This block is motivated by the use of symmetry in the detection of TB. It takes a simple difference between feature maps and their laterally-flipped counterparts, and broadcasts that difference back to the feature maps. There is some blurring done (pooling) to smooth out some slight asymmetry, since the CXRs are not guarenteed to be 100% symmetric anyway.

`src/tb_classifier/models/blocks/flipr.py`. Given features `x`:

```
asym = avgpool3x3(x - flip_horizontal(x))
gate = sigmoid(conv1x1(asym))          # per-pixel scalar in (0, 1)
out  = x * (1 + gate)                  # residual-friendly amplification
```


## Configs

`experiments/configs/default.yaml` is the canonical run. The model section has
two fields (`num_classes`, `pretrained`); the rest of the YAML is data and
training hyperparams. Copy it to add new experiments.

## Dev

```bash
uv run python scripts/check_data.py                  # dataloader smoke test
uv run ruff check . && uv run ruff format .          # lint / format
```

## Layout

```
src/tb_classifier/
├── models/
│   ├── blocks/flipr.py      # FlipR asymmetry gate
│   ├── backbone.py          # 3-stage ResNet18 + FlipR
│   └── classifier.py        # backbone + linear head
├── data/
│   ├── dataset.py           # TBX11K folder dataset
│   ├── transforms.py        # albumentations pipelines
│   └── loaders.py
├── training/
│   └── lightning_module.py  # TBLitModule
├── utils/metrics.py         # AUROC, sensitivity, specificity
└── config.py                # YAML → dataclasses

experiments/
├── configs/                 # per-run YAMLs
└── results/                 # gitignored (checkpoints, W&B run dirs)

scripts/{train,evaluate,check_data}.py
data/                        # gitignored
```
