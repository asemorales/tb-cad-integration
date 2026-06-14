# Classification

This stage classifies a chest X-ray as healthy, sick non-TB, or TB. It holds nine models: FlipR and eight baselines. FlipR is a standalone module with its own training stack. The eight baselines share one engine, `code/_mlx`, and differ only by a command-line flag, so each baseline folder carries just its run command, config, and results.

### Models

| Folder | Engine | Parameters |
| --- | --- | ---: |
| `flipr` | standalone | 2.8M |
| `efficientnet-b0` | `_mlx` | 4.0M |
| `mobilenetv3-large` | `_mlx` | 4.2M |
| `drax-mobilenetv3-large` | `_mlx` | 4.8M |
| `densenet121` | `_mlx` | 7.0M |
| `resnet18` | `_mlx` | 11.2M |
| `draxnet` | `_mlx` | 16.5M |
| `resnet50` | `_mlx` | 23.5M |
| `convnext-tiny` | `_mlx` | 27.8M |

### Running a model

Each folder has a `run.sh` that trains and then benchmarks on the held-out test split. Trained weights are not distributed, so the script retrains from scratch. All runs use seed 42, 100 epochs, 512x512 inputs, and the best-validation checkpoint.

```bash
cd resnet18 && ./run.sh        # any _mlx baseline
cd flipr && ./run.sh           # FlipR (PyTorch Lightning)
```

The baselines read the dataset from `../../../dataset/classification`, which holds `train/`, `val/`, and `test/` over the three class folders. FlipR reads the same path through its own `experiments/configs/default.yaml`.

### Results

Each folder keeps its run artifacts in `results/`: `metrics.csv`, `confusion_matrix.csv`, `confusion_matrix.png`, and `roc_curve.png`. The FlipR results are transcribed from the original run rather than regenerated, because its checkpoint was not retained; see `flipr/results/NOTES.md`.
