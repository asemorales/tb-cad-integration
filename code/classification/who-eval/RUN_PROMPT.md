# Handoff prompt: run M2 (WHO 2025) inference + metrics

Paste everything below the line into a fresh Claude Code session started in
`/home/ase/code/alive-medical-platform`.

---

Run the M2 WHO 2025 operating-point evaluation for the nine TB classifiers. All
scripts already exist under `code/classification/who-eval/` (read its `README.md`
first). Do not modify the manuscript. Do not commit anything. Your job is to run
inference, validate the weights reproduce known results, and report the numbers.

## Run it

```bash
bash code/classification/who-eval/run_all.sh
```

This dumps per-image `P(tb)` for each model (`<model>/results/predictions.csv`),
computes the WHO indices (`<model>/results/who_metrics.csv`), and writes the
summary table (`code/classification/who-eval/who_summary.csv`) plus the ROC
figure (`figures/classifier_who_roc.pdf`). Inference is GPU (CUDA, GTX 1050 Ti),
roughly 1-2 min per model. The eight torchvision/Drax models use the mlx venv
(`code/_mlx/.venv`, imported via `PYTHONPATH=code/_mlx`, already set in the
script); FlipR uses its own venv (`code/classification/flipr/.venv`).

If a model fails, run that one model's `predict_*.py` line by hand (see
`run_all.sh`) to see the error, fix, and continue. The others are independent.

## Validate before trusting any number (mandatory, do not skip)

These weights were relocated into the repo and the FlipR checkpoint was supplied
separately, so confirm by reproduction, never assume:

1. **mlx models** — the argmax of each `predictions.csv` must reproduce that
   model's committed `confusion_matrix.csv` exactly. Check every mlx model:

   ```bash
   code/_mlx/.venv/bin/python - <<'PY'
   import csv, numpy as np
   from pathlib import Path
   from sklearn.metrics import confusion_matrix
   base = Path("code/classification")
   folders = ["efficientnet-b0","mobilenetv3-large","densenet121","resnet18",
              "resnet50","convnext-tiny","draxnet","drax-mobilenetv3-large"]
   for f in folders:
       r = base/f/"results"
       rows = list(csv.DictReader(open(r/"predictions.csv")))
       labels = [int(x["label"]) for x in rows]
       preds  = [int(np.argmax([float(x["p_healthy"]),float(x["p_sick_non_tb"]),float(x["p_tb"])])) for x in rows]
       got = confusion_matrix(labels, preds, labels=[0,1,2])
       want = np.array([[int(v) for v in row[1:]] for row in list(csv.reader(open(r/"confusion_matrix.csv")))[1:]])
       print(f"{f:<24} {'MATCH' if (got==want).all() else 'MISMATCH'}")
   PY
   ```

   Every line must say MATCH. A MISMATCH means the relocated weight is not the
   one that produced the paper's table; report it, do not paper over it.

2. **FlipR** — recompute per-class sensitivity / specificity / AUROC from
   `code/classification/flipr/results/predictions.csv` and confirm they match the
   table in `code/classification/flipr/README.md` (tb sensitivity 0.9417,
   specificity 0.9991, AUROC 0.9984; macro AUROC 0.9988; accuracy 0.9897). If the
   numbers do not match within rounding, the checkpoint or preprocessing is off:
   report the discrepancy honestly rather than presenting the numbers as WHO
   results.

## Report

Show the `who_summary.csv` table (AUC, pAUC(40-60% spec), spec@90%sens per model)
and state whether all cross-checks passed. Note the standing context: these
numbers are reported in WHO's format but are not directly comparable to WHO's
approved-product figures (single-source, curated, class-balanced test set vs
WHO's multi-region withheld libraries). Do not claim the models "pass" WHO bars.

Be honest, not claim-supporting: report what the runs actually show, including any
mismatch or failure.
