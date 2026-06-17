# M2: WHO 2025 operating-point evaluation

Restates the classifier results in the terms the WHO 2025 CAD policy statement
(ISBN 9789240110373, Annex 3 FIND performance report) uses to benchmark TB
triage software, instead of the argmax accuracy the original benchmark reported.
This is the quantitative half of reviewer item M2.

## What is computed

The task is binary TB triage. The score is **P(tb)** from the 3-class softmax,
with `tb` positive and `healthy` + `sick-non-tb` pooled as negative. (Using
`1 - P(healthy)` would be wrong: it counts sick-non-tb as positive.)

For each classifier:

| index | meaning |
| --- | --- |
| `auc_tb` | area under the binary TB ROC |
| `pauc_spec_40_60` | mean sensitivity over the 40-60% specificity band (FIND standardised partial AUC, normalised by the 0.2 band width so it reads as a mean TPR) |
| `spec_at_90_sens` | highest specificity at sensitivity >= 90%, the single operating point WHO scores every product at |

WHO standardises every product to the one 90%-sensitivity operating point, so no
multi-threshold table is produced. WHO reports discrimination only, so no
probability-calibration diagram is produced.

These numbers are **not** directly comparable to WHO's approved-product figures
(AUC 0.72-0.84, spec@90%sens 26-56%): this test set is single-source, curated and
class-balanced relative to WHO's multi-region withheld libraries. The point of M2
is to report in WHO's format and state the non-comparability, not to claim a pass.

## Layout and file conventions

One inference path per framework, one shared metric, one summary. Every model
emits the same `predictions.csv`, so the metric step is identical for all nine.

```
who-eval/
  who_metrics.py     # predictions.csv -> who_metrics.csv  (framework-agnostic core)
  predict_mlx.py     # one mlx model    -> results/predictions.csv  (mlx venv)
  predict_flipr.py   # FlipR checkpoint -> results/predictions.csv  (flipr venv)
  summarize.py       # all who_metrics.csv -> who_summary.csv + ROC figure
  run_all.sh         # runs everything end to end
```

Per-model outputs land next to the existing benchmark artifacts, matching the
repo convention (`<model>/results/`):

```
<model>/results/predictions.csv   # label,p_healthy,p_sick_non_tb,p_tb  (one row per test image)
<model>/results/who_metrics.csv   # metric,value
```

Combined outputs:

```
who-eval/who_summary.csv          # one row per model (the M2 table)
figures/classifier_who_roc.pdf    # binary-TB ROC, all classifiers, 90%-sens line
```

## Reproduce

```bash
bash code/classification/who-eval/run_all.sh
```

The eight torchvision/Drax models run through the mlx package and reuse the exact
benchmark preprocessing (`load_checkpoint_bundle` + `load_standard_classification_directory`).
FlipR runs through its Lightning package and reuses its trained val/test transform
(`tb_classifier.data.build_test_loader`: longest-max resize to 512, pad to square,
ImageNet normalisation).

Class index order is fixed everywhere: `0=healthy, 1=sick-non-tb, 2=tb`.

## Cross-checks

The moved mlx weights and the FlipR checkpoint are validated by reproduction, not
trusted blind:

- **mlx models:** the argmax of `predictions.csv` must reproduce each model's
  committed `confusion_matrix.csv` exactly. If it does, the weight is the one
  that produced the paper's numbers.
- **FlipR:** the per-class sensitivity / specificity / AUROC recomputed from
  `predictions.csv` must match the table in `../flipr/README.md`
  (tb sensitivity 0.9417, specificity 0.9991, AUROC 0.9984).
