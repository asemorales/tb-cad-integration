"""WHO 2025 CAD operating-point metrics from a per-image prediction file.

This is the framework-agnostic core of the M2 evaluation. It reads one
``predictions.csv`` (the common interchange format written by ``predict_mlx.py``
and ``predict_flipr.py``) and reports the three indices the WHO 2025 CAD policy
statement (ISBN 9789240110373, Annex 3 FIND report) uses to benchmark TB triage
software:

  - AUC                  area under the binary TB ROC
  - pAUC(40-60% spec)    mean sensitivity over the 40-60% specificity band,
                         the FIND standardised partial AUC (range width 0.2,
                         normalised so a value is a mean TPR in [0, 1])
  - spec@90%sens         highest specificity at sensitivity >= 90%, the single
                         standardised operating point WHO scores every product at

The binary TB score is P(tb) from the 3-class softmax, with tb positive and both
healthy and sick-non-tb pooled as negative. 1 - P(healthy) is wrong here: it
would count sick-non-tb as positive.

predictions.csv schema (header row, one row per test image):
    label,p_healthy,p_sick_non_tb,p_tb
where label is the integer class index (0=healthy, 1=sick-non-tb, 2=tb).

Usage:
    python who_metrics.py <results_dir>      # reads <results_dir>/predictions.csv,
                                             # writes <results_dir>/who_metrics.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Class index order is fixed across every classifier in this study
# (alphabetical folder order: healthy, sick-non-tb, tb).
TB_INDEX = 2
SPEC_LO, SPEC_HI = 0.40, 0.60   # WHO/FIND partial-AUC specificity band
SENS_TARGET = 0.90              # WHO standardised operating point


def read_predictions(predictions_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, p_tb) from a predictions.csv. Reads P(tb) by header name."""
    labels: list[int] = []
    p_tb: list[float] = []
    with open(predictions_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        if "p_tb" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"{predictions_csv} must have 'label' and 'p_tb' columns, got {reader.fieldnames}"
            )
        for row in reader:
            labels.append(int(row["label"]))
            p_tb.append(float(row["p_tb"]))
    return np.asarray(labels, dtype=int), np.asarray(p_tb, dtype=float)


def who_indices(
    labels: np.ndarray,
    p_tb: np.ndarray,
    *,
    spec_lo: float = SPEC_LO,
    spec_hi: float = SPEC_HI,
    sens_target: float = SENS_TARGET,
) -> dict[str, float]:
    """Compute the WHO/FIND TB-triage indices for binary score P(tb)."""
    y = (labels == TB_INDEX).astype(int)   # tb positive, {healthy, sick-non-tb} negative
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both TB and non-TB images to compute ROC metrics.")

    auc = float(roc_auc_score(y, p_tb))
    fpr, tpr, thresholds = roc_curve(y, p_tb)
    spec = 1.0 - fpr

    # spec@90%sens: highest specificity among operating points whose sensitivity
    # is at least the 90% target (the WHO-standardised point).
    at_target = tpr >= sens_target
    if at_target.any():
        idx_pool = np.where(at_target)[0]
        best = idx_pool[np.argmax(spec[idx_pool])]
        spec_at_sens = float(spec[best])
        threshold_at_sens = float(thresholds[best])
        sens_at_point = float(tpr[best])
    else:
        spec_at_sens = 0.0
        threshold_at_sens = float("nan")
        sens_at_point = float(tpr.max())

    # pAUC over the 40-60% specificity band, normalised to a mean sensitivity.
    # spec in [lo, hi] is fpr in [1-hi, 1-lo]; integrate tpr over that fpr band
    # (linear ROC interpolation) and divide by the band width.
    fpr_lo, fpr_hi = 1.0 - spec_hi, 1.0 - spec_lo
    grid = np.linspace(fpr_lo, fpr_hi, 2001)
    tpr_on_grid = np.interp(grid, fpr, tpr)
    pauc_mean_sens = float(np.trapezoid(tpr_on_grid, grid) / (fpr_hi - fpr_lo))

    return {
        "auc_tb": auc,
        "pauc_spec_40_60": pauc_mean_sens,
        "spec_at_90_sens": spec_at_sens,
        "sens_at_operating_point": sens_at_point,
        "threshold_at_90_sens": threshold_at_sens,
        "n_positive": float(n_pos),
        "n_negative": float(n_neg),
        "prevalence": float(n_pos / (n_pos + n_neg)),
        "n_total": float(n_pos + n_neg),
    }


def write_who_metrics(metrics: dict[str, float], out_csv: Path) -> None:
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in (
            "auc_tb",
            "pauc_spec_40_60",
            "spec_at_90_sens",
            "sens_at_operating_point",
            "threshold_at_90_sens",
            "prevalence",
            "n_positive",
            "n_negative",
            "n_total",
        ):
            writer.writerow([key, f"{metrics[key]:.6f}"])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python who_metrics.py <results_dir>")
    results_dir = Path(sys.argv[1])
    predictions_csv = results_dir / "predictions.csv"
    labels, p_tb = read_predictions(predictions_csv)
    metrics = who_indices(labels, p_tb)
    out_csv = results_dir / "who_metrics.csv"
    write_who_metrics(metrics, out_csv)
    print(
        f"{results_dir.parent.name:>26}  "
        f"AUC={metrics['auc_tb']:.4f}  "
        f"pAUC(40-60% spec)={metrics['pauc_spec_40_60']:.4f}  "
        f"spec@90%sens={metrics['spec_at_90_sens']:.4f}  "
        f"-> {out_csv}"
    )


if __name__ == "__main__":
    main()
