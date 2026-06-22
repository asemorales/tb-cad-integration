"""Binary-TB ACC + AUC on an external set (Montgomery or Shenzhen) from a predictions.csv.

Reuses the common interchange format written by who-eval's predict_mlx.py /
predict_flipr.py (header: label,p_healthy,p_sick_non_tb,p_tb; class order fixed at
0=healthy, 1=sick-non-tb, 2=tb). Montgomery and Shenzhen are binary CXR sets
(normal vs TB), so a staged external set only contains the healthy (normal) and tb
folders; label is therefore 0 or 2.

The task is binary TB triage, identical framing to the WHO eval: the positive class
is TB, the score is P(tb) from the 3-class softmax, and healthy + sick-non-tb pool
as negative (here only healthy is present among the negatives). Reported:

  - acc        binary accuracy, TB-positive iff the 3-class argmax is tb
  - auc        area under the binary TB ROC, score = P(tb)
  - sensitivity, specificity at the argmax decision
  - n_total, n_tb, n_normal

Usage:
    python mc_sz_metrics.py <results_dir> [source_label]
        # reads <results_dir>/predictions.csv, writes <results_dir>/mc_sz_metrics.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

TB_INDEX = 2  # fixed class order: 0=healthy, 1=sick-non-tb, 2=tb


def read_predictions(predictions_csv: Path):
    """Return (labels, p_matrix, prob_columns) from a predictions.csv."""
    labels: list[int] = []
    probs: list[list[float]] = []
    with open(predictions_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        prob_cols = [c for c in reader.fieldnames if c.startswith("p_")]
        if "label" not in reader.fieldnames or "p_tb" not in prob_cols:
            raise ValueError(
                f"{predictions_csv} must have 'label' and 'p_tb' columns, got {reader.fieldnames}"
            )
        for row in reader:
            labels.append(int(row["label"]))
            probs.append([float(row[c]) for c in prob_cols])
    return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float), prob_cols


def binary_metrics(labels: np.ndarray, prob_matrix: np.ndarray, prob_cols: list[str]) -> dict[str, float]:
    tb_col = prob_cols.index("p_tb")
    p_tb = prob_matrix[:, tb_col]
    y = (labels == TB_INDEX).astype(int)          # TB positive
    pred = (prob_matrix.argmax(axis=1) == tb_col).astype(int)  # 3-class argmax -> is-TB

    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both TB and normal images to compute binary metrics.")

    acc = float((pred == y).mean())
    auc = float(roc_auc_score(y, p_tb))
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sensitivity = tp / n_pos
    specificity = tn / n_neg

    return {
        "acc": acc,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "n_tb": float(n_pos),
        "n_normal": float(n_neg),
        "n_total": float(n_pos + n_neg),
    }


def write_metrics(metrics: dict[str, float], out_csv: Path) -> None:
    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in ("acc", "auc", "sensitivity", "specificity", "n_tb", "n_normal", "n_total"):
            writer.writerow([key, f"{metrics[key]:.6f}"])


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python mc_sz_metrics.py <results_dir> [source_label]")
    results_dir = Path(sys.argv[1])
    source = sys.argv[2] if len(sys.argv) > 2 else results_dir.name
    labels, probs, cols = read_predictions(results_dir / "predictions.csv")
    metrics = binary_metrics(labels, probs, cols)
    write_metrics(metrics, results_dir / "mc_sz_metrics.csv")
    print(
        f"{source:>6}  ACC={metrics['acc']:.4f}  AUC={metrics['auc']:.4f}  "
        f"sens={metrics['sensitivity']:.4f}  spec={metrics['specificity']:.4f}  "
        f"(TB={int(metrics['n_tb'])}, normal={int(metrics['n_normal'])})"
    )


if __name__ == "__main__":
    main()
