"""Binary-TB confusion matrices for the MC/SZ out-of-distribution sweep.

For every external predictions.csv (header: label,p_healthy,p_sick_non_tb,p_tb;
class order 0=healthy, 1=sick-non-tb, 2=tb) this writes the 2x2 binary-TB
confusion matrix at the model's native (argmax) decision. Positive = TB; negative
= healthy + sick-non-tb pooled (MC/SZ contain only healthy among the negatives).
This is the breakdown behind the headline ACC/AUC: it shows WHICH error a model
makes under domain shift, i.e. sensitivity collapse (misses TB, large FN) vs
specificity collapse (over-flags normals, large FP).

Outputs:
  - <results_dir>/confusion_matrix.csv          per model/source (tp,fn,fp,tn,...)
  - code/classification/mc-sz-eval/mc_sz_confusion.csv   combined, one row per (model,source)

Usage:
    python confusion_matrix.py            # sweep all code/classification/*/results/external/*/
    python confusion_matrix.py <dir> ...  # only the named results dirs
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from mc_sz_metrics import read_predictions, TB_INDEX

HERE = Path(__file__).resolve().parent
CLS = HERE.parent


def confusion(labels: np.ndarray, prob_matrix: np.ndarray, prob_cols: list[str]) -> dict:
    tb_col = prob_cols.index("p_tb")
    y = (labels == TB_INDEX).astype(int)
    pred = (prob_matrix.argmax(axis=1) == tb_col).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n_pos, n_neg = tp + fn, fp + tn
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": tp / n_pos if n_pos else float("nan"),
        "specificity": tn / n_neg if n_neg else float("nan"),
        "fn_rate": fn / n_pos if n_pos else float("nan"),   # missed TB
        "fp_rate": fp / n_neg if n_neg else float("nan"),   # over-flagged normal
        "n_tb": n_pos, "n_normal": n_neg,
    }


def write_per_dir(metrics: dict, out_csv: Path) -> None:
    # human-readable 2x2 plus the derived rates
    with open(out_csv, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["", "pred_normal", "pred_tb"])
        w.writerow(["actual_tb", metrics["fn"], metrics["tp"]])
        w.writerow(["actual_normal", metrics["tn"], metrics["fp"]])
        w.writerow([])
        w.writerow(["metric", "value"])
        for k in ("sensitivity", "specificity", "fn_rate", "fp_rate", "n_tb", "n_normal"):
            v = metrics[k]
            w.writerow([k, f"{v:.6f}" if isinstance(v, float) else v])


def discover() -> list[Path]:
    return sorted(CLS.glob("*/results/external/*/predictions.csv"))


def main() -> None:
    if len(sys.argv) > 1:
        dirs = [Path(a) for a in sys.argv[1:]]
        preds = [d / "predictions.csv" for d in dirs]
    else:
        preds = discover()

    rows = []
    for pcsv in preds:
        if not pcsv.exists():
            print(f"  skip (no predictions.csv): {pcsv}")
            continue
        # .../<model>/results/external/<source>/predictions.csv
        source = pcsv.parent.name
        model = pcsv.parents[3].name
        labels, probs, cols = read_predictions(pcsv)
        m = confusion(labels, probs, cols)
        write_per_dir(m, pcsv.parent / "confusion_matrix.csv")
        rows.append({"model": model, "source": source, **m})
        print(
            f"{model:>24} {source:>3}  TP={m['tp']:>3} FN={m['fn']:>3} "
            f"FP={m['fp']:>3} TN={m['tn']:>3}  sens={m['sensitivity']:.3f} "
            f"spec={m['specificity']:.3f}"
        )

    rows.sort(key=lambda r: (r["model"], r["source"]))
    combined = HERE / "mc_sz_confusion.csv"
    fields = ["model", "source", "tp", "fn", "fp", "tn",
              "sensitivity", "specificity", "fn_rate", "fp_rate", "n_tb", "n_normal"]
    with open(combined, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"\nwrote {combined} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
