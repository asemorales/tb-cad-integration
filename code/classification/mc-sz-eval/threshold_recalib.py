"""Per-site threshold recalibration for the MC/SZ out-of-distribution sweep.

Motivation. The zero-shot MC/SZ collapse mixes two distinct failures: (1) a genuine
loss of *ranking* ability (AUC drop, not fixable by thresholding) and (2) a wrong
*operating point* inherited from the source domain (the 3-class argmax boundary does
not transfer, producing sensitivity collapse on MC and specificity collapse on SZ).
Failure (2) is what any real deployment fixes first, by re-thresholding P(tb) on a
small local sample. This script measures how much of the collapse is (2) vs (1).

Method. For each (model, source) we read the existing external predictions.csv (no
re-inference), stratified-split it 50/50 into a calibration slice and a held-out test
slice (seed 42), fit a decision threshold on the calibration slice, and report the
recovered operating point on the held-out test slice. Two clinically meaningful
thresholds are fit:

  - youden : argmax(sensitivity + specificity - 1) on calibration (balanced point).
  - who90  : the lowest P(tb) threshold reaching >= 90% sensitivity on calibration,
             matching the WHO 2025 triage convention (fix sensitivity, read off
             specificity). If 90% is unreachable, the max-sensitivity threshold is used.

The baseline is the model's native zero-shot 3-class argmax decision on the SAME
held-out test slice (apples to apples). AUC is reported on the test slice too: it is
threshold-independent, so it is identical across baseline/youden/who90 by construction.
That is the point. Recovered ACC/sens/spec show the operating-point part is salvageable
per site; the unchanged, still-low AUC is the genuine residual discrimination loss.

Outputs:
  - code/classification/mc-sz-eval/mc_sz_recalib.csv   one row per (model, source)

Usage:
    python threshold_recalib.py            # all models found under code/classification/*/results/external/*
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from mc_sz_metrics import read_predictions, TB_INDEX

HERE = Path(__file__).resolve().parent
CLS = HERE.parent
SEED = 42
WHO_TARGET_SENS = 0.90


def stratified_half(y: np.ndarray, seed: int = SEED):
    """Return (cal_idx, test_idx): a stratified 50/50 split preserving class balance."""
    rng = np.random.default_rng(seed)
    cal, test = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        cal.extend(idx[:half].tolist())
        test.extend(idx[half:].tolist())
    return np.array(sorted(cal)), np.array(sorted(test))


def sens_spec_acc(y: np.ndarray, pred: np.ndarray):
    pos, neg = y == 1, y == 0
    tp = int((pred[pos] == 1).sum()); fn = int((pred[pos] == 0).sum())
    tn = int((pred[neg] == 0).sum()); fp = int((pred[neg] == 1).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / len(y)
    return sens, spec, acc


def youden_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    thrs = np.unique(scores)
    best_thr, best_j = 0.5, -1.0
    for t in thrs:
        s, sp, _ = sens_spec_acc(y, (scores >= t).astype(int))
        j = s + sp - 1.0
        if j > best_j:
            best_j, best_thr = j, float(t)
    return best_thr


def who90_threshold(scores: np.ndarray, y: np.ndarray, target: float = WHO_TARGET_SENS) -> float:
    """Lowest threshold whose calibration sensitivity >= target; else max-sensitivity threshold."""
    thrs = np.unique(scores)
    feasible = []
    for t in thrs:
        s, _, _ = sens_spec_acc(y, (scores >= t).astype(int))
        if s >= target:
            feasible.append((t, s))
    if feasible:
        # highest threshold still meeting the target -> best specificity among feasible
        return float(max(t for t, _ in feasible))
    # target unreachable: take the threshold giving max sensitivity (lowest, =0)
    return float(thrs.min())


def discover():
    return sorted(CLS.glob("*/results/external/*/predictions.csv"))


def main() -> None:
    rows = []
    for pcsv in discover():
        source = pcsv.parent.name
        model = pcsv.parents[3].name
        labels, probs, cols = read_predictions(pcsv)
        tb_col = cols.index("p_tb")
        p_tb = probs[:, tb_col]
        y = (labels == TB_INDEX).astype(int)
        argmax_pred = (probs.argmax(axis=1) == tb_col).astype(int)

        cal_idx, test_idx = stratified_half(y)
        yc, ptc = y[cal_idx], p_tb[cal_idx]
        yt, ptt = y[test_idx], p_tb[test_idx]

        auc_test = float(roc_auc_score(yt, ptt))

        # baseline: native 3-class argmax on the held-out test slice
        b_sens, b_spec, b_acc = sens_spec_acc(yt, argmax_pred[test_idx])

        # youden, fit on calibration, evaluated on test
        yt_thr = youden_threshold(ptc, yc)
        y_sens, y_spec, y_acc = sens_spec_acc(yt, (ptt >= yt_thr).astype(int))

        # who90, fit on calibration, evaluated on test
        w_thr = who90_threshold(ptc, yc)
        w_sens_cal, _, _ = sens_spec_acc(yc, (ptc >= w_thr).astype(int))
        w_sens, w_spec, w_acc = sens_spec_acc(yt, (ptt >= w_thr).astype(int))

        rows.append({
            "model": model, "source": source,
            "n_cal": len(cal_idx), "n_test": len(test_idx),
            "auc_test": auc_test,
            "zeroshot_acc": b_acc, "zeroshot_sens": b_sens, "zeroshot_spec": b_spec,
            "youden_thr": yt_thr, "youden_acc": y_acc, "youden_sens": y_sens, "youden_spec": y_spec,
            "who90_thr": w_thr, "who90_sens_cal": w_sens_cal,
            "who90_acc": w_acc, "who90_sens": w_sens, "who90_spec": w_spec,
        })
        print(
            f"{model:>24} {source:>3}  AUC={auc_test:.3f} | "
            f"zeroshot acc={b_acc:.3f}(se={b_sens:.2f},sp={b_spec:.2f}) -> "
            f"youden acc={y_acc:.3f}(se={y_sens:.2f},sp={y_spec:.2f}) | "
            f"who90 se={w_sens:.2f},sp={w_spec:.2f}"
        )

    fields = ["model", "source", "n_cal", "n_test", "auc_test",
              "zeroshot_acc", "zeroshot_sens", "zeroshot_spec",
              "youden_thr", "youden_acc", "youden_sens", "youden_spec",
              "who90_thr", "who90_sens_cal", "who90_acc", "who90_sens", "who90_spec"]
    out = HERE / "mc_sz_recalib.csv"
    with open(out, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
