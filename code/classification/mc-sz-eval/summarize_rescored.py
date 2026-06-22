"""Regenerate the MC/SZ OOD summary reporting BOTH scoring conventions side by side:

  p_tb       : the raw 3-class TB posterior (original sweep score)
  binary_tb  : p_tb / (p_healthy + p_tb), the renormalized binary posterior -- the
               principled score for a binary healthy-vs-TB external set. Adopted
               uniformly (not cherry-picked per model). See rescore_probe.py.

AUC is reported under both (threshold-free, the honest headline). The operating-point
metrics (acc/sens/spec) are reported for binary_tb at its natural 0.5 threshold, and
for p_tb at the model's native 3-class argmax (the original sweep convention), so the
table is directly comparable to the first sweep. Reads existing predictions.csv only;
no re-inference. Writes mc_sz_summary_rescored.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from mc_sz_metrics import read_predictions, TB_INDEX

HERE = Path(__file__).resolve().parent
CLS = HERE.parent
EPS = 1e-9


def ss_acc(y, pred):
    pos, neg = y == 1, y == 0
    tp = int((pred[pos] == 1).sum()); fn = int((pred[pos] == 0).sum())
    tn = int((pred[neg] == 0).sum()); fp = int((pred[neg] == 1).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / len(y)
    return acc, sens, spec


def main() -> None:
    rows = []
    for pcsv in sorted(CLS.glob("*/results/external/*/predictions.csv")):
        source = pcsv.parent.name
        model = pcsv.parents[3].name
        labels, probs, cols = read_predictions(pcsv)
        y = (labels == TB_INDEX).astype(int)
        p_h = probs[:, cols.index("p_healthy")]
        p_t = probs[:, cols.index("p_tb")]
        s_bin = p_t / (p_h + p_t + EPS)

        auc_ptb = float(roc_auc_score(y, p_t))
        auc_bin = float(roc_auc_score(y, s_bin))

        # p_tb operating point: native 3-class argmax (original sweep convention)
        argmax_pred = (probs.argmax(axis=1) == cols.index("p_tb")).astype(int)
        acc_p, sens_p, spec_p = ss_acc(y, argmax_pred)
        # binary_tb operating point: natural 0.5 threshold on renormalized posterior
        acc_b, sens_b, spec_b = ss_acc(y, (s_bin >= 0.5).astype(int))

        rows.append({
            "model": model, "source": source,
            "auc_p_tb": auc_ptb, "auc_binary_tb": auc_bin,
            "auc_gain": auc_bin - auc_ptb,
            "acc_p_tb": acc_p, "sens_p_tb": sens_p, "spec_p_tb": spec_p,
            "acc_binary_tb": acc_b, "sens_binary_tb": sens_b, "spec_binary_tb": spec_b,
            "n_tb": int(y.sum()), "n_normal": int((1 - y).sum()),
        })

    rows.sort(key=lambda r: (r["model"], r["source"]))
    fields = ["model", "source", "auc_p_tb", "auc_binary_tb", "auc_gain",
              "acc_p_tb", "sens_p_tb", "spec_p_tb",
              "acc_binary_tb", "sens_binary_tb", "spec_binary_tb",
              "n_tb", "n_normal"]
    out = HERE / "mc_sz_summary_rescored.csv"
    with open(out, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in fields})

    # console digest
    print(f"{'model':>24} {'src':>3}  {'AUC p_tb':>8} {'AUC bin':>8} {'gain':>7}  "
          f"{'bin acc':>7} {'bin se':>6} {'bin sp':>6}")
    for r in rows:
        print(f"{r['model']:>24} {r['source']:>3}  {r['auc_p_tb']:>8.3f} "
              f"{r['auc_binary_tb']:>8.3f} {r['auc_gain']:>+7.3f}  "
              f"{r['acc_binary_tb']:>7.3f} {r['sens_binary_tb']:>6.2f} {r['spec_binary_tb']:>6.2f}")
    mean_p = np.mean([r["auc_p_tb"] for r in rows])
    mean_b = np.mean([r["auc_binary_tb"] for r in rows])
    print(f"\n  mean AUC: p_tb={mean_p:.4f}  binary_tb={mean_b:.4f}  (+{mean_b-mean_p:.4f})")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
