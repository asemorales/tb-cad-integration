"""Probe alternative binary-TB scores on the MC/SZ external sets (no retraining).

We currently rank with p_tb from the 3-class head. MC/SZ negatives are healthy-only,
so a plausible OOD failure is that abnormal-but-not-recognized-as-TB images land in the
sick_non_tb bucket: the model sees "abnormal" but picks the wrong abnormal class, which
deflates p_tb even though healthy-vs-TB is separable. This probe recomputes AUC (ranking,
the binding metric) under several score definitions derived from the SAME existing
predictions.csv (zero cost, no GPU, no retraining):

  p_tb            current score (3-class TB posterior)
  not_healthy     1 - p_healthy  = p_sick_non_tb + p_tb   (P abnormal)
  binary_tb       p_tb / (p_healthy + p_tb)               (drop sick_non_tb mass, renormalize)
  margin          p_tb - p_healthy                         (healthy-vs-TB log-odds-ish)

AUC is threshold-free, so any change here is a genuine change in discriminative ranking,
not a recalibration artifact. Outputs mc_sz_rescore.csv (one row per model/source).
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


def scores(probs: np.ndarray, cols: list[str]) -> dict[str, np.ndarray]:
    p_h = probs[:, cols.index("p_healthy")]
    p_t = probs[:, cols.index("p_tb")]
    return {
        "p_tb": p_t,
        "not_healthy": 1.0 - p_h,
        "binary_tb": p_t / (p_h + p_t + EPS),
        "margin": p_t - p_h,
    }


def main() -> None:
    rows = []
    score_names = ["p_tb", "not_healthy", "binary_tb", "margin"]
    for pcsv in sorted(CLS.glob("*/results/external/*/predictions.csv")):
        source = pcsv.parent.name
        model = pcsv.parents[3].name
        labels, probs, cols = read_predictions(pcsv)
        y = (labels == TB_INDEX).astype(int)
        s = scores(probs, cols)
        aucs = {name: float(roc_auc_score(y, s[name])) for name in score_names}
        best = max(score_names, key=lambda n: aucs[n])
        rows.append({"model": model, "source": source, **aucs,
                     "best_score": best, "gain_vs_p_tb": aucs[best] - aucs["p_tb"]})
        print(
            f"{model:>24} {source:>3}  p_tb={aucs['p_tb']:.3f}  "
            f"not_healthy={aucs['not_healthy']:.3f}  binary_tb={aucs['binary_tb']:.3f}  "
            f"margin={aucs['margin']:.3f}  -> best={best} (+{aucs[best]-aucs['p_tb']:+.3f})"
        )

    out = HERE / "mc_sz_rescore.csv"
    fields = ["model", "source", *score_names, "best_score", "gain_vs_p_tb"]
    with open(out, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in fields})

    # aggregate
    import statistics
    for name in score_names:
        vals = [r[name] for r in rows]
        print(f"  mean AUC {name:>12}: {statistics.mean(vals):.4f}")
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
