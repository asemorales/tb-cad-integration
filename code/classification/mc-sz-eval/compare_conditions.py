"""Unified no-retrain OOD recovery table: compare AUC across conditions and scores.

For every model x source it reads whatever prediction files exist and reports AUC under
both the raw p_tb score and the adopted binary_tb score, for each condition:

  zeroshot   results/external/<src>/predictions.csv           (original sweep)
  adabn      results/external_adabn/<src>/predictions.csv      (BatchNorm adaptation)
  lungcrop   results/external_lungcrop/<src>/predictions.csv   (lung-field crop)

AUC is the honest headline (threshold-free). The "best" column names the single best
condition+score per cell, and best_gain is its lift over the zero-shot p_tb baseline.
Writes mc_sz_recovery.csv and prints a digest. Missing files are skipped, not errored,
so this runs while the sweeps are still in progress.
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
CONDITIONS = {"zeroshot": "external", "adabn": "external_adabn", "lungcrop": "external_lungcrop"}
MODELS = ["efficientnet-b0", "mobilenetv3-large", "mobilenetv3-small", "densenet121",
          "resnet18", "resnet50", "convnext-tiny", "draxnet", "drax-mobilenetv3-large",
          "lighttbnet", "flipr"]


def aucs(pcsv: Path):
    labels, probs, cols = read_predictions(pcsv)
    y = (labels == TB_INDEX).astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return None
    p_h = probs[:, cols.index("p_healthy")] if "p_healthy" in cols else None
    p_t = probs[:, cols.index("p_tb")]
    a_ptb = float(roc_auc_score(y, p_t))
    a_bin = float(roc_auc_score(y, p_t / (p_h + p_t + EPS))) if p_h is not None else a_ptb
    return a_ptb, a_bin


def main() -> None:
    rows = []
    for model in MODELS:
        for src in ("mc", "sz"):
            cell = {"model": model, "source": src}
            present = False
            for cond, sub in CONDITIONS.items():
                pcsv = CLS / model / "results" / sub / src / "predictions.csv"
                res = aucs(pcsv) if pcsv.exists() else None
                if res:
                    present = True
                    cell[f"{cond}_p_tb"] = res[0]
                    cell[f"{cond}_binary_tb"] = res[1]
                else:
                    cell[f"{cond}_p_tb"] = ""
                    cell[f"{cond}_binary_tb"] = ""
            if not present:
                continue
            base = cell.get("zeroshot_p_tb")
            candidates = {k: v for k, v in cell.items()
                          if k not in ("model", "source") and isinstance(v, float)}
            best_key = max(candidates, key=candidates.get)
            cell["best"] = best_key
            cell["best_auc"] = candidates[best_key]
            cell["best_gain_vs_zeroshot_p_tb"] = (
                candidates[best_key] - base if isinstance(base, float) else "")
            rows.append(cell)

    cols = ["model", "source",
            "zeroshot_p_tb", "zeroshot_binary_tb",
            "adabn_p_tb", "adabn_binary_tb",
            "lungcrop_p_tb", "lungcrop_binary_tb",
            "best", "best_auc", "best_gain_vs_zeroshot_p_tb"]
    out = HERE / "mc_sz_recovery.csv"
    with open(out, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c, ""))
                        for c in cols})

    # digest
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else "  -  "
    print(f"{'model':>22} {'src':>3} | {'zs_ptb':>6} {'zs_bin':>6} | "
          f"{'ab_ptb':>6} {'ab_bin':>6} | {'lc_ptb':>6} {'lc_bin':>6} | best (+gain)")
    for r in rows:
        g = r["best_gain_vs_zeroshot_p_tb"]
        gs = f"+{g:.3f}" if isinstance(g, float) else ""
        print(f"{r['model']:>22} {r['source']:>3} | "
              f"{fmt(r['zeroshot_p_tb'])} {fmt(r['zeroshot_binary_tb'])} | "
              f"{fmt(r['adabn_p_tb'])} {fmt(r['adabn_binary_tb'])} | "
              f"{fmt(r['lungcrop_p_tb'])} {fmt(r['lungcrop_binary_tb'])} | "
              f"{r['best']} {gs}")
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
