"""Generate the out-of-distribution TB ROC figure for FlipR.

Plots three binary TB ROC curves on one axis: in-distribution TBX11K test,
Montgomery, and Shenzhen. Positive class is TB (label == 2) scored by P(tb);
healthy and sick-non-tb are pooled as negative in-distribution. This makes the
zero-shot collapse that Table IV quantifies legible: the in-distribution curve
saturates while the two external curves fall toward the diagonal.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "code/classification/flipr/results"

sys.path.insert(0, str(ROOT / "figures"))
from paper_style import apply_style, style_axes  # noqa: E402

apply_style()

# One color per dataset across the whole paper, matching the detection figure:
# TBX11K blue, Montgomery red, Shenzhen near-black (#474747, the detection
# figure's "all classes" color). Red and black separate the two overlapping
# zero-shot curves cleanly.
SOURCES = [
    ("TBX11K test (in-distribution)", RES / "predictions.csv", "#0C5DA5"),
    ("Montgomery (zero-shot)", RES / "external/mc/predictions.csv", "#FF2C00"),
    ("Shenzhen (zero-shot)", RES / "external/sz/predictions.csv", "#474747"),
]

fig, ax = plt.subplots(figsize=(5.2, 4.4))
for name, path, color in SOURCES:
    df = pd.read_csv(path)
    y = (df["label"].to_numpy() == 2).astype(int)
    score = df["p_tb"].to_numpy()
    fpr, tpr, _ = roc_curve(y, score)
    auc = roc_auc_score(y, score)
    ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{name} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], ls=":", lw=0.8, color="0.6")
ax.axhline(0.90, ls="--", lw=0.8, color="0.4", label="90% sensitivity (WHO benchmark)")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_xlabel("1 - specificity (false positive rate)")
ax.set_ylabel("sensitivity (true positive rate)")
ax.set_title("FlipR binary TB ROC: in-distribution vs unseen sites")
ax.legend(loc="lower right", fontsize=8, frameon=False)
style_axes(ax)
fig.tight_layout()
out = ROOT / "figures/classifier_ood_roc/classifier_ood_roc.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
