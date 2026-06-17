"""Combine per-model WHO metrics into one table and a combined ROC figure.

Reads every classifier's ``results/who_metrics.csv`` (written by ``who_metrics.py``)
and ``results/predictions.csv`` (written by the predictors), then emits:

  - who-eval/who_summary.csv          one row per model: AUC, pAUC(40-60% spec),
                                      spec@90%sens (the M2 deliverable table)
  - figures/classifier_who_roc.pdf    binary-TB ROC for every classifier on one
                                      axis, with the 90%-sensitivity line marked

Run from the classification directory with any venv that has numpy / sklearn /
matplotlib (the mlx or FlipR venv both work):
    code/_mlx/.venv/bin/python who-eval/summarize.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from who_metrics import SENS_TARGET, read_predictions, who_indices

# Display order and labels for the nine classifiers (folder -> paper name).
MODELS: list[tuple[str, str]] = [
    ("efficientnet-b0", "EfficientNet-B0"),
    ("mobilenetv3-large", "MobileNetV3-Large"),
    ("densenet121", "DenseNet-121"),
    ("resnet18", "ResNet-18"),
    ("resnet50", "ResNet-50"),
    ("convnext-tiny", "ConvNeXt-Tiny"),
    ("draxnet", "DraxNet"),
    ("drax-mobilenetv3-large", "Drax-MobileNetV3-Large"),
    ("flipr", "FlipR"),
]

HERE = Path(__file__).resolve().parent
CLASSIFICATION_DIR = HERE.parent
REPO = CLASSIFICATION_DIR.parents[1]
FIGDIR = REPO / "figures"


def read_metrics_csv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out[row["metric"]] = float(row["value"])
    return out


def main() -> None:
    rows: list[tuple[str, dict[str, float]]] = []
    roc_data: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    for folder, display in MODELS:
        results_dir = CLASSIFICATION_DIR / folder / "results"
        metrics_csv = results_dir / "who_metrics.csv"
        predictions_csv = results_dir / "predictions.csv"
        if not metrics_csv.is_file():
            print(f"skip {display}: no {metrics_csv}")
            continue
        metrics = read_metrics_csv(metrics_csv)
        rows.append((display, metrics))

        if predictions_csv.is_file():
            from sklearn.metrics import roc_curve

            labels, p_tb = read_predictions(predictions_csv)
            y = (labels == 2).astype(int)
            fpr, tpr, _ = roc_curve(y, p_tb)
            roc_data.append((display, fpr, tpr, metrics["auc_tb"]))

    # Summary table.
    summary_csv = HERE / "who_summary.csv"
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "auc_tb", "pauc_spec_40_60", "spec_at_90_sens"])
        for display, metrics in rows:
            writer.writerow(
                [
                    display,
                    f"{metrics['auc_tb']:.4f}",
                    f"{metrics['pauc_spec_40_60']:.4f}",
                    f"{metrics['spec_at_90_sens']:.4f}",
                ]
            )
    print(f"wrote {summary_csv}")
    print(f"\n{'model':<24} {'AUC':>8} {'pAUC(40-60)':>13} {'spec@90sens':>13}")
    for display, metrics in rows:
        print(
            f"{display:<24} {metrics['auc_tb']:>8.4f} "
            f"{metrics['pauc_spec_40_60']:>13.4f} {metrics['spec_at_90_sens']:>13.4f}"
        )

    # Combined ROC figure.
    if roc_data:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 6))
        for display, fpr, tpr, auc_value in roc_data:
            ax.plot(fpr, tpr, lw=1.4, label=f"{display} (AUC={auc_value:.3f})")
        ax.axhline(SENS_TARGET, color="gray", ls="--", lw=1, label=f"{SENS_TARGET:.0%} sensitivity")
        ax.plot([0, 1], [0, 1], color="lightgray", ls=":", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("1 - specificity (false positive rate)")
        ax.set_ylabel("sensitivity (true positive rate)")
        ax.set_title("Binary TB discrimination (P(tb)) on the held-out test set")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        FIGDIR.mkdir(parents=True, exist_ok=True)
        out_pdf = FIGDIR / "classifier_who_roc.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
