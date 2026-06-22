"""Combine every model's per-source mc_sz_metrics.csv into one table.

Scans code/classification/*/results/external/{mc,sz}/mc_sz_metrics.csv (written by
run_all.sh + mc_sz_metrics.py) and emits mc-sz-eval/mc_sz_summary.csv with one row
per (model, source): the external-set binary-TB ACC and AUC plus sensitivity,
specificity, and counts. This is the Montgomery/Shenzhen out-of-distribution table.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLS = HERE.parent
OUT = HERE / "mc_sz_summary.csv"

FIELDS = ["acc", "auc", "sensitivity", "specificity", "n_tb", "n_normal", "n_total"]
SOURCE_ORDER = {"mc": 0, "sz": 1}


def read_metrics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            values[row["metric"]] = float(row["value"])
    return values


def main() -> None:
    rows = []
    for metrics_csv in sorted(CLS.glob("*/results/external/*/mc_sz_metrics.csv")):
        model = metrics_csv.parents[3].name           # <model>/results/external/<source>/mc_sz_metrics.csv
        source = metrics_csv.parent.name              # mc | sz
        m = read_metrics(metrics_csv)
        rows.append((model, source, m))

    rows.sort(key=lambda r: (r[0], SOURCE_ORDER.get(r[1], 9)))

    with open(OUT, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "source", *FIELDS])
        for model, source, m in rows:
            writer.writerow([model, source, *(f"{m[f]:.6f}" for f in FIELDS)])

    print(f"wrote {len(rows)} rows -> {OUT}")
    # Echo a compact table.
    print(f"{'model':<26}{'src':<5}{'ACC':>8}{'AUC':>8}{'sens':>8}{'spec':>8}")
    for model, source, m in rows:
        print(f"{model:<26}{source:<5}{m['acc']:>8.4f}{m['auc']:>8.4f}{m['sensitivity']:>8.4f}{m['specificity']:>8.4f}")


if __name__ == "__main__":
    main()
