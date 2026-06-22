"""Quantify the dataset domain gap between the training set (TBX11K-derived) and the
two external sets (Montgomery, Shenzhen). Reported as a paper contribution: the
cross-dataset shift in CXR appearance is rarely quantified in the TB-screening
literature, yet it is what bounds zero-shot generalization here.

For each domain we read every image as 8-bit grayscale and accumulate:
  - a normalized 256-bin intensity histogram (mean over images),
  - per-image mean / std intensity (then summarized across the domain),
  - image dimension stats (median W x H) -- framing/scanner differences.

Between domains we report distribution distances on the mean intensity histogram:
  - Wasserstein-1 (earth-mover) distance,
  - Jensen-Shannon divergence.
Larger distance = larger appearance shift = harder zero-shot transfer.

Domains:
  train : dataset/classification/train/{healthy,tb}   (the model's training distribution)
  mc    : .tmp/external/mc/test/{healthy,tb}
  sz    : .tmp/external/sz/test/{healthy,tb}

Outputs:
  - mc-sz-eval/domain_gap_stats.csv      per-domain intensity/size summary
  - mc-sz-eval/domain_gap_distances.csv  pairwise Wasserstein + JS on mean histograms
  - mc-sz-eval/domain_gap_hist.png       overlaid mean histograms (figure for the paper)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CLS = HERE.parent
REPO = CLS.parent.parent

sys.path.insert(0, str(REPO / "figures"))
from paper_style import apply_style, style_axes  # noqa: E402

apply_style()
BINS = 256
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DOMAINS = {
    "train": [REPO / "dataset/classification/train/healthy",
              REPO / "dataset/classification/train/tb"],
    "mc": [REPO / ".tmp/external/mc/test/healthy",
           REPO / ".tmp/external/mc/test/tb"],
    "sz": [REPO / ".tmp/external/sz/test/healthy",
           REPO / ".tmp/external/sz/test/tb"],
}


def iter_images(dirs):
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXTS:
                yield p


def domain_profile(dirs):
    hist = np.zeros(BINS, dtype=np.float64)
    means, stds, widths, heights = [], [], [], []
    n = 0
    for p in iter_images(dirs):
        try:
            im = Image.open(p).convert("L")
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        arr = np.asarray(im, dtype=np.uint8)
        h, _ = np.histogram(arr, bins=BINS, range=(0, 255))
        hist += h / arr.size            # per-image normalized, then summed
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))
        widths.append(im.width); heights.append(im.height)
        n += 1
    hist /= max(n, 1)                   # mean normalized histogram
    return {
        "n": n,
        "hist": hist,
        "mean_intensity": float(np.mean(means)) if means else float("nan"),
        "std_intensity_within": float(np.mean(stds)) if stds else float("nan"),
        "mean_intensity_sd_across": float(np.std(means)) if means else float("nan"),
        "median_w": float(np.median(widths)) if widths else float("nan"),
        "median_h": float(np.median(heights)) if heights else float("nan"),
    }


def main() -> None:
    profiles = {}
    for name, dirs in DOMAINS.items():
        print(f"profiling {name} ...")
        profiles[name] = domain_profile(dirs)
        p = profiles[name]
        print(f"  {name}: n={p['n']} mean={p['mean_intensity']:.1f} "
              f"within-sd={p['std_intensity_within']:.1f} "
              f"median_dim={p['median_w']:.0f}x{p['median_h']:.0f}")

    # per-domain stats
    with open(HERE / "domain_gap_stats.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["domain", "n", "mean_intensity", "std_intensity_within",
                    "mean_intensity_sd_across", "median_w", "median_h"])
        for name, p in profiles.items():
            w.writerow([name, p["n"], f"{p['mean_intensity']:.4f}",
                        f"{p['std_intensity_within']:.4f}",
                        f"{p['mean_intensity_sd_across']:.4f}",
                        f"{p['median_w']:.0f}", f"{p['median_h']:.0f}"])

    # pairwise distances on the mean histograms
    centers = np.arange(BINS)
    pairs = [("train", "mc"), ("train", "sz"), ("mc", "sz")]
    with open(HERE / "domain_gap_distances.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["pair", "wasserstein", "jensenshannon"])
        for a, b in pairs:
            ha = profiles[a]["hist"] / profiles[a]["hist"].sum()
            hb = profiles[b]["hist"] / profiles[b]["hist"].sum()
            wd = float(wasserstein_distance(centers, centers, ha, hb))
            js = float(jensenshannon(ha, hb, base=2))
            w.writerow([f"{a}_vs_{b}", f"{wd:.4f}", f"{js:.4f}"])
            print(f"  {a} vs {b}: Wasserstein={wd:.2f}  JS={js:.4f}")

    # figure
    plt.figure(figsize=(7, 4))
    colors = {"train": "#0C5DA5", "mc": "#FF2C00", "sz": "#474747"}  # per-dataset, matching the detection figure
    labels = {"train": "TBX11K (train)", "mc": "Montgomery", "sz": "Shenzhen"}
    for name, p in profiles.items():
        hh = p["hist"] / p["hist"].sum()
        plt.plot(centers, hh, label=f"{labels[name]} (n={p['n']})",
                 color=colors[name], linewidth=1.6)
    # Log y-axis: the pixel-0 (padding/background) bin holds ~9-15% of pixels and
    # dwarfs every tissue bin on a linear scale, hiding the actual appearance shift.
    plt.yscale("log")
    plt.ylim(1e-5, 0.2)
    plt.xlabel("pixel intensity (8-bit)")
    plt.ylabel("normalized frequency (log scale)")
    plt.title("CXR intensity distribution shift across datasets")
    plt.legend(frameon=False)
    style_axes(plt.gca())
    plt.tight_layout()
    fig_path = HERE / "domain_gap_hist.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nwrote domain_gap_stats.csv, domain_gap_distances.csv, {fig_path.name}")


if __name__ == "__main__":
    main()
