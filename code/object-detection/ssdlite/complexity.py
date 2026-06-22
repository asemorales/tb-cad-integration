"""Params, MACs, CPU latency/memory, and disk size for SSDLite320.

Writes a row in the SAME schema as
``code/model-complexity/ondevice_benchmark_detection.csv`` so the number drops
into the manuscript table. SSDLite is measured at its NATIVE 320x320 (see
train.py for why), so the resolution differs from the YOLO rows at 512; the
handoff/README disclose this. NMS is excluded from the MAC count, matching the
thop/Ultralytics convention for the YOLO rows.

    code/_mlx/.venv/bin/python code/object-detection/ssdlite/complexity.py
"""

from __future__ import annotations

import csv
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from train import build_model

HERE = Path(__file__).resolve().parent
IMG = 320  # SSDLite native input
WARMUP, ITERS, THREADS_MULTI = 8, 40, 6


class _TensorIn(nn.Module):
    """Wrap the detection model so fvcore can trace it from a single tensor."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m([x[i] for i in range(x.shape[0])])


def _peak_rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def _latency(model, imgs, threads) -> dict:
    torch.set_num_threads(threads)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(imgs)
        s = []
        for _ in range(ITERS):
            t = time.perf_counter()
            model(imgs)
            s.append((time.perf_counter() - t) * 1000.0)
    s.sort()
    n = len(s)
    return {"median": s[n // 2], "p10": s[int(0.1 * n)], "p90": s[min(n - 1, int(0.9 * n))]}


def main() -> None:
    from fvcore.nn import FlopCountAnalysis

    model = build_model().eval()
    params = sum(p.numel() for p in model.parameters())

    x = torch.randn(1, 3, IMG, IMG)
    with torch.no_grad():
        fca = FlopCountAnalysis(_TensorIn(model), x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        macs = fca.total()

    imgs = [torch.randn(3, IMG, IMG)]
    l1 = _latency(model, imgs, 1)
    ln = _latency(model, imgs, THREADS_MULTI)

    best = HERE / "results" / "weights" / "best.pt"
    disk_mb = best.stat().st_size / 1e6 if best.exists() else float("nan")
    peak = _peak_rss_mb()

    row = {
        "model": "SSDLite320-MNv3-Large",
        "params_m": params / 1e6,
        "macs_g": macs / 1e9,
        "disk_mb": disk_mb,
        "lat_1t": l1,
        "lat_nt": ln,
        "peak_rss_mb": peak,
    }
    print(f"SSDLite320 @ {IMG}x{IMG} (native): params {row['params_m']:.2f}M  "
          f"MACs {row['macs_g']:.3f}G  disk {disk_mb:.1f}MB  "
          f"lat@1T {l1['median']:.1f}ms  lat@{THREADS_MULTI}T {ln['median']:.1f}ms  "
          f"peakRSS {peak:.0f}MB")

    out = HERE / "results" / "ondevice_benchmark_ssdlite.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "model", "params_m", "macs_g", "disk_mb",
            "lat_1t_median_ms", "lat_1t_p10_ms", "lat_1t_p90_ms",
            f"lat_{THREADS_MULTI}t_median_ms", f"lat_{THREADS_MULTI}t_p10_ms", f"lat_{THREADS_MULTI}t_p90_ms",
            "peak_rss_mb", "input_res",
        ])
        w.writerow([
            row["model"], f"{row['params_m']:.4f}", f"{row['macs_g']:.4f}", f"{disk_mb:.3f}",
            f"{l1['median']:.3f}", f"{l1['p10']:.3f}", f"{l1['p90']:.3f}",
            f"{ln['median']:.3f}", f"{ln['p10']:.3f}", f"{ln['p90']:.3f}",
            f"{peak:.1f}", IMG,
        ])
    print(f"wrote {out}")
    print("NOTE: measured at 320 (SSDLite native); YOLO rows are at 512. Disclose in the table.")


if __name__ == "__main__":
    main()
