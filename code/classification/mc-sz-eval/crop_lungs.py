"""Stage lung-cropped copies of the external sets (segment once, reuse for all models).

Segmentation is model-independent and is the slow step, so we crop every MC/SZ image
a single time into a parallel staged dataset, then the normal predict_mlx.py sweep runs
fast over the crops with the IDENTICAL preprocessing path (Image.open + transform).

Lung masks: torchxrayvision pretrained PSPNet (chestx_det), off-the-shelf localizer,
no training. We take the union of left/right lung masks, pad the bounding box by 5% of
each side, and crop. Empty mask -> keep the full image (counted as fallback). This
removes most burned-in text / borders / field-of-view differences (Zech et al., 2018)
that drive the cross-dataset shift quantified in domain_gap.py.

Output tree mirrors the staged sets so predict_mlx.py --dataset works unchanged:
    .tmp/external/<src>_lungcrop/test/<class>/<name>.png

Usage:
    python crop_lungs.py [mc sz]      # default: both
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import torchxrayvision as xrv
import torchxrayvision.utils as xrv_utils

REPO = Path(__file__).resolve().parents[3]
EXT = REPO / ".tmp/external"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PAD_FRAC = 0.05
DEVICE = "cpu"


def lung_bbox(seg_model, gray: np.ndarray):
    x = xrv_utils.normalize(gray.astype(np.float32), 255)[None, ...]
    x = torch.from_numpy(x)
    x = torch.nn.functional.interpolate(
        x[None], size=(512, 512), mode="bilinear", align_corners=False)[0]
    with torch.no_grad():
        out = seg_model(x[None].to(DEVICE))[0].cpu().numpy()
    lung_idx = [i for i, t in enumerate(seg_model.targets) if "Lung" in t]
    if not lung_idx:
        return None
    mask = (out[lung_idx] > 0).any(axis=0)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    H, W = gray.shape
    sy, sx = H / 512.0, W / 512.0
    y0 = max(0, int(ys.min() * sy - PAD_FRAC * H))
    y1 = min(H, int(ys.max() * sy + PAD_FRAC * H))
    x0 = max(0, int(xs.min() * sx - PAD_FRAC * W))
    x1 = min(W, int(xs.max() * sx + PAD_FRAC * W))
    if y1 - y0 < 16 or x1 - x0 < 16:
        return None
    return y0, y1, x0, x1


def main() -> None:
    srcs = sys.argv[1:] or ["mc", "sz"]
    seg = xrv.baseline_models.chestx_det.PSPNet().to(DEVICE).eval()
    for src in srcs:
        in_root = EXT / src / "test"
        out_root = EXT / f"{src}_lungcrop" / "test"
        if not in_root.exists():
            print(f"  skip {src}: {in_root} missing")
            continue
        n, fb = 0, 0
        for cls_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
            out_cls = out_root / cls_dir.name
            out_cls.mkdir(parents=True, exist_ok=True)
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() not in EXTS:
                    continue
                dst = out_cls / (img.stem + ".png")
                if dst.exists():
                    n += 1
                    continue
                gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    fb += 1
                    continue
                bb = lung_bbox(seg, gray)
                crop = gray if bb is None else gray[bb[0]:bb[1], bb[2]:bb[3]]
                if bb is None:
                    fb += 1
                Image.fromarray(crop).save(dst)
                n += 1
            print(f"  {src}/{cls_dir.name}: {len(list(out_cls.glob('*.png')))} crops")
        print(f"{src}: staged {n} images ({fb} fallback to full) -> {out_root}")


if __name__ == "__main__":
    main()
