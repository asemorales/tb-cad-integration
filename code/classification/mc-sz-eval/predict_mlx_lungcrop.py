"""Lung-field cropping at test time, then predict (no retraining).

Cross-dataset CXR shortcuts (burned-in text, laterality markers, black borders,
differing field-of-view) are a documented cause of poor generalization (Zech et al.,
2018). Montgomery/Shenzhen frame the thorax very differently from the TBX11K-derived
training set (median 4020x4892 / 2744x2937 vs 512x512). Cropping each image to its
lung bounding box before inference removes most of that domain-specific border content
while keeping the diagnostic region, and changes the input, so it can move AUC.

Lung masks come from torchxrayvision's pretrained PSPNet segmentation model
(chestx_det), used purely as an off-the-shelf localizer (no training). We take the
union of the left+lung masks, expand to a padded bounding box, crop, then hand the
crop to the SAME mlx preprocessing/inference path as predict_mlx.py. If segmentation
fails for an image (empty mask), we fall back to the full image and count it.

Writes <output>/predictions.csv (standard schema). Run inside the mlx venv with
torchxrayvision + scikit-image installed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from mlx.modes.image_classification.data import (
    _default_classification_transform,
    resolve_evaluation_dir,
)
from mlx.modes.image_classification.utils import load_checkpoint_bundle

import torchxrayvision as xrv
import torchxrayvision.utils as xrv_utils

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PAD_FRAC = 0.05  # pad the lung bbox by 5% of image size on each side


def _slug(name: str) -> str:
    return name.replace("-", "_")


def lung_bbox(seg_model, img_gray: np.ndarray, device) -> tuple | None:
    """Return (y0,y1,x0,x1) lung bounding box from the xrv seg model, or None."""
    # xrv expects [-1024,1024]-normalized single-channel 512x512
    x = xrv_utils.normalize(img_gray.astype(np.float32), 255)
    x = x[None, ...]                       # 1xHxW
    x = torch.from_numpy(x)
    x = torch.nn.functional.interpolate(
        x[None], size=(512, 512), mode="bilinear", align_corners=False)[0]
    with torch.no_grad():
        out = seg_model(x[None].to(device))[0].cpu().numpy()  # CxHxW logits
    targets = seg_model.targets
    lung_idx = [i for i, t in enumerate(targets) if "Lung" in t]
    if not lung_idx:
        return None
    mask = (out[lung_idx] > 0).any(axis=0)         # 512x512 bool
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    # map 512-space bbox back to original resolution
    H, W = img_gray.shape
    sy, sx = H / 512.0, W / 512.0
    y0, y1 = ys.min() * sy, ys.max() * sy
    x0, x1 = xs.min() * sx, xs.max() * sx
    py, px = PAD_FRAC * H, PAD_FRAC * W
    y0 = max(0, int(y0 - py)); y1 = min(H, int(y1 + py))
    x0 = max(0, int(x0 - px)); x1 = min(W, int(x1 + px))
    if y1 - y0 < 16 or x1 - x0 < 16:
        return None
    return y0, y1, x0, x1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True, help="external root containing test/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = {"model": args.model, "model_path": args.weights,
              "dataset_path": args.dataset, "device": args.device, "batch_size": 1}
    model, metadata = load_checkpoint_bundle(config)
    model = model.to(args.device).eval()
    classes = metadata["classes"]
    input_size = tuple(metadata["input_size"])
    colored = metadata["colored"]
    # EXACT predict_mlx.py preprocessing (Resize + [Grayscale] + ToTensor + Normalize)
    transform = _default_classification_transform(input_size=input_size, colored=colored)

    seg = xrv.baseline_models.chestx_det.PSPNet().to(args.device).eval()

    eval_dir = Path(resolve_evaluation_dir(args.dataset))
    # iterate class folders in the fixed label order
    samples = []
    for label_idx, cname in enumerate(classes):
        d = eval_dir / cname
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXTS:
                samples.append((p, label_idx))

    rows = []
    n_fallback = 0
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for path, label in samples:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            n_fallback += 1
            continue
        bb = lung_bbox(seg, gray, args.device)
        if bb is None:
            crop = gray
            n_fallback += 1
        else:
            y0, y1, x0, x1 = bb
            crop = gray[y0:y1, x0:x1]
        pil = Image.fromarray(crop)
        pil = pil.convert("RGB") if colored else pil.convert("L")
        tensor = transform(pil)
        with torch.no_grad():
            logits = model(tensor[None].to(args.device))
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
        rows.append((int(label), *(float(v) for v in prob)))

    out_csv = Path(args.output) / "predictions.csv"
    header = ["label", *(f"p_{_slug(n)}" for n in classes)]
    with open(out_csv, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(header)
        for r in rows:
            w.writerow([r[0], *(f"{v:.6f}" for v in r[1:])])
    print(f"{args.model:>26}  lung-crop  {len(rows)} rows ({n_fallback} fallback) -> {out_csv}")


if __name__ == "__main__":
    main()
