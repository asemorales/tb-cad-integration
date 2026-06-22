"""Convert the YOLO-format TB detection set to COCO JSON for NanoDet.

NanoDet (RangiLyu/nanodet) trains from COCO-style annotations. This writes one
JSON per split, referencing the existing images in place (no copy), with the two
TB classes. Background images (empty label files) are included with zero
annotations, which is correct: NanoDet trains on them as negatives.

    python yolo_to_coco.py --root ../../../dataset/object-detection --out annotations

Produces annotations/instances_train.json and annotations/instances_val.json.
Categories: id 1 ActiveTuberculosis, id 2 ObsoletePulmonaryTuberculosis
(COCO category ids are 1-based; NanoDet maps them internally).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg")
CATEGORIES = [
    {"id": 1, "name": "ActiveTuberculosis", "supercategory": "tb"},
    {"id": 2, "name": "ObsoletePulmonaryTuberculosis", "supercategory": "tb"},
]


def convert_split(root: Path, split: str, out_dir: Path) -> Path:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images, annotations = [], []
    ann_id = 1
    img_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    for img_id, path in enumerate(img_paths, start=1):
        with Image.open(path) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": path.name, "width": w, "height": h})
        label_path = labels_dir / f"{path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            bw_abs, bh_abs = bw * w, bh * h
            if bw_abs < 1.0 or bh_abs < 1.0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls) + 1,  # YOLO 0/1 -> COCO 1/2
                "bbox": [x, y, bw_abs, bh_abs],  # COCO xywh
                "area": bw_abs * bh_abs,
                "iscrowd": 0,
            })
            ann_id += 1
    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"instances_{split}.json"
    out_path.write_text(json.dumps(coco))
    print(f"{split}: {len(images)} images, {len(annotations)} annotations -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../../../dataset/object-detection")
    ap.add_argument("--out", default="annotations")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    for split in ("train", "val"):
        convert_split(root, split, out_dir)


if __name__ == "__main__":
    main()
