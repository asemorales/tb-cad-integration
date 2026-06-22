"""YOLO-format dataset adapter for torchvision detection models.

The detection dataset is stored YOLO-style:

    dataset/object-detection/
        images/{train,val}/<stem>.png
        labels/{train,val}/<stem>.txt   # one "cls cx cy w h" per line, normalized

torchvision detection models want, per image, a target dict with absolute-pixel
xyxy boxes and Int64 labels in 1..num_classes-1 (0 is reserved for background).
YOLO classes {0,1} therefore map to torchvision labels {1,2}, and num_classes=3.

Background images (empty .txt, ~91% of this set) yield boxes of shape [0,4] and
labels of shape [0]; torchvision SSD trains on those as all-negative samples,
which is what we want for a TB set dominated by no-finding frames.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg")


class YoloDetectionDataset(Dataset):
    def __init__(self, root: str | Path, split: str, transforms=None):
        root = Path(root)
        self.images_dir = root / "images" / split
        self.labels_dir = root / "labels" / split
        self.transforms = transforms
        self.items = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS
        )
        if not self.items:
            raise FileNotFoundError(f"no images under {self.images_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def _load_target(self, stem: str, w: int, h: int) -> dict:
        label_path = self.labels_dir / f"{stem}.txt"
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                cls, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                # clip and drop degenerate boxes
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(w), x2), min(float(h), y2)
                if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)  # YOLO 0/1 -> tv 1/2 (0=background)
        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        return {"boxes": boxes_t, "labels": labels_t}

    def __getitem__(self, idx: int):
        path = self.items[idx]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        target = self._load_target(path.stem, w, h)
        target["image_id"] = torch.tensor([idx])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def collate_fn(batch):
    """Detection batches are tuples of (list[image], list[target])."""
    return tuple(zip(*batch))
