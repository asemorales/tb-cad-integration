"""TBX11K 3-class classification dataset (folder layout).

Expected directory layout under ``root``:

    <root>/<split>/<class>/<image>.png

with ``split`` in {"train", "val", "test"} and ``class`` in
{"healthy", "sick-non-tb", "tb"}. Class label mapping:
    0 = healthy
    1 = sick non-TB
    2 = TB
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_HEALTHY = 0
LABEL_SICK = 1
LABEL_TB = 2

_CLASS_TO_LABEL: dict[str, int] = {
    "healthy": LABEL_HEALTHY,
    "sick-non-tb": LABEL_SICK,
    "tb": LABEL_TB,
}

_VALID_SPLITS = ("train", "val", "test")
_IMG_EXTS = (".png", ".jpg", ".jpeg")


class TBDataset(Dataset):
    """Folder-backed TBX11K 3-class classification dataset.

    Parameters
    ----------
    root:
        Path to the dataset root containing ``train/``, ``val/``, ``test/``.
    split:
        One of {"train", "val", "test"}.
    transform:
        Albumentations Compose (or any callable) accepting
        ``image=np.ndarray`` and returning a dict with an ``image`` key
        (typically a tensor). If None, the raw RGB uint8 array is returned
        as a CHW float tensor in [0, 1].
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.transform = transform

        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.records: list[dict] = []
        for class_name, label in _CLASS_TO_LABEL.items():
            class_dir = self.split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in _IMG_EXTS:
                    continue
                self.records.append(
                    {
                        "path": path,
                        "label": label,
                        "file_name": f"{class_name}/{path.name}",
                    }
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        image = cv2.imread(str(rec["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {rec['path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "label": int(rec["label"]),
            "file_name": rec["file_name"],
        }

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
        for r in self.records:
            counts[r["label"]] += 1
        return counts
