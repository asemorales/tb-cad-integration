"""Minimal detection transforms (PIL image + target dict -> tensor + target).

Kept dependency-free (no torchvision.transforms.v2 reliance) so it works on the
pinned stack. Training uses a horizontal flip; both splits convert to a float
tensor in [0,1]. The SSD model's own GeneralizedRCNNTransform then resizes to its
native input and applies ImageNet normalization, so we do not resize here.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1).item() < self.p:
            w = image.width if hasattr(image, "width") else image.shape[-1]
            image = F.hflip(image)
            boxes = target["boxes"]
            if boxes.numel():
                boxes = boxes.clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


def build_transforms(train: bool) -> Compose:
    ts = [ToTensor()]
    if train:
        ts.append(RandomHorizontalFlip(0.5))
    return Compose(ts)
