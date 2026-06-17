"""DataLoader factories for the TB classifier."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from ..config import DataConfig
from .dataset import TBDataset
from .transforms import get_train_transforms, get_val_transforms


def build_dataloaders(
    cfg: DataConfig | None = None,
    *,
    root: str | Path | None = None,
    image_size: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    augment: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders.

    Pass either a ``DataConfig`` (preferred — comes from the YAML) or
    individual kwargs for ad-hoc / test use. Kwargs override the config
    when both are provided.
    """
    cfg = cfg or DataConfig()
    root = root if root is not None else cfg.root
    image_size = image_size if image_size is not None else cfg.image_size
    batch_size = batch_size if batch_size is not None else cfg.batch_size
    num_workers = num_workers if num_workers is not None else cfg.num_workers
    pin_memory = pin_memory if pin_memory is not None else cfg.pin_memory
    augment = augment if augment is not None else cfg.augment

    train_tf = get_train_transforms(image_size) if augment else get_val_transforms(image_size)
    train_ds = TBDataset(root, split="train", transform=train_tf)
    val_ds = TBDataset(root, split="val", transform=get_val_transforms(image_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def build_test_loader(
    cfg: DataConfig | None = None,
    *,
    root: str | Path | None = None,
    image_size: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    cfg = cfg or DataConfig()
    root = root if root is not None else cfg.root
    image_size = image_size if image_size is not None else cfg.image_size
    batch_size = batch_size if batch_size is not None else cfg.batch_size
    num_workers = num_workers if num_workers is not None else cfg.num_workers
    pin_memory = pin_memory if pin_memory is not None else cfg.pin_memory

    test_ds = TBDataset(root, split="test", transform=get_val_transforms(image_size))
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
