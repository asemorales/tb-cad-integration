"""YAML-backed run configuration.

Each experiment is described by a single YAML file under
``experiments/configs/``. Load it with :func:`load_config` to get a typed
:class:`Config` object that can be passed around (or unpacked into
``build_dataloaders``, ``build_classifier``, etc.).
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/tbx11k"
    image_size: int = 512
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    # When False, the train split uses the val pipeline (resize + pad + normalize only),
    # disabling all random augmentations. Val/test are deterministic regardless.
    augment: bool = True


@dataclass
class ModelConfig:
    num_classes: int = 3
    pretrained: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "cosine" | "none"
    precision: str = "32"  # "32" | "16-mixed" | "bf16-mixed"
    seed: int = 42
    # "balanced" → inverse-frequency from train counts; null → unweighted
    class_weights: str | None = "balanced"
    log_dir: str = "experiments/results"
    wandb_project: str = "tb-classifier"
    wandb_run_name: str | None = None


@dataclass
class Config:
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _instantiate(cls: type, raw: dict[str, Any]):
    """Build a (possibly nested) dataclass from a plain dict, rejecting unknown keys."""
    if not is_dataclass(cls):
        return raw
    valid = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(valid)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in valid:
        if name not in raw:
            continue
        value = raw[name]
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _instantiate(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping at the top level")
    raw.setdefault("name", path.stem)
    return _instantiate(Config, raw)
