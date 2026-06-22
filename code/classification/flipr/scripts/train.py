"""Training entry point: fine-tune a TB classifier from a YAML config.

Usage:
    uv run python scripts/train.py --config experiments/configs/default.yaml
    uv run python scripts/train.py --config experiments/configs/default.yaml --fast-dev-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from tb_classifier.config import TrainingConfig, load_config
from tb_classifier.data import build_dataloaders
from tb_classifier.training import DriveBackupModelCheckpoint, TBLitModule


def _compute_class_weights(cfg: TrainingConfig, train_loader) -> torch.Tensor | None:
    """Inverse-frequency weights normalized so the mean weight is 1."""
    if cfg.class_weights is None:
        return None
    if cfg.class_weights != "balanced":
        raise ValueError(
            f"Unsupported class_weights mode: {cfg.class_weights!r} (expected 'balanced' or null)"
        )
    counts = train_loader.dataset.class_counts()
    total = sum(counts.values())
    n_classes = len(counts)
    return torch.tensor(
        [total / (n_classes * counts[i]) for i in range(n_classes)],
        dtype=torch.float32,
    )


def _loss_of(path: Path) -> float:
    """Sort key for ``<name>-loss=0.12345-epoch=NN.ckpt`` filenames."""
    try:
        return float(path.stem.split("loss=")[1].split("-")[0])
    except (IndexError, ValueError):
        return float("inf")


def _resolve_resume(resume: str | None, *dirs: Path) -> str | None:
    """Turn ``--resume`` into a concrete .ckpt path.

    ``"auto"`` searches the given dirs (in priority order, e.g. Drive backup
    before local) and resumes from the most recent complete epoch (``last.ckpt``),
    falling back to the highest-AUROC top-k checkpoint, or None to start fresh.
    Any other value is treated as an explicit path and returned as-is.
    """
    if resume != "auto":
        return resume
    for d in dirs:
        if d is None or not d.is_dir():
            continue
        last = d / "last.ckpt"
        if last.exists():
            return str(last)
        epoch_ckpts = list(d.glob("epoch=*.ckpt"))
        if epoch_ckpts:
            return str(min(epoch_ckpts, key=_loss_of))
    print("  --resume auto: no checkpoint found, starting from scratch")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/configs/default.yaml",
        help="Path to YAML run config.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a single train+val batch for wiring sanity (no W&B run).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Path to a .ckpt file to resume from (restores model, optimizer, scheduler, "
            "epoch), or 'auto' to pick the latest from the Drive backup / local ckpt dir."
        ),
    )
    parser.add_argument(
        "--drive-ckpt-dir",
        default=None,
        help="Mirror checkpoints to this dir (e.g. mounted Drive). Overrides config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    L.seed_everything(cfg.training.seed, workers=True)

    train_loader, val_loader = build_dataloaders(cfg.data)
    class_weights = _compute_class_weights(cfg.training, train_loader)
    if class_weights is not None:
        print(f"  class_weights (balanced): {class_weights.tolist()}")
    module = TBLitModule(cfg.model, cfg.training, class_weights=class_weights)

    run_name = cfg.training.wandb_run_name or cfg.name
    run_dir = Path(cfg.training.log_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    drive_ckpt_dir = args.drive_ckpt_dir or cfg.training.drive_ckpt_dir
    backup_dir = Path(drive_ckpt_dir) / run_name if drive_ckpt_dir else None

    logger = False
    if not args.fast_dev_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(
            project=cfg.training.wandb_project,
            name=run_name,
            save_dir=str(run_dir),
            config={"config_path": str(args.config), **cfg.__dict__},
        )

    if backup_dir is not None:
        print(f"  mirroring checkpoints to: {backup_dir}")
    callbacks = [
        DriveBackupModelCheckpoint(
            dirpath=str(ckpt_dir),
            backup_dir=backup_dir,
            filename=f"{run_name}-loss={{val/loss:.5f}}-epoch={{epoch:02d}}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=20,
        default_root_dir=str(run_dir),
    )
    resume_path = _resolve_resume(args.resume, backup_dir, ckpt_dir)
    if resume_path:
        print(f"  resuming from: {resume_path}")
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_path,
    )


if __name__ == "__main__":
    main()