"""LightningModule wrapping the TB classifier for fine-tuning."""

from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..config import ModelConfig, TrainingConfig
from ..models import build_classifier
from ..utils.metrics import build_metrics, flatten_metrics


class TBLitModule(L.LightningModule):
    def __init__(
        self,
        model_cfg: ModelConfig,
        training_cfg: TrainingConfig,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        # `load_from_checkpoint` will round-trip class_weights through hparams, where
        # tensors get serialized as lists — accept either form.
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        # Hyperparameters end up in the W&B run config + checkpoint payload.
        hparams: dict = {"model": vars(model_cfg), "training": vars(training_cfg)}
        if class_weights is not None:
            hparams["class_weights"] = class_weights.tolist()
        self.save_hyperparameters(hparams)

        self.model_cfg = model_cfg
        self.training_cfg = training_cfg

        self.model = build_classifier(model_cfg)
        # CrossEntropyLoss registers `weight` as a buffer; Lightning moves it with the module.
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.train_metrics = build_metrics(model_cfg.num_classes, prefix="train/")
        self.val_metrics = build_metrics(model_cfg.num_classes, prefix="val/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: dict, metrics) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = batch["image"]
        labels = batch["label"]
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        metrics.update(logits.detach(), labels)
        return loss, logits, labels

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._step(batch, self.train_metrics)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["image"].size(0))
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._step(batch, self.val_metrics)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["image"].size(0))
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(flatten_metrics(self.train_metrics.compute()), prog_bar=False)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(flatten_metrics(self.val_metrics.compute()), prog_bar=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.training_cfg.lr,
            weight_decay=self.training_cfg.weight_decay,
        )
        if self.training_cfg.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.training_cfg.epochs)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        if self.training_cfg.scheduler == "none":
            return optimizer
        raise ValueError(f"Unknown scheduler: {self.training_cfg.scheduler!r}")
