"""TB classifier: 3-stage FlipR ResNet18 backbone + linear head."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .backbone import Backbone


class TBClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        self.backbone = Backbone(pretrained=pretrained)
        self.head = nn.Linear(Backbone.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_classifier(cfg: ModelConfig) -> TBClassifier:
    return TBClassifier(num_classes=cfg.num_classes, pretrained=cfg.pretrained)
