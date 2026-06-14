"""3-stage ResNet18 with a FlipR asymmetry gate after layer2.

This is the only backbone — the project's whole architectural bet is the
FlipR block sitting between layer2 and layer3 of a depth-truncated ResNet18.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .blocks.flipr import FlipRBlock


class Backbone(nn.Module):
    """ResNet18 truncated to 3 stages with a FlipRBlock inserted after layer2.

    Output is a pooled ``(N, 256)`` feature vector. The 256 matches the last
    BasicBlock's ``conv2.out_channels`` in ResNet18 ``layer3``.
    """

    feat_dim: int = 256

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        src = resnet18(weights=weights)

        self.conv1 = src.conv1
        self.bn1 = src.bn1
        self.relu = src.relu
        self.maxpool = src.maxpool

        self.layer1 = src.layer1
        self.layer2 = src.layer2
        self.layer3 = src.layer3

        # FlipR sits between layer2 (out=128) and layer3.
        self.flipr = FlipRBlock(in_channels=128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flipr(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)
