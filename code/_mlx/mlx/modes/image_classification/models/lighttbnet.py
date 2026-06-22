"""LightTBNet (Capellan-Martin et al., ISBI 2023, arXiv:2309.02140).

A lightweight residual CNN for tuberculosis screening on chest X-rays. The paper
specifies the *structure* but not the per-stage channel widths, so the channel
progression below is this repo's design choice, tuned to land near the paper's
reported ~1.467M parameters for the recommended N=4 configuration. Everything the
paper does specify is reproduced:

- N residual blocks (default N=4, the paper's best val-AUC / low-compute setting).
  Each block: two 3x3 convolutions, each followed by BatchNorm and ReLU, with a
  1x1-convolution skip connection, then a 2x2 stride-2 max-pool at the block end.
- a 1x1 convolution that reduces dimensionality, then
- an MLP head of two fully-connected layers.

It is built here as a standard mlx classifier (variable num_classes, 1- or
3-channel stem, adaptive pooling so the same architecture trains at this repo's
512x512 protocol as well as the paper's native 256x256). Loss/optimizer are
supplied by the mlx training loop, not the model, so LightTBNet trains under the
same unified baseline recipe as the other classifiers (the fair head-to-head the
paper's "retrain under this protocol" caveat calls for).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

# Per-block output channels. Paper does not specify widths; this progression is
# our design, chosen so N=4 sits near the paper's ~1.467M-parameter figure.
DEFAULT_BLOCK_CHANNELS: tuple[int, ...] = (32, 64, 128, 256)
REDUCTION_CHANNELS = 128   # output width of the 1x1 dimensionality-reduction conv
MLP_HIDDEN = 64            # hidden width of the two-layer MLP head


class LightTBResidualBlock(nn.Module):
    """Two 3x3 convs (BN+ReLU) with a 1x1-conv skip, then 2x2 max-pool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 conv skip branch to match channel counts (identity when shapes agree).
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + identity)
        return self.pool(x)


class LightTBNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        block_channels: Sequence[int] = DEFAULT_BLOCK_CHANNELS,
        reduction_channels: int = REDUCTION_CHANNELS,
        mlp_hidden: int = MLP_HIDDEN,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev = in_channels
        for channels in block_channels:
            blocks.append(LightTBResidualBlock(prev, channels))
            prev = channels
        self.features = nn.Sequential(*blocks)

        # 1x1 conv that reduces dimensionality before the classifier head.
        self.reduce = nn.Sequential(
            nn.Conv2d(prev, reduction_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(reduction_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(reduction_channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.reduce(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_lighttbnet(*, num_classes: int, colored: bool, pretrained: bool, config: dict | None = None):
    """Build LightTBNet. `config['lighttbnet_blocks']` sets N (default 4)."""
    config = config or {}
    num_blocks = int(config.get("lighttbnet_blocks", 4))
    channels = tuple(DEFAULT_BLOCK_CHANNELS[:num_blocks]) if num_blocks <= len(DEFAULT_BLOCK_CHANNELS) else (
        DEFAULT_BLOCK_CHANNELS + tuple(DEFAULT_BLOCK_CHANNELS[-1] for _ in range(num_blocks - len(DEFAULT_BLOCK_CHANNELS)))
    )
    if pretrained:
        # No published LightTBNet ImageNet/CXR weights to load; train from scratch.
        pass
    return LightTBNet(
        in_channels=3 if colored else 1,
        num_classes=num_classes,
        block_channels=channels,
    )
