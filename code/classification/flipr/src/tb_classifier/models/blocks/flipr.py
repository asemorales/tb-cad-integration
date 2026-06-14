"""FlipR: a learned left–right asymmetry gate for chest X-ray features.

Many TB findings on a chest X-ray are unilateral (cavitations, infiltrates,
pleural effusions). FlipR exploits that by amplifying features that differ
from their horizontally-flipped counterpart and dampening features that
look the same on both sides. It is meant to sit between mid-level stages of
a backbone, after the receptive field is large enough for "left vs right
lung" to be meaningful but before the network collapses spatial structure.

Given features ``x`` of shape ``(N, C, H, W)``:

    asym = avgpool3x3(x - flip_horizontal(x))   # raw asymmetry signal, (N, C, H, W)
    gate = sigmoid(conv1x1(asym))               # per-pixel scalar in (0, 1), (N, 1, H, W)
    out  = x * (1 + gate)                       # residual-friendly amplification

Properties:
- Adds ``C + 1`` parameters (one 1×1 conv from C to 1) regardless of input size.
- Degrades to identity when the gate saturates at 0, so it can only help or no-op.
- Symmetric to a horizontal flip of the input as long as the upstream stem is
  symmetric, so it does not break standard horizontal-flip augmentation.
"""

import torch
from torch import nn


class FlipRBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
        self.blur = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flipped = torch.flip(x, dims=[-1])
        asym = self.blur(x - x_flipped)
        gate = self.act(self.conv(asym))
        return x * (1 + gate)
