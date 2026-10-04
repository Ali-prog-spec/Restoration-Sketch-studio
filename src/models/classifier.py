"""Corruption classifier (Task 2) — also the initialisation of the Task 3 gate.

Stages of [conv-BN-ReLU, conv-BN-ReLU, 2x2 max-pool]. The first stage keeps full
resolution before pooling so that pixel-level cues (isolated salt/pepper pixels,
loss of high-frequency detail from blur) are still visible to the filters.

The pooled feature is the concatenation of global AVERAGE and global MAX pooling:
average pooling captures global statistics (blur lowers edge energy everywhere),
while max pooling responds to sparse, extreme local evidence (a black rectangle,
isolated white/black pixels) that averaging would dilute.

Outputs raw logits of shape (B, 4) in the order clean, salt_pepper, blur, occlusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CorruptionClassifier(nn.Module):
    def __init__(self, base_channels: int = 32, channel_mults=(1, 2, 4, 8), dropout: float = 0.3,
                 num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        self.config = dict(base_channels=base_channels, channel_mults=list(channel_mults),
                           dropout=dropout, num_classes=num_classes, in_channels=in_channels)
        layers = []
        cin = in_channels
        for m in channel_mults:
            c = base_channels * m
            layers += [
                nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            cin = c
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * cin, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.fc(self.dropout(pooled))


def build_classifier(cfg: dict) -> CorruptionClassifier:
    return CorruptionClassifier(**cfg)
