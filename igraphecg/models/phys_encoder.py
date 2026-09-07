"""Physical-parameter observer encoder (ROUND2 §8 Step3): 12-lead ECG -> z (unbounded params).

Reuses the ResNet1D backbone with the output dimension set to PARAM_DIM; decoder.pspace.theta
then maps z to the bounded theta.
"""
from __future__ import annotations

import torch.nn as nn

from .resnet1d import BasicBlock1d
from .surrogate_decoder import PARAM_DIM


class PhysEncoder(nn.Module):
    def __init__(self, in_ch: int = 12, out_dim: int = PARAM_DIM,
                 widths: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, widths[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(widths[0]), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers, prev = [], widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers += [BasicBlock1d(prev, w, stride=stride), BasicBlock1d(w, w, stride=1)]
            prev = w
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(prev, out_dim))

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)   # z (unbounded)
