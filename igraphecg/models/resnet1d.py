"""1D ResNet baseline (B1): 12-lead median beat [B, 12, T] in, 4-class logits out."""
from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class ResNet1D(nn.Module):
    def __init__(self, in_ch: int = 12, n_classes: int = 4,
                 widths: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, widths[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers = []
        prev = widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers.append(BasicBlock1d(prev, w, stride=stride))
            layers.append(BasicBlock1d(w, w, stride=1))
            prev = w
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(prev, n_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
