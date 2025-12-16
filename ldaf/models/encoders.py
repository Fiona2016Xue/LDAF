from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, groups=in_ch, dilation=dilation, padding=padding)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class TinyTCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, layers: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        chs = [in_dim] + [hidden_dim] * layers
        blocks = []
        for i in range(layers):
            blocks.append(DepthwiseSeparableConv1d(chs[i], chs[i+1], kernel_size=kernel_size, dilation=2**i))
            blocks.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B,T,d] -> [B,d,T]
        h = self.net(x)        # [B,D,T]
        return h.mean(dim=-1)  # [B,D]
