from __future__ import annotations

import torch
import torch.nn as nn


class GatedMultimodalFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * hidden_dim, 1)

    def forward(self, h_flow: torch.Tensor, h_tele: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([h_flow, h_tele], dim=-1)))
        return g * h_flow + (1.0 - g) * h_tele
