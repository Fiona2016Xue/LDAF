from __future__ import annotations

import torch
import torch.nn as nn


class TopologyAwareConditioning(nn.Module):
    def __init__(self, hidden_dim: int, topo_dim: int):
        super().__init__()
        self.gamma = nn.Linear(topo_dim, hidden_dim)
        self.beta = nn.Linear(topo_dim, hidden_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(z)
        beta = self.beta(z)
        return gamma * h + beta
