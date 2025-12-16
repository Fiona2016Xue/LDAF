from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import TinyTCNEncoder
from .gmf import GatedMultimodalFusion
from .tac import TopologyAwareConditioning


class LDAF(nn.Module):
    def __init__(self, d_flow: int, d_tele: int, topo_dim: int, hidden_dim: int = 64, layers: int = 3, dropout: float = 0.1,
                 use_gmf: bool = True, use_tac: bool = True):
        super().__init__()
        self.use_gmf = use_gmf
        self.use_tac = use_tac

        self.enc_flow = TinyTCNEncoder(d_flow, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
        self.enc_tele = TinyTCNEncoder(d_tele, hidden_dim=hidden_dim, layers=layers, dropout=dropout)

        self.gmf = GatedMultimodalFusion(hidden_dim) if use_gmf else None
        self.tac = TopologyAwareConditioning(hidden_dim, topo_dim) if use_tac else None
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim) if not use_gmf else None

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x_flow: torch.Tensor, x_tele: torch.Tensor, z_topo: torch.Tensor) -> torch.Tensor:
        h_flow = self.enc_flow(x_flow)
        h_tele = self.enc_tele(x_tele)

        if self.use_gmf:
            h = self.gmf(h_flow, h_tele)
        else:
            h = torch.relu(self.proj(torch.cat([h_flow, h_tele], dim=-1)))

        if self.use_tac:
            h = self.tac(h, z_topo)

        return self.classifier(h)
