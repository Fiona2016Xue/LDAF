from __future__ import annotations

import torch
import torch.nn as nn


def channel_l1_importance(weight: torch.Tensor) -> torch.Tensor:
    return weight.abs().view(weight.shape[0], -1).sum(dim=1)


def structured_prune_linear(layer: nn.Linear, sparsity: float) -> nn.Linear:
    assert 0.0 <= sparsity < 1.0
    W = layer.weight.data
    b = layer.bias.data if layer.bias is not None else None

    imp = channel_l1_importance(W)
    k = int((1.0 - sparsity) * W.shape[0])
    k = max(1, k)

    keep = torch.topk(imp, k=k, largest=True).indices.sort().values
    new = nn.Linear(layer.in_features, k, bias=(b is not None))
    new.weight.data.copy_(W[keep])
    if b is not None:
        new.bias.data.copy_(b[keep])
    return new
