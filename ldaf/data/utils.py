from __future__ import annotations

import torch
from torch.utils.data import Dataset, random_split


def split_dataset(ds: Dataset, ratios=(0.7, 0.15, 0.15), seed: int = 42):
    assert abs(sum(ratios) - 1.0) < 1e-6
    n = len(ds)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val, n_test], generator=g)
