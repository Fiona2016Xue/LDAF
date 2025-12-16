from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticConfig:
    n_samples: int = 12000
    window: int = 64
    d_flow: int = 16
    d_tele: int = 12
    topo_dim: int = 8
    anomaly_ratio: float = 0.15
    seed: int = 123


class SyntheticIoTDataset(Dataset):
    def __init__(self, cfg: SyntheticConfig):
        super().__init__()
        rng = np.random.default_rng(cfg.seed)

        n, T = cfg.n_samples, cfg.window
        z = rng.normal(size=(n, cfg.topo_dim)).astype(np.float32)

        x_flow = rng.normal(size=(n, T, cfg.d_flow)).astype(np.float32)
        x_tele = rng.normal(size=(n, T, cfg.d_tele)).astype(np.float32)

        x_flow += (z[:, None, : min(cfg.d_flow, cfg.topo_dim)] * 0.15).astype(np.float32)
        x_tele += (z[:, None, : min(cfg.d_tele, cfg.topo_dim)] * 0.10).astype(np.float32)

        y = (rng.uniform(size=n) < cfg.anomaly_ratio).astype(np.int64)

        for i in np.where(y == 1)[0]:
            mode = rng.integers(0, 3)
            if mode == 0:
                t0 = rng.integers(0, T)
                x_flow[i, t0:t0+3] += rng.normal(loc=3.0, scale=1.0, size=(min(3, T-t0), cfg.d_flow)).astype(np.float32)
            elif mode == 1:
                drift = np.linspace(0, 2.0, T, dtype=np.float32)[:, None]
                x_tele[i] += drift
            else:
                t = np.arange(T, dtype=np.float32)
                x_flow[i] += (0.8 * np.sin(2 * np.pi * t / 8.0))[:, None].astype(np.float32)

        self.x_flow = torch.tensor(x_flow)
        self.x_tele = torch.tensor(x_tele)
        self.z_topo = torch.tensor(z)
        self.y = torch.tensor(y)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.x_flow[idx], self.x_tele[idx], self.z_topo[idx], self.y[idx]
