from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ldaf.data.synthetic import SyntheticConfig, SyntheticIoTDataset
from ldaf.data.utils import split_dataset
from ldaf.metrics import compute_metrics
from ldaf.models.ldaf import LDAF
from ldaf.utils import get_device, load_yaml, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--latency_runs", type=int, default=200)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    ys, ps = [], []
    for x_f, x_t, z, y in loader:
        x_f = x_f.to(device)
        x_t = x_t.to(device)
        z = z.to(device)
        logits = model(x_f, x_t, z)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        ys.append(y.cpu().numpy())
        ps.append(prob.cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(ps)
    return compute_metrics(y_true, y_score, threshold=threshold)


@torch.no_grad()
def measure_latency_ms(model, sample, device, runs=200):
    model.eval()
    x_f, x_t, z = sample
    x_f = x_f.unsqueeze(0).to(device)
    x_t = x_t.unsqueeze(0).to(device)
    z = z.unsqueeze(0).to(device)

    for _ in range(20):
        _ = model(x_f, x_t, z)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(runs):
        _ = model(x_f, x_t, z)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) * 1000.0 / runs


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))

    dcfg = SyntheticConfig(
        n_samples=int(cfg["data"]["n_samples"]),
        window=int(cfg["data"]["window"]),
        d_flow=int(cfg["data"]["d_flow"]),
        d_tele=int(cfg["data"]["d_tele"]),
        topo_dim=int(cfg["data"]["topo_dim"]),
        anomaly_ratio=float(cfg["data"]["anomaly_ratio"]),
        seed=int(cfg["seed"]),
    )
    ds = SyntheticIoTDataset(dcfg)
    _, _, ds_test = split_dataset(ds, seed=int(cfg["seed"]))
    test_loader = DataLoader(ds_test, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    mcfg = cfg["model"]
    model = LDAF(
        d_flow=dcfg.d_flow,
        d_tele=dcfg.d_tele,
        topo_dim=dcfg.topo_dim,
        hidden_dim=int(mcfg["hidden_dim"]),
        layers=int(mcfg["layers"]),
        dropout=float(mcfg["dropout"]),
        use_gmf=bool(mcfg["use_gmf"]),
        use_tac=bool(mcfg["use_tac"]),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    metrics = evaluate(model, test_loader, device=device, threshold=float(cfg["eval"]["threshold"]))
    print("Test metrics:", metrics)

    x_f, x_t, z, _ = ds_test[0]
    latency_ms = measure_latency_ms(model, (x_f, x_t, z), device=device, runs=int(args.latency_runs))
    print(f"Latency: {latency_ms:.3f} ms/sample on device={device}")


if __name__ == "__main__":
    main()
