from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ldaf.data.synthetic import SyntheticConfig, SyntheticIoTDataset
from ldaf.data.utils import split_dataset
from ldaf.metrics import compute_metrics
from ldaf.models.ldaf import LDAF
from ldaf.utils import get_device, load_yaml, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
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


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / cfg["output"]["best_ckpt"]

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
    ds_train, ds_val, ds_test = split_dataset(ds, seed=int(cfg["seed"]))

    train_loader = DataLoader(ds_train, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
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

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

    w_pos = float(cfg["train"].get("class_weight_pos", 1.0))
    class_weights = torch.tensor([1.0, w_pos], device=device)

    best_auprc = -1.0
    patience = int(cfg["train"]["early_stop_patience"])
    wait = 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for x_f, x_t, z, y in pbar:
            x_f = x_f.to(device)
            x_t = x_t.to(device)
            z = z.to(device)
            y = y.to(device)

            logits = model(x_f, x_t, z)
            loss = torch.nn.functional.cross_entropy(logits, y, weight=class_weights)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        val_metrics = evaluate(model, val_loader, device=device, threshold=float(cfg["eval"]["threshold"]))
        print(f"[Epoch {epoch}] val: AUROC={val_metrics['AUROC']:.4f} AUPRC={val_metrics['AUPRC']:.4f} F1={val_metrics['F1']:.4f}")

        if val_metrics["AUPRC"] > best_auprc:
            best_auprc = val_metrics["AUPRC"]
            wait = 0
            torch.save({"model": model.state_dict(), "config": cfg}, best_path)
            print(f"  Saved best checkpoint to: {best_path}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device=device, threshold=float(cfg["eval"]["threshold"]))
    print(f"[Best] test: AUROC={test_metrics['AUROC']:.4f} AUPRC={test_metrics['AUPRC']:.4f} F1={test_metrics['F1']:.4f} FPR@95TPR={test_metrics['FPR@95TPR']:.4f}")


if __name__ == "__main__":
    main()
