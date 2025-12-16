# LDAF — Lightweight Deep Anomaly Detection Framework (IoT)

This repository provides a **reproducible PyTorch scaffold** for the paper-style framework **LDAF**, featuring:

- **GMF**: *Gated Multimodal Fusion* (network-flow + device telemetry)
- **TAC**: *Topology-Aware Conditioning* (FiLM-style conditioning on graph descriptors)
- **Compression pipeline** (scaffolds): **KD → Structured Pruning → QAT**
- Training/evaluation scripts with standard detection metrics (**AUROC/AUPRC/F1**) and basic efficiency metrics (latency on CPU).

> Note: This is a research scaffold intended to be adapted to real datasets (e.g., N-BaIoT, TON_IoT). A synthetic dataset generator is included so you can run end-to-end immediately.

---

## Project structure

```text
LDAF/
├── ldaf/
│   ├── data/
│   │   ├── synthetic.py          # synthetic multimodal dataset + topology descriptors
│   │   └── utils.py              # normalization, train/val/test split helpers
│   ├── models/
│   │   ├── encoders.py           # Tiny-TCN encoder
│   │   ├── gmf.py                # Gated Multimodal Fusion
│   │   ├── tac.py                # Topology-Aware Conditioning (FiLM)
│   │   └── ldaf.py               # Full LDAF model (GMF + TAC + classifier)
│   ├── compression/
│   │   ├── distill.py            # Knowledge Distillation utilities
│   │   ├── pruning.py            # Channel pruning (structured) utilities
│   │   └── qat.py                # Quantization-aware training helpers
│   ├── metrics.py                # AUROC/AUPRC/F1/FPR@95
│   └── utils.py                  # config, seeding, device selection, timers
├── configs/
│   └── default.yaml              # default experimental config
├── scripts/
│   ├── train.py                  # training entrypoint
│   └── test.py                   # evaluation entrypoint
├── requirements.txt
└── pyproject.toml
```

---

## Quickstart (synthetic data)

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train

```bash
python scripts/train.py --config configs/default.yaml
```

### 3) Test

```bash
python scripts/test.py --config configs/default.yaml --ckpt outputs/best.pt
```

Outputs (logs + checkpoint) are written to `outputs/`.

---

## Adapting to N-BaIoT / TON_IoT

1. Implement a dataset class under `ldaf/data/` that returns:
   - `x_flow`: `Tensor[T, d_f]`
   - `x_tele`: `Tensor[T, d_s]`
   - `z_topo`: `Tensor[m]`
   - `y`: label `{0,1}`
2. Replace `SyntheticIoTDataset` usage in `scripts/train.py` with your dataset.
3. Keep evaluation unchanged—metrics are dataset-agnostic.

---

## License

MIT License. See `LICENSE`.
