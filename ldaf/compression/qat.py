from __future__ import annotations

import torch.nn as nn


def prepare_qat(model: nn.Module) -> nn.Module:
    try:
        import torch.ao.quantization as tq
        model.train()
        model.qconfig = tq.get_default_qat_qconfig("fbgemm")
        tq.prepare_qat(model, inplace=True)
    except Exception:
        pass
    return model


def convert_quantized(model: nn.Module) -> nn.Module:
    try:
        import torch.ao.quantization as tq
        model.eval()
        tq.convert(model, inplace=True)
    except Exception:
        pass
    return model
