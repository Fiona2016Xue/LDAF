from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor,
                      y_true: torch.Tensor,
                      alpha: float = 0.5,
                      temperature: float = 2.0,
                      class_weights: torch.Tensor | None = None) -> torch.Tensor:
    ce = F.cross_entropy(student_logits, y_true, weight=class_weights)
    t = temperature
    p_t = F.softmax(teacher_logits / t, dim=-1)
    log_p_s = F.log_softmax(student_logits / t, dim=-1)
    kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (t * t)
    return (1.0 - alpha) * ce + alpha * kd
