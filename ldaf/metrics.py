from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, roc_curve


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    f1 = f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    fpr95 = fpr_at_tpr(y_true, y_score, target_tpr=0.95)
    return {"AUROC": float(auroc), "AUPRC": float(auprc), "F1": float(f1), "FPR@95TPR": float(fpr95)}


def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, target_tpr, side="left")
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])
