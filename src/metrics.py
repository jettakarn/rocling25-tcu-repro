from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_va(y_true_v, y_pred_v, y_true_a, y_pred_a) -> dict[str, float]:
    return {
        "mae_v": mae(y_true_v, y_pred_v),
        "mae_a": mae(y_true_a, y_pred_a),
        "pcc_v": pcc(y_true_v, y_pred_v),
        "pcc_a": pcc(y_true_a, y_pred_a),
    }


def format_metrics(m: dict[str, float]) -> str:
    return (
        f"MAE_V={m['mae_v']:.3f}  MAE_A={m['mae_a']:.3f}  "
        f"PCC_V={m['pcc_v']:.3f}  PCC_A={m['pcc_a']:.3f}"
    )
