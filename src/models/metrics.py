"""Point-forecast metrics (definitions in results/METHODS_FOR_REPORT.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    ybar = float(np.mean(y))
    ss_tot = float(np.sum((y - ybar) ** 2))
    return {
        "n": int(len(y)),
        "rmse": rmse, "mae": mae,
        "mape": float(np.mean(np.abs(e) / y) * 100),
        "r2": float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        "rmse_pct": rmse / ybar * 100, "mae_pct": mae / ybar * 100,
        "bias": float(np.mean(p - y)),
    }


def metrics_table(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pred.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key, strict=True)), **metrics(g.y_true, g.y_pred)})
    return pd.DataFrame(rows)
