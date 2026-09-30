"""Masked, cross-sectional prediction metrics for stock-ranking experiments."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import rankdata


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation, or NaN when it is undefined."""
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _safe_mean(values: list[float]) -> float:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return float(valid.mean()) if valid.size else np.nan


def _information_ratio(values: list[float]) -> float:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size < 2:
        return np.nan
    std = valid.std(ddof=1)
    return float(valid.mean() / std) if std > 0 else np.nan


def evaluate_predictions(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    top_k: int = 10,
) -> Dict[str, float]:
    """Evaluate returns shaped ``[stocks, dates]`` without masked-value leakage.

    IC and RankIC are calculated independently for every date across tradable
    stocks.  ICIR and RankICIR use the time-series standard deviation (ddof=1).
    """
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    valid = mask & np.isfinite(prediction) & np.isfinite(target)
    if not valid.any():
        raise ValueError("no valid observations")

    errors = prediction[valid] - target[valid]
    daily_ic: list[float] = []
    daily_rank_ic: list[float] = []
    daily_top_k_return: list[float] = []

    for day in range(prediction.shape[1]):
        day_valid = valid[:, day]
        pred_day = prediction[day_valid, day]
        target_day = target[day_valid, day]
        daily_ic.append(_correlation(pred_day, target_day))
        daily_rank_ic.append(_correlation(rankdata(pred_day), rankdata(target_day)))
        if pred_day.size:
            chosen = np.argsort(pred_day)[-min(top_k, pred_day.size):]
            daily_top_k_return.append(float(target_day[chosen].mean()))

    return {
        "mse": float(np.mean(errors**2)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "ic": _safe_mean(daily_ic),
        "icir": _information_ratio(daily_ic),
        "rank_ic": _safe_mean(daily_rank_ic),
        "rank_icir": _information_ratio(daily_rank_ic),
        "top_k_mean_return": _safe_mean(daily_top_k_return),
        "n_observations": int(valid.sum()),
        "n_days": int(np.any(valid, axis=0).sum()),
    }
