"""Leakage-safe access to the preprocessed StockMixer EOD datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EODData:
    features: np.ndarray
    masks: np.ndarray
    returns: np.ndarray
    prices: np.ndarray
    # Point-in-time universe flags; ``None`` means every stock is always eligible.
    member: Optional[np.ndarray] = None
    # True when inputs are raw price levels that must be scaled inside each window.
    window_normalised: bool = False
    # Number of leading channels that are price levels; the rest are volume-like.
    price_channels: int = 4

    @property
    def stock_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def date_count(self) -> int:
        return int(self.features.shape[1])


def load_nasdaq(dataset_dir: str | Path) -> EODData:
    """Load NASDAQ data exactly once and validate its aligned dimensions."""
    base = Path(dataset_dir)
    names = {"features": "eod_data.pkl", "masks": "mask_data.pkl", "returns": "gt_data.pkl", "prices": "price_data.pkl"}
    loaded = {}
    for key, filename in names.items():
        with (base / filename).open("rb") as handle:
            loaded[key] = pickle.load(handle)
    data = EODData(**loaded)
    expected = data.features.shape[:2]
    if any(array.shape != expected for array in (data.masks, data.returns, data.prices)):
        raise ValueError("dataset arrays do not share [stock, date] dimensions")
    if not np.isfinite(data.features).all():
        raise ValueError("features contain non-finite values")
    return data


MOVING_AVERAGE_DAYS = (5, 10, 20, 30)


def moving_average_features(close: np.ndarray) -> np.ndarray:
    """``[MA5, MA10, MA20, MA30, close]`` as in the original StockMixer NASDAQ data.

    Each average uses past and current closes only and needs at least half of
    its window to have traded; the result is NaN wherever the close is NaN.
    """
    frame = pd.DataFrame(close.T)
    channels = [frame.rolling(days, min_periods=(days + 1) // 2).mean().to_numpy().T for days in MOVING_AVERAGE_DAYS]
    stacked = np.stack(channels + [close], axis=2).astype(np.float32)
    stacked[np.isnan(close)] = np.nan
    return stacked


def load_panel(dataset_dir: str | Path, features: str = "ohlcv") -> EODData:
    """Load a panel written by ``research.ashare_build``."""
    with np.load(Path(dataset_dir) / "panel.npz", allow_pickle=False) as panel:
        raw = panel["features"]
        if features not in ("ohlcv", "ma"):
            raise ValueError(f"unknown feature set {features}")
        data = EODData(
            features=raw if features == "ohlcv" else moving_average_features(raw[:, :, 3]),
            masks=panel["mask"],
            returns=panel["returns"],
            prices=raw[:, :, 3],
            member=panel["member"],
            window_normalised=True,
            price_channels=4 if features == "ohlcv" else 5,
        )
    expected = data.features.shape[:2]
    if any(array.shape != expected for array in (data.masks, data.returns, data.member)):
        raise ValueError("panel arrays do not share [stock, date] dimensions")
    return data


def load_dataset(dataset_dir: str | Path, features: str = "ohlcv") -> EODData:
    """Pick the loader from the files present in ``dataset_dir``."""
    if (Path(dataset_dir) / "panel.npz").exists():
        return load_panel(dataset_dir, features)
    return load_nasdaq(dataset_dir)


def forward_return(data: EODData, base_index: int, horizon: int):
    """Compounded return over the ``horizon`` days after ``base_index`` and whether all of them traded."""
    block = slice(base_index + 1, base_index + horizon + 1)
    traded = data.masks[:, block].min(axis=1) > 0
    if horizon == 1:
        # Returned unchanged so that one-day runs stay bit-identical to earlier results.
        return data.returns[:, block].astype(np.float32), traded
    compounded = np.prod(1.0 + data.returns[:, block], axis=1) - 1.0
    return np.where(traded, compounded, 0.0).astype(np.float32)[:, None], traded


def _window_normalised_batch(data: EODData, offset: int, lookback: int, horizon: int):
    """Scale prices by the last close and volume by the mean volume of the window."""
    base_index = offset + lookback - 1
    target, traded = forward_return(data, base_index, horizon)
    window = data.features[:, offset : offset + lookback, :]
    complete = np.isfinite(window).all(axis=(1, 2))
    tradable = complete & traded & data.member[:, base_index]
    prices = data.price_channels
    last_close = np.where(complete, data.prices[:, base_index], 1.0)[:, None]
    scaled = window[:, :, :prices] / last_close[:, :, None]
    if prices < window.shape[2]:
        mean_volume = np.where(complete, window[:, :, prices:].mean(axis=(1, 2)), 1.0)[:, None]
        scaled = np.concatenate([scaled, window[:, :, prices:] / mean_volume[:, :, None]], axis=2)
    # Stocks with a gap in the window are excluded by the mask; give them neutral inputs.
    scaled = np.where(complete[:, None, None], scaled, 1.0).astype(np.float32)
    return (
        scaled,
        tradable[:, None].astype(np.float32),
        np.ones((window.shape[0], 1), dtype=np.float32),
        target,
    )


def batch_at(data: EODData, offset: int, lookback: int, horizon: int = 1):
    """Return one date's inputs and target, using only information through t."""
    target_index = offset + lookback + horizon - 1
    if offset < 0 or target_index >= data.date_count:
        raise IndexError("offset and lookback exceed dataset bounds")
    if data.window_normalised:
        return _window_normalised_batch(data, offset, lookback, horizon)
    tradable = data.masks[:, offset : target_index + 1].min(axis=1, keepdims=True)
    return (
        data.features[:, offset : offset + lookback, :],
        tradable.astype(np.float32),
        data.prices[:, offset + lookback - 1 : offset + lookback],
        forward_return(data, offset + lookback - 1, horizon)[0],
    )
