"""Test whether a predictive sigma carries information beyond trailing volatility.

Rank statistics are computed per trading day across stocks and then averaged
over days, mirroring how IC is computed.  Confidence intervals use a moving
block bootstrap over days, because daily statistics are persistent and
same-day observations share the market factor.  Calibration scores (PICP,
ENCE) are pooled over all stock-days.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata

from research.data import load_dataset

logger = logging.getLogger(__name__)
COVERAGES = (1.0, 0.8, 0.6, 0.4, 0.2)
Z_90 = 1.6448536269514722
MIN_SPREAD = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", default=None, help="defaults to the dataset recorded in the run config")
    parser.add_argument("--volatility-window", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--sigma", choices=("mc_std", "aleatoric_std", "total"), default="mc_std")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def trailing_volatility(returns: np.ndarray, masks: np.ndarray, first_target: int, days: int, window: int) -> np.ndarray:
    """Std of the ``window`` returns that end the day before each target day.

    A return is only used when the stock traded on that day and the day
    before; otherwise it was computed against a placeholder price.
    """
    usable = masks.astype(bool).copy()
    usable[:, 1:] &= masks[:, :-1].astype(bool)
    usable[:, 0] = False
    clean = np.where(usable, returns, np.nan)
    columns = []
    for target in range(first_target, first_target + days):
        block = clean[:, target - window : target]
        enough = np.isfinite(block).sum(axis=1) >= 2
        column = np.full(block.shape[0], np.nan)
        column[enough] = np.nanstd(block[enough], axis=1)
        columns.append(column)
    return np.stack(columns, axis=1)


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def partial_rank_correlation(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Rank correlation of x and y after removing the rank of ``control`` from both."""
    if x.size < 4:
        return np.nan
    ranks = [rankdata(values) for values in (x, y, control)]
    design = np.column_stack([np.ones_like(ranks[2]), ranks[2]])
    residuals = [values - design @ np.linalg.lstsq(design, values, rcond=None)[0] for values in ranks[:2]]
    if np.ptp(residuals[0]) == 0 or np.ptp(residuals[1]) == 0:
        return np.nan
    return float(np.corrcoef(residuals[0], residuals[1])[0, 1])


def block_indices(days: int, block: int, draws: int, generator: np.random.Generator) -> np.ndarray:
    """Day indices of a circular moving-block bootstrap, shaped [draws, days]."""
    blocks = int(np.ceil(days / block))
    starts = generator.integers(0, days, size=(draws, blocks))
    indices = (starts[:, :, None] + np.arange(block)[None, None, :]) % days
    return indices.reshape(draws, -1)[:, :days]


def mean_with_interval(series: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    """Mean over days with a 95% interval from pre-drawn, shared day indices."""
    finite = np.isfinite(series)
    if not finite.any():
        raise ValueError("no finite daily values to summarise")
    samples = np.nanmean(np.where(finite, series, np.nan)[indices], axis=1)
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return {"mean": float(series[finite].mean()), "ci_low": float(low), "ci_high": float(high), "days": int(finite.sum())}


def selective_rank_ic(score: np.ndarray, target: np.ndarray, uncertainty: np.ndarray, tie_breaker: np.ndarray, valid: np.ndarray, coverage: float) -> float:
    """RankIC on the ``coverage`` share of eligible stocks with the lowest uncertainty."""
    index = np.flatnonzero(valid)
    order = np.lexsort((tie_breaker[index], uncertainty[index]))
    keep = index[order[: max(3, int(round(coverage * index.size)))]]
    return rank_correlation(score[keep], target[keep])


def expected_normalised_calibration_error(sigma: np.ndarray, error: np.ndarray, bins: int = 10) -> Dict[str, float]:
    order = np.argsort(sigma, kind="stable")
    terms, skipped = [], 0
    for chunk in np.array_split(order, bins):
        predicted = np.sqrt(np.mean(sigma[chunk] ** 2))
        if predicted < MIN_SPREAD:
            skipped += 1
            continue
        terms.append(abs(predicted - np.sqrt(np.mean(error[chunk] ** 2))) / predicted)
    return {"ence": float(np.mean(terms)) if terms else float("nan"), "bins_skipped": skipped}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run = Path(args.run_dir)
    config = json.loads((run / "config.json").read_text())
    if args.volatility_window > config["lookback"]:
        raise ValueError("volatility window must not exceed the model lookback, which bounds the tradability mask")
    arrays = dict(np.load(run / "test_predictions.npz"))
    needed = ("mc_std", "aleatoric_std") if args.sigma == "total" else (args.sigma,)
    if any(key not in arrays for key in needed):
        raise ValueError(f"run lacks {needed}; it was trained without the matching option")
    generator = np.random.default_rng(args.seed)

    sigma = np.sqrt(sum(arrays[key] ** 2 for key in needed))
    mean, target = arrays["prediction"], arrays["target"]
    point = arrays.get("point_prediction", mean)
    data = load_dataset(args.dataset or config["dataset"], config.get("features", "ohlcv"))
    if data.stock_count != mean.shape[0]:
        raise ValueError("dataset and predictions disagree on the number of stocks")
    volatility = trailing_volatility(data.returns, data.masks, config["valid_end"], mean.shape[1], args.volatility_window)
    mask = arrays["mask"].astype(bool) & np.isfinite(volatility)
    error = np.abs(target - mean)
    random_order = generator.random(mean.shape)
    days = range(mask.shape[1])
    indices = block_indices(mask.shape[1], args.block_days, args.bootstrap, generator)

    def per_day(function, *fields: np.ndarray) -> np.ndarray:
        return np.asarray([function(*(field[mask[:, day], day] for field in fields)) for day in days])

    series = {
        "sigma_vs_abs_error": per_day(rank_correlation, sigma, error),
        "volatility_vs_abs_error": per_day(rank_correlation, volatility, error),
        "sigma_vs_volatility": per_day(rank_correlation, sigma, volatility),
        "sigma_vs_abs_prediction": per_day(rank_correlation, sigma, np.abs(mean)),
        "sigma_vs_abs_error_given_volatility": per_day(partial_rank_correlation, sigma, error, volatility),
    }
    series["sigma_minus_volatility_vs_abs_error"] = series["sigma_vs_abs_error"] - series["volatility_vs_abs_error"]

    report: Dict[str, object] = {
        "run_dir": str(run),
        "block_days": args.block_days,
        "stock_days": int(mask.sum()),
        "scale": {
            "abs_target_mean": float(np.abs(target[mask]).mean()),
            "abs_prediction_mean": float(np.abs(mean[mask]).mean()),
            "sigma_mean": float(sigma[mask].mean()),
            "trailing_volatility_mean": float(volatility[mask].mean()),
        },
        "rank_correlations": {name: mean_with_interval(values, indices) for name, values in series.items()},
        "calibration_against_abs_error": {
            "picp_90_gaussian_sigma": float((error[mask] <= Z_90 * sigma[mask]).mean()),
            "picp_90_gaussian_volatility": float((error[mask] <= Z_90 * volatility[mask]).mean()),
            "sigma": expected_normalised_calibration_error(sigma[mask], error[mask]),
            "trailing_volatility": expected_normalised_calibration_error(volatility[mask], error[mask]),
        },
    }

    curves: Dict[str, List[Dict[str, float]]] = {name: [] for name in ("sigma", "trailing_volatility", "random", "sigma_minus_trailing_volatility")}
    for coverage in COVERAGES:
        daily = {
            name: np.asarray([selective_rank_ic(point[:, d], target[:, d], ordering[:, d], random_order[:, d], mask[:, d], coverage) for d in days])
            for name, ordering in (("sigma", sigma), ("trailing_volatility", volatility), ("random", random_order))
        }
        daily["sigma_minus_trailing_volatility"] = daily["sigma"] - daily["trailing_volatility"]
        for name, values in daily.items():
            curves[name].append({"coverage": coverage, **mean_with_interval(values, indices)})
    report["rank_ic_by_coverage"] = curves

    report["sigma"] = args.sigma
    (run / f"uncertainty_eval_{args.sigma}.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
