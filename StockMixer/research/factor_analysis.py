"""What does the model rank on beyond simple price factors (X1), and is sigma tied to |prediction| (X2)?

Per trading day the prediction ranks are regressed on the ranks of short-term
reversal and volatility factors.  The RankIC of the residual is the part of
the signal that those factors do not explain.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from research.aggregate import load_runs
from research.data import EODData, load_dataset
from research.uncertainty_eval import block_indices, mean_with_interval, rank_correlation, trailing_volatility

logger = logging.getLogger(__name__)
REVERSAL_WINDOWS = (1, 5, 16)
VOLATILITY_WINDOW = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def simple_factors(data: EODData, first_target: int, days: int) -> Dict[str, np.ndarray]:
    """Factors known at the close of the day before each target day, oriented so that high is good."""
    targets = range(first_target, first_target + days)
    factors = {
        f"reversal_{window}d": np.stack([-(np.prod(1.0 + data.returns[:, t - window : t], axis=1) - 1.0) for t in targets], axis=1)
        for window in REVERSAL_WINDOWS
    }
    factors[f"low_volatility_{VOLATILITY_WINDOW}d"] = -trailing_volatility(data.returns, data.masks, first_target, days, VOLATILITY_WINDOW)
    return factors


def residual_ranks(prediction: np.ndarray, factors: List[np.ndarray]) -> np.ndarray:
    """Residual of the prediction rank after a least-squares fit on the factor ranks."""
    design = np.column_stack([np.ones(prediction.size)] + [rankdata(factor) for factor in factors])
    ranks = rankdata(prediction)
    return ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]


def analyse_prediction(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, factors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Daily series of raw RankIC, factor-neutral RankIC and the rank correlation with each factor."""
    days = prediction.shape[1]
    series = {name: np.full(days, np.nan) for name in ("rank_ic", "neutral_rank_ic", *[f"corr_{name}" for name in factors])}
    for day in range(days):
        valid = mask[:, day] & np.all([np.isfinite(factor[:, day]) for factor in factors.values()], axis=0)
        if valid.sum() < len(factors) + 3 or np.ptp(prediction[valid, day]) == 0:
            continue
        today = [factor[valid, day] for factor in factors.values()]
        series["rank_ic"][day] = rank_correlation(prediction[valid, day], target[valid, day])
        series["neutral_rank_ic"][day] = rank_correlation(residual_ranks(prediction[valid, day], today), target[valid, day])
        for name, factor in zip(factors, today):
            series[f"corr_{name}"][day] = rank_correlation(prediction[valid, day], factor)
    return series


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    generator = np.random.default_rng(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, coupling = [], []
    for runs in sorted(load_runs(args.run_dirs).values(), key=lambda members: members[0]["label"]):
        config = runs[0]["config"]
        data = load_dataset(config["dataset"], config.get("features", "ohlcv"))
        target, mask = runs[0]["target"], runs[0]["mask"].astype(bool)
        factors = simple_factors(data, config["valid_end"], target.shape[1])
        indices = block_indices(target.shape[1], args.block_days, args.bootstrap, generator)
        points = [run.get("point_prediction", run["prediction"]) for run in runs]

        per_seed = [analyse_prediction(point, target, mask, factors) for point in points]
        ensemble = analyse_prediction(np.mean(points, axis=0), target, mask, factors)
        for name in per_seed[0]:
            seed_means = np.asarray([np.nanmean(series[name]) for series in per_seed])
            interval = mean_with_interval(ensemble[name], indices)
            rows.append({
                "group": runs[0]["label"], "quantity": name, "seeds": len(runs),
                "single_model_mean": float(seed_means.mean()),
                "single_model_std": float(seed_means.std(ddof=1)) if len(runs) > 1 else float("nan"),
                "ensemble": interval["mean"], "ensemble_ci_low": interval["ci_low"], "ensemble_ci_high": interval["ci_high"],
            })

        if not rows or all(row["group"] != "simple factors" for row in rows):
            for name, factor in factors.items():
                series = np.asarray([rank_correlation(factor[mask[:, d] & np.isfinite(factor[:, d]), d], target[mask[:, d] & np.isfinite(factor[:, d]), d]) for d in range(target.shape[1])])
                interval = mean_with_interval(series, indices)
                rows.append({"group": "simple factors", "quantity": f"rank_ic_{name}", "seeds": 0, "ensemble": interval["mean"], "ensemble_ci_low": interval["ci_low"], "ensemble_ci_high": interval["ci_high"]})

        if "mc_std" in runs[0]:
            volatility = -factors[f"low_volatility_{VOLATILITY_WINDOW}d"]
            for run, point in zip(runs, points):
                eligible = mask & np.isfinite(volatility)
                days = range(target.shape[1])
                coupling.append({
                    "group": run["label"], "seed": run["config"]["seed"],
                    "sigma_vs_abs_prediction": float(np.nanmean([rank_correlation(run["mc_std"][eligible[:, d], d], np.abs(point[eligible[:, d], d])) for d in days])),
                    "sigma_vs_volatility": float(np.nanmean([rank_correlation(run["mc_std"][eligible[:, d], d], volatility[eligible[:, d], d]) for d in days])),
                    "sigma_mean": float(run["mc_std"][eligible].mean()),
                })

    table = pd.DataFrame(rows)
    table.to_csv(output / "factor_neutral.csv", index=False)
    logger.info("X1 factor-neutral RankIC\n%s", table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if coupling:
        detail = pd.DataFrame(coupling)
        detail.to_csv(output / "sigma_coupling_by_seed.csv", index=False)
        summary = detail.groupby("group")[["sigma_vs_abs_prediction", "sigma_vs_volatility", "sigma_mean"]].agg(["mean", "std", "min", "max"])
        summary.to_csv(output / "sigma_coupling_summary.csv")
        logger.info("X2 sigma coupling by seed\n%s", detail.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        logger.info("X2 summary\n%s", summary.to_string(float_format=lambda value: f"{value:.4f}"))
    (output / "factor_analysis_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
