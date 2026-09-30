"""Add the observation-noise term of Gal & Ghahramani (2016) to the MC Dropout variance (X7).

The paper's predictive variance is ``1/tau`` plus the sample variance of the
stochastic passes.  Here the noise term is fitted on the validation period by
maximum Gaussian log-likelihood and scored once on the test period, next to
variance models that use no MC information at all.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from research.aggregate import load_runs
from research.data import load_dataset
from research.uncertainty_eval import Z_90, block_indices, mean_with_interval, trailing_volatility

logger = logging.getLogger(__name__)
LOG_2PI = float(np.log(2.0 * np.pi))
MIN_VARIANCE = 1e-10

# name -> (uses the model mean, variance as a function of (theta, mc_variance, volatility_squared))
MODELS: Dict[str, Tuple[bool, Callable[[float, np.ndarray, np.ndarray], np.ndarray]]] = {
    "mc_only": (True, lambda theta, mc, vol: mc),
    "mc_plus_noise": (True, lambda theta, mc, vol: theta + mc),
    "mc_scaled": (True, lambda theta, mc, vol: theta * mc),
    "constant": (False, lambda theta, mc, vol: np.full_like(mc, theta)),
    "volatility": (False, lambda theta, mc, vol: theta * vol),
    "volatility_with_model_mean": (True, lambda theta, mc, vol: theta * vol),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def log_likelihood(error: np.ndarray, variance: np.ndarray) -> np.ndarray:
    variance = np.maximum(variance, MIN_VARIANCE)
    return -0.5 * (LOG_2PI + np.log(variance) + error**2 / variance)


def fit_theta(error: np.ndarray, mc: np.ndarray, vol: np.ndarray, variance: Callable) -> float:
    """One positive parameter, searched on a log scale."""
    result = minimize_scalar(lambda log_theta: -log_likelihood(error, variance(np.exp(log_theta), mc, vol)).mean(), bounds=(-25.0, 15.0), method="bounded")
    return float(np.exp(result.x))


def split_arrays(run: Dict, split: str, data, window: int) -> Dict[str, np.ndarray]:
    config = run["config"]
    arrays = dict(np.load(Path(run["path"]) / f"{split}_predictions.npz"))
    first_target = config["train_end"] if split == "valid" else config["valid_end"]
    volatility = trailing_volatility(data.returns, data.masks, first_target, arrays["target"].shape[1], window)
    mask = arrays["mask"].astype(bool) & np.isfinite(volatility) & (volatility > 0)
    return {"mean": arrays["prediction"], "mc": arrays["mc_std"] ** 2, "vol": volatility**2, "target": arrays["target"], "mask": mask}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    generator = np.random.default_rng(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for runs in load_runs(args.run_dirs).values():
        if "mc_std" not in runs[0]:
            continue
        data = load_dataset(runs[0]["config"]["dataset"], runs[0]["config"].get("features", "ohlcv"))
        for run in runs:
            valid, test = (split_arrays(run, split, data, run["config"]["lookback"]) for split in ("valid", "test"))
            indices = block_indices(test["target"].shape[1], args.block_days, args.bootstrap, generator)
            daily: Dict[str, np.ndarray] = {}
            for name, (uses_mean, variance) in MODELS.items():
                def pieces(split: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    centre = split["mean"] if uses_mean else 0.0
                    return (split["target"] - centre)[split["mask"]], split["mc"][split["mask"]], split["vol"][split["mask"]]

                theta = 1.0 if name == "mc_only" else fit_theta(*pieces(valid), variance)
                error, mc, vol = pieces(test)
                spread = np.maximum(variance(theta, mc, vol), MIN_VARIANCE)
                scores = np.full(test["mask"].shape, np.nan)
                scores[test["mask"]] = log_likelihood(error, spread)
                daily[name] = np.nanmean(scores, axis=0)
                rows.append({
                    "group": run["label"], "seed": run["config"]["seed"], "model": name, "theta": theta,
                    "test_log_likelihood": float(np.nanmean(scores)),
                    "test_coverage_90": float((np.abs(error) <= Z_90 * np.sqrt(spread)).mean()),
                    "mean_sigma": float(np.sqrt(spread).mean()),
                    "mc_share_of_variance": float((mc / spread).mean()) if name == "mc_plus_noise" else float("nan"),
                })
            for other in ("volatility", "volatility_with_model_mean", "constant", "mc_scaled"):
                interval = mean_with_interval(daily["mc_plus_noise"] - daily[other], indices)
                rows.append({"group": run["label"], "seed": run["config"]["seed"], "model": f"mc_plus_noise_minus_{other}", "test_log_likelihood": interval["mean"], "ci_low": interval["ci_low"], "ci_high": interval["ci_high"]})

    table = pd.DataFrame(rows)
    table.to_csv(output / "noise_term.csv", index=False)
    (output / "noise_term_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    summary = table.groupby(["group", "model"], sort=False).mean(numeric_only=True).drop(columns="seed")
    logger.info("X7 by group\n%s", summary.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
