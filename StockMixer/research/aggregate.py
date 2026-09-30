"""Summarise multi-seed runs: mean/std per group, ensembles, and tests.

Every run directory must hold ``config.json`` and ``test_predictions.npz`` as
written by ``research.run_experiment``.  Runs are grouped by every setting
except the seed, so different methods are never pooled.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

from research.metrics import evaluate_predictions

logger = logging.getLogger(__name__)
REPORTED = ("ic", "icir", "rank_ic", "rank_icir", "rmse", "top_k_mean_return")
IGNORED_KEYS = ("seed", "output_dir", "device", "device_used")
LABEL_KEYS = ("dropout", "dropout_sites", "select_metric", "mc_samples", "label", "features", "horizon")
# Values implied by runs that were made before an option existed.
DEFAULTS = {"dropout_sites": "all", "select_metric": "mse", "select_smoothing": 1, "min_epochs": 0, "label": "raw", "features": "ohlcv", "heteroscedastic": False, "beta": 0.5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True, help="directories that contain run folders")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-dropout", type=float, default=0.0)
    parser.add_argument("--newey-west-lags", type=int, default=5)
    return parser.parse_args()


def group_key(config: Dict) -> Tuple:
    settings = {**DEFAULTS, **config}
    return tuple(sorted((key, json.dumps(value)) for key, value in settings.items() if key not in IGNORED_KEYS))


def group_label(config: Dict) -> str:
    settings = {**DEFAULTS, **config}
    if settings["dropout"] == 0:
        return f"dropout=0.0 select={settings['select_metric']} label={settings['label']} features={settings['features']} horizon={settings['horizon']}"
    return " ".join(f"{key.replace('_metric', '').replace('dropout_', '')}={settings[key]}" for key in LABEL_KEYS)


def load_runs(roots: List[str]) -> Dict[Tuple, List[Dict]]:
    groups: Dict[Tuple, List[Dict]] = {}
    for root in roots:
        for config_path in sorted(Path(root).glob("*/config.json")):
            prediction_path = config_path.parent / "test_predictions.npz"
            if not prediction_path.exists():
                logger.warning("skipping unfinished run %s", config_path.parent)
                continue
            config = json.loads(config_path.read_text())
            run = {"path": str(config_path.parent), "config": config, "label": group_label(config), **dict(np.load(prediction_path))}
            groups.setdefault(group_key(config), []).append(run)
    reference = next(iter(groups.values()))[0] if groups else None
    for members in groups.values():
        for run in members:
            if not (np.array_equal(run["target"], reference["target"]) and np.array_equal(run["mask"], reference["mask"])):
                raise ValueError(f"{run['path']} was evaluated on different test data than {reference['path']}")
    return groups


def daily_rank_ic(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = []
    for day in range(prediction.shape[1]):
        valid = mask[:, day].astype(bool)
        if valid.sum() < 2 or np.ptp(prediction[valid, day]) == 0:
            values.append(np.nan)
            continue
        values.append(np.corrcoef(rankdata(prediction[valid, day]), rankdata(target[valid, day]))[0, 1])
    return np.asarray(values)


def newey_west_t(series: np.ndarray, lags: int) -> float:
    """t-statistic of the mean with a Bartlett-kernel HAC variance."""
    count = series.size
    if count < lags + 2:
        return float("nan")
    centred = series - series.mean()
    variance = centred @ centred / count
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * (centred[lag:] @ centred[:-lag]) / count
    return float(series.mean() / np.sqrt(variance / count)) if variance > 0 else float("nan")


def prediction_variants(run: Dict) -> Dict[str, str]:
    variants = {"point": "point_prediction" if "point_prediction" in run else "prediction"}
    if "mc_std" in run:
        variants["mc_mean"] = "prediction"
    return variants


def summarise_group(runs: List[Dict]) -> List[Dict]:
    """Return rows for each prediction type and for its seed ensemble."""
    rows = []
    for label, key in prediction_variants(runs[0]).items():
        metrics = [evaluate_predictions(run[key], run["target"], run["mask"]) for run in runs]
        row = {"group": runs[0]["label"], "prediction": label, "seeds": len(runs)}
        for metric in REPORTED:
            values = np.asarray([item[metric] for item in metrics])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        rows.append(row)
        ensemble = np.mean([run[key] for run in runs], axis=0)
        ensemble_metrics = evaluate_predictions(ensemble, runs[0]["target"], runs[0]["mask"])
        rows.append({"group": runs[0]["label"], "prediction": f"{label}_ensemble", "seeds": len(runs), **{f"{metric}_mean": ensemble_metrics[metric] for metric in REPORTED}})
    return rows


def compare_with_baseline(runs: List[Dict], baseline: List[Dict], lags: int) -> List[Dict]:
    """Test a group against the baseline at day level and at seed level.

    The day-level tests hold the trained models fixed, so they ignore training
    randomness.  The seed-level Welch test treats each seed as one observation
    and is the one to quote for "method A beats method B".
    """
    def seed_series(members: List[Dict], key: str) -> np.ndarray:
        return np.stack([daily_rank_ic(run[key], run["target"], run["mask"]) for run in members])

    reference = seed_series(baseline, prediction_variants(baseline[0])["point"])
    rows = []
    for label, key in prediction_variants(runs[0]).items():
        candidate = seed_series(runs, key)
        difference = np.nan_to_num(np.nanmean(candidate, axis=0) - np.nanmean(reference, axis=0), nan=0.0)
        day_t, day_p = stats.ttest_1samp(difference, 0.0)
        seed_means = (np.nanmean(candidate, axis=1), np.nanmean(reference, axis=1))
        welch = stats.ttest_ind(seed_means[0], seed_means[1], equal_var=False) if min(map(len, seed_means)) > 1 else None
        rows.append({
            "group": runs[0]["label"], "prediction": label, "baseline": baseline[0]["label"],
            "rank_ic_difference": float(difference.mean()), "days": int(difference.size),
            "day_paired_t": float(day_t), "day_paired_p": float(day_p), "day_newey_west_t": newey_west_t(difference, lags),
            "seeds": len(runs), "baseline_seeds": len(baseline),
            "seed_welch_t": float(welch.statistic) if welch else float("nan"),
            "seed_welch_p": float(welch.pvalue) if welch else float("nan"),
        })
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    groups = load_runs(args.run_dirs)
    if not groups:
        raise ValueError("no finished runs found")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(groups.values(), key=lambda members: members[0]["label"])
    formatter = {"float_format": lambda value: f"{value:.4f}"}

    summary = pd.DataFrame([row for members in ordered for row in summarise_group(members)])
    summary.to_csv(output / "summary.csv", index=False)
    logger.info("summary\n%s", summary.to_string(index=False, **formatter))

    tests = []
    for members in ordered:
        config = {**DEFAULTS, **members[0]["config"]}
        if config["dropout"] == args.baseline_dropout:
            continue
        baselines = [
            other for other in ordered
            if {**DEFAULTS, **other[0]["config"]}["dropout"] == args.baseline_dropout
            and {**DEFAULTS, **other[0]["config"]}["select_metric"] == config["select_metric"]
        ]
        if len(baselines) != 1:
            logger.warning("no unique baseline with select_metric=%s for %s", config["select_metric"], members[0]["label"])
            continue
        tests.extend(compare_with_baseline(members, baselines[0], args.newey_west_lags))
    if tests:
        table = pd.DataFrame(tests)
        table.to_csv(output / "tests_vs_baseline.csv", index=False)
        logger.info("tests vs baseline\n%s", table.to_string(index=False, **formatter))


if __name__ == "__main__":
    main()
