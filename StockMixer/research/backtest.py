"""Cost-aware top-k backtest of uncertainty-adjusted stock selection.

Each group of seeds is combined into one forecast (the seed mean).  Rule
parameters are chosen on the validation period by net Sharpe and then applied
unchanged to the test period.  Eligibility uses information available at the
close of the decision day only; a holding that does not trade on the target
day earns a zero return.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from research.aggregate import load_runs
from research.data import EODData, load_dataset
from research.uncertainty_eval import block_indices, trailing_volatility

logger = logging.getLogger(__name__)
TRADING_DAYS = 252
ETAS = (0.25, 0.5, 1.0)
DROP_SHARES = (0.1, 0.2, 0.4)
# Fixed in advance (plan X3, X5); not tuned.
GATE_LOOKBACK, GATE_QUANTILE, GATE_RANDOM_DRAWS = 60, 0.8, 200
CONSENSUS_TOP_SHARE, CONSENSUS_MIN_SEEDS = 0.2, 4
# Turnover control (plan X8), chosen on the NASDAQ validation period and then frozen.
SCORE_AVERAGE_DAYS, EXIT_RANK = 5, 200
# Uncertainty-based sizing and filtering on top of the turnover-controlled book (plan D2).
SIZING_GAMMAS, SIZING_CAP = (0.5, 1.0), 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--cost-bps", type=float, nargs="+", default=[0.0, 10.0, 20.0], help="one-way cost per unit traded")
    parser.add_argument("--tuning-cost-bps", type=float, default=10.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def decision_eligibility(data: EODData, panel_dir: Path, first_target: int, days: int, lookback: int) -> np.ndarray:
    """Stocks that can be bought at the close of the day before each target day."""
    columns = []
    limit_up = None
    if data.window_normalised:
        with np.load(panel_dir / "panel.npz", allow_pickle=False) as panel:
            limit_up = panel["limit_up"]
    for target in range(first_target, first_target + days):
        base = target - 1
        if data.window_normalised:
            complete = np.isfinite(data.features[:, target - lookback : target, :]).all(axis=(1, 2))
            columns.append(complete & data.member[:, base] & (data.masks[:, base] > 0) & ~limit_up[:, base])
        else:
            columns.append(data.masks[:, target - lookback : target].min(axis=1) > 0)
    return np.stack(columns, axis=1)


def realised_returns(data: EODData, first_target: int, days: int) -> np.ndarray:
    block = slice(first_target, first_target + days)
    return np.where(data.masks[:, block] > 0, data.returns[:, block], 0.0)


def zscore(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score per day over eligible stocks."""
    masked = np.where(eligible, values, np.nan)
    centre = np.nanmean(masked, axis=0, keepdims=True)
    spread = np.nanstd(masked, axis=0, keepdims=True)
    return (values - centre) / np.where(spread > 0, spread, 1.0)


def rule_scores(mu: np.ndarray, sigma: np.ndarray, eligible: np.ndarray, rule: str, parameter: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return the ranking score and the candidate set of one rule."""
    if rule == "mu":
        return mu, eligible
    if rule == "penalty":
        return zscore(mu, eligible) - parameter * zscore(sigma, eligible), eligible
    if rule == "filter":
        threshold = np.nanquantile(np.where(eligible, sigma, np.nan), 1.0 - parameter, axis=0, keepdims=True)
        return mu, eligible & (sigma <= threshold)
    raise ValueError(f"unknown rule {rule}")


def consensus_candidates(points: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Stocks that enough seeds place in the top share of the eligible cross-section."""
    votes = np.zeros(eligible.shape, dtype=int)
    for point in points:
        threshold = np.nanquantile(np.where(eligible, point, np.nan), 1.0 - CONSENSUS_TOP_SHARE, axis=0, keepdims=True)
        votes += eligible & (point >= threshold)
    return eligible & (votes >= min(CONSENSUS_MIN_SEEDS, points.shape[0]))


def market_gate(history: np.ndarray, current: np.ndarray) -> np.ndarray:
    """True on days when trading is allowed.

    ``history`` is the market-level uncertainty of the days that precede
    ``current``.  A day is closed when its value exceeds the chosen quantile
    of the previous ``GATE_LOOKBACK`` days; the value itself is known at the
    close of the decision day.
    """
    series = np.concatenate([history, current])
    start = history.size
    allowed = np.ones(current.size, dtype=bool)
    for day in range(current.size):
        past = series[max(0, start + day - GATE_LOOKBACK) : start + day]
        past = past[np.isfinite(past)]
        if past.size >= GATE_LOOKBACK // 2:
            allowed[day] = series[start + day] <= np.quantile(past, GATE_QUANTILE)
    return allowed


def market_level(sigma: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    return np.nanmedian(np.where(eligible, sigma, np.nan), axis=0)


def top_k_weights(score: np.ndarray, candidates: np.ndarray, top_k: int) -> np.ndarray:
    weights = np.zeros_like(score, dtype=float)
    for day in range(score.shape[1]):
        index = np.flatnonzero(candidates[:, day] & np.isfinite(score[:, day]))
        if index.size == 0:
            continue
        chosen = index[np.argsort(score[index, day], kind="stable")[-min(top_k, index.size):]]
        weights[chosen, day] = 1.0 / chosen.size
    return weights


def trailing_mean_score(score: np.ndarray, days: int) -> np.ndarray:
    """Mean of the forecasts made on the last ``days`` decision days, including today."""
    smoothed = np.empty_like(score, dtype=float)
    for day in range(score.shape[1]):
        smoothed[:, day] = score[:, max(0, day - days + 1) : day + 1].mean(axis=1)
    return smoothed


def buffered_weights(score: np.ndarray, candidates: np.ndarray, top_k: int, exit_rank: int) -> np.ndarray:
    """Equal-weight holdings that are sold only when their rank falls below ``exit_rank``."""
    weights = np.zeros_like(score, dtype=float)
    held: list = []
    for day in range(score.shape[1]):
        index = np.flatnonzero(candidates[:, day] & np.isfinite(score[:, day]))
        order = index[np.argsort(-score[index, day], kind="stable")]
        rank = {stock: position for position, stock in enumerate(order)}
        held = [stock for stock in held if rank.get(stock, exit_rank) < exit_rank]
        kept = set(held)
        for stock in order:
            if len(held) >= top_k:
                break
            if stock not in kept:
                held.append(stock)
        if held:
            weights[held, day] = 1.0 / len(held)
    return weights


def inverse_uncertainty_weights(holdings: np.ndarray, sigma: np.ndarray, gamma: float) -> np.ndarray:
    """Re-weight the held stocks by ``1 / sigma**gamma``, capped at a multiple of equal weight."""
    weights = np.zeros_like(holdings, dtype=float)
    for day in range(holdings.shape[1]):
        held = np.flatnonzero(holdings[:, day] > 0)
        if held.size == 0:
            continue
        raw = 1.0 / np.maximum(sigma[held, day], 1e-8) ** gamma
        raw = raw / raw.sum()
        cap = SIZING_CAP / held.size
        for _ in range(held.size):
            over = raw > cap + 1e-12
            if not over.any():
                break
            free = ~over & (raw < cap)
            raw[over] = cap
            if free.any():
                raw[free] *= (1.0 - cap * (~free).sum()) / raw[free].sum()
        weights[held, day] = raw / raw.sum()
    return weights


def controlled_book(inputs: Dict[str, np.ndarray], top_k: int, source: str, rule: str, value: float) -> np.ndarray:
    """Turnover-controlled holdings, optionally filtered or sized by a smoothed sigma."""
    score = trailing_mean_score(inputs["mu"], SCORE_AVERAGE_DAYS)
    if rule == "equal":
        return buffered_weights(score, inputs["eligible"], top_k, EXIT_RANK)
    sigma = trailing_mean_score(inputs[f"sigma_{source}"], SCORE_AVERAGE_DAYS)
    if rule == "filter":
        threshold = np.nanquantile(np.where(inputs["eligible"], sigma, np.nan), 1.0 - value, axis=0, keepdims=True)
        return buffered_weights(score, inputs["eligible"] & (sigma <= threshold), top_k, EXIT_RANK)
    if rule == "size":
        return inverse_uncertainty_weights(buffered_weights(score, inputs["eligible"], top_k, EXIT_RANK), sigma, value)
    raise ValueError(f"unknown rule {rule}")


def sharpe_difference(candidate: np.ndarray, reference: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    samples = np.asarray([sharpe(candidate[row]) - sharpe(reference[row]) for row in indices])
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return {"sharpe_minus_equal": sharpe(candidate) - sharpe(reference), "difference_ci_low": float(low), "difference_ci_high": float(high)}


def portfolio_returns(weights: np.ndarray, returns: np.ndarray, cost_bps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Daily net return and traded amount of a portfolio rebalanced to ``weights`` every day."""
    previous = np.concatenate([np.zeros((weights.shape[0], 1)), weights[:, :-1]], axis=1)
    traded = np.abs(weights - previous).sum(axis=0)
    gross = (weights * returns).sum(axis=0)
    return gross - traded * cost_bps / 1e4, traded


def sharpe(series: np.ndarray) -> float:
    spread = series.std(ddof=1)
    return float(series.mean() / spread * np.sqrt(TRADING_DAYS)) if spread > 0 else float("nan")


def performance(net: np.ndarray, traded: np.ndarray, benchmark: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    wealth = np.cumprod(1.0 + net)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    samples = np.asarray([sharpe(net[row]) for row in indices])
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return {
        "annual_return": float(net.mean() * TRADING_DAYS),
        "annual_volatility": float(net.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(net),
        "sharpe_ci_low": float(low),
        "sharpe_ci_high": float(high),
        "max_drawdown": float(drawdown.min()),
        "annual_excess_return": float((net - benchmark).mean() * TRADING_DAYS),
        "daily_one_way_turnover": float(traded.mean() / 2.0),
    }


def split_inputs(runs: List[Dict], split: str, data: EODData, dataset_dir: Path, window: int) -> Dict[str, np.ndarray]:
    """Seed-mean forecast, the three sigma estimates, eligibility and realised returns of one split."""
    config = runs[0]["config"]
    arrays = [dict(np.load(Path(run["path"]) / f"{split}_predictions.npz")) for run in runs]
    points = np.stack([item.get("point_prediction", item["prediction"]) for item in arrays])
    days = points.shape[2]
    first_target = config["train_end"] if split == "valid" else config["valid_end"]
    sigmas = {
        "seed_ensemble": points.std(axis=0, ddof=1) if len(runs) > 1 else None,
        "mc_dropout": np.mean([item["mc_std"] for item in arrays], axis=0) if "mc_std" in arrays[0] else None,
        "aleatoric": np.mean([item["aleatoric_std"] for item in arrays], axis=0) if "aleatoric_std" in arrays[0] else None,
        "trailing_volatility": trailing_volatility(data.returns, data.masks, first_target, days, window),
    }
    eligible = decision_eligibility(data, dataset_dir, first_target, days, config["lookback"])
    for sigma in sigmas.values():
        if sigma is not None:
            eligible &= np.isfinite(sigma)
    return {"points": points, "mu": points.mean(axis=0), "eligible": eligible, "returns": realised_returns(data, first_target, days), **{f"sigma_{name}": sigma for name, sigma in sigmas.items() if sigma is not None}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    generator = np.random.default_rng(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for runs in sorted(load_runs(args.run_dirs).values(), key=lambda members: members[0]["label"]):
        config = runs[0]["config"]
        dataset_dir = Path(config["dataset"])
        data = load_dataset(dataset_dir, config.get("features", "ohlcv"))
        splits = {split: split_inputs(runs, split, data, dataset_dir, config["lookback"]) for split in ("valid", "test")}
        sources = [key[len("sigma_"):] for key in splits["test"] if key.startswith("sigma_")]
        candidates = [("mu", "none", 0.0)] + [(rule, source, value) for source in sources for rule, grid in (("penalty", ETAS), ("filter", DROP_SHARES)) for value in grid]

        def evaluate(split: str, rule: str, source: str, value: float, cost: float) -> Tuple[np.ndarray, np.ndarray]:
            inputs = splits[split]
            sigma = inputs.get(f"sigma_{source}", inputs["mu"])
            score, allowed = rule_scores(inputs["mu"], sigma, inputs["eligible"], rule, value)
            return portfolio_returns(top_k_weights(score, allowed, args.top_k), inputs["returns"], cost)

        tuned: Dict[Tuple[str, str], float] = {}
        for rule, source, value in candidates:
            net, _ = evaluate("valid", rule, source, value, args.tuning_cost_bps)
            if (rule, source) not in tuned or sharpe(net) > tuned[(rule, source)][1]:
                tuned[(rule, source)] = (value, sharpe(net))

        test = splits["test"]
        benchmark_weights = test["eligible"] / np.maximum(test["eligible"].sum(axis=0, keepdims=True), 1)
        benchmark = (benchmark_weights * test["returns"]).sum(axis=0)
        indices = block_indices(benchmark.size, args.block_days, args.bootstrap, generator)
        for (rule, source), (value, valid_sharpe) in tuned.items():
            for cost in args.cost_bps:
                net, traded = evaluate("test", rule, source, value, cost)
                rows.append({"group": runs[0]["label"], "seeds": len(runs), "rule": rule, "sigma": source, "parameter": value, "valid_sharpe_at_tuning_cost": valid_sharpe, "cost_bps": cost, "test_days": int(net.size), **performance(net, traded, benchmark, indices)})
        base_weights = top_k_weights(test["mu"], test["eligible"], args.top_k)
        controlled = buffered_weights(trailing_mean_score(test["mu"], SCORE_AVERAGE_DAYS), test["eligible"], args.top_k, EXIT_RANK)
        for cost in args.cost_bps:
            net, traded = portfolio_returns(controlled, test["returns"], cost)
            rows.append({"group": runs[0]["label"], "seeds": len(runs), "rule": "turnover_control", "sigma": "none", "parameter": EXIT_RANK, "cost_bps": cost, "test_days": int(net.size), **performance(net, traded, benchmark, indices)})
        cost = args.tuning_cost_bps
        reference, _ = portfolio_returns(controlled, test["returns"], cost)
        for source in sources:
            for rule, grid in (("size", SIZING_GAMMAS), ("filter", DROP_SHARES)):
                scores = {value: sharpe(portfolio_returns(controlled_book(splits["valid"], args.top_k, source, rule, value), splits["valid"]["returns"], cost)[0]) for value in grid}
                value = max(scores, key=scores.get)
                net, traded = portfolio_returns(controlled_book(test, args.top_k, source, rule, value), test["returns"], cost)
                rows.append({"group": runs[0]["label"], "seeds": len(runs), "rule": f"controlled_{rule}", "sigma": source, "parameter": value, "valid_sharpe_at_tuning_cost": scores[value], "cost_bps": cost, "test_days": int(net.size), **performance(net, traded, benchmark, indices), **sharpe_difference(net, reference, indices)})
        consensus = consensus_candidates(test["points"], test["eligible"])
        net, traded = portfolio_returns(top_k_weights(test["mu"], consensus, args.top_k), test["returns"], cost)
        rows.append({"group": runs[0]["label"], "seeds": len(runs), "rule": "consensus", "sigma": "seed_votes", "parameter": CONSENSUS_MIN_SEEDS, "cost_bps": cost, "test_days": int(net.size), "mean_holdings": float((top_k_weights(test["mu"], consensus, args.top_k) > 0).sum(axis=0).mean()), **performance(net, traded, benchmark, indices)})
        for source in sources:
            allowed = market_gate(market_level(splits["valid"][f"sigma_{source}"], splits["valid"]["eligible"]), market_level(test[f"sigma_{source}"], test["eligible"]))
            net, traded = portfolio_returns(base_weights * allowed[None, :], test["returns"], cost)
            rows.append({"group": runs[0]["label"], "seeds": len(runs), "rule": "market_gate", "sigma": source, "parameter": GATE_QUANTILE, "cost_bps": cost, "test_days": int(net.size), "days_out_share": float(1.0 - allowed.mean()), **performance(net, traded, benchmark, indices)})
            closed = int((~allowed).sum())
            draws = []
            for _ in range(GATE_RANDOM_DRAWS):
                random_allowed = np.ones(allowed.size, dtype=bool)
                random_allowed[generator.choice(allowed.size, size=closed, replace=False)] = False
                draws.append(sharpe(portfolio_returns(base_weights * random_allowed[None, :], test["returns"], cost)[0]))
            rows.append({"group": runs[0]["label"], "seeds": len(runs), "rule": "random_gate", "sigma": source, "parameter": GATE_QUANTILE, "cost_bps": cost, "test_days": int(net.size), "days_out_share": float(1.0 - allowed.mean()), "sharpe": float(np.mean(draws)), "sharpe_ci_low": float(np.percentile(draws, 2.5)), "sharpe_ci_high": float(np.percentile(draws, 97.5))})
        rows.append({"group": runs[0]["label"], "seeds": len(runs), "rule": "equal_weight_benchmark", "sigma": "none", "parameter": 0.0, "cost_bps": 0.0, "test_days": int(benchmark.size), **performance(benchmark, np.zeros_like(benchmark), benchmark, indices)})

    table = pd.DataFrame(rows)
    table.to_csv(output / "backtest.csv", index=False)
    (output / "backtest_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    logger.info("backtest\n%s", table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
