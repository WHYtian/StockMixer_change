"""Development-only ranking-risk audit; never opens test predictions.

Base checkpoints used full validation data for selection: meta-model chronological
cross-fitting does not make this an independent out-of-sample performance claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from research.data import load_dataset
from research.uncertainty_eval import block_indices, mean_with_interval, rank_correlation, trailing_volatility


def percentile(x: np.ndarray) -> np.ndarray:
    return (rankdata(x, method="average") - 0.5) / x.size


def random_rank_loss(p: np.ndarray, n: int) -> np.ndarray:
    """Exact expected distance to a uniform random midpoint rank, including ties."""
    grid = (np.arange(n) + 0.5) / n
    # Prefix sums avoid an N x N temporary and also handle tied forecast ranks.
    prefix = np.r_[0.0, np.cumsum(grid)]
    k = np.searchsorted(grid, p, side="right")
    return (p * k - prefix[k] + prefix[n] - prefix[k] - p * (n - k)) / n


def expanding_folds(days: int, initial: int, step: int, horizon: int = 1):
    if initial < horizon or step < 1 or days <= initial:
        raise ValueError("invalid expanding-window boundaries")
    for start in range(initial, days, step):
        # Column t has a forward window starting at t and maturing at t+h-1.
        fit = np.arange(start - horizon + 1)
        evaluation = np.arange(start, min(days, start + step))
        yield fit, evaluation


def quadratic(x: np.ndarray) -> np.ndarray:
    return np.column_stack([x] + [x[:, i] * x[:, j] for i in range(x.shape[1]) for j in range(i, x.shape[1])])


def ridge_predict(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                  evaluation: np.ndarray, penalty: float = 0.001) -> np.ndarray:
    """Weighted quadratic ridge; both scaling stages fitted on training only."""
    if penalty <= 0 or not np.isfinite(x).all() or not np.isfinite(evaluation).all():
        raise ValueError("positive penalty and finite features required")
    w = weights / weights.sum()
    center = w @ x
    scale = np.maximum(np.sqrt(w @ ((x - center) ** 2)), 1e-8)
    train = quadratic((x - center) / scale)
    test = quadratic((evaluation - center) / scale)
    center2 = w @ train
    scale2 = np.maximum(np.sqrt(w @ ((train - center2) ** 2)), 1e-8)
    train = (train - center2) / scale2
    test = (test - center2) / scale2
    intercept = float(w @ y)
    gram = train.T @ (w[:, None] * train)
    coef = np.linalg.solve(gram + penalty * np.eye(train.shape[1]), train.T @ (w * (y - intercept)))
    return np.clip(intercept + test @ coef, 0.0, 1.0)


def deciles(p: np.ndarray) -> np.ndarray:
    return np.minimum((p * 10).astype(int), 9)


def within_decile_correlation(risk: np.ndarray, loss: np.ndarray, p: np.ndarray) -> float:
    # Correlate ranks after removing forecast-decile means, not pooled stock-days.
    a, b = rankdata(risk), rankdata(loss)
    group = deciles(p)
    for bucket in np.unique(group):
        take = group == bucket
        a[take] -= a[take].mean()
        b[take] -= b[take].mean()
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def select(risk: np.ndarray, p: np.ndarray, fraction: float, stratified: bool,
           tie_breaker: np.ndarray) -> np.ndarray:
    groups = deciles(p) if stratified else np.zeros(p.size, dtype=int)
    selected = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        order = np.lexsort((tie_breaker[indices], risk[indices]))
        count = max(1, int(round(indices.size * fraction)))
        selected.extend(indices[order[:count]])
    return np.asarray(selected, dtype=int)


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(runs: list[Path], dataset: Path):
    configs, arrays, hashes = [], [], {}
    for run in runs:
        config_path, prediction_path = run / "config.json", run / "valid_predictions.npz"
        configs.append(json.loads(config_path.read_text()))
        with np.load(prediction_path, allow_pickle=False) as values:
            arrays.append({key: values[key] for key in values.files})
        hashes[str(config_path)] = fingerprint(config_path)
        hashes[str(prediction_path)] = fingerprint(prediction_path)
    config = configs[0]
    if len(runs) < 2 or len({c["seed"] for c in configs}) != len(runs):
        raise ValueError("require at least two distinct seeds")
    if config.get("horizon", 1) != 1 or config.get("limit_days", 0):
        raise ValueError("pilot supports full horizon-1 validation arrays only")
    keys = ("dataset", "train_end", "valid_end", "lookback", "horizon", "dropout", "dropout_sites", "label", "features", "heteroscedastic", "limit_days")
    for other in configs[1:]:
        if any(config.get(key) != other.get(key) for key in keys):
            raise ValueError("run configurations do not match")
    for other in arrays[1:]:
        if not np.array_equal(other["target"], arrays[0]["target"], equal_nan=True) or not np.array_equal(other["mask"], arrays[0]["mask"]):
            raise ValueError("targets or masks do not match across seeds")
    if not all("point_prediction" in a and "mc_std" in a for a in arrays):
        raise ValueError("point_prediction and mc_std required for every seed")
    point = np.stack([a["point_prediction"] for a in arrays]).astype(float)
    mc = np.stack([a["mc_std"] for a in arrays]).astype(float)
    target = arrays[0]["target"].astype(float)
    if point.shape[1:] != target.shape or mc.shape != point.shape:
        raise ValueError("prediction shapes do not match")
    if target.shape[1] != config["valid_end"] - config["train_end"]:
        raise ValueError("validation date count does not match config")
    if np.any(mc < 0):
        raise ValueError("negative MC standard deviation")
    data = load_dataset(dataset, config.get("features", "ohlcv"))
    if target.shape[0] != data.stock_count:
        raise ValueError("stock count mismatch")
    actual = data.returns[:, config["train_end"]:config["valid_end"]]
    saved_mask = arrays[0]["mask"].astype(bool)
    if not np.allclose(actual[saved_mask], target[saved_mask], rtol=1e-5, atol=1e-7):
        raise ValueError("dataset targets do not match saved target columns")
    volatility = trailing_volatility(data.returns, data.masks, config["train_end"], target.shape[1], 16)
    eligible = saved_mask & np.isfinite(volatility) & np.isfinite(target)
    eligible &= np.isfinite(point).all(axis=0) & np.isfinite(mc).all(axis=0)
    files = [dataset / "panel.npz"] if (dataset / "panel.npz").exists() else [dataset / name for name in ("eod_data.pkl", "gt_data.pkl", "mask_data.pkl", "price_data.pkl")]
    for file in files:
        hashes[str(file)] = fingerprint(file)
    return config, point, mc, target, volatility, eligible, hashes


def make_days(point, mc, target, volatility, eligible):
    days = []
    generator = np.random.default_rng(20260930)
    for day in range(target.shape[1]):
        keep = np.flatnonzero(eligible[:, day])
        if keep.size < 30:
            raise ValueError("too few eligible stocks for decile evaluation")
        samples = point[:, keep, day]
        score = samples.mean(axis=0)
        p = percentile(score)
        truth = target[keep, day]
        loss = np.abs(p - percentile(truth))
        vol = volatility[keep, day]
        mc_std = np.sqrt(np.mean(mc[:, keep, day] ** 2, axis=0))
        ens_std = samples.std(axis=0, ddof=1)
        rank_std = np.stack([percentile(sample) for sample in samples]).std(axis=0, ddof=1)
        raw = {"mc": mc_std, "ensemble": ens_std, "rank": rank_std}
        controls = np.column_stack([p, score, np.abs(score), np.log(np.maximum(vol, 1e-12)), percentile(vol)])
        extras = {key: np.column_stack([np.log(np.maximum(value, 1e-12)) if key != "rank" else value, percentile(value)]) for key, value in raw.items()}
        features = {"control": controls}
        features.update({f"plus_{key}": np.column_stack([controls, value]) for key, value in extras.items()})
        features["plus_all"] = np.column_stack([controls, *extras.values()])
        risks = {"geometry": random_rank_loss(p, keep.size), "volatility": vol}
        risks.update({f"raw_{key}": value for key, value in raw.items()})
        days.append(dict(stocks=keep, score=score, truth=truth, p=p, loss=loss,
                         features=features, risks=risks, tie=generator.random(keep.size)))
    return days


def audit(days, initial=96, step=52, penalty=0.001):
    rows, folds, saved = [], [], []
    for fold, (fit, evaluation) in enumerate(expanding_folds(len(days), initial, step)):
        folds.append({"fold": fold, "fit_first": int(fit[0]), "fit_last": int(fit[-1]), "eval_first": int(evaluation[0]), "eval_last": int(evaluation[-1])})
        y = np.concatenate([days[d]["loss"] for d in fit])
        weights = np.concatenate([np.full(days[d]["loss"].size, 1 / days[d]["loss"].size) for d in fit])
        predictions = {}
        for method in days[0]["features"]:
            x = np.concatenate([days[d]["features"][method] for d in fit])
            z = np.concatenate([days[d]["features"][method] for d in evaluation])
            values = ridge_predict(x, y, weights, z, penalty)
            predictions[method] = np.split(values, np.cumsum([days[d]["loss"].size for d in evaluation])[:-1])
        for j, day in enumerate(evaluation):
            entry = days[day]
            risks = {**entry["risks"], **{method: values[j] for method, values in predictions.items()}}
            saved.append({"day": int(day), "stocks": entry["stocks"], "loss": entry["loss"], "p": entry["p"], **risks})
            for method, risk in risks.items():
                row = {"fold": fold, "day": int(day), "method": method, "stocks": entry["loss"].size,
                       "rho": rank_correlation(risk, entry["loss"]),
                       "rho_within_decile": within_decile_correlation(risk, entry["loss"], entry["p"]),
                       "mse": float(np.mean((risk - entry["loss"]) ** 2)) if method in predictions or method == "geometry" else float("nan"),
                       "full_rank_ic": rank_correlation(entry["score"], entry["truth"])}
                for stratified, suffix in ((False, "global"), (True, "stratified")):
                    take = select(risk, entry["p"], .6, stratified, entry["tie"])
                    row[f"loss60_{suffix}"] = float(entry["loss"][take].mean())
                    row[f"rankic60_{suffix}"] = rank_correlation(entry["score"][take], entry["truth"][take])
                    row[f"retained_{suffix}"] = int(take.size)
                rows.append(row)
    return rows, folds, saved


def summarise(rows, draws=2000, block=10):
    methods = list(dict.fromkeys(row["method"] for row in rows))
    by_method = {method: [row for row in rows if row["method"] == method] for method in methods}
    indices = block_indices(len(by_method[methods[0]]), block, draws, np.random.default_rng(20260930))
    metrics = ("mse", "rho", "rho_within_decile", "loss60_stratified", "rankic60_stratified", "loss60_global", "rankic60_global")
    summary = {"methods": {}, "paired_augmented_minus_control": {}, "fold_means": {}}
    for method in methods:
        summary["methods"][method] = {}
        for metric in metrics:
            values = np.asarray([r[metric] for r in by_method[method]])
            if np.isfinite(values).any():
                summary["methods"][method][metric] = mean_with_interval(values, indices)
        summary["fold_means"][method] = {str(fold): {metric: float(np.mean([r[metric] for r in by_method[method] if r["fold"] == fold])) for metric in summary["methods"][method]} for fold in sorted({r["fold"] for r in rows})}
        if method.startswith("plus_"):
            summary["paired_augmented_minus_control"][method] = {}
            for metric in metrics:
                delta = np.asarray([a[metric] - b[metric] for a, b in zip(by_method[method], by_method["control"])])
                if np.isfinite(delta).any():
                    summary["paired_augmented_minus_control"][method][metric] = mean_with_interval(delta, indices)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("use a new output directory; no overwrites")
    config, point, mc, target, vol, eligible, hashes = load_inputs(args.run_dirs, args.dataset)
    hashes[str(Path(__file__).resolve())] = fingerprint(Path(__file__))
    hashes[str(args.protocol.resolve())] = fingerprint(args.protocol)
    days = make_days(point, mc, target, vol, eligible)
    rows, folds, saved = audit(days)
    summary = summarise(rows)
    manifest = {"status": "development_only_not_independent_oos", "warning": "Base checkpoints were selected on the entire validation period. Test outputs were not opened.", "run_dirs": [str(p) for p in args.run_dirs], "dataset": str(args.dataset), "source_config": config,
                "initial_days": 96, "step_days": 52, "ridge_penalty": .001, "retention": .6,
                "bootstrap_draws": 2000, "block_days": 10, "hashes": hashes, "folds": folds,
                "stock_days": sum(d["loss"].size for d in days), "evaluated_stock_days": sum(d["loss"].size for d in saved),
                "rank_disagreement_source": "five-seed deterministic point ranks, NOT MC draws"}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "protocol.md").write_text(args.protocol.read_text())
    with (args.output_dir / "daily_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    packed = {key: np.concatenate([entry[key] for entry in saved]) for key in saved[0] if key != "day"}
    packed["day"] = np.concatenate([np.full(entry["loss"].size, entry["day"]) for entry in saved])
    np.savez_compressed(args.output_dir / "crossfit_risks.npz", **packed)
    print(json.dumps({"output": str(args.output_dir), "status": manifest["status"], "folds": folds, "contrasts": summary["paired_augmented_minus_control"]}, indent=2))


if __name__ == "__main__":
    main()
