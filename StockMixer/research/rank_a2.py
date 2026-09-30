"""A2 retrospective audit with separated base selection and risk fitting.

The frozen protocol is plan/protocol_rank_uncertainty_a2_20260930.md.
This does not implement DEUP's irreducible-noise subtraction.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from research.data import load_dataset
from research.model import StockMixerDropout, stockmixer_loss
from research.rank_uncertainty import audit, deciles, fingerprint, make_days
from research.run_experiment import evaluate, seed_everything, smoothed_score, split_offsets, to_tensors
from research.uncertainty_eval import block_indices, trailing_volatility

SEEDS = tuple(range(20260813, 20260818))
WINDOWS = ((504, 630, 756, 1008), (756, 882, 1008, 1245))
AUGMENTATIONS = ("plus_mc", "plus_ensemble", "plus_rank", "plus_all")


def boundaries(window):
    train_end, selection_end, risk_end, eval_end = window
    if not 16 < train_end < selection_end < risk_end < eval_end:
        raise ValueError("four strictly ordered time boundaries required")
    return {
        "train": split_offsets(0, train_end, 16, 1, 0),
        "selection": split_offsets(train_end, selection_end, 16, 1, 0),
        "post_selection": split_offsets(selection_end, eval_end, 16, 1, 0),
    }


def data_hashes(dataset):
    return {name: fingerprint(dataset / name) for name in
            ("eod_data.pkl", "gt_data.pkl", "mask_data.pkl", "price_data.pkl")}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train_window(dataset, root, protocol, window_id, seed, device, smoke=False):
    if seed not in SEEDS or window_id not in range(len(WINDOWS)):
        raise ValueError("unregistered seed/window")
    window = WINDOWS[window_id]
    output = root / f"window{window_id}" / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    config = dict(window=window_id, seed=seed, boundaries=list(window), smoke=smoke,
                  dropout=.1, dropout_sites="mixer", horizon=1, lookback=16,
                  label="raw", learning_rate=.001, alpha=.1, mc_samples=2 if smoke else 50,
                  epochs=2 if smoke else 300, patience=30, min_epochs=30,
                  selection_metric="rank_ic", selection_smoothing=5,
                  protocol_sha256=fingerprint(protocol), dataset_sha256=data_hashes(dataset),
                  source_sha256={p.name: fingerprint(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    dump(output / "config.json", config)
    offsets = boundaries(window)
    if smoke:
        offsets = {key: values[:2] for key, values in offsets.items()}
    data = load_dataset(dataset)
    if data.date_count != 1245 or data.stock_count != 1026:
        raise ValueError("A2 requires the registered NASDAQ dimensions")
    seed_everything(seed)
    args = SimpleNamespace(lookback=16, horizon=1)
    model = StockMixerDropout(data.stock_count, 16, data.features.shape[2], 20, 3,
                             .1, head_dropout=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best, selected, stalled, history = -float("inf"), 0, 0, []
    with (output / "epochs.jsonl").open("x", buffering=1) as log:
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            random.shuffle(offsets["train"])
            losses = []
            for offset in offsets["train"]:
                features, mask, price, target = to_tensors(data, offset, args, device)
                optimizer.zero_grad()
                loss, *_ = stockmixer_loss(model(features), target, price, mask, .1)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            metrics, *_ = evaluate(model, data, offsets["selection"], args, device)
            row = dict(epoch=epoch, train_loss=float(np.mean(losses)),
                       **{f"valid_{k}": v for k, v in metrics.items()})
            history.append(row)
            score = smoothed_score(history, "rank_ic", 1 if smoke else 5)
            row["selection_score"] = score if np.isfinite(score) else None
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if score > best:
                best, selected, stalled = score, epoch, 0
                torch.save(model.state_dict(), output / "best_model.pt")
            else:
                stalled += 1
            if stalled >= 30 and epoch >= 30:
                break
    if selected == 0:
        raise ValueError("no finite checkpoint-selection score")
    model.load_state_dict(torch.load(output / "best_model.pt", weights_only=True, map_location=device))
    # No post-selection targets have been evaluated during checkpoint selection.
    seed_everything(seed + 100000 + window_id)
    _, prediction, target, mask, std, point, _ = evaluate(
        model, data, offsets["post_selection"], args, device, config["mc_samples"])
    np.savez_compressed(output / "post_selection_predictions.npz", prediction=prediction,
                        point_prediction=point, target=target, mask=mask, mc_std=std,
                        target_day=np.asarray(offsets["post_selection"]) + 16)
    dump(output / "history.json", history)
    dump(output / "complete.json", dict(selected_epoch=selected, epochs_completed=len(history),
         selection_score=best, checkpoint_sha256=fingerprint(output / "best_model.pt"),
         predictions_sha256=fingerprint(output / "post_selection_predictions.npz")))
    print(json.dumps(dict(window=window_id, seed=seed, status="complete", epochs=len(history))), flush=True)


def validate_config(config, window_id, seed, protocol_hash):
    expected = dict(window=window_id, seed=seed, boundaries=list(WINDOWS[window_id]),
                    smoke=False, dropout=.1, dropout_sites="mixer", horizon=1,
                    lookback=16, label="raw", learning_rate=.001, alpha=.1,
                    mc_samples=50, epochs=300, patience=30, min_epochs=30,
                    selection_metric="rank_ic", selection_smoothing=5,
                    protocol_sha256=protocol_hash)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("run does not match the frozen A2 protocol")


def load_window(root, dataset, protocol, window_id):
    configs, arrays, inputs = [], [], {}
    for seed in SEEDS:
        run = root / f"window{window_id}" / f"seed{seed}"
        config = json.loads((run / "config.json").read_text())
        validate_config(config, window_id, seed, fingerprint(protocol))
        complete = json.loads((run / "complete.json").read_text())
        path = run / "post_selection_predictions.npz"
        if fingerprint(path) != complete["predictions_sha256"]:
            raise ValueError("prediction fingerprint changed")
        if fingerprint(run / "best_model.pt") != complete["checkpoint_sha256"]:
            raise ValueError("checkpoint fingerprint changed")
        inputs[str(path)] = complete["predictions_sha256"]
        inputs[str(run / "config.json")] = fingerprint(run / "config.json")
        with np.load(path, allow_pickle=False) as packed:
            arrays.append({k: packed[k] for k in packed.files})
        configs.append(config)
    actual_hashes = data_hashes(dataset)
    if any(c["dataset_sha256"] != actual_hashes or c["source_sha256"] != configs[0]["source_sha256"] for c in configs):
        raise ValueError("data or training-source mismatch")
    start, end = WINDOWS[window_id][1], WINDOWS[window_id][3]
    for a in arrays:
        if not np.array_equal(a["target_day"], np.arange(start, end)):
            raise ValueError("post-selection target dates misaligned")
        for key in ("target", "mask"):
            if not np.array_equal(a[key], arrays[0][key], equal_nan=True):
                raise ValueError("seed arrays misaligned")
    point = np.stack([a["point_prediction"] for a in arrays]).astype(float)
    mc = np.stack([a["mc_std"] for a in arrays]).astype(float)
    target, mask = arrays[0]["target"], arrays[0]["mask"].astype(bool)
    data = load_dataset(dataset)
    if point.shape != (5, data.stock_count, end-start) or mc.shape != point.shape or np.any(mc < 0):
        raise ValueError("prediction shapes/uncertainty invalid")
    if not np.allclose(target[mask], data.returns[:, start:end][mask], rtol=1e-5, atol=1e-7):
        raise ValueError("saved labels differ from dataset")
    vol = trailing_volatility(data.returns, data.masks, start, end-start, 16)
    eligible = mask & np.isfinite(target) & np.isfinite(vol)
    eligible &= np.isfinite(point).all(axis=0) & np.isfinite(mc).all(axis=0)
    return make_days(point, mc, target, vol, eligible), inputs, configs


def add_placebos(days, seed=20261001):
    """Jointly shuffle extra columns within date/forecast-decile, without labels."""
    rng = np.random.default_rng(seed)
    for day in days:
        group = deciles(day["p"])
        width = day["features"]["control"].shape[1]
        for method in AUGMENTATIONS:
            x = day["features"][method].copy()
            for bucket in np.unique(group):
                take = np.flatnonzero(group == bucket)
                x[take, width:] = x[rng.permutation(take), width:]
            day["features"][method.replace("plus_", "placebo_")] = x


def stratified_blocks(labels, draws=10000, block=10):
    rng = np.random.default_rng(20260930)
    groups = [np.flatnonzero(labels == k) for k in np.unique(labels)]
    return np.concatenate([g[block_indices(g.size, block, draws, rng)] for g in groups], axis=1)


def interval(values, indices, alpha=.05):
    if not np.isfinite(values).all():
        return None
    lo, hi = np.quantile(values[indices].mean(axis=1), [alpha/2, 1-alpha/2])
    return dict(mean=float(np.mean(values)), lower=float(lo), upper=float(hi))


def summarise_a2(rows, draws=10000):
    methods = list(dict.fromkeys(r["method"] for r in rows))
    table = {m: sorted([r for r in rows if r["method"] == m], key=lambda r: r["target_day"]) for m in methods}
    dates = [r["target_day"] for r in table["control"]]
    if len(set(dates)) != len(dates):
        raise ValueError("evaluation dates overlap")
    if any([r["target_day"] for r in v] != dates for v in table.values()):
        raise ValueError("methods must use identical evaluation dates")
    labels = np.asarray([r["outer_window"] for r in table["control"]])
    indices = stratified_blocks(labels, draws)
    metrics = ("mse", "rho", "rho_within_decile", "loss60_stratified", "rankic60_stratified", "loss60_global", "rankic60_global", "full_rank_ic")
    result = dict(status="retrospective_temporally_separated_not_untouched_data", evaluation_days=len(dates),
                  family_size=8, adjusted_interval_coverage=.99375, methods={}, contrasts={}, candidates_passing=[])
    for method, entries in table.items():
        result["methods"][method] = {metric: interval(np.asarray([r[metric] for r in entries]), indices) for metric in metrics}
    for method in AUGMENTATIONS:
        passed = True
        for reference in ("control", method.replace("plus_", "placebo_")):
            differences = np.asarray([a["mse"]-b["mse"] for a, b in zip(table[method], table[reference])])
            adjusted = interval(differences, indices, .05/8)
            window_means = {str(k): float(np.mean(differences[labels == k])) for k in np.unique(labels)}
            result["contrasts"][method + "_minus_" + reference] = dict(
                ordinary_95=interval(differences, indices), adjusted_99375=adjusted, window_means=window_means)
            passed &= adjusted is not None and adjusted["upper"] < 0
            if reference == "control":
                passed &= all(x < 0 for x in window_means.values())
        if passed:
            result["candidates_passing"].append(method)
    return result


def analyse(root, dataset, protocol, output):
    if output.exists():
        raise FileExistsError("use a new analysis directory")
    rows, fold_records, hashes, configs = [], [], {}, []
    for window_id, window in enumerate(WINDOWS):
        days, inputs, source_configs = load_window(root, dataset, protocol, window_id)
        add_placebos(days)
        current, folds, _ = audit(days, initial=window[2]-window[1], step=63)
        for row in current:
            row["outer_window"] = window_id
            row["target_day"] = row["day"] + window[1]
        rows.extend(current)
        fold_records.append(dict(window=window_id, target_day_origin=window[1], folds=folds))
        hashes.update(inputs)
        configs.extend(source_configs)
    result = summarise_a2(rows)
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "summary.json", result)
    dump(output / "manifest.json", dict(protocol_sha256=fingerprint(protocol), input_hashes=hashes,
         analysis_source_sha256={p.name: fingerprint(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
         source_configs=configs, folds=fold_records, bootstrap_draws=10000, block_days=10,
         evaluated_stock_days=sum(r["stocks"] for r in rows if r["method"] == "control")))
    (output / "protocol.md").write_text(protocol.read_text())
    with (output / "daily_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(output=str(output), days=result["evaluation_days"], passing=result["candidates_passing"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "analyse"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "train":
        if args.seed is None:
            parser.error("train requires --seed")
        for window_id in range(len(WINDOWS)):
            train_window(args.dataset, args.root, args.protocol, window_id, args.seed, torch.device(args.device), args.smoke)
    else:
        if args.output is None:
            parser.error("analyse requires --output")
        analyse(args.root, args.dataset, args.protocol, args.output)


if __name__ == "__main__":
    main()
