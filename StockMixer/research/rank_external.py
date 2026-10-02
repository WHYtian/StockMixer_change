"""External-market ranking-reliability audit on frozen A-share test predictions.

The base models were selected on 2024 validation data; this script never opens
their validation predictions. It reuses the A2 risk audit with one frozen test
window and records the limitation that this historical holdout was inspected in
the earlier project.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.data import load_dataset
from research.rank_a2 import add_placebos, summarise_a2
from research.rank_uncertainty import audit, fingerprint, make_days
from research.uncertainty_eval import trailing_volatility

SEEDS = tuple(range(20260813, 20260818))
TRAIN_END, VALID_END = 2189, 2431


def load_frozen(run_root: Path, dataset: Path):
    arrays = []
    hashes = {}
    for seed in SEEDS:
        run = run_root / f"dropout0.1_seed{seed}"
        config = json.loads((run / "config.json").read_text())
        expected = {"seed": seed, "train_end": TRAIN_END, "valid_end": VALID_END,
                    "dropout": 0.1, "dropout_sites": "mixer", "select_metric": "rank_ic",
                    "select_smoothing": 5, "min_epochs": 30, "mc_samples": 50,
                    "horizon": 1}
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError(f"run {seed} does not match frozen A-share configuration")
        path = run / "test_predictions.npz"
        with np.load(path, allow_pickle=False) as packed:
            arrays.append({key: packed[key] for key in packed.files})
        hashes[str(path)] = fingerprint(path)
        hashes[str(run / "config.json")] = fingerprint(run / "config.json")
    for other in arrays[1:]:
        if not np.array_equal(other["target"], arrays[0]["target"], equal_nan=True) or not np.array_equal(other["mask"], arrays[0]["mask"]):
            raise ValueError("seed test arrays are not aligned")
    point = np.stack([a["point_prediction"] for a in arrays]).astype(float)
    mc = np.stack([a["mc_std"] for a in arrays]).astype(float)
    target = arrays[0]["target"].astype(float)
    mask = arrays[0]["mask"].astype(bool)
    data = load_dataset(dataset)
    if point.shape != (5, data.stock_count, data.date_count - VALID_END) or mc.shape != point.shape:
        raise ValueError("frozen test shapes do not match A-share panel")
    if not np.allclose(target[mask], data.returns[:, VALID_END:][mask], rtol=1e-5, atol=1e-7):
        raise ValueError("saved test labels differ from A-share panel")
    volatility = trailing_volatility(data.returns, data.masks, VALID_END, target.shape[1], 16)
    eligible = mask & np.isfinite(target) & np.isfinite(volatility)
    eligible &= np.isfinite(point).all(axis=0) & np.isfinite(mc).all(axis=0)
    hashes[str(dataset / "panel.npz")] = fingerprint(dataset / "panel.npz")
    return make_days(point, mc, target, volatility, eligible), hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("analysis output must not already exist")
    days, hashes = load_frozen(args.run_root, args.dataset)
    add_placebos(days, seed=20261001)
    rows, folds, _ = audit(days, initial=126, step=63)
    for row in rows:
        row["outer_window"] = 0
        row["target_day"] = row["day"] + VALID_END
    summary = summarise_a2(rows, draws=10000)
    summary.update({"market": "A-share CSI300+CSI500 historical PIT panel",
                    "status": "retrospective_external_holdout_not_untouched_data",
                    "validation_end": VALID_END, "evaluation_days": len(days),
                    "evaluated_stock_days": sum(r["stocks"] for r in rows if r["method"] == "control")})
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output / "protocol.md").write_text(args.protocol.read_text())
    (args.output / "manifest.json").write_text(json.dumps({
        "market": "A-share CSI300+CSI500", "run_root": str(args.run_root),
        "dataset": str(args.dataset), "validation_end": VALID_END,
        "hashes": hashes, "protocol_sha256": fingerprint(args.protocol),
        "folds": folds, "bootstrap_draws": 10000, "block_days": 10,
        "status": summary["status"]}, indent=2) + "\n")
    import csv
    with (args.output / "daily_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "days": len(days),
                      "stock_days": summary["evaluated_stock_days"],
                      "passing": summary["candidates_passing"]}, indent=2))


if __name__ == "__main__":
    main()
