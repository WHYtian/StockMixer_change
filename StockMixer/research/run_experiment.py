"""Train StockMixer and evaluate final point/MC-dropout predictions reproducibly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from research.data import EODData, batch_at, load_dataset
from research.mc_dropout import predict_mc_returns
from research.metrics import evaluate_predictions
from research.model import StockMixerDropout, cross_sectional_zscore, heteroscedastic_loss, split_output, stockmixer_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/NASDAQ")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropout-sites", choices=("all", "mixer"), default="all", help="mixer: no dropout on the inputs of the output heads")
    parser.add_argument("--select-metric", choices=("mse", "rank_ic"), default="mse", help="validation metric used for early stopping and model selection")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-end", type=int, default=756)
    parser.add_argument("--valid-end", type=int, default=1008)
    parser.add_argument("--mc-samples", type=int, default=0)
    parser.add_argument("--features", choices=("ohlcv", "ma"), default="ohlcv", help="input channels of panel datasets")
    parser.add_argument("--label", choices=("raw", "cs_zscore"), default="raw", help="training target; evaluation always uses raw returns")
    parser.add_argument("--heteroscedastic", action="store_true", help="add a return-variance head trained with beta-NLL")
    parser.add_argument("--beta", type=float, default=0.5, help="beta of the beta-NLL loss")
    parser.add_argument("--select-smoothing", type=int, default=1, help="epochs in the trailing mean of the selection metric")
    parser.add_argument("--min-epochs", type=int, default=0, help="early stopping cannot end training before this epoch")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit-days", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def split_offsets(start: int, end: int, lookback: int, horizon: int, limit: int) -> list[int]:
    # The first target period starts at ``start``, so target periods of adjacent splits never overlap.
    first_offset = max(0, start - lookback)
    offsets = list(range(first_offset, end - lookback - horizon + 1))
    return offsets[:limit] if limit else offsets


def to_tensors(data: EODData, offset: int, args: argparse.Namespace, device: torch.device):
    return tuple(
        torch.from_numpy(array).to(device=device, dtype=torch.float32)
        for array in batch_at(data, offset, args.lookback, args.horizon)
    )


def evaluate(model, data, offsets, args, device, mc_samples=0):
    predictions, points, targets, masks, stds, aleatoric = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for offset in offsets:
            features, mask, price, target = to_tensors(data, offset, args, device)
            predicted_price, log_variance = split_output(model(features))
            prediction = (predicted_price - price) / price
            points.append(prediction.cpu().numpy())
            if log_variance is not None:
                aleatoric.append((0.5 * log_variance).exp().cpu().numpy())
            if mc_samples:
                prediction, standard_deviation, _ = predict_mc_returns(model, features, price, mc_samples)
                stds.append(standard_deviation.cpu().numpy())
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
            masks.append(mask.cpu().numpy())
    prediction = np.concatenate(predictions, axis=1)
    target = np.concatenate(targets, axis=1)
    mask = np.concatenate(masks, axis=1)
    std = np.concatenate(stds, axis=1) if stds else None
    point = np.concatenate(points, axis=1)
    metrics = evaluate_predictions(prediction, target, mask)
    if mc_samples:
        # Keep the deterministic forward pass so MC-mean and point ranking can be compared.
        metrics.update({f"point_{key}": value for key, value in evaluate_predictions(point, target, mask).items()})
    noise = np.concatenate(aleatoric, axis=1) if aleatoric else None
    return metrics, prediction, target, mask, std, point, noise


def smoothed_score(history: list[dict], metric: str, window: int) -> float:
    """Trailing mean of the selection metric; ``-inf`` until ``window`` epochs exist."""
    if len(history) < window:
        return float("-inf")
    values = [row[f"valid_{metric}"] for row in history[-window:]]
    score = float(np.mean(values)) if np.all(np.isfinite(values)) else float("-inf")
    return score if metric == "rank_ic" else -score


def main() -> None:
    args = parse_args()
    if args.select_smoothing < 1:
        raise ValueError("select-smoothing must be at least 1")
    if args.mc_samples == 1:
        raise ValueError("mc-samples must be 0 or at least 2")
    if args.mc_samples and args.dropout == 0:
        raise ValueError("MC Dropout needs dropout > 0; sigma would be exactly zero")
    if args.horizon < 1:
        raise ValueError("horizon must be at least 1")
    seed_everything(args.seed)
    device = get_device(args.device)
    data = load_dataset(args.dataset, args.features)
    if not (0 < args.train_end < args.valid_end <= data.date_count):
        raise ValueError("invalid split boundaries")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({**vars(args), "device_used": str(device)}, indent=2) + "\n")

    train_offsets = split_offsets(0, args.train_end, args.lookback, args.horizon, args.limit_days)
    valid_offsets = split_offsets(args.train_end, args.valid_end, args.lookback, args.horizon, args.limit_days)
    test_offsets = split_offsets(args.valid_end, data.date_count, args.lookback, args.horizon, args.limit_days)
    if not all((train_offsets, valid_offsets, test_offsets)):
        raise ValueError("one or more experiment splits are empty")
    model = StockMixerDropout(data.stock_count, args.lookback, data.features.shape[2], 20, 3, args.dropout, head_dropout=args.dropout_sites == "all", heteroscedastic=args.heteroscedastic).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_score, stalled, history = float("-inf"), 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_offsets)
        losses = []
        for offset in train_offsets:
            features, mask, price, target = to_tensors(data, offset, args, device)
            optimizer.zero_grad()
            if args.label == "cs_zscore":
                target = cross_sectional_zscore(target, mask)
            predicted_price, log_variance = split_output(model(features))
            if log_variance is None:
                loss, _, _, _ = stockmixer_loss(predicted_price, target, price, mask, args.alpha)
            else:
                loss = heteroscedastic_loss(predicted_price, log_variance, target, price, mask, args.alpha, args.beta)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        valid_metrics, *_ = evaluate(model, data, valid_offsets, args, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"valid_{key}": value for key, value in valid_metrics.items()}}
        history.append(row)
        score = smoothed_score(history, args.select_metric, args.select_smoothing)
        row["selection_score"] = score if np.isfinite(score) else None
        print(json.dumps(row), flush=True)
        # Epoch 1 always writes a checkpoint so that a stale file is never loaded.
        if score > best_score or epoch == 1:
            best_score, stalled = max(score, best_score), 0
            torch.save(model.state_dict(), output / "best_model.pt")
        else:
            stalled += 1
            if stalled >= args.patience and epoch >= args.min_epochs:
                break

    model.load_state_dict(torch.load(output / "best_model.pt", map_location=device, weights_only=True))
    # Validation predictions are kept so that portfolio rules can be tuned without touching the test set.
    for split, offsets in (("valid", valid_offsets), ("test", test_offsets)):
        metrics, prediction, target, mask, std, point, noise = evaluate(model, data, offsets, args, device, args.mc_samples)
        (output / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        arrays = {"prediction": prediction, "target": target, "mask": mask}
        if std is not None:
            arrays["mc_std"] = std
            arrays["point_prediction"] = point
        if noise is not None:
            arrays["aleatoric_std"] = noise
        np.savez_compressed(output / f"{split}_predictions.npz", **arrays)
    test_metrics = metrics
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"test": test_metrics, "epochs_completed": len(history)}, indent=2))


if __name__ == "__main__":
    main()
