"""Build an A-share ``[stock, date]`` panel with point-in-time index membership.

Source: a Qlib-format dump (``calendars/day.txt``, ``instruments/<index>.txt``,
``features/<symbol>/<field>.day.bin``).  Prices in the dump are already
multiplied by the adjustment factor, so ``close[t] / close[t-1] - 1`` is the
total return.  Nothing is normalised with full-sample statistics; scaling
happens per look-back window at load time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIELDS = ("open", "high", "low", "close", "volume")
CHINEXT_REFORM = "2020-08-24"
LIMIT_TOLERANCE = 0.003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qlib-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--indices", nargs="+", default=["csi300", "csi500"])
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2026-09-28")
    parser.add_argument("--min-listed-days", type=int, default=60)
    return parser.parse_args()


def read_field(qlib_dir: Path, symbol: str, field: str, calendar: pd.Index) -> pd.Series:
    """Read one Qlib binary: a float32 start index followed by the values."""
    raw = np.fromfile(qlib_dir / "features" / symbol / f"{field}.day.bin", dtype="<f4")
    start = int(raw[0])
    return pd.Series(raw[1:], index=calendar[start : start + raw.size - 1])


def read_membership(qlib_dir: Path, index: str) -> pd.DataFrame:
    table = pd.read_csv(qlib_dir / "instruments" / f"{index}.txt", sep="\t", header=None, names=["symbol", "start", "end"], parse_dates=["start", "end"])
    table["symbol"] = table["symbol"].str.lower()
    return table


def membership_matrix(table: pd.DataFrame, symbols: List[str], dates: pd.DatetimeIndex) -> np.ndarray:
    """``[stock, date]`` flags, true while the stock is an index member."""
    row = {symbol: position for position, symbol in enumerate(symbols)}
    matrix = np.zeros((len(symbols), len(dates)), dtype=bool)
    for symbol, start, end in table.itertuples(index=False):
        if symbol in row:
            matrix[row[symbol], dates.searchsorted(start) : dates.searchsorted(end, side="right")] = True
    return matrix


def limit_rate(symbol: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """Daily price-limit rate by board; ST stocks are not identified."""
    code = symbol[2:]
    rate = np.full(len(dates), 0.10)
    if code.startswith("68"):
        rate[:] = 0.20
    elif code.startswith("30"):
        rate[dates >= pd.Timestamp(CHINEXT_REFORM)] = 0.20
    return rate


def build_stock(frame: pd.DataFrame, symbol: str, dates: pd.DatetimeIndex, min_listed_days: int) -> Dict[str, np.ndarray]:
    """``frame`` holds the stock's full history so that listing age is exact."""
    traded = frame["close"].notna() & (frame["volume"] > 0)
    close = frame["close"].where(traded)
    returns = close / close.ffill().shift(1) - 1.0
    traded_yesterday = traded.shift(1, fill_value=False)
    listed_days = traded.cumsum()
    # A resumption-day return spans the whole halt, so it is not a one-day return.
    mask = traded & traded_yesterday & returns.notna() & (listed_days > min_listed_days)

    at_high = pd.Series(np.isclose(frame["close"], frame["high"], rtol=1e-5), index=frame.index)
    at_low = pd.Series(np.isclose(frame["close"], frame["low"], rtol=1e-5), index=frame.index)
    full_dates = pd.DatetimeIndex(frame.index)
    rate = pd.Series(limit_rate(symbol, full_dates), index=frame.index)
    limit_up = traded_yesterday & at_high & (returns >= rate - LIMIT_TOLERANCE)
    limit_down = traded_yesterday & at_low & (returns <= -rate + LIMIT_TOLERANCE)

    window = frame.index.isin(dates)
    return {
        "features": frame.loc[window, list(FIELDS)].where(traded[window]).to_numpy(dtype=np.float32),
        "mask": mask[window].to_numpy(dtype=np.float32),
        "returns": returns[window].fillna(0.0).to_numpy(dtype=np.float32),
        "limit_up": limit_up[window].to_numpy(dtype=bool),
        "limit_down": limit_down[window].to_numpy(dtype=bool),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    qlib_dir = Path(args.qlib_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    calendar = pd.DatetimeIndex(pd.read_csv(qlib_dir / "calendars" / "day.txt", header=None)[0])
    dates = calendar[(calendar >= args.start_date) & (calendar <= args.end_date)]
    tables = {index: read_membership(qlib_dir, index) for index in args.indices}
    in_period = [table[(table["end"] >= dates[0]) & (table["start"] <= dates[-1])] for table in tables.values()]
    symbols = sorted(set(pd.concat(in_period)["symbol"]))
    logger.info("%d trading days, %d stocks that were ever a member of %s", len(dates), len(symbols), args.indices)

    built = []
    for symbol in symbols:
        frame = pd.DataFrame({field: read_field(qlib_dir, symbol, field, calendar) for field in FIELDS}).reindex(calendar)
        built.append(build_stock(frame, symbol, dates, args.min_listed_days))

    panel = {key: np.stack([arrays[key] for arrays in built]) for key in built[0]}
    for index, table in tables.items():
        panel[f"member_{index}"] = membership_matrix(table, symbols, dates)
    panel["member"] = np.any([panel[f"member_{index}"] for index in tables], axis=0)
    panel["dates"] = dates.strftime("%Y-%m-%d").to_numpy()
    panel["symbols"] = np.asarray(symbols)
    target = output / "panel.npz"
    np.savez_compressed(target, **panel)

    eligible = panel["member"] & panel["mask"].astype(bool)
    manifest = qlib_dir.parent / "qlib_bin.manifest.json"
    meta = {
        **vars(args),
        "source_manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
        "stocks": len(symbols),
        "dates": len(dates),
        "first_date": str(dates[0].date()),
        "last_date": str(dates[-1].date()),
        "members_per_day": {index: [int(panel[f"member_{index}"].sum(axis=0).min()), int(panel[f"member_{index}"].sum(axis=0).max())] for index in tables},
        "eligible_per_day_min_median_max": [int(np.min(eligible.sum(axis=0))), float(np.median(eligible.sum(axis=0))), int(np.max(eligible.sum(axis=0)))],
        "member_days_tradable_share": float(eligible.sum() / panel["member"].sum()),
        "limit_up_share_of_member_days": float((panel["limit_up"] & panel["member"]).sum() / panel["member"].sum()),
        "limit_down_share_of_member_days": float((panel["limit_down"] & panel["member"]).sum() / panel["member"].sum()),
        "fields": list(FIELDS),
        "panel_sha256": hashlib.sha256(target.read_bytes()).hexdigest()[:12],
        "year_start_index": {str(year): int(dates.searchsorted(pd.Timestamp(f"{year}-01-01"))) for year in range(dates[0].year, dates[-1].year + 1)},
        "known_limits": "ST status is not identified; limit flags come from adjusted returns and close at high/low.",
    }
    (output / "panel_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("panel: %s", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
