#!/usr/bin/env python3
"""Match raw broker fills into round-trip trades.

Reads data/broker_fills.jsonl, outputs data/broker_trades.jsonl.
Position tracking per (account_id, instrument). VWAP entry/exit pricing.

Usage:
    python scripts/tools/trade_matcher.py                    # match all fills
    python scripts/tools/trade_matcher.py --date 2026-03-06  # specific date
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
FILLS_PATH = DATA_DIR / "broker_fills.jsonl"
TRADES_PATH = DATA_DIR / "broker_trades.jsonl"


def load_fills(*, path: Path = FILLS_PATH) -> list[dict]:
    """Load fills from JSONL."""
    if not path.exists():
        return []
    fills = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            fills.append(json.loads(line))
    return fills


def classify_trade_type(hold_seconds: float) -> str:
    if hold_seconds < 300:
        return "scalp"
    elif hold_seconds < 3600:
        return "swing"
    return "position"


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def match_fills_to_trades(fills: list[dict]) -> list[dict]:
    """Convert a list of fills into round-trip trade records.

    Algorithm:
    1. Group fills by (account_id, instrument)
    2. Sort by timestamp within each group
    3. Track running position; emit trade when position returns to zero or flips
    """
    # Group by (account_id, instrument)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for fill in fills:
        key = (fill["account_id"], fill["instrument"])
        groups[key].append(fill)

    all_trades = []

    for (account_id, instrument), group_fills in groups.items():
        group_fills.sort(key=lambda f: f["timestamp"])

        position = 0  # positive = long, negative = short
        entry_fills: list[dict] = []
        trade_counter = 0

        for fill in group_fills:
            side = fill["side"].upper()
            size = fill["size"]
            signed = size if side == "BUY" else -size

            new_position = position + signed

            # Same direction as existing position — accumulate
            if position == 0 or (position > 0 and signed > 0) or (position < 0 and signed < 0):
                entry_fills.append(fill)
                position = new_position
                continue

            # Opposite direction — partial or full close, or flip
            if abs(signed) <= abs(position):
                # Partial or full close
                position = new_position
                if position == 0:
                    # Full close — emit trade
                    trade_counter += 1
                    trade = _build_trade(entry_fills, fill, account_id, instrument, trade_counter)
                    all_trades.append(trade)
                    entry_fills = []
                # If partial close, don't emit yet
            else:
                # Position flip — close current, open new
                trade_counter += 1
                trade = _build_trade(entry_fills, fill, account_id, instrument, trade_counter)
                all_trades.append(trade)

                # Remaining size opens new position
                entry_fills = [fill]  # The flip fill starts new position
                position = new_position

    return all_trades


def _build_trade(
    entry_fills: list[dict],
    exit_fill: dict,
    account_id: int,
    instrument: str,
    counter: int,
) -> dict:
    """Build a round-trip trade record from entry fills + exit fill."""
    # VWAP entry price
    total_entry_value = sum(f["price"] * f["size"] for f in entry_fills)
    total_entry_size = sum(f["size"] for f in entry_fills)
    entry_price_avg = total_entry_value / total_entry_size if total_entry_size else 0

    entry_time = entry_fills[0]["timestamp"]
    exit_time = exit_fill["timestamp"]

    entry_dt = _parse_ts(entry_time)
    exit_dt = _parse_ts(exit_time)
    hold_seconds = (exit_dt - entry_dt).total_seconds()

    direction = "LONG" if entry_fills[0]["side"].upper() == "BUY" else "SHORT"

    total_fees = sum(f.get("fees", 0) for f in entry_fills) + exit_fill.get("fees", 0)
    total_pnl = sum(f.get("pnl", 0) for f in entry_fills) + exit_fill.get("pnl", 0)

    date_str = entry_dt.strftime("%Y-%m-%d")
    trade_id = f"{entry_fills[0]['broker']}-{account_id}-{date_str}-{counter:03d}"

    return {
        "trade_id": trade_id,
        "broker": entry_fills[0]["broker"],
        "account_name": entry_fills[0].get("account_name", str(account_id)),
        "instrument": instrument,
        "direction": direction,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price_avg": round(entry_price_avg, 6),
        "exit_price_avg": round(exit_fill["price"], 6),
        "size": total_entry_size,
        "pnl_dollar": total_pnl,
        "fees": total_fees,
        "hold_seconds": hold_seconds,
        "num_fills": len(entry_fills) + 1,
        "trade_type": classify_trade_type(hold_seconds),
        "source": "manual",
        "strategy_id": None,
    }


def save_trades(trades: list[dict], *, path: Path = TRADES_PATH) -> int:
    """Append trades to JSONL. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for trade in trades:
            fh.write(json.dumps(trade) + "\n")
    return len(trades)


def main():
    parser = argparse.ArgumentParser(description="Match fills to round-trip trades")
    parser.add_argument("--date", help="Filter fills by date (YYYY-MM-DD)")
    args = parser.parse_args()

    fills = load_fills()
    if args.date:
        fills = [f for f in fills if f.get("timestamp", "").startswith(args.date)]

    print(f"Loaded {len(fills)} fills")
    trades = match_fills_to_trades(fills)
    print(f"Matched {len(trades)} round-trip trades")

    if trades:
        n = save_trades(trades)
        print(f"Saved {n} trades to {TRADES_PATH}")

        for t in trades:
            prefix = "+" if t["pnl_dollar"] >= 0 else ""
            print(
                f"  {t['trade_id']}: {t['direction']} {t['instrument']} "
                f"x{t['size']} {prefix}${t['pnl_dollar']:.0f} ({t['trade_type']}, "
                f"{t['hold_seconds']:.0f}s)"
            )


if __name__ == "__main__":
    main()
