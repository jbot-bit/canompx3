"""
Shared utilities for alternative strategy analysis scripts.

Used by analyze_double_break.py and analyze_gap_fade.py.
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import CostSpec, get_cost_spec, to_r_multiple
from pipeline.paths import GOLD_DB_PATH


def load_daily_features(db_path: Path, start: date, end: date,
                        orb_minutes: int = 5) -> pd.DataFrame:
    """Load daily_features for a date range."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT *
            FROM daily_features
            WHERE symbol = 'MGC'
              AND orb_minutes = ?
              AND trading_day BETWEEN ? AND ?
            ORDER BY trading_day
        """, [orb_minutes, start, end]).fetchdf()
    finally:
        con.close()
    return df


def load_bars_for_day(db_path: Path, trading_day: date) -> pd.DataFrame:
    """Load 1-minute bars for one trading day (09:00 Brisbane boundary).

    Trading day boundary: 23:00 UTC previous calendar day to 23:00 UTC trading day.
    """
    from pipeline.build_daily_features import compute_trading_day_utc_range
    start_utc, end_utc = compute_trading_day_utc_range(trading_day)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = 'MGC'
              AND ts_utc >= ? AND ts_utc < ?
            ORDER BY ts_utc
        """, [start_utc, end_utc]).fetchdf()
    finally:
        con.close()
    return df


def compute_walk_forward_windows(
    test_start: date,
    test_end: date,
    train_months: int = 12,
    step_months: int = 1,
) -> list[dict]:
    """Generate rolling walk-forward windows.

    Each window: train_months of training, then 1 month OOS test.
    Steps forward by step_months each time.
    """
    windows = []
    current_test_start = test_start

    while current_test_start < test_end:
        # Test period: current month
        test_end_month = _add_months(current_test_start, step_months)
        if test_end_month > test_end:
            test_end_month = test_end

        # Train period: train_months before test start
        train_start = _add_months(current_test_start, -train_months)
        train_end = current_test_start - timedelta(days=1)

        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": current_test_start,
            "test_end": test_end_month - timedelta(days=1),
        })

        current_test_start = test_end_month

    return windows


def compute_strategy_metrics(pnls: np.ndarray) -> dict | None:
    """Compute trading stats from array of R-multiples.

    Returns dict with n, wr, expr, sharpe, maxdd, total or None if empty.
    """
    n = len(pnls)
    if n == 0:
        return None
    wr = float((pnls > 0).sum() / n)
    expr = float(pnls.mean())
    std = float(pnls.std())
    sharpe = expr / std if std > 0 else 0.0
    cumul = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumul)
    maxdd = float((cumul - peak).min())
    total = float(pnls.sum())
    return {
        "n": n, "wr": wr, "expr": expr, "sharpe": sharpe,
        "maxdd": maxdd, "total": total,
    }


def resolve_bar_outcome(
    bars: pd.DataFrame,
    entry_price: float,
    stop_price: float,
    target_price: float,
    direction: str,
    start_idx: int,
) -> dict | None:
    """Scan bars forward from start_idx for stop/target hit.

    Gate C: If stop AND target hit on same bar, resolve as LOSS (conservative).

    Returns dict with outcome, exit_price, exit_bar_idx, pnl_points or None if
    neither stop nor target hit.
    """
    is_long = direction == "long"

    for i in range(start_idx, len(bars)):
        bar = bars.iloc[i]
        bar_high = bar["high"]
        bar_low = bar["low"]

        if is_long:
            stop_hit = bar_low <= stop_price
            target_hit = bar_high >= target_price
        else:
            stop_hit = bar_high >= stop_price
            target_hit = bar_low <= target_price

        # Gate C: ambiguous bar -> LOSS
        if stop_hit and target_hit:
            pnl_points = stop_price - entry_price if is_long else entry_price - stop_price
            return {
                "outcome": "loss",
                "exit_price": stop_price,
                "exit_bar_idx": i,
                "pnl_points": pnl_points,
            }

        if stop_hit:
            pnl_points = stop_price - entry_price if is_long else entry_price - stop_price
            return {
                "outcome": "loss",
                "exit_price": stop_price,
                "exit_bar_idx": i,
                "pnl_points": pnl_points,
            }

        if target_hit:
            pnl_points = target_price - entry_price if is_long else entry_price - target_price
            return {
                "outcome": "win",
                "exit_price": target_price,
                "exit_bar_idx": i,
                "pnl_points": pnl_points,
            }

    return None  # No resolution within session


def save_results(results: dict, path: Path) -> None:
    """Save results dict to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def _add_months(d: date, months: int) -> date:
    """Add months to a date, clamping to valid day."""
    month = d.month + months
    year = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = min(d.day, max_day)
    return date(year, month, day)
