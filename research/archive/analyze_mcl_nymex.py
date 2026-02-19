#!/usr/bin/env python3
"""
MCL (Micro Crude Oil) NYMEX session ORB breakout + fade research.

Tests 15m and 30m ORB strategies on the 2300 Brisbane session (13:00 UTC = NYMEX open),
where crude oil has real liquidity.

Also tests ORB fade (mean-reversion) entries exploiting MCL's high double-break rate.

Usage:
    python scripts/analyze_mcl_nymex.py
    python scripts/analyze_mcl_nymex.py --db-path C:/db/gold.db
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pipeline.cost_model import get_cost_spec, to_r_multiple
from pipeline.build_daily_features import compute_trading_day_utc_range
from research._alt_strategy_utils import compute_strategy_metrics, annualize_sharpe

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UTC_TZ = ZoneInfo("UTC")
SPEC = get_cost_spec("MCL")

# MCL cost model: point_value=$100, total_friction=$5.24
FRICTION = SPEC.total_friction  # $5.24

# Session: NYMEX open = 13:00 UTC
SESSION_START_HOUR_UTC = 13

# Date range
START_DATE = date(2021, 7, 11)
END_DATE = date(2026, 2, 10)

# ORB durations to test (minutes)
ORB_DURATIONS = [15, 30]

# Minimum ORB range: ORB_range * $100 >= $30  =>  ORB_range >= 0.30 points
MIN_ORB_RANGE_POINTS = 0.30

# RR targets
RR_TARGETS = [1.0, 1.5, 2.0, 2.5]

# CB values for E1
CB_VALUES = [1, 2]

# Weekday names
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_trading_days(db_path: Path) -> list[date]:
    """Get all trading days with MCL bars in the date range."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT DISTINCT CAST(ts_utc AS DATE) AS bar_date
            FROM bars_1m
            WHERE symbol = 'MCL'
              AND ts_utc >= ? AND ts_utc < ?
            ORDER BY bar_date
        """, [
            datetime(START_DATE.year, START_DATE.month, START_DATE.day, tzinfo=UTC_TZ),
            datetime(END_DATE.year, END_DATE.month, END_DATE.day, 23, 59, 59, tzinfo=UTC_TZ),
        ]).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]

def load_session_bars(db_path: Path, trading_day: date) -> pd.DataFrame:
    """Load MCL 1m bars from 13:00 UTC to 23:00 UTC for the given calendar day.

    The NYMEX session runs from 13:00 UTC.  Trading day ends at 23:00 UTC
    (09:00 Brisbane next day).
    """
    session_start = datetime(
        trading_day.year, trading_day.month, trading_day.day,
        SESSION_START_HOUR_UTC, 0, 0, tzinfo=UTC_TZ,
    )
    session_end = datetime(
        trading_day.year, trading_day.month, trading_day.day,
        23, 0, 0, tzinfo=UTC_TZ,
    )

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = 'MCL'
              AND ts_utc >= ? AND ts_utc < ?
            ORDER BY ts_utc
        """, [session_start, session_end]).fetchdf()
    finally:
        con.close()
    return df

# ---------------------------------------------------------------------------
# ORB computation
# ---------------------------------------------------------------------------
def compute_orb(bars: pd.DataFrame, orb_minutes: int) -> dict | None:
    """Compute ORB high/low from the first orb_minutes bars of the session.

    Returns dict with orb_high, orb_low, orb_range, orb_end_idx or None.
    """
    if len(bars) < orb_minutes:
        return None

    orb_bars = bars.iloc[:orb_minutes]
    orb_high = float(orb_bars["high"].max())
    orb_low = float(orb_bars["low"].min())
    orb_range = orb_high - orb_low

    if orb_range <= 0:
        return None

    return {
        "orb_high": orb_high,
        "orb_low": orb_low,
        "orb_range": orb_range,
        "orb_end_idx": orb_minutes,  # first bar AFTER ORB window
    }

# ---------------------------------------------------------------------------
# Breakout simulation (E1 with CB confirm)
# ---------------------------------------------------------------------------
def simulate_breakout(
    bars: pd.DataFrame,
    orb: dict,
    rr: float,
    cb: int,
    direction_filter: str,  # "long", "short", or "both"
) -> list[dict]:
    """Simulate E1 breakout entries with CB confirm bars.

    Returns list of trade dicts (may be 0 or 1 per day).
    """
    orb_high = orb["orb_high"]
    orb_low = orb["orb_low"]
    orb_range = orb["orb_range"]
    start_idx = orb["orb_end_idx"]

    if start_idx >= len(bars):
        return []

    trades = []

    # Scan for first break
    confirm_count_long = 0
    confirm_count_short = 0
    break_found = False

    for i in range(start_idx, len(bars)):
        bar_close = bars.iloc[i]["close"]

        # Long break: close above ORB high
        if bar_close > orb_high and direction_filter in ("long", "both"):
            confirm_count_long += 1
            if confirm_count_long >= cb and not break_found:
                # Entry at next bar open
                if i + 1 < len(bars):
                    entry_price = float(bars.iloc[i + 1]["open"])
                    stop_price = orb_low
                    target_price = entry_price + rr * orb_range
                    trade = _resolve_trade(
                        bars, i + 1, entry_price, stop_price, target_price,
                        "long", orb_range,
                    )
                    if trade is not None:
                        trades.append(trade)
                    break_found = True
                    break
        else:
            confirm_count_long = 0

        # Short break: close below ORB low
        if bar_close < orb_low and direction_filter in ("short", "both"):
            confirm_count_short += 1
            if confirm_count_short >= cb and not break_found:
                if i + 1 < len(bars):
                    entry_price = float(bars.iloc[i + 1]["open"])
                    stop_price = orb_high
                    target_price = entry_price - rr * orb_range
                    trade = _resolve_trade(
                        bars, i + 1, entry_price, stop_price, target_price,
                        "short", orb_range,
                    )
                    if trade is not None:
                        trades.append(trade)
                    break_found = True
                    break
        else:
            confirm_count_short = 0

    return trades

# ---------------------------------------------------------------------------
# Fade simulation (mean-reversion)
# ---------------------------------------------------------------------------
def simulate_fade(
    bars: pd.DataFrame,
    orb: dict,
    rr: float,
    cb: int,
    target_mode: str,  # "midpoint" or "opposite"
) -> list[dict]:
    """Simulate fade (mean-reversion) entries -- enter OPPOSITE to break.

    If ORB breaks long -> go SHORT
    If ORB breaks short -> go LONG
    Stop = ORB extreme + 0.3 * ORB range (buffer for fakeout)
    Target midpoint: ORB midpoint
    Target opposite: opposite ORB level
    """
    orb_high = orb["orb_high"]
    orb_low = orb["orb_low"]
    orb_range = orb["orb_range"]
    orb_mid = (orb_high + orb_low) / 2.0
    start_idx = orb["orb_end_idx"]

    if start_idx >= len(bars):
        return []

    trades = []
    confirm_count_long = 0
    confirm_count_short = 0
    break_found = False

    for i in range(start_idx, len(bars)):
        bar_close = bars.iloc[i]["close"]

        # ORB breaks long -> fade SHORT
        if bar_close > orb_high:
            confirm_count_long += 1
            if confirm_count_long >= cb and not break_found:
                if i + 1 < len(bars):
                    entry_price = float(bars.iloc[i + 1]["open"])
                    stop_price = orb_high + 0.3 * orb_range  # buffer above high
                    if target_mode == "midpoint":
                        target_price = orb_mid
                    else:  # opposite
                        target_price = orb_low
                    # Only take if target is below entry
                    if target_price < entry_price:
                        trade = _resolve_trade(
                            bars, i + 1, entry_price, stop_price, target_price,
                            "short", orb_range,
                        )
                        if trade is not None:
                            trades.append(trade)
                    break_found = True
                    break
        else:
            confirm_count_long = 0

        # ORB breaks short -> fade LONG
        if bar_close < orb_low:
            confirm_count_short += 1
            if confirm_count_short >= cb and not break_found:
                if i + 1 < len(bars):
                    entry_price = float(bars.iloc[i + 1]["open"])
                    stop_price = orb_low - 0.3 * orb_range  # buffer below low
                    if target_mode == "midpoint":
                        target_price = orb_mid
                    else:  # opposite
                        target_price = orb_high
                    # Only take if target is above entry
                    if target_price > entry_price:
                        trade = _resolve_trade(
                            bars, i + 1, entry_price, stop_price, target_price,
                            "long", orb_range,
                        )
                        if trade is not None:
                            trades.append(trade)
                    break_found = True
                    break
        else:
            confirm_count_short = 0

    return trades

# ---------------------------------------------------------------------------
# Trade resolution
# ---------------------------------------------------------------------------
def _resolve_trade(
    bars: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    direction: str,
    orb_range: float,
) -> dict | None:
    """Scan bars from entry_idx for stop/target hit.

    Stop takes priority on ambiguous bars (conservative).
    EOD exit on last bar if neither hit.
    """
    is_long = direction == "long"
    risk_points = abs(entry_price - stop_price)

    if risk_points <= 0:
        return None

    for i in range(entry_idx, len(bars)):
        bar_high = float(bars.iloc[i]["high"])
        bar_low = float(bars.iloc[i]["low"])

        if is_long:
            stop_hit = bar_low <= stop_price
            target_hit = bar_high >= target_price
        else:
            stop_hit = bar_high >= stop_price
            target_hit = bar_low <= target_price

        # Ambiguous bar -> LOSS (conservative)
        if stop_hit and target_hit:
            pnl_points = stop_price - entry_price if is_long else entry_price - stop_price
            return _build_trade(entry_price, stop_price, pnl_points, risk_points,
                                direction, "stop", orb_range)

        if stop_hit:
            pnl_points = stop_price - entry_price if is_long else entry_price - stop_price
            return _build_trade(entry_price, stop_price, pnl_points, risk_points,
                                direction, "stop", orb_range)

        if target_hit:
            pnl_points = target_price - entry_price if is_long else entry_price - target_price
            return _build_trade(entry_price, target_price, pnl_points, risk_points,
                                direction, "target", orb_range)

    # EOD exit at last bar close
    last_close = float(bars.iloc[-1]["close"])
    pnl_points = last_close - entry_price if is_long else entry_price - last_close
    return _build_trade(entry_price, last_close, pnl_points, risk_points,
                        direction, "eod", orb_range)

def _build_trade(
    entry_price: float,
    exit_price: float,
    pnl_points: float,
    risk_points: float,
    direction: str,
    outcome: str,
    orb_range: float,
) -> dict:
    """Build a trade result dict with gross and net P&L."""
    gross_pnl_dollars = pnl_points * SPEC.point_value
    net_pnl_dollars = gross_pnl_dollars - FRICTION

    # R-multiples: risk_dollars = risk_points * point_value + friction
    risk_dollars = risk_points * SPEC.point_value + FRICTION
    gross_r = (pnl_points * SPEC.point_value) / risk_dollars if risk_dollars > 0 else 0.0
    net_r = net_pnl_dollars / risk_dollars if risk_dollars > 0 else 0.0

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "direction": direction,
        "outcome": outcome,
        "pnl_points": pnl_points,
        "gross_pnl_dollars": gross_pnl_dollars,
        "net_pnl_dollars": net_pnl_dollars,
        "gross_r": gross_r,
        "net_r": net_r,
        "orb_range": orb_range,
    }

# ---------------------------------------------------------------------------
# Metrics formatting
# ---------------------------------------------------------------------------
def format_metrics(label: str, gross_pnls: np.ndarray, net_pnls: np.ndarray,
                   years: float) -> str:
    """Format one line of the results table."""
    gross = compute_strategy_metrics(gross_pnls)
    net = compute_strategy_metrics(net_pnls)
    if gross is None or net is None:
        return f"{label:<28s}   0    --       --       --     --      --      --"

    gross = annualize_sharpe(gross, years)
    net = annualize_sharpe(net, years)

    return (
        f"{label:<28s} {gross['n']:>4d}  {net['wr']*100:5.1f}%  "
        f"{gross['expr']:+7.3f}  {net['expr']:+7.3f}  "
        f"{net['sharpe']:6.3f}  {net['maxdd']:6.1f}  {net['total']:+6.1f}"
    )

def print_header():
    """Print results table header."""
    print(f"{'Variant':<28s} {'N':>4s}  {'WR':>5s}   {'GrossExpR':>7s}  "
          f"{'NetExpR':>7s}  {'Sharpe':>6s}  {'MaxDD':>6s}  {'TotalR':>6s}")
    print("-" * 90)

def print_weekday_breakdown(trades_by_weekday: dict[int, list[dict]], label: str):
    """Print per-weekday results for the given trade list."""
    print(f"\nBy weekday ({label}):")
    for wd in range(5):
        trades = trades_by_weekday.get(wd, [])
        if not trades:
            print(f"  {WEEKDAY_NAMES[wd]}  N=0")
            continue
        net_pnls = np.array([t["net_r"] for t in trades])
        n = len(net_pnls)
        wr = float((net_pnls > 0).sum() / n) * 100
        expr = float(net_pnls.mean())
        tag = "  <-- EIA day" if wd == 2 else ""
        print(f"  {WEEKDAY_NAMES[wd]}  N={n:<4d}  WR={wr:5.1f}%  ExpR={expr:+.3f}{tag}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MCL NYMEX ORB breakout/fade research")
    parser.add_argument("--db-path", type=str, default="C:/db/gold.db",
                        help="Path to DuckDB database")
    args = parser.parse_args()
    db_path = Path(args.db_path)

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    print(f"Database: {db_path}")
    print(f"Instrument: MCL (Micro Crude Oil)")
    print(f"Session: NYMEX open (13:00 UTC / 2300 Brisbane)")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Cost model: point_value=${SPEC.point_value:.0f}, "
          f"friction=${FRICTION:.2f} RT")
    print(f"Min ORB range: {MIN_ORB_RANGE_POINTS:.2f} pts "
          f"(= ${MIN_ORB_RANGE_POINTS * SPEC.point_value:.0f})")
    print()

    # -----------------------------------------------------------------------
    # Load all unique calendar days with MCL bars
    # -----------------------------------------------------------------------
    print("Loading trading days...")
    trading_days = load_trading_days(db_path)
    print(f"  Found {len(trading_days)} calendar days with MCL bars")

    # Filter to weekdays only and within date range
    valid_days = [
        d for d in trading_days
        if START_DATE <= d <= END_DATE and d.weekday() < 5
    ]
    print(f"  Weekdays in range: {len(valid_days)}")

    years = (END_DATE - START_DATE).days / 365.25

    # -----------------------------------------------------------------------
    # Pre-load all session bars
    # -----------------------------------------------------------------------
    print("Loading session bars (13:00-23:00 UTC) for each day...")
    day_bars = {}
    skipped_no_bars = 0
    for d in valid_days:
        bars = load_session_bars(db_path, d)
        if len(bars) >= 5:  # need at least a few bars
            day_bars[d] = bars
        else:
            skipped_no_bars += 1

    print(f"  Days with session bars: {len(day_bars)}")
    if skipped_no_bars > 0:
        print(f"  Skipped (no/few bars): {skipped_no_bars}")
    print()

    # -----------------------------------------------------------------------
    # Run simulations for each ORB duration
    # -----------------------------------------------------------------------
    for orb_minutes in ORB_DURATIONS:
        # Compute ORBs
        orbs = {}
        skipped_small = 0
        skipped_no_orb = 0
        for d, bars in day_bars.items():
            orb = compute_orb(bars, orb_minutes)
            if orb is None:
                skipped_no_orb += 1
                continue
            # Cost filter: ORB range * $100 >= $30
            if orb["orb_range"] * SPEC.point_value < 30.0:
                skipped_small += 1
                continue
            orbs[d] = orb

        print(f"{'='*60}")
        print(f"=== NYMEX {orb_minutes}m ORB BREAKOUT ===")
        print(f"{'='*60}")
        print(f"Days with valid ORB: {len(orbs)}  "
              f"(skipped: {skipped_no_orb} no-orb, {skipped_small} too-small)")
        print()

        # -------------------------------------------------------------------
        # BREAKOUT simulation
        # -------------------------------------------------------------------
        print_header()

        best_breakout_label = None
        best_breakout_expr = -999.0
        best_breakout_trades = []

        for rr in RR_TARGETS:
            for cb_val in CB_VALUES:
                for dir_filter in ["long", "short", "both"]:
                    dir_tag = dir_filter.upper()
                    label = f"E1_CB{cb_val}_RR{rr}__{dir_tag}"

                    all_trades = []
                    for d, orb in orbs.items():
                        trades = simulate_breakout(
                            day_bars[d], orb, rr, cb_val, dir_filter,
                        )
                        for t in trades:
                            t["weekday"] = d.weekday()
                        all_trades.extend(trades)

                    gross_pnls = np.array([t["gross_r"] for t in all_trades]) if all_trades else np.array([])
                    net_pnls = np.array([t["net_r"] for t in all_trades]) if all_trades else np.array([])

                    print(format_metrics(label, gross_pnls, net_pnls, years))

                    if len(net_pnls) > 0:
                        net_expr = float(net_pnls.mean())
                        if net_expr > best_breakout_expr:
                            best_breakout_expr = net_expr
                            best_breakout_label = label
                            best_breakout_trades = all_trades

        # Weekday breakdown for best breakout
        if best_breakout_trades:
            by_wd = {}
            for t in best_breakout_trades:
                wd = t["weekday"]
                by_wd.setdefault(wd, []).append(t)
            print_weekday_breakdown(by_wd, best_breakout_label)

        print()

        # -------------------------------------------------------------------
        # FADE simulation
        # -------------------------------------------------------------------
        print(f"{'='*60}")
        print(f"=== NYMEX {orb_minutes}m ORB FADE ===")
        print(f"{'='*60}")
        print()
        print_header()

        best_fade_label = None
        best_fade_expr = -999.0
        best_fade_trades = []

        for target_mode in ["midpoint", "opposite"]:
            for cb_val in CB_VALUES:
                # RR is implicit in fade (determined by ORB geometry), but we
                # still label by target mode
                label = f"FADE_CB{cb_val}_{target_mode.upper()}"

                all_trades = []
                for d, orb in orbs.items():
                    trades = simulate_fade(
                        day_bars[d], orb, 1.0, cb_val, target_mode,
                    )
                    for t in trades:
                        t["weekday"] = d.weekday()
                    all_trades.extend(trades)

                gross_pnls = np.array([t["gross_r"] for t in all_trades]) if all_trades else np.array([])
                net_pnls = np.array([t["net_r"] for t in all_trades]) if all_trades else np.array([])

                print(format_metrics(label, gross_pnls, net_pnls, years))

                if len(net_pnls) > 0:
                    net_expr = float(net_pnls.mean())
                    if net_expr > best_fade_expr:
                        best_fade_expr = net_expr
                        best_fade_label = label
                        best_fade_trades = all_trades

        # Weekday breakdown for best fade
        if best_fade_trades:
            by_wd = {}
            for t in best_fade_trades:
                wd = t["weekday"]
                by_wd.setdefault(wd, []).append(t)
            print_weekday_breakdown(by_wd, best_fade_label)

        print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("DONE. Review net ExpR and Sharpe to identify viable variants.")
    print("Wednesday (EIA inventory report day) may show different behavior.")
    print()

if __name__ == "__main__":
    main()
