#!/usr/bin/env python3
"""Alternative Stop & Re-entry Research. 2026-02-24.

Two modes:
A) Pullback entry (E3 CB1) with tighter stops — same ORB-edge entry, smaller risk
B) Second-chance re-entry after failed first breakout — re-break same direction

Usage:
    python research/research_alt_stops.py
    python research/research_alt_stops.py --mode pullback --instrument MGC
    python research/research_alt_stops.py --mode reentry --instrument MES
"""

from __future__ import annotations

import argparse
import sys
import time

# Force unbuffered stdout for background runs
sys.stdout.reconfigure(line_buffering=True)
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pipeline.asset_configs import get_enabled_sessions
from pipeline.cost_model import COST_SPECS, CostSpec
from trading_app.entry_rules import detect_entry_with_confirm_bars

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTIVE_INSTRUMENTS = ["MGC", "MES", "MNQ", "M2K"]
MIN_TRADES = 30
FDR_ALPHA = 0.05

# Mode A: Stop fractions (of ORB range). 1.00 = standard E3 baseline.
STOP_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
PULLBACK_RR_TARGETS = [1.0, 1.5, 2.0, 3.0]

# Mode B: Failure window sizes (bars after initial break)
FAIL_WINDOWS = [5, 10, 20]
REENTRY_RR_TARGETS = [1.0, 1.5, 2.0]

MAX_TRADE_BARS = 240  # 4 hours


# ---------------------------------------------------------------------------
# StrategyResult
# ---------------------------------------------------------------------------
@dataclass
class StrategyResult:
    """Result of a single strategy backtest."""

    mode: str
    instrument: str
    variant: str
    params: dict
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    avg_pnl_r: float = 0.0
    total_pnl_r: float = 0.0
    sharpe: float = 0.0
    max_dd_r: float = 0.0
    p_value: float = 1.0
    yearly_results: dict = field(default_factory=dict)
    pnl_series: list = field(default_factory=list)
    trade_dates: list = field(default_factory=list)

    @property
    def classification(self) -> str:
        if self.n_trades < 30:
            return "INVALID"
        elif self.n_trades < 100:
            return "REGIME"
        elif self.n_trades < 200:
            return "PRELIMINARY"
        return "CORE"


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------
def compute_max_dd(pnl_series: list[float]) -> float:
    if not pnl_series:
        return 0.0
    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return round(float(np.max(dd)), 4) if len(dd) > 0 else 0.0


def yearly_breakdown(dates: list, pnls: list[float]) -> dict:
    if not dates or not pnls:
        return {}
    df = pd.DataFrame({"date": dates, "pnl_r": pnls})
    df["year"] = pd.to_datetime(df["date"]).dt.year
    result = {}
    for year, grp in df.groupby("year"):
        n = len(grp)
        if n < 5:
            continue
        result[int(year)] = {
            "n": n,
            "avg_r": round(float(grp["pnl_r"].mean()), 4),
            "total_r": round(float(grp["pnl_r"].sum()), 2),
            "wr": round(float((grp["pnl_r"] > 0).mean()), 4),
        }
    return result


def apply_friction_r(pnl_points: float, risk_points: float, spec: CostSpec) -> float:
    """Convert points PnL to R-multiple after friction."""
    if risk_points <= 0:
        return 0.0
    return (pnl_points - spec.friction_in_points) / risk_points


def finalize_result(
    mode: str,
    instrument: str,
    variant: str,
    params: dict,
    trades: list[float],
    trade_dates: list,
) -> StrategyResult:
    """Compute stats from a list of R-multiple trades."""
    n = len(trades)
    if n < 1:
        return StrategyResult(mode=mode, instrument=instrument, variant=variant, params=params)
    arr = np.array(trades)
    wins = int(np.sum(arr > 0))
    avg_r = float(np.mean(arr))
    total_r = float(np.sum(arr))
    std_r = float(np.std(arr, ddof=1)) if n > 1 else 1.0
    sharpe = avg_r / std_r if std_r > 0 else 0.0

    if n >= 2 and std_r > 0:
        t_stat, p_val = stats.ttest_1samp(arr, 0.0)
        p_val = float(p_val / 2) if t_stat > 0 else 1.0 - float(p_val / 2)
    else:
        p_val = 1.0

    return StrategyResult(
        mode=mode,
        instrument=instrument,
        variant=variant,
        params=params,
        n_trades=n,
        n_wins=wins,
        win_rate=round(wins / n, 4),
        avg_pnl_r=round(avg_r, 4),
        total_pnl_r=round(total_r, 2),
        sharpe=round(sharpe, 4),
        max_dd_r=compute_max_dd(trades),
        p_value=round(p_val, 6),
        yearly_results=yearly_breakdown(trade_dates, trades),
        pnl_series=trades,
        trade_dates=trade_dates,
    )


def bh_fdr(p_values: list[float], alpha: float = 0.05) -> list[tuple[int, float, bool]]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev_adj = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        adj_p = min(prev_adj, p * n / rank, 1.0)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p
    return [(i, adjusted[i], adjusted[i] < alpha) for i in range(n)]


def simulate_trade(
    bars_after: pd.DataFrame,
    direction: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    max_bars: int = MAX_TRADE_BARS,
) -> tuple[float, str]:
    """Simulate a trade on 1m bars.

    Returns (pnl_points, exit_type). Stop checked before target (conservative).
    """
    bars_arr = bars_after[["high", "low", "close"]].values
    n_bars = min(len(bars_arr), max_bars)

    for i in range(n_bars):
        h, l, c = bars_arr[i]
        if direction == 1:  # long
            if l <= stop_price:
                return (stop_price - entry_price), "stop"
            if h >= target_price:
                return (target_price - entry_price), "target"
        else:  # short
            if h >= stop_price:
                return (entry_price - stop_price), "stop"
            if l <= target_price:
                return (entry_price - target_price), "target"

    if n_bars > 0:
        last_close = bars_arr[n_bars - 1][2]
        return (last_close - entry_price) * direction, "time"
    return 0.0, "time"


def compute_td_end(trading_day) -> datetime:
    """Trading day ends at 23:00 UTC (= 09:00 next day Brisbane)."""
    if isinstance(trading_day, pd.Timestamp):
        td = trading_day.date()
    elif isinstance(trading_day, datetime):
        td = trading_day.date()
    elif hasattr(trading_day, "year"):
        td = trading_day
    else:
        td = pd.Timestamp(trading_day).date()
    return datetime(td.year, td.month, td.day, 23, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_instrument_bars(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
) -> pd.DataFrame:
    """Load all 1m bars with trading_day column."""
    print(f"    Loading 1m bars for {instrument}...", end=" ", flush=True)
    t0 = time.time()
    df = con.execute(
        """
        SELECT ts_utc, open, high, low, close, volume,
               CAST((ts_utc AT TIME ZONE 'Australia/Brisbane'
                     - INTERVAL '9 hours') AS DATE) AS trading_day
        FROM bars_1m
        WHERE symbol = ?
        ORDER BY ts_utc
    """,
        [instrument],
    ).fetchdf()
    elapsed = time.time() - t0
    print(f"{len(df):,} bars in {elapsed:.1f}s")
    return df


def load_daily_features(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    sessions: list[str],
) -> pd.DataFrame:
    """Load daily features (orb_minutes=5) with ORB columns for sessions."""
    # Get available columns
    all_cols = {
        c[0]
        for c in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='daily_features'"
        ).fetchall()
    }

    cols = ["trading_day", "atr_20"]
    for label in sessions:
        for suffix in ["_high", "_low", "_size", "_break_dir", "_break_ts"]:
            col = f"orb_{label}{suffix}"
            if col in all_cols:
                cols.append(col)

    col_str = ", ".join(cols)
    df = con.execute(
        f"""
        SELECT {col_str}
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
        ORDER BY trading_day
    """,
        [instrument],
    ).fetchdf()
    return df


# =============================================================================
# MODE A: Pullback with Tighter Stop
# =============================================================================


def run_pullback_tighter_stop(
    daily_df: pd.DataFrame,
    bars_by_day: dict,
    instrument: str,
    orb_label: str,
    stop_frac: float,
    rr_target: float,
    spec: CostSpec,
) -> StrategyResult:
    """E3 CB1 entry at ORB edge, stop at fraction of ORB range.

    stop_frac=0.25: stop 25% of ORB from entry (very tight)
    stop_frac=1.00: stop at opposite ORB edge (standard E3 baseline)
    """
    variant = f"{orb_label}_E3_frac{stop_frac:.2f}_RR{rr_target}"
    params = {"orb_label": orb_label, "stop_frac": stop_frac, "rr_target": rr_target, "entry_model": "E3"}
    mode = "pullback"

    h_col = f"orb_{orb_label}_high"
    l_col = f"orb_{orb_label}_low"
    dir_col = f"orb_{orb_label}_break_dir"
    ts_col = f"orb_{orb_label}_break_ts"

    # Check columns exist
    for c in [h_col, l_col, dir_col, ts_col]:
        if c not in daily_df.columns:
            return StrategyResult(mode=mode, instrument=instrument, variant=variant, params=params)

    min_orb_pts = spec.friction_in_points * 2.5

    trades = []
    trade_dates = []

    for _, row in daily_df.iterrows():
        break_dir = row.get(dir_col)
        break_ts = row.get(ts_col)
        orb_high = row.get(h_col)
        orb_low = row.get(l_col)
        trading_day = row["trading_day"]

        if pd.isna(break_dir) or pd.isna(break_ts):
            continue
        if pd.isna(orb_high) or pd.isna(orb_low):
            continue

        orb_size = float(orb_high) - float(orb_low)
        if orb_size < min_orb_pts or orb_size <= 0:
            continue

        day_bars = bars_by_day.get(trading_day)
        if day_bars is None or len(day_bars) == 0:
            continue

        td_end = compute_td_end(trading_day)

        # Use production E3 CB1 entry detection
        signal = detect_entry_with_confirm_bars(
            bars_df=day_bars,
            orb_break_ts=break_ts,
            orb_high=float(orb_high),
            orb_low=float(orb_low),
            break_dir=break_dir,
            confirm_bars=1,
            detection_window_end=td_end,
            entry_model="E3",
        )

        if not signal.triggered:
            continue

        entry_price = signal.entry_price

        # Compute alternative stop
        if break_dir == "long":
            stop_price = entry_price - stop_frac * orb_size
            direction = 1
        else:
            stop_price = entry_price + stop_frac * orb_size
            direction = -1

        risk_points = abs(entry_price - stop_price)
        if risk_points <= 0 or risk_points < spec.friction_in_points * 2:
            continue

        target_price = entry_price + direction * rr_target * risk_points

        # Simulate from fill bar onward (entry is intra-bar limit fill)
        post_entry = day_bars[day_bars["ts_utc"] >= pd.Timestamp(signal.entry_ts)]
        if len(post_entry) == 0:
            continue

        pnl_pts, _ = simulate_trade(
            post_entry,
            direction,
            entry_price,
            stop_price,
            target_price,
            max_bars=MAX_TRADE_BARS,
        )

        pnl_r = apply_friction_r(pnl_pts, risk_points, spec)
        trades.append(pnl_r)
        trade_dates.append(trading_day)

    return finalize_result(mode, instrument, variant, params, trades, trade_dates)


# =============================================================================
# MODE B: Second-Chance Re-entry After Failed Breakout
# =============================================================================


def run_second_chance(
    daily_df: pd.DataFrame,
    bars_by_day: dict,
    instrument: str,
    orb_label: str,
    n_fail_bars: int,
    rr_target: float,
    spec: CostSpec,
) -> StrategyResult:
    """Enter on second break after first break fails within N bars.

    1. ORB breaks direction D
    2. Within n_fail_bars, price closes back inside ORB -> failure
    3. Scan for second close outside ORB in same direction D
    4. Enter at next bar open (E1-style)
    5. Stop at swing extreme between failure and 2nd break
    6. If swing stop > full ORB, cap at full ORB
    """
    variant = f"{orb_label}_reentry_N{n_fail_bars}_RR{rr_target}"
    params = {"orb_label": orb_label, "n_fail_bars": n_fail_bars, "rr_target": rr_target}
    mode = "reentry"

    h_col = f"orb_{orb_label}_high"
    l_col = f"orb_{orb_label}_low"
    dir_col = f"orb_{orb_label}_break_dir"
    ts_col = f"orb_{orb_label}_break_ts"

    for c in [h_col, l_col, dir_col, ts_col]:
        if c not in daily_df.columns:
            return StrategyResult(mode=mode, instrument=instrument, variant=variant, params=params)

    min_orb_pts = spec.friction_in_points * 2.5

    trades = []
    trade_dates = []

    for _, row in daily_df.iterrows():
        break_dir = row.get(dir_col)
        break_ts = row.get(ts_col)
        orb_high = row.get(h_col)
        orb_low = row.get(l_col)
        trading_day = row["trading_day"]

        if pd.isna(break_dir) or pd.isna(break_ts):
            continue
        if pd.isna(orb_high) or pd.isna(orb_low):
            continue

        orb_high = float(orb_high)
        orb_low = float(orb_low)
        orb_size = orb_high - orb_low
        if orb_size < min_orb_pts or orb_size <= 0:
            continue

        day_bars = bars_by_day.get(trading_day)
        if day_bars is None or len(day_bars) == 0:
            continue

        # Bars after initial break
        post_break = day_bars[day_bars["ts_utc"] > break_ts]
        if len(post_break) < n_fail_bars + 5:
            continue

        # Step 2: Look for failure — close back inside ORB within N bars
        scan_window = post_break.iloc[:n_fail_bars]
        fail_idx = None

        closes = scan_window["close"].values
        if break_dir == "long":
            for j in range(len(closes)):
                if closes[j] < orb_high:
                    fail_idx = j
                    break
        elif break_dir == "short":
            for j in range(len(closes)):
                if closes[j] > orb_low:
                    fail_idx = j
                    break

        if fail_idx is None:
            continue  # Break held — no failure

        # Step 3: Look for second break in same direction after failure
        remaining = post_break.iloc[fail_idx + 1 :]
        if len(remaining) < 3:
            continue

        rebreak_idx = None
        rem_closes = remaining["close"].values
        if break_dir == "long":
            for k in range(len(rem_closes)):
                if rem_closes[k] > orb_high:
                    rebreak_idx = k
                    break
        elif break_dir == "short":
            for k in range(len(rem_closes)):
                if rem_closes[k] < orb_low:
                    rebreak_idx = k
                    break

        if rebreak_idx is None:
            continue  # No second break

        # Step 4: Entry at next bar open after 2nd confirmation
        entry_bars = remaining.iloc[rebreak_idx + 1 :]
        if len(entry_bars) == 0:
            continue

        entry_price = float(entry_bars.iloc[0]["open"])

        # Step 5: Stop at swing extreme between failure and 2nd break
        swing_bars = remaining.iloc[: rebreak_idx + 1]
        if break_dir == "long":
            direction = 1
            swing_low = float(swing_bars["low"].min())
            stop_price = max(swing_low, orb_low)  # cap at full ORB
        else:
            direction = -1
            swing_high = float(swing_bars["high"].max())
            stop_price = min(swing_high, orb_high)  # cap at full ORB

        risk_points = abs(entry_price - stop_price)
        if risk_points <= 0 or risk_points < spec.friction_in_points * 2:
            continue

        target_price = entry_price + direction * rr_target * risk_points

        # Simulate from entry bar (entry at bar open, bar range is post-entry)
        pnl_pts, _ = simulate_trade(
            entry_bars,
            direction,
            entry_price,
            stop_price,
            target_price,
            max_bars=MAX_TRADE_BARS,
        )

        pnl_r = apply_friction_r(pnl_pts, risk_points, spec)
        trades.append(pnl_r)
        trade_dates.append(trading_day)

    return finalize_result(mode, instrument, variant, params, trades, trade_dates)


# =============================================================================
# Main Runner
# =============================================================================


def run_all(
    db_path: str,
    mode_filter: Optional[str] = None,
    instrument_filter: Optional[str] = None,
) -> list[StrategyResult]:
    """Run all alternative stop and re-entry strategies."""
    results = []
    con = duckdb.connect(db_path, read_only=True)

    instruments = ACTIVE_INSTRUMENTS
    if instrument_filter:
        instruments = [instrument_filter.upper()]

    try:
        for instrument in instruments:
            spec = COST_SPECS.get(instrument)
            if spec is None:
                print(f"  No cost model for {instrument}, skipping")
                continue

            sessions = get_enabled_sessions(instrument)
            if not sessions:
                print(f"  No enabled sessions for {instrument}, skipping")
                continue

            print(f"\n{'=' * 70}")
            print(f"  INSTRUMENT: {instrument} ({len(sessions)} sessions)")
            print(f"{'=' * 70}")

            bars_df = load_instrument_bars(con, instrument)
            if len(bars_df) < 10000:
                print(f"    Insufficient 1m bars ({len(bars_df)}), skipping")
                continue

            daily_df = load_daily_features(con, instrument, sessions)
            if len(daily_df) < 200:
                print(f"    Insufficient daily features ({len(daily_df)}), skipping")
                continue

            bars_by_day = {day: grp for day, grp in bars_df.groupby("trading_day")}

            # =============================================================
            # MODE A: Pullback + Tighter Stop
            # =============================================================
            if not mode_filter or mode_filter == "pullback":
                print(f"\n  --- Pullback + Tighter Stop ({instrument}) ---")
                for label in sessions:
                    for stop_frac in STOP_FRACTIONS:
                        for rr in PULLBACK_RR_TARGETS:
                            r = run_pullback_tighter_stop(
                                daily_df,
                                bars_by_day,
                                instrument,
                                label,
                                stop_frac=stop_frac,
                                rr_target=rr,
                                spec=spec,
                            )
                            results.append(r)
                            if r.n_trades >= MIN_TRADES:
                                sig = "*" if r.p_value < 0.05 else " "
                                print(
                                    f"    {sig} {label:18s} "
                                    f"frac={stop_frac:.2f} RR={rr:.1f} "
                                    f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                                    f"ExpR={r.avg_pnl_r:+.4f} "
                                    f"p={r.p_value:.4f}"
                                )

            # =============================================================
            # MODE B: Second-Chance Re-entry
            # =============================================================
            if not mode_filter or mode_filter == "reentry":
                print(f"\n  --- Second-Chance Re-entry ({instrument}) ---")
                for label in sessions:
                    for n_fail in FAIL_WINDOWS:
                        for rr in REENTRY_RR_TARGETS:
                            r = run_second_chance(
                                daily_df,
                                bars_by_day,
                                instrument,
                                label,
                                n_fail_bars=n_fail,
                                rr_target=rr,
                                spec=spec,
                            )
                            results.append(r)
                            if r.n_trades >= MIN_TRADES:
                                sig = "*" if r.p_value < 0.05 else " "
                                print(
                                    f"    {sig} {label:18s} "
                                    f"N_fail={n_fail:2d} RR={rr:.1f} "
                                    f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                                    f"ExpR={r.avg_pnl_r:+.4f} "
                                    f"p={r.p_value:.4f}"
                                )

            del bars_df, bars_by_day, daily_df

        # =================================================================
        # FDR CORRECTION
        # =================================================================
        valid_results = [r for r in results if r.n_trades >= MIN_TRADES]
        print_fdr_results(valid_results)
        print_baseline_comparison(valid_results)
        print_honest_summary(valid_results)

    finally:
        con.close()

    return results


# =============================================================================
# Reporting
# =============================================================================


def print_fdr_results(valid_results: list[StrategyResult]) -> None:
    print("\n" + "=" * 70)
    print("BH FDR CORRECTION")
    print("=" * 70)

    if not valid_results:
        print(f"\nNo strategies had N >= {MIN_TRADES}")
        return

    p_values = [r.p_value for r in valid_results]
    fdr_results = bh_fdr(p_values, alpha=FDR_ALPHA)

    n_tested = len(valid_results)
    n_sig = sum(1 for _, _, sig in fdr_results if sig)
    print(f"\nTests with N >= {MIN_TRADES}: {n_tested}")
    print(f"FDR-significant at alpha={FDR_ALPHA}: {n_sig}")

    if n_sig > 0:
        print(
            f"\n{'Mode':<12s} {'Inst':<6s} {'Variant':<42s} "
            f"{'N':>5s} {'WR':>7s} {'ExpR':>8s} {'S':>6s} "
            f"{'raw_p':>8s} {'adj_p':>8s}"
        )
        print("-" * 110)
        for idx, adj_p, sig in fdr_results:
            if sig:
                r = valid_results[idx]
                print(
                    f"{r.mode:<12s} {r.instrument:<6s} "
                    f"{r.variant:<42s} "
                    f"{r.n_trades:5d} {r.win_rate:7.2%} "
                    f"{r.avg_pnl_r:+8.4f} {r.sharpe:6.3f} "
                    f"{r.p_value:8.4f} {adj_p:8.4f}"
                )
    else:
        print("\nNo strategies survived FDR correction.")


def print_baseline_comparison(valid_results: list[StrategyResult]) -> None:
    print("\n" + "=" * 70)
    print("TIGHTER STOP vs STANDARD BASELINE (pullback mode)")
    print("=" * 70)

    pullback = [r for r in valid_results if r.mode == "pullback"]
    if not pullback:
        print("\n  No valid pullback results.")
        return

    groups = defaultdict(list)
    for r in pullback:
        key = (r.instrument, r.params["orb_label"], r.params["rr_target"])
        groups[key].append(r)

    print(
        f"\n{'Inst':<6s} {'Session':<18s} {'RR':>4s} | "
        f"{'frac=0.25':>14s} {'frac=0.50':>14s} "
        f"{'frac=0.75':>14s} {'frac=1.00':>14s}"
    )
    print("-" * 90)

    for key in sorted(groups.keys()):
        inst, session, rr = key
        fracs = {r.params["stop_frac"]: r for r in groups[key]}
        line = f"{inst:<6s} {session:<18s} {rr:4.1f} |"
        for f in STOP_FRACTIONS:
            r = fracs.get(f)
            if r and r.n_trades >= MIN_TRADES:
                line += f" {r.avg_pnl_r:+.3f}({r.n_trades:3d})"
            else:
                line += "            ---"
        print(line)


def print_honest_summary(valid_results: list[StrategyResult]) -> None:
    print("\n" + "=" * 70)
    print("HONEST SUMMARY (per RESEARCH_RULES.md)")
    print("=" * 70)

    # Check for FDR survivors
    p_values = [r.p_value for r in valid_results]
    fdr_results = bh_fdr(p_values, alpha=FDR_ALPHA) if valid_results else []
    survived = []
    if fdr_results:
        survived = [valid_results[i] for i, (_, _, sig) in enumerate(fdr_results) if sig]

    print("\nSURVIVED SCRUTINY:")
    if survived:
        for r in survived:
            yy = r.yearly_results
            yy_str = ", ".join(f"{y}:{d['avg_r']:+.3f}" for y, d in sorted(yy.items()))
            print(f"  - {r.mode} | {r.instrument} | {r.variant}")
            print(f"    N={r.n_trades}, ExpR={r.avg_pnl_r:+.4f}, Sharpe={r.sharpe:.3f}, p={r.p_value:.4f}")
            print(f"    Year-by-year: {yy_str}")
    else:
        print("  None.")

    n_failed = len(valid_results) - len(survived)
    print(f"\nDID NOT SURVIVE: {n_failed} strategies tested")

    # Per-mode summary
    for mode_name in ["pullback", "reentry"]:
        mode_results = [r for r in valid_results if r.mode == mode_name]
        if not mode_results:
            continue
        best = min(mode_results, key=lambda r: r.p_value)
        avg_exp = np.mean([r.avg_pnl_r for r in mode_results])
        print(f"\n  {mode_name} ({len(mode_results)} valid tests):")
        print(f"    Average ExpR across all: {avg_exp:+.4f}")
        print(
            f"    Best: {best.instrument} {best.variant} "
            f"N={best.n_trades} ExpR={best.avg_pnl_r:+.4f} "
            f"Sharpe={best.sharpe:.3f} p={best.p_value:.4f}"
        )

        if mode_name == "pullback":
            frac_avgs = {}
            for f in STOP_FRACTIONS:
                frac_r = [r for r in mode_results if r.params.get("stop_frac") == f]
                if frac_r:
                    frac_avgs[f] = np.mean([r.avg_pnl_r for r in frac_r])
            if frac_avgs:
                print(
                    "    Avg ExpR by stop fraction: "
                    + ", ".join(f"{f:.2f}={v:+.4f}" for f, v in sorted(frac_avgs.items()))
                )

    print("\nCAVEATS:")
    print("  - Tighter stops increase stop-out rate. Check WR alongside ExpR.")
    print("  - Second-chance has selection bias: only fires after failure + re-break.")
    print("  - Swing stop in reentry varies widely — distribution matters.")
    print("  - All results are IN-SAMPLE. Walk-forward needed before 'validated'.")
    print("  - MNQ/M2K have fewer years — year-by-year less reliable.")

    print("\nNEXT STEPS:")
    print("  - If any fraction improves: test E0 CB1 with same tighter stops")
    print("  - If reentry works: add to outcome_builder as new entry model")
    print("  - Sensitivity: +/-1 fraction step around best (e.g., 0.40-0.60)")
    print("  - DST regime split for affected sessions (0900/1800/0030/2300)")


# =============================================================================
# Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Alternative Stop & Re-entry Research")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "gold.db")
    parser.add_argument("--mode", type=str, default=None, choices=["pullback", "reentry"], help="Run only one mode")
    parser.add_argument(
        "--instrument", type=str, default=None, choices=ACTIVE_INSTRUMENTS, help="Run only one instrument"
    )
    args = parser.parse_args()

    print("Alternative Stop & Re-entry Research")
    print(f"Database: {args.db_path}")
    print(f"Mode: {args.mode or 'ALL'}")
    print(f"Instrument: {args.instrument or 'ALL'}")
    print(f"Date: {date.today()}")
    n_pullback = len(STOP_FRACTIONS) * len(PULLBACK_RR_TARGETS)
    n_reentry = len(FAIL_WINDOWS) * len(REENTRY_RR_TARGETS)
    print(f"Grid per session: {n_pullback} pullback + {n_reentry} reentry")
    print()

    t0 = time.time()
    results = run_all(
        str(args.db_path),
        mode_filter=args.mode,
        instrument_filter=args.instrument,
    )
    elapsed = time.time() - t0

    total = len(results)
    valid = sum(1 for r in results if r.n_trades >= MIN_TRADES)
    print(f"\nTotal tests: {total}, Valid (N>={MIN_TRADES}): {valid}")
    print(f"Runtime: {elapsed:.0f}s ({elapsed / 60:.1f}min)")


if __name__ == "__main__":
    main()
