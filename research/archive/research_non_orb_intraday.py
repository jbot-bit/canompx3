#!/usr/bin/env python3
"""Non-ORB Strategy Research — Phase 2: Intraday Strategies.

Tests 3 strategy archetypes that scan 1m bars:
  1. Failed Breakout Fade (mean reversion after failed ORB break)
  2. Late-Session Reversal (end-of-session profit-taking reversion)
  3. VWAP Reversion (deviation from session VWAP)

Usage:
    python research/research_non_orb_intraday.py --db-path gold.db
    python research/research_non_orb_intraday.py --db-path gold.db --archetype fade
    python research/research_non_orb_intraday.py --db-path gold.db --instrument MGC
"""

from __future__ import annotations

import argparse
import sys
import time
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import COST_SPECS, CostSpec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_INSTRUMENTS = ["MGC", "MES", "MNQ", "M2K", "MCL", "M6E", "SIL"]

MIN_TRADES = 30  # REGIME minimum per RESEARCH_RULES.md
FDR_ALPHA = 0.05

# ORB labels to test for Failed Breakout Fade (sessions with meaningful breaks)
FADE_ORB_LABELS = ["1000", "1100", "1800", "0030", "US_EQUITY_OPEN"]

# Session blocks for Late-Session Reversal and VWAP Reversion
# Times are in UTC (Brisbane - 10 hours). Session end = next group boundary.
SESSION_BLOCKS = {
    "asia": {
        "start_utc_h": 0,  # 10:00 Brisbane = 00:00 UTC
        "end_utc_h": 8,  # 18:00 Brisbane = 08:00 UTC
        "duration_hours": 8,
    },
    "london": {
        "start_utc_h": 8,  # 18:00 Brisbane = 08:00 UTC
        "end_utc_h": 13,  # 23:00 Brisbane = 13:00 UTC
        "duration_hours": 5,
    },
    "us": {
        "start_utc_h": 13,  # 23:00 Brisbane = 13:00 UTC
        "end_utc_h": 19,  # 05:00 Brisbane = 19:00 UTC
        "duration_hours": 6,
    },
}

# Maximum bars after entry for time stop (Failed Breakout Fade)
MAX_TRADE_BARS = 240  # 4 hours


# ---------------------------------------------------------------------------
# StrategyResult (same as Phase 1)
# ---------------------------------------------------------------------------
@dataclass
class StrategyResult:
    """Result of a single strategy backtest."""

    archetype: str
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
    archetype: str,
    instrument: str,
    variant: str,
    params: dict,
    trades: list[float],
    trade_dates: list,
) -> StrategyResult:
    """Compute stats from a list of R-multiple trades."""
    n = len(trades)
    if n < 1:
        return StrategyResult(archetype=archetype, instrument=instrument, variant=variant, params=params)

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
        archetype=archetype,
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


def compute_orb_correlation(
    con: duckdb.DuckDBPyConnection,
    result: StrategyResult,
    instrument: str,
) -> Optional[float]:
    """Compute correlation between strategy daily PnL and representative ORB PnL."""
    if not result.trade_dates or not result.pnl_series:
        return None
    orb_df = con.execute(
        """
        SELECT o.trading_day, o.pnl_r
        FROM orb_outcomes o
        WHERE o.symbol = ? AND o.orb_label = '1000'
          AND o.entry_model = 'E0' AND o.confirm_bars = 1
          AND o.rr_target = 1.0 AND o.orb_minutes = 5
          AND o.outcome IN ('WIN', 'LOSS')
        ORDER BY o.trading_day
    """,
        [instrument],
    ).fetchdf()
    if orb_df.empty:
        orb_df = con.execute(
            """
            SELECT o.trading_day, o.pnl_r
            FROM orb_outcomes o
            WHERE o.symbol = ? AND o.orb_label = '0900'
              AND o.entry_model = 'E1' AND o.confirm_bars = 1
              AND o.rr_target = 1.0 AND o.orb_minutes = 5
              AND o.outcome IN ('WIN', 'LOSS')
            ORDER BY o.trading_day
        """,
            [instrument],
        ).fetchdf()
    if orb_df.empty:
        return None
    strat_df = pd.DataFrame({"trading_day": result.trade_dates, "strat_pnl": result.pnl_series})
    strat_df["trading_day"] = pd.to_datetime(strat_df["trading_day"])
    orb_df["trading_day"] = pd.to_datetime(orb_df["trading_day"])
    merged = strat_df.merge(orb_df, on="trading_day", how="inner")
    if len(merged) < 20:
        return None
    corr = merged["strat_pnl"].corr(merged["pnl_r"])
    return round(float(corr), 4) if not pd.isna(corr) else None


# ---------------------------------------------------------------------------
# Trade Simulation Helper
# ---------------------------------------------------------------------------
def simulate_trade(
    bars_after: pd.DataFrame,
    direction: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    max_bars: int = MAX_TRADE_BARS,
) -> tuple[float, str]:
    """Simulate a trade on 1m bars.

    Returns (pnl_points, exit_type) where exit_type is 'target', 'stop', or 'time'.
    PnL is signed: positive = profit in trade direction.
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

    # Time stop: exit at last bar's close
    if n_bars > 0:
        last_close = bars_arr[n_bars - 1][2]
        return (last_close - entry_price) * direction, "time"
    return 0.0, "time"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_instrument_bars(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
) -> pd.DataFrame:
    """Load all 1m bars for an instrument with trading_day column."""
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
) -> pd.DataFrame:
    """Load daily features (orb_minutes=5) with ORB columns for all labels."""
    # Build column list for ORB labels we need
    cols = ["trading_day", "atr_20", "us_dst", "uk_dst", "daily_open", "daily_close", "daily_high", "daily_low"]
    for label in FADE_ORB_LABELS:
        cols.extend(
            [
                f"orb_{label}_high",
                f"orb_{label}_low",
                f"orb_{label}_size",
                f"orb_{label}_break_dir",
                f"orb_{label}_break_ts",
            ]
        )
    # Session range stats for Late-Session Reversal
    cols.extend(
        [
            "session_asia_high",
            "session_asia_low",
            "session_london_high",
            "session_london_low",
            "session_ny_high",
            "session_ny_low",
        ]
    )
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
# ARCHETYPE 1: Failed Breakout Fade
# =============================================================================


def run_failed_breakout_fade(
    daily_df: pd.DataFrame,
    bars_by_day: dict,
    instrument: str,
    orb_label: str,
    n_return_bars: int = 10,
    rr_target: float = 1.0,
    spec: CostSpec | None = None,
) -> StrategyResult:
    """Failed Breakout Fade: after ORB break fails and reverses, trade opposite.

    Mechanism: Trapped breakout traders create reverse flow when they stop out
    after a failed ORB break. Double-break days show this pattern.

    Entry: After ORB breaks direction D, if price returns inside range within
    N bars, enter opposite to D.
    Target: Entry ± RR × risk in fade direction.
    Stop: ORB boundary in original break direction.
    Time stop: 4 hours after entry.
    """
    variant = f"{orb_label}_N{n_return_bars}_RR{rr_target}"
    params = {"orb_label": orb_label, "n_return_bars": n_return_bars, "rr_target": rr_target}
    archetype = "failed_breakout_fade"

    if spec is None:
        spec = COST_SPECS.get(instrument)
    if spec is None:
        return StrategyResult(archetype=archetype, instrument=instrument, variant=variant, params=params)

    h_col = f"orb_{orb_label}_high"
    l_col = f"orb_{orb_label}_low"
    dir_col = f"orb_{orb_label}_break_dir"
    ts_col = f"orb_{orb_label}_break_ts"
    size_col = f"orb_{orb_label}_size"

    min_risk_pts = spec.friction_in_points * 2.5  # skip if risk < 2.5× friction

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
        if orb_high <= orb_low:
            continue

        day_bars = bars_by_day.get(trading_day)
        if day_bars is None or len(day_bars) == 0:
            continue

        # Get bars after break timestamp
        # break_ts is timezone-aware; bars ts_utc should be too
        mask = day_bars["ts_utc"] > break_ts
        post_break = day_bars[mask]
        if len(post_break) < n_return_bars + 2:
            continue

        # Scan first N bars for re-entry
        scan_bars = post_break.iloc[:n_return_bars]
        entry_bar_idx = None

        if break_dir == "long":
            # Price broke above ORB high. Look for close back below ORB high.
            for j in range(len(scan_bars)):
                if scan_bars.iloc[j]["close"] < orb_high:
                    entry_bar_idx = j
                    break
        elif break_dir == "short":
            # Price broke below ORB low. Look for close back above ORB low.
            for j in range(len(scan_bars)):
                if scan_bars.iloc[j]["close"] > orb_low:
                    entry_bar_idx = j
                    break

        if entry_bar_idx is None:
            continue

        # Set up trade
        entry_price = scan_bars.iloc[entry_bar_idx]["close"]

        if break_dir == "long":
            # Fade: go SHORT
            direction = -1
            stop_price = orb_high
            risk_pts = stop_price - entry_price  # positive (stop above entry)
        else:
            # Fade: go LONG
            direction = 1
            stop_price = orb_low
            risk_pts = entry_price - stop_price  # positive (stop below entry)

        if risk_pts < min_risk_pts or risk_pts <= 0:
            continue

        target_price = entry_price + direction * rr_target * risk_pts

        # Simulate on bars after entry
        remaining_bars = post_break.iloc[entry_bar_idx + 1 :]
        if len(remaining_bars) == 0:
            continue

        pnl_pts, exit_type = simulate_trade(
            remaining_bars,
            direction,
            entry_price,
            stop_price,
            target_price,
            max_bars=MAX_TRADE_BARS,
        )

        pnl_r = apply_friction_r(pnl_pts, risk_pts, spec)
        trades.append(pnl_r)
        trade_dates.append(trading_day)

    return finalize_result(archetype, instrument, variant, params, trades, trade_dates)


# =============================================================================
# ARCHETYPE 2: Late-Session Reversal
# =============================================================================


def _get_session_bars(
    day_bars: pd.DataFrame,
    block_name: str,
    trading_day,
) -> pd.DataFrame:
    """Filter 1m bars to a session block by UTC hour range."""
    block = SESSION_BLOCKS[block_name]
    start_h = block["start_utc_h"]
    end_h = block["end_utc_h"]

    # Extract UTC hour from timestamp
    hours = day_bars["ts_utc"].dt.hour
    if start_h < end_h:
        mask = (hours >= start_h) & (hours < end_h)
    else:
        # Crosses midnight (e.g., US session 13:00-19:00 doesn't cross,
        # but if it did we'd handle it)
        mask = (hours >= start_h) | (hours < end_h)
    return day_bars[mask]


def run_late_session_reversal(
    daily_df: pd.DataFrame,
    bars_by_day: dict,
    instrument: str,
    block_name: str,
    extension_threshold: float = 1.0,
    entry_offset_min: int = 60,
    spec: CostSpec | None = None,
) -> StrategyResult:
    """Late-Session Reversal: fade extended moves near session close.

    Mechanism: End-of-session profit-taking and position squaring causes
    reversion from intraday extremes.

    Entry: At T-N min before session close, if price is extended > threshold×ATR
    from session midpoint, trade opposite direction.
    Target: Return to session midpoint.
    Stop: Further extension by 0.5×ATR.
    Time stop: Session close.
    """
    variant = f"{block_name}_ext{extension_threshold}_T-{entry_offset_min}"
    params = {"block": block_name, "extension_threshold": extension_threshold, "entry_offset_min": entry_offset_min}
    archetype = "late_session_reversal"

    if spec is None:
        spec = COST_SPECS.get(instrument)
    if spec is None:
        return StrategyResult(archetype=archetype, instrument=instrument, variant=variant, params=params)

    block = SESSION_BLOCKS[block_name]
    session_end_h = block["end_utc_h"]

    trades = []
    trade_dates = []

    for _, row in daily_df.iterrows():
        atr = row.get("atr_20")
        trading_day = row["trading_day"]
        if pd.isna(atr) or atr <= 0:
            continue

        day_bars = bars_by_day.get(trading_day)
        if day_bars is None:
            continue

        session_bars = _get_session_bars(day_bars, block_name, trading_day)
        if len(session_bars) < 60:  # need at least 1 hour of bars
            continue

        # Determine entry time: T-N minutes before session end
        # Session end in UTC hour
        session_end_utc = session_bars["ts_utc"].iloc[-1]  # last bar of session

        entry_time = session_end_utc - pd.Timedelta(minutes=entry_offset_min)

        # Get bars up to entry time (for computing midpoint — no look-ahead)
        pre_entry = session_bars[session_bars["ts_utc"] <= entry_time]
        if len(pre_entry) < 30:
            continue

        # Compute running session stats up to entry time
        session_high = pre_entry["high"].max()
        session_low = pre_entry["low"].min()
        session_mid = (session_high + session_low) / 2.0

        # Current price at entry time
        entry_price = pre_entry.iloc[-1]["close"]

        # Check extension from midpoint
        extension = abs(entry_price - session_mid) / atr
        if extension < extension_threshold:
            continue

        # Determine direction: fade the extension
        if entry_price > session_mid:
            direction = -1  # short: price is above mid, expect reversion down
        else:
            direction = 1  # long: price is below mid, expect reversion up

        # Risk and targets
        stop_distance = 0.5 * atr
        stop_price = entry_price - direction * stop_distance
        target_price = session_mid  # revert to midpoint

        risk_pts = stop_distance
        if risk_pts < spec.friction_in_points * 2:
            continue

        # Simulate on remaining session bars
        post_entry = session_bars[session_bars["ts_utc"] > entry_time]
        if len(post_entry) == 0:
            continue

        pnl_pts, exit_type = simulate_trade(
            post_entry,
            direction,
            entry_price,
            stop_price,
            target_price,
            max_bars=entry_offset_min + 10,  # session end + small buffer
        )

        pnl_r = apply_friction_r(pnl_pts, risk_pts, spec)
        trades.append(pnl_r)
        trade_dates.append(trading_day)

    return finalize_result(archetype, instrument, variant, params, trades, trade_dates)


# =============================================================================
# ARCHETYPE 3: VWAP Reversion
# =============================================================================


def run_vwap_reversion(
    daily_df: pd.DataFrame,
    bars_by_day: dict,
    instrument: str,
    block_name: str,
    sigma_mult: float = 2.0,
    target_type: str = "full",
    spec: CostSpec | None = None,
) -> StrategyResult:
    """VWAP Reversion: trade back toward VWAP when deviation is extreme.

    Mechanism: Institutional orders benchmark to VWAP. Extended deviation
    triggers institutional flow back to VWAP.

    Entry: When price deviates > N×σ from running VWAP, trade toward VWAP.
    Target: Return to VWAP (full) or VWAP ± 0.5σ (partial).
    Stop: Deviation extends by 1σ from entry.
    Time stop: Session close.
    """
    variant = f"{block_name}_sig{sigma_mult}_{target_type}"
    params = {"block": block_name, "sigma_mult": sigma_mult, "target_type": target_type}
    archetype = "vwap_reversion"

    if spec is None:
        spec = COST_SPECS.get(instrument)
    if spec is None:
        return StrategyResult(archetype=archetype, instrument=instrument, variant=variant, params=params)

    min_vwap_bars = 60  # need 1 hour of bars before VWAP signal meaningful

    trades = []
    trade_dates = []

    for _, row in daily_df.iterrows():
        trading_day = row["trading_day"]
        atr = row.get("atr_20")
        if pd.isna(atr) or atr <= 0:
            continue

        day_bars = bars_by_day.get(trading_day)
        if day_bars is None:
            continue

        session_bars = _get_session_bars(day_bars, block_name, trading_day)
        if len(session_bars) < min_vwap_bars + 30:
            continue

        # Filter out zero-volume bars for VWAP computation
        vol_bars = session_bars[session_bars["volume"] > 0].copy()
        if len(vol_bars) < min_vwap_bars:
            continue

        # Compute running VWAP
        cum_pv = (vol_bars["close"].values * vol_bars["volume"].values).cumsum()
        cum_vol = vol_bars["volume"].values.cumsum()
        vwap = cum_pv / cum_vol

        # Compute running deviation and std
        deviation = vol_bars["close"].values - vwap
        # Expanding std with min_periods
        dev_std = pd.Series(deviation).expanding(min_periods=min_vwap_bars).std().values

        # Find first signal bar (only take one trade per session per day)
        entry_found = False
        for j in range(min_vwap_bars, len(vol_bars)):
            if dev_std[j] <= 0 or np.isnan(dev_std[j]):
                continue
            z_score = abs(deviation[j]) / dev_std[j]
            if z_score < sigma_mult:
                continue

            # Signal! Set up trade.
            entry_price = vol_bars.iloc[j]["close"]
            current_vwap = vwap[j]
            current_std = dev_std[j]

            if entry_price > current_vwap:
                direction = -1  # short: price above VWAP, expect reversion down
            else:
                direction = 1  # long: price below VWAP, expect reversion up

            # Stop: deviation extends by 1σ
            stop_price = entry_price - direction * current_std
            risk_pts = current_std

            if risk_pts < spec.friction_in_points * 2 or risk_pts <= 0:
                continue

            # Target
            if target_type == "full":
                target_price = current_vwap
            else:  # partial
                target_price = current_vwap + direction * 0.5 * current_std

            # Remaining bars after entry
            remaining_idx = vol_bars.index[j + 1 :] if j + 1 < len(vol_bars) else pd.Index([])
            if len(remaining_idx) == 0:
                continue

            # Use ALL session bars after entry for simulation (not just vol>0)
            entry_ts = vol_bars.iloc[j]["ts_utc"]
            remaining = session_bars[session_bars["ts_utc"] > entry_ts]
            if len(remaining) == 0:
                continue

            pnl_pts, exit_type = simulate_trade(
                remaining,
                direction,
                entry_price,
                stop_price,
                target_price,
                max_bars=MAX_TRADE_BARS,
            )

            pnl_r = apply_friction_r(pnl_pts, risk_pts, spec)
            trades.append(pnl_r)
            trade_dates.append(trading_day)
            entry_found = True
            break  # one trade per session per day

    return finalize_result(archetype, instrument, variant, params, trades, trade_dates)


# =============================================================================
# Main Runner
# =============================================================================


def run_all(
    db_path: str,
    archetype_filter: Optional[str] = None,
    instrument_filter: Optional[str] = None,
) -> list[StrategyResult]:
    """Run all Phase 2 intraday strategies."""
    results = []
    con = duckdb.connect(db_path, read_only=True)

    instruments = ALL_INSTRUMENTS
    if instrument_filter:
        instruments = [instrument_filter]

    try:
        for instrument in instruments:
            spec = COST_SPECS.get(instrument)
            if spec is None:
                print(f"  No cost model for {instrument}, skipping")
                continue

            print(f"\n{'=' * 70}")
            print(f"  INSTRUMENT: {instrument}")
            print(f"{'=' * 70}")

            # Load data once per instrument
            bars_df = load_instrument_bars(con, instrument)
            if len(bars_df) < 10000:
                print(f"    Insufficient 1m bars ({len(bars_df)}), skipping")
                continue

            daily_df = load_daily_features(con, instrument)
            if len(daily_df) < 200:
                print(f"    Insufficient daily features ({len(daily_df)}), skipping")
                continue

            # Group bars by trading day for O(1) lookup
            bars_by_day = {day: grp for day, grp in bars_df.groupby("trading_day")}

            # =================================================================
            # ARCHETYPE 1: Failed Breakout Fade
            # =================================================================
            if not archetype_filter or archetype_filter == "fade":
                print(f"\n  --- Failed Breakout Fade ({instrument}) ---")
                for label in FADE_ORB_LABELS:
                    for n_bars in [5, 10, 15, 20]:
                        for rr in [0.5, 1.0, 1.5]:
                            r = run_failed_breakout_fade(
                                daily_df,
                                bars_by_day,
                                instrument,
                                label,
                                n_return_bars=n_bars,
                                rr_target=rr,
                                spec=spec,
                            )
                            results.append(r)
                            if r.n_trades >= MIN_TRADES:
                                sig = "*" if r.p_value < 0.05 else " "
                                print(
                                    f"    {sig} {label:18s} N_ret={n_bars:2d} "
                                    f"RR={rr:.1f} N={r.n_trades:4d} "
                                    f"WR={r.win_rate:.2%} ExpR={r.avg_pnl_r:+.4f} "
                                    f"p={r.p_value:.4f}"
                                )

            # =================================================================
            # ARCHETYPE 2: Late-Session Reversal
            # =================================================================
            if not archetype_filter or archetype_filter == "reversal":
                print(f"\n  --- Late-Session Reversal ({instrument}) ---")
                for block in SESSION_BLOCKS:
                    for threshold in [0.5, 0.75, 1.0, 1.5]:
                        for offset in [60, 90]:
                            r = run_late_session_reversal(
                                daily_df,
                                bars_by_day,
                                instrument,
                                block,
                                extension_threshold=threshold,
                                entry_offset_min=offset,
                                spec=spec,
                            )
                            results.append(r)
                            if r.n_trades >= MIN_TRADES:
                                sig = "*" if r.p_value < 0.05 else " "
                                print(
                                    f"    {sig} {block:8s} ext={threshold:.2f} "
                                    f"T-{offset} N={r.n_trades:4d} "
                                    f"WR={r.win_rate:.2%} ExpR={r.avg_pnl_r:+.4f} "
                                    f"p={r.p_value:.4f}"
                                )

            # =================================================================
            # ARCHETYPE 3: VWAP Reversion
            # =================================================================
            if not archetype_filter or archetype_filter == "vwap":
                print(f"\n  --- VWAP Reversion ({instrument}) ---")
                for block in SESSION_BLOCKS:
                    for sigma in [1.5, 2.0, 2.5, 3.0]:
                        for ttype in ["full", "partial"]:
                            r = run_vwap_reversion(
                                daily_df,
                                bars_by_day,
                                instrument,
                                block,
                                sigma_mult=sigma,
                                target_type=ttype,
                                spec=spec,
                            )
                            results.append(r)
                            if r.n_trades >= MIN_TRADES:
                                sig = "*" if r.p_value < 0.05 else " "
                                print(
                                    f"    {sig} {block:8s} sig={sigma:.1f} "
                                    f"{ttype:7s} N={r.n_trades:4d} "
                                    f"WR={r.win_rate:.2%} ExpR={r.avg_pnl_r:+.4f} "
                                    f"p={r.p_value:.4f}"
                                )

            # Free memory
            del bars_df, bars_by_day, daily_df

        # =====================================================================
        # FDR CORRECTION
        # =====================================================================
        print("\n" + "=" * 70)
        print("BH FDR CORRECTION (Phase 2 — Intraday)")
        print("=" * 70)

        valid_results = [r for r in results if r.n_trades >= MIN_TRADES]
        fdr_results = []
        if valid_results:
            p_values = [r.p_value for r in valid_results]
            fdr_results = bh_fdr(p_values, alpha=FDR_ALPHA)

            n_tested = len(valid_results)
            n_sig = sum(1 for _, _, sig in fdr_results if sig)
            print(f"\nTests with N >= {MIN_TRADES}: {n_tested}")
            print(f"FDR-significant at alpha={FDR_ALPHA}: {n_sig}")

            if n_sig > 0:
                print(
                    f"\n{'Archetype':<25s} {'Instrument':<8s} {'Variant':<35s} "
                    f"{'N':>5s} {'WR':>7s} {'ExpR':>8s} {'Sharpe':>7s} "
                    f"{'raw_p':>8s} {'adj_p':>8s}"
                )
                print("-" * 120)
                for idx, adj_p, sig in fdr_results:
                    if sig:
                        r = valid_results[idx]
                        print(
                            f"{r.archetype:<25s} {r.instrument:<8s} "
                            f"{r.variant:<35s} "
                            f"{r.n_trades:5d} {r.win_rate:7.2%} "
                            f"{r.avg_pnl_r:+8.4f} {r.sharpe:7.3f} "
                            f"{r.p_value:8.4f} {adj_p:8.4f}"
                        )
            else:
                print("\nNo strategies survived FDR correction.")
        else:
            print(f"\nNo strategies had N >= {MIN_TRADES}")

        # =====================================================================
        # ORB CORRELATION
        # =====================================================================
        print("\n" + "=" * 70)
        print("ORB CORRELATION ANALYSIS")
        print("=" * 70)

        for r in valid_results:
            if r.p_value < 0.05:
                corr = compute_orb_correlation(con, r, r.instrument)
                if corr is not None:
                    label = "UNCORRELATED" if abs(corr) < 0.3 else "CORRELATED"
                    print(f"  {r.archetype:25s} {r.instrument:8s} {r.variant:35s} corr={corr:+.3f} [{label}]")

        # =====================================================================
        # HONEST SUMMARY
        # =====================================================================
        print("\n" + "=" * 70)
        print("HONEST SUMMARY (per RESEARCH_RULES.md)")
        print("=" * 70)

        survived = []
        if fdr_results:
            survived = [valid_results[i] for i, (_, _, sig) in enumerate(fdr_results) if sig]
        failed = [r for r in valid_results if r not in survived]

        print("\nSURVIVED SCRUTINY:")
        if survived:
            for r in survived:
                print(
                    f"  - {r.archetype} | {r.instrument} | {r.variant} | "
                    f"N={r.n_trades}, ExpR={r.avg_pnl_r:+.4f}, p={r.p_value:.4f}"
                )
        else:
            print("  None.")

        print(f"\nDID NOT SURVIVE: {len(failed)} strategies tested, none FDR-significant")

        # Per-archetype summary
        for arch in ["failed_breakout_fade", "late_session_reversal", "vwap_reversion"]:
            arch_results = [r for r in valid_results if r.archetype == arch]
            if arch_results:
                best = min(arch_results, key=lambda r: r.p_value)
                avg_exp = np.mean([r.avg_pnl_r for r in arch_results])
                print(f"\n  {arch} ({len(arch_results)} valid tests):")
                print(f"    Average ExpR across all: {avg_exp:+.4f}")
                print(
                    f"    Best: {best.instrument} {best.variant} "
                    f"N={best.n_trades} ExpR={best.avg_pnl_r:+.4f} "
                    f"p={best.p_value:.4f}"
                )

        print("\nCAVEATS:")
        print("  - Session block times are fixed UTC (DST may shift by ±1hr)")
        print("  - Failed Breakout Fade minimum risk filter may skip tight ORBs")
        print("  - VWAP requires volume data; thin sessions may have noisy VWAP")
        print("  - Gold (MGC) trends intraday — mean reversion archetypes may underperform")
        print("  - MNQ/SIL have only 2 years of data — insufficient regime diversity")
        print("  - All results are IN-SAMPLE. Walk-forward needed before 'validated'.")

        print("\nNEXT STEPS:")
        print("  - Combine Phase 1 + Phase 2 results for global FDR")
        print("  - For any FDR survivors: sensitivity analysis (±20% on parameters)")
        print("  - Compute full correlation matrix with ORB portfolio daily PnL")

    finally:
        con.close()

    return results


# =============================================================================
# Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Non-ORB Strategy Research — Phase 2: Intraday")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "gold.db")
    parser.add_argument(
        "--archetype", type=str, default=None, choices=["fade", "reversal", "vwap"], help="Run only one archetype"
    )
    parser.add_argument("--instrument", type=str, default=None, choices=ALL_INSTRUMENTS, help="Run only one instrument")
    args = parser.parse_args()

    print("Non-ORB Strategy Research — Phase 2: Intraday Strategies")
    print(f"Database: {args.db_path}")
    print(f"Archetype filter: {args.archetype or 'ALL'}")
    print(f"Instrument filter: {args.instrument or 'ALL'}")
    print(f"Date: {date.today()}")
    print()

    t0 = time.time()
    results = run_all(
        str(args.db_path),
        archetype_filter=args.archetype,
        instrument_filter=args.instrument,
    )
    elapsed = time.time() - t0

    total = len(results)
    valid = sum(1 for r in results if r.n_trades >= MIN_TRADES)
    print(f"\nTotal tests: {total}, Valid (N>={MIN_TRADES}): {valid}")
    print(f"Runtime: {elapsed:.0f}s ({elapsed / 60:.1f}min)")


if __name__ == "__main__":
    main()
