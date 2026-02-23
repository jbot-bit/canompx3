#!/usr/bin/env python3
"""Non-ORB Strategy Research — Phase 1: Daily-Level Strategies.

Tests 3 strategy archetypes that use daily_features data only:
  1. Multi-Day Momentum (trend following, multi-day holds)
  2. Vol Expansion Fade (mean reversion after ATR explosion)
  3. Cross-Instrument Vol Signal (leader vol → follower breakout)

Usage:
    python research/research_non_orb_daily.py --db-path gold.db
    python research/research_non_orb_daily.py --db-path gold.db --archetype momentum
    python research/research_non_orb_daily.py --db-path gold.db --instrument MGC
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date
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

# Minimum data requirements
MIN_TRADING_DAYS = 200   # need at least 200 days of daily data
MIN_TRADES = 30          # REGIME minimum per RESEARCH_RULES.md

# BH FDR significance threshold
FDR_ALPHA = 0.05


@dataclass
class StrategyResult:
    """Result of a single strategy backtest."""
    archetype: str
    instrument: str
    variant: str         # human-readable description of the specific config
    params: dict         # full parameter dict
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    avg_pnl_r: float = 0.0       # mean PnL in R-multiples
    total_pnl_r: float = 0.0
    sharpe: float = 0.0           # PnL_R Sharpe (not annualized)
    max_dd_r: float = 0.0
    p_value: float = 1.0          # t-test H0: mean PnL_R = 0
    yearly_results: dict = field(default_factory=dict)
    pnl_series: list = field(default_factory=list)  # daily PnL_R for correlation
    trade_dates: list = field(default_factory=list)

    @property
    def classification(self) -> str:
        if self.n_trades < 30:
            return "INVALID"
        elif self.n_trades < 100:
            return "REGIME"
        elif self.n_trades < 200:
            return "PRELIMINARY"
        else:
            return "CORE"


def compute_max_dd(pnl_series: list[float]) -> float:
    """Compute maximum drawdown from a PnL series in R-multiples."""
    if not pnl_series:
        return 0.0
    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return round(float(np.max(dd)), 4) if len(dd) > 0 else 0.0


def yearly_breakdown(dates: list, pnls: list[float]) -> dict:
    """Compute year-by-year PnL stats."""
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
    friction_pts = spec.friction_in_points
    # Risk in R-multiples: risk_points is the stop distance.
    # PnL in R = (pnl_points - friction_pts) / risk_points
    if risk_points <= 0:
        return 0.0
    return (pnl_points - friction_pts) / risk_points


# =============================================================================
# ARCHETYPE 1: Multi-Day Momentum
# =============================================================================

def run_multiday_momentum(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    lookback: int = 5,
    hold_days: int = 2,
    entry_quantile: float = 0.2,
) -> StrategyResult:
    """Multi-day momentum: buy after strong up-moves, sell after strong down-moves.

    Mechanism: Macro trends persist over multiple days (Moskowitz et al., 2012).
    Entry: If N-day return is in top/bottom quintile, enter at next day's open.
    Exit: Fixed hold period.
    """
    variant = f"lookback={lookback}_hold={hold_days}_q={entry_quantile}"

    # Get daily OHLC data
    df = con.execute("""
        SELECT trading_day, daily_open, daily_close, atr_20
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
          AND daily_open IS NOT NULL
          AND daily_close IS NOT NULL
          AND atr_20 > 0
        ORDER BY trading_day
    """, [instrument]).fetchdf()

    if len(df) < MIN_TRADING_DAYS:
        return StrategyResult(
            archetype="multi_day_momentum", instrument=instrument,
            variant=variant, params={"lookback": lookback, "hold_days": hold_days,
                                      "entry_quantile": entry_quantile}
        )

    spec = COST_SPECS.get(instrument)
    if spec is None:
        return StrategyResult(
            archetype="multi_day_momentum", instrument=instrument,
            variant=variant, params={"lookback": lookback, "hold_days": hold_days,
                                      "entry_quantile": entry_quantile}
        )

    df = df.sort_values("trading_day").reset_index(drop=True)

    # Compute N-day returns
    df["return_n"] = df["daily_close"].pct_change(lookback)

    # Compute rolling quantile thresholds (expanding window to avoid lookahead)
    # Use expanding window with min_periods = lookback * 5 for stability
    min_obs = max(lookback * 5, 60)
    df["q_high"] = df["return_n"].expanding(min_periods=min_obs).quantile(1 - entry_quantile)
    df["q_low"] = df["return_n"].expanding(min_periods=min_obs).quantile(entry_quantile)

    trades = []
    trade_dates = []
    i = min_obs + lookback  # start after enough data for quantile estimation

    while i < len(df) - hold_days:
        ret = df.iloc[i]["return_n"]
        q_hi = df.iloc[i]["q_high"]
        q_lo = df.iloc[i]["q_low"]
        atr = df.iloc[i]["atr_20"]

        if pd.isna(ret) or pd.isna(q_hi) or pd.isna(q_lo) or pd.isna(atr) or atr <= 0:
            i += 1
            continue

        direction = 0
        if ret >= q_hi:
            direction = 1   # long (momentum continuation)
        elif ret <= q_lo:
            direction = -1  # short (momentum continuation)

        if direction == 0:
            i += 1
            continue

        # Entry at next day's open
        entry_idx = i + 1
        exit_idx = min(i + 1 + hold_days, len(df) - 1)

        entry_price = df.iloc[entry_idx]["daily_open"]
        exit_price = df.iloc[exit_idx]["daily_close"]
        entry_date = df.iloc[entry_idx]["trading_day"]

        if pd.isna(entry_price) or pd.isna(exit_price):
            i += 1
            continue

        # PnL in points
        pnl_pts = (exit_price - entry_price) * direction
        # Risk = 1.5 * ATR (stop distance)
        risk_pts = 1.5 * atr

        # Convert to R-multiple with friction
        pnl_r = apply_friction_r(pnl_pts, risk_pts, spec)

        trades.append(pnl_r)
        trade_dates.append(entry_date)

        # Skip hold period (no overlapping trades)
        i = exit_idx + 1

    n = len(trades)
    if n < 1:
        return StrategyResult(
            archetype="multi_day_momentum", instrument=instrument,
            variant=variant, params={"lookback": lookback, "hold_days": hold_days,
                                      "entry_quantile": entry_quantile}
        )

    arr = np.array(trades)
    wins = int(np.sum(arr > 0))
    avg_r = float(np.mean(arr))
    total_r = float(np.sum(arr))
    std_r = float(np.std(arr, ddof=1)) if n > 1 else 1.0
    sharpe = avg_r / std_r if std_r > 0 else 0.0

    # t-test: H0 mean = 0
    if n >= 2 and std_r > 0:
        t_stat, p_val = stats.ttest_1samp(arr, 0.0)
        # One-sided p-value (we want positive expectancy)
        p_val = float(p_val / 2) if t_stat > 0 else 1.0 - float(p_val / 2)
    else:
        p_val = 1.0

    return StrategyResult(
        archetype="multi_day_momentum", instrument=instrument,
        variant=variant,
        params={"lookback": lookback, "hold_days": hold_days,
                "entry_quantile": entry_quantile},
        n_trades=n, n_wins=wins, win_rate=round(wins / n, 4),
        avg_pnl_r=round(avg_r, 4), total_pnl_r=round(total_r, 2),
        sharpe=round(sharpe, 4), max_dd_r=compute_max_dd(trades),
        p_value=round(p_val, 6),
        yearly_results=yearly_breakdown(trade_dates, trades),
        pnl_series=trades, trade_dates=trade_dates,
    )


# =============================================================================
# ARCHETYPE 2: Vol Expansion Fade
# =============================================================================

def run_vol_expansion_fade(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    atr_ratio_threshold: float = 1.3,
    fade_mode: str = "fade_prior",   # "fade_prior" or "fade_toward_mean"
) -> StrategyResult:
    """Vol expansion fade: after ATR explodes, mean-revert next day.

    Mechanism: Volatility clusters but also mean-reverts. After a high-vol day
    (ATR5/ATR20 > threshold), the next day tends to have smaller range and
    price tends to revert toward the prior day's midpoint.

    Entry: Day after vol expansion, trade opposite to prior day's direction.
    Stop: 1.0 * ATR(20). Target: 0.5 * ATR(20) mean reversion.
    Exit: EOD (next close).
    """
    variant = f"atr_ratio>{atr_ratio_threshold}_{fade_mode}"

    df = con.execute("""
        SELECT trading_day, daily_open, daily_high, daily_low, daily_close,
               atr_20, prev_day_direction, prev_day_close, prev_day_high, prev_day_low
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
          AND daily_open IS NOT NULL AND daily_close IS NOT NULL
          AND atr_20 > 0
        ORDER BY trading_day
    """, [instrument]).fetchdf()

    if len(df) < MIN_TRADING_DAYS:
        return StrategyResult(
            archetype="vol_expansion_fade", instrument=instrument,
            variant=variant, params={"atr_ratio_threshold": atr_ratio_threshold,
                                     "fade_mode": fade_mode}
        )

    spec = COST_SPECS.get(instrument)
    if spec is None:
        return StrategyResult(
            archetype="vol_expansion_fade", instrument=instrument,
            variant=variant, params={"atr_ratio_threshold": atr_ratio_threshold,
                                     "fade_mode": fade_mode}
        )

    df = df.sort_values("trading_day").reset_index(drop=True)

    # Compute ATR(5) as rolling 5-day average of daily range
    df["daily_range"] = df["daily_high"] - df["daily_low"]
    df["atr_5"] = df["daily_range"].rolling(5, min_periods=5).mean()
    df["atr_ratio"] = df["atr_5"] / df["atr_20"]

    trades = []
    trade_dates = []

    for i in range(6, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        atr_ratio = row["atr_ratio"]
        atr_20 = row["atr_20"]
        prev_dir = row["prev_day_direction"]

        if pd.isna(atr_ratio) or pd.isna(atr_20) or atr_20 <= 0:
            continue
        if atr_ratio < atr_ratio_threshold:
            continue

        # Determine fade direction
        if fade_mode == "fade_prior":
            if prev_dir == "UP":
                direction = -1  # fade the prior up day
            elif prev_dir == "DOWN":
                direction = 1   # fade the prior down day
            else:
                continue
        else:
            # fade toward the mean of prior day's range
            prior_mid = (row["prev_day_high"] + row["prev_day_low"]) / 2
            if pd.isna(prior_mid) or pd.isna(next_row["daily_open"]):
                continue
            if next_row["daily_open"] > prior_mid:
                direction = -1
            else:
                direction = 1

        entry_price = next_row["daily_open"]
        exit_price = next_row["daily_close"]
        entry_date = next_row["trading_day"]

        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        pnl_pts = (exit_price - entry_price) * direction
        risk_pts = 1.0 * atr_20  # stop at 1 ATR

        pnl_r = apply_friction_r(pnl_pts, risk_pts, spec)
        trades.append(pnl_r)
        trade_dates.append(entry_date)

    n = len(trades)
    if n < 1:
        return StrategyResult(
            archetype="vol_expansion_fade", instrument=instrument,
            variant=variant, params={"atr_ratio_threshold": atr_ratio_threshold,
                                     "fade_mode": fade_mode}
        )

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
        archetype="vol_expansion_fade", instrument=instrument,
        variant=variant,
        params={"atr_ratio_threshold": atr_ratio_threshold, "fade_mode": fade_mode},
        n_trades=n, n_wins=wins, win_rate=round(wins / n, 4),
        avg_pnl_r=round(avg_r, 4), total_pnl_r=round(total_r, 2),
        sharpe=round(sharpe, 4), max_dd_r=compute_max_dd(trades),
        p_value=round(p_val, 6),
        yearly_results=yearly_breakdown(trade_dates, trades),
        pnl_series=trades, trade_dates=trade_dates,
    )


# =============================================================================
# ARCHETYPE 3: Cross-Instrument Vol Signal
# =============================================================================

def run_cross_instrument_vol(
    con: duckdb.DuckDBPyConnection,
    leader: str,
    follower: str,
    vol_threshold: float = 1.5,
    rr_target: float = 1.5,
) -> StrategyResult:
    """Cross-instrument vol signal: leader's high vol → follower breakout.

    Mechanism: Volatility transmits across asset classes with time lag.
    If leader had high overnight_range (> threshold * ATR), trade follower's
    next session with wider targets (expecting more energy).

    This is an ORB enhancer, not a standalone edge. We test it as:
    "On days when leader vol is high, is follower's daily return more positive?"
    Direction: trade WITH the leader's direction (momentum transmission).
    """
    variant = f"{leader}->{follower}_vol>{vol_threshold}_rr{rr_target}"

    # Get leader and follower daily data aligned by trading_day
    df = con.execute("""
        SELECT l.trading_day,
               l.daily_close - l.daily_open AS leader_return,
               l.daily_high - l.daily_low AS leader_range,
               l.atr_20 AS leader_atr,
               l.overnight_range AS leader_overnight_range,
               f.daily_open AS follower_open,
               f.daily_close AS follower_close,
               f.daily_high AS follower_high,
               f.daily_low AS follower_low,
               f.atr_20 AS follower_atr
        FROM daily_features l
        JOIN daily_features f
          ON l.trading_day = f.trading_day
          AND l.orb_minutes = f.orb_minutes
        WHERE l.symbol = ? AND f.symbol = ?
          AND l.orb_minutes = 5
          AND l.atr_20 > 0 AND f.atr_20 > 0
          AND f.daily_open IS NOT NULL AND f.daily_close IS NOT NULL
        ORDER BY l.trading_day
    """, [leader, follower]).fetchdf()

    if len(df) < MIN_TRADING_DAYS:
        return StrategyResult(
            archetype="cross_instrument_vol", instrument=f"{leader}->{follower}",
            variant=variant, params={"leader": leader, "follower": follower,
                                     "vol_threshold": vol_threshold, "rr_target": rr_target}
        )

    spec = COST_SPECS.get(follower)
    if spec is None:
        return StrategyResult(
            archetype="cross_instrument_vol", instrument=f"{leader}->{follower}",
            variant=variant, params={"leader": leader, "follower": follower,
                                     "vol_threshold": vol_threshold, "rr_target": rr_target}
        )

    trades = []
    trade_dates = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        leader_range = prev["leader_range"]
        leader_atr = prev["leader_atr"]
        leader_ret = prev["leader_return"]
        follower_atr = row["follower_atr"]

        if pd.isna(leader_range) or pd.isna(leader_atr) or leader_atr <= 0:
            continue
        if pd.isna(follower_atr) or follower_atr <= 0:
            continue

        # Check if leader had high vol yesterday
        vol_ratio = leader_range / leader_atr
        if vol_ratio < vol_threshold:
            continue

        # Direction: follow leader's prior day direction
        if pd.isna(leader_ret):
            continue
        direction = 1 if leader_ret > 0 else -1

        entry_price = row["follower_open"]
        exit_price = row["follower_close"]
        entry_date = row["trading_day"]

        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        pnl_pts = (exit_price - entry_price) * direction
        risk_pts = follower_atr  # 1 ATR stop

        pnl_r = apply_friction_r(pnl_pts, risk_pts, spec)
        trades.append(pnl_r)
        trade_dates.append(entry_date)

    n = len(trades)
    if n < 1:
        return StrategyResult(
            archetype="cross_instrument_vol", instrument=f"{leader}->{follower}",
            variant=variant, params={"leader": leader, "follower": follower,
                                     "vol_threshold": vol_threshold, "rr_target": rr_target}
        )

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
        archetype="cross_instrument_vol", instrument=f"{leader}->{follower}",
        variant=variant,
        params={"leader": leader, "follower": follower,
                "vol_threshold": vol_threshold, "rr_target": rr_target},
        n_trades=n, n_wins=wins, win_rate=round(wins / n, 4),
        avg_pnl_r=round(avg_r, 4), total_pnl_r=round(total_r, 2),
        sharpe=round(sharpe, 4), max_dd_r=compute_max_dd(trades),
        p_value=round(p_val, 6),
        yearly_results=yearly_breakdown(trade_dates, trades),
        pnl_series=trades, trade_dates=trade_dates,
    )


# =============================================================================
# BH FDR Correction
# =============================================================================

def bh_fdr(p_values: list[float], alpha: float = 0.05) -> list[tuple[int, float, bool]]:
    """Benjamini-Hochberg FDR correction.

    Returns: list of (original_index, adjusted_p, is_significant)
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n

    # Step through sorted p-values
    prev_adj = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        adj_p = min(prev_adj, p * n / rank)
        adj_p = min(adj_p, 1.0)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p

    return [(i, adjusted[i], adjusted[i] < alpha) for i in range(n)]


# =============================================================================
# ORB Correlation Analysis
# =============================================================================

def compute_orb_correlation(
    con: duckdb.DuckDBPyConnection,
    result: StrategyResult,
    instrument: str,
) -> Optional[float]:
    """Compute correlation between strategy daily PnL and representative ORB PnL."""
    if not result.trade_dates or not result.pnl_series:
        return None

    # Get a representative ORB strategy's daily PnL for this instrument
    # Use 1000 session E0 CB1 RR1.0 NO_FILTER as the baseline ORB
    orb_df = con.execute("""
        SELECT o.trading_day, o.pnl_r
        FROM orb_outcomes o
        WHERE o.symbol = ? AND o.orb_label = '1000'
          AND o.entry_model = 'E0' AND o.confirm_bars = 1
          AND o.rr_target = 1.0 AND o.orb_minutes = 5
          AND o.outcome IN ('WIN', 'LOSS')
        ORDER BY o.trading_day
    """, [instrument]).fetchdf()

    if orb_df.empty:
        # Try 0900 for MGC
        orb_df = con.execute("""
            SELECT o.trading_day, o.pnl_r
            FROM orb_outcomes o
            WHERE o.symbol = ? AND o.orb_label = '0900'
              AND o.entry_model = 'E1' AND o.confirm_bars = 1
              AND o.rr_target = 1.0 AND o.orb_minutes = 5
              AND o.outcome IN ('WIN', 'LOSS')
            ORDER BY o.trading_day
        """, [instrument]).fetchdf()

    if orb_df.empty:
        return None

    # Align by date
    strat_df = pd.DataFrame({
        "trading_day": result.trade_dates,
        "strat_pnl": result.pnl_series,
    })
    strat_df["trading_day"] = pd.to_datetime(strat_df["trading_day"])
    orb_df["trading_day"] = pd.to_datetime(orb_df["trading_day"])

    merged = strat_df.merge(orb_df, on="trading_day", how="inner")
    if len(merged) < 20:
        return None

    corr = merged["strat_pnl"].corr(merged["pnl_r"])
    return round(float(corr), 4) if not pd.isna(corr) else None


# =============================================================================
# Main Runner
# =============================================================================

def run_all(db_path: str, archetype_filter: Optional[str] = None,
            instrument_filter: Optional[str] = None) -> list[StrategyResult]:
    """Run all Phase 1 strategies."""
    results = []
    con = duckdb.connect(db_path, read_only=True)

    instruments = ALL_INSTRUMENTS
    if instrument_filter:
        instruments = [instrument_filter]

    try:
        # =====================================================================
        # ARCHETYPE 1: Multi-Day Momentum
        # =====================================================================
        if not archetype_filter or archetype_filter == "momentum":
            print("=" * 70)
            print("ARCHETYPE 1: Multi-Day Momentum")
            print("=" * 70)
            for inst in instruments:
                for lookback in [3, 5, 10, 20]:
                    for hold in [1, 2, 3, 5]:
                        r = run_multiday_momentum(
                            con, inst, lookback=lookback, hold_days=hold,
                            entry_quantile=0.2,
                        )
                        results.append(r)
                        if r.n_trades >= MIN_TRADES:
                            sig = "*" if r.p_value < 0.05 else " "
                            print(f"  {sig} {inst:4s} L={lookback:2d} H={hold} "
                                  f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                                  f"ExpR={r.avg_pnl_r:+.4f} Sharpe={r.sharpe:.3f} "
                                  f"p={r.p_value:.4f}")

        # =====================================================================
        # ARCHETYPE 2: Vol Expansion Fade
        # =====================================================================
        if not archetype_filter or archetype_filter == "vol_fade":
            print("\n" + "=" * 70)
            print("ARCHETYPE 2: Vol Expansion Fade")
            print("=" * 70)
            for inst in instruments:
                for threshold in [1.2, 1.3, 1.5, 1.8]:
                    for mode in ["fade_prior", "fade_toward_mean"]:
                        r = run_vol_expansion_fade(
                            con, inst, atr_ratio_threshold=threshold,
                            fade_mode=mode,
                        )
                        results.append(r)
                        if r.n_trades >= MIN_TRADES:
                            sig = "*" if r.p_value < 0.05 else " "
                            print(f"  {sig} {inst:4s} ATR>{threshold:.1f} {mode:20s} "
                                  f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                                  f"ExpR={r.avg_pnl_r:+.4f} Sharpe={r.sharpe:.3f} "
                                  f"p={r.p_value:.4f}")

        # =====================================================================
        # ARCHETYPE 3: Cross-Instrument Vol Signal
        # =====================================================================
        if not archetype_filter or archetype_filter == "cross_vol":
            print("\n" + "=" * 70)
            print("ARCHETYPE 3: Cross-Instrument Vol Signal")
            print("=" * 70)
            # Define leader-follower pairs
            pairs = [
                ("MGC", "MES"), ("MGC", "MNQ"), ("MGC", "M2K"),
                ("M6E", "MGC"), ("M6E", "MES"),
                ("MCL", "MES"), ("MCL", "MGC"),
                ("MES", "MGC"), ("MNQ", "MGC"),
                ("MGC", "MCL"), ("MGC", "SIL"),
            ]
            if instrument_filter:
                pairs = [(l, f) for l, f in pairs
                         if l == instrument_filter or f == instrument_filter]

            for leader, follower in pairs:
                for vol_thresh in [1.0, 1.5, 2.0]:
                    r = run_cross_instrument_vol(
                        con, leader, follower, vol_threshold=vol_thresh,
                    )
                    results.append(r)
                    if r.n_trades >= MIN_TRADES:
                        sig = "*" if r.p_value < 0.05 else " "
                        print(f"  {sig} {leader:4s}->{follower:4s} vol>{vol_thresh:.1f} "
                              f"N={r.n_trades:4d} WR={r.win_rate:.2%} "
                              f"ExpR={r.avg_pnl_r:+.4f} Sharpe={r.sharpe:.3f} "
                              f"p={r.p_value:.4f}")

        # =====================================================================
        # FDR CORRECTION
        # =====================================================================
        print("\n" + "=" * 70)
        print("BH FDR CORRECTION")
        print("=" * 70)

        valid_results = [r for r in results if r.n_trades >= MIN_TRADES]
        if valid_results:
            p_values = [r.p_value for r in valid_results]
            fdr_results = bh_fdr(p_values, alpha=FDR_ALPHA)

            n_tested = len(valid_results)
            n_sig = sum(1 for _, _, sig in fdr_results if sig)
            print(f"\nTests with N >= {MIN_TRADES}: {n_tested}")
            print(f"FDR-significant at alpha={FDR_ALPHA}: {n_sig}")

            if n_sig > 0:
                print(f"\n{'Archetype':<25s} {'Instrument':<15s} {'Variant':<40s} "
                      f"{'N':>5s} {'WR':>7s} {'ExpR':>8s} {'Sharpe':>7s} "
                      f"{'raw_p':>8s} {'adj_p':>8s}")
                print("-" * 130)
                for idx, adj_p, sig in fdr_results:
                    if sig:
                        r = valid_results[idx]
                        print(f"{r.archetype:<25s} {r.instrument:<15s} "
                              f"{r.variant:<40s} "
                              f"{r.n_trades:5d} {r.win_rate:7.2%} "
                              f"{r.avg_pnl_r:+8.4f} {r.sharpe:7.3f} "
                              f"{r.p_value:8.4f} {adj_p:8.4f}")
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
            if r.p_value < 0.05:  # only correlate promising ones
                # Extract base instrument for correlation
                inst = r.instrument.split("->")[-1] if "->" in r.instrument else r.instrument
                corr = compute_orb_correlation(con, r, inst)
                if corr is not None:
                    label = "UNCORRELATED" if abs(corr) < 0.3 else "CORRELATED"
                    print(f"  {r.archetype:25s} {r.instrument:15s} "
                          f"{r.variant:40s} corr={corr:+.3f} [{label}]")

        # =====================================================================
        # HONEST SUMMARY
        # =====================================================================
        print("\n" + "=" * 70)
        print("HONEST SUMMARY (per RESEARCH_RULES.md)")
        print("=" * 70)

        survived = [r for i, (_, adj_p, sig) in enumerate(fdr_results)
                    if sig for r in [valid_results[i]]] if valid_results else []
        failed = [r for r in valid_results
                  if r not in survived and r.n_trades >= MIN_TRADES]

        print("\nSURVIVED SCRUTINY:")
        if survived:
            for r in survived:
                print(f"  - {r.archetype} | {r.instrument} | {r.variant} | "
                      f"N={r.n_trades}, ExpR={r.avg_pnl_r:+.4f}, p={r.p_value:.4f}")
        else:
            print("  None.")

        print(f"\nDID NOT SURVIVE: {len(failed)} strategies tested but not FDR-significant")

        print("\nCAVEATS:")
        print("  - Multi-day momentum requires overnight holding (prop-firm incompatible)")
        print("  - Vol expansion fade has small N for high thresholds")
        print("  - Cross-instrument pairs limited by overlapping data coverage")
        print("  - MNQ and SIL have only 2 years — insufficient regime diversity")
        print("  - All results are IN-SAMPLE. Walk-forward needed before 'validated'.")

        print("\nNEXT STEPS:")
        print("  - For FDR survivors: run sensitivity analysis (±20% on parameters)")
        print("  - Run Phase 2 (intraday strategies): Failed Breakout Fade, "
              "Late-Session Reversal, VWAP Reversion")
        print("  - Compute full correlation matrix with ORB portfolio daily PnL")

    finally:
        con.close()

    return results


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Non-ORB Strategy Research — Phase 1")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "gold.db")
    parser.add_argument("--archetype", type=str, default=None,
                        choices=["momentum", "vol_fade", "cross_vol"],
                        help="Run only one archetype")
    parser.add_argument("--instrument", type=str, default=None,
                        choices=ALL_INSTRUMENTS,
                        help="Run only one instrument")
    args = parser.parse_args()

    print("Non-ORB Strategy Research — Phase 1: Daily-Level Strategies")
    print(f"Database: {args.db_path}")
    print(f"Archetype filter: {args.archetype or 'ALL'}")
    print(f"Instrument filter: {args.instrument or 'ALL'}")
    print(f"Date: {date.today()}")
    print()

    results = run_all(
        str(args.db_path),
        archetype_filter=args.archetype,
        instrument_filter=args.instrument,
    )

    total = len(results)
    valid = sum(1 for r in results if r.n_trades >= MIN_TRADES)
    print(f"\nTotal tests: {total}, Valid (N>={MIN_TRADES}): {valid}")


if __name__ == "__main__":
    main()
