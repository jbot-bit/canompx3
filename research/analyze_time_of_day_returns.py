#!/usr/bin/env python3
"""
Time-of-Day Returns analysis for Gold (MGC).

Question: Which hours have systematic directional bias in gold?

For each hour (0-23 UTC):
  - Compute mean/median hourly return and stddev
  - Group by session (Asia 23:00-06:00 UTC, London 07:00-13:00, NY 13:00-21:00)
  - Break into terciles by ATR regime (low/med/high vol)
  - Use 5m bars for efficiency
  - Hourly return = close of last 5m bar in hour vs close of first 5m bar in hour
  - Test: if you buy at hour X open and sell at hour X close, what's the ExpR?

Read-only: does NOT write to the database.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import get_cost_spec
from scripts._alt_strategy_utils import compute_strategy_metrics

DB_PATH = Path(r"C:\db\gold.db")
SPEC = get_cost_spec("MGC")

# Screen on recent data first; expand to full dataset only if signal found
SCREEN_START = "2024-01-01"

# Session definitions (UTC hours)
def get_session(hour_utc: int) -> str:
    """Classify UTC hour into session."""
    if hour_utc >= 23 or hour_utc < 7:
        return "Asia"
    elif 7 <= hour_utc < 13:
        return "London"
    elif 13 <= hour_utc < 21:
        return "NY"
    else:
        return "Off-hours"


def load_5m_bars(db_path: Path) -> pd.DataFrame:
    """Load 5m bars for MGC (screened date range)."""
    print("Loading 5m bars...", end=" ", flush=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_5m
            WHERE symbol = 'MGC'
              AND ts_utc >= ?::TIMESTAMPTZ
            ORDER BY ts_utc
        """, [f"{SCREEN_START} 00:00:00+00"]).fetchdf()
    finally:
        con.close()
    print(f"{len(df):,} bars loaded.")
    return df


def load_atr(db_path: Path) -> pd.DataFrame:
    """Load ATR_20 from daily_features."""
    print("Loading daily ATR...", end=" ", flush=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT trading_day, atr_20
            FROM daily_features
            WHERE symbol = 'MGC'
              AND orb_minutes = 5
              AND atr_20 IS NOT NULL
              AND trading_day >= ?
            ORDER BY trading_day
        """, [SCREEN_START]).fetchdf()
    finally:
        con.close()
    print(f"{len(df):,} days with ATR.")
    return df


def compute_hourly_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly returns from 5m bars.

    For each hour: return = last bar close - first bar open in that hour.
    """
    print("Computing hourly returns...")

    # Ensure UTC
    if bars["ts_utc"].dt.tz is not None:
        bars["ts_utc"] = bars["ts_utc"].dt.tz_convert("UTC")
    else:
        bars["ts_utc"] = bars["ts_utc"].dt.tz_localize("UTC")

    bars["hour_utc"] = bars["ts_utc"].dt.hour
    bars["date"] = bars["ts_utc"].dt.date

    # Group by (date, hour) - get first open and last close
    hourly = bars.groupby(["date", "hour_utc"]).agg(
        open_price=("open", "first"),
        close_price=("close", "last"),
        high=("high", "max"),
        low=("low", "min"),
        n_bars=("close", "count"),
    ).reset_index()

    # Only keep hours with at least 6 bars (30 min of data minimum)
    hourly = hourly[hourly["n_bars"] >= 6].copy()

    hourly["return_pts"] = hourly["close_price"] - hourly["open_price"]
    hourly["range_pts"] = hourly["high"] - hourly["low"]

    print(f"  {len(hourly):,} hourly observations across {hourly['date'].nunique():,} days.")
    return hourly


def merge_atr_regimes(hourly: pd.DataFrame, atr_df: pd.DataFrame) -> pd.DataFrame:
    """Merge ATR and assign tercile regimes."""
    atr_df = atr_df.copy()
    atr_df["date"] = pd.to_datetime(atr_df["trading_day"]).dt.date

    merged = hourly.merge(atr_df[["date", "atr_20"]], on="date", how="inner")

    # Terciles
    tercile_edges = merged["atr_20"].quantile([0.333, 0.667]).values
    conditions = [
        merged["atr_20"] <= tercile_edges[0],
        merged["atr_20"] <= tercile_edges[1],
        merged["atr_20"] > tercile_edges[1],
    ]
    merged["atr_regime"] = np.select(conditions, ["Low", "Med", "High"], default="Med")

    print(f"  ATR terciles: Low <= {tercile_edges[0]:.1f}, Med <= {tercile_edges[1]:.1f}, High > {tercile_edges[1]:.1f}")
    return merged


def print_hourly_stats(data: pd.DataFrame) -> None:
    """Print stats for each hour."""
    print("\n" + "=" * 100)
    print("HOURLY RETURN STATISTICS (all regimes)")
    print("=" * 100)
    header = f"{'Hour':>4} {'Session':<10} {'N':>7} {'Mean':>8} {'Median':>8} {'Std':>8} {'Up%':>6} {'T-stat':>7} {'Signif':>6}"
    print(header)
    print("-" * 100)

    for hour in range(24):
        subset = data[data["hour_utc"] == hour]
        if len(subset) == 0:
            continue

        n = len(subset)
        returns = subset["return_pts"].values
        mean_r = np.mean(returns)
        median_r = np.median(returns)
        std_r = np.std(returns, ddof=1) if n > 1 else 0
        up_pct = (returns > 0).sum() / n * 100
        t_stat = mean_r / (std_r / np.sqrt(n)) if std_r > 0 and n > 1 else 0
        signif = "***" if abs(t_stat) > 2.576 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.645 else ""
        session = get_session(hour)

        print(f"{hour:>4} {session:<10} {n:>7,} {mean_r:>+8.3f} {median_r:>+8.3f} {std_r:>8.3f} {up_pct:>5.1f}% {t_stat:>+7.2f} {signif:>6}")


def print_session_summary(data: pd.DataFrame) -> None:
    """Print aggregated session stats."""
    print("\n" + "=" * 100)
    print("SESSION AGGREGATE STATS")
    print("=" * 100)

    for session_name in ["Asia", "London", "NY", "Off-hours"]:
        subset = data[data["session"] == session_name]
        if len(subset) == 0:
            continue

        n = len(subset)
        returns = subset["return_pts"].values
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1) if n > 1 else 0
        up_pct = (returns > 0).sum() / n * 100
        total = np.sum(returns)

        print(f"\n  {session_name} (N={n:,}):")
        print(f"    Mean: {mean_r:+.4f} pts | Std: {std_r:.4f} | Up%: {up_pct:.1f}% | Total: {total:+.1f} pts")


def print_regime_breakdown(data: pd.DataFrame) -> None:
    """Print hourly stats broken down by ATR regime."""
    print("\n" + "=" * 100)
    print("HOURLY RETURNS BY ATR REGIME")
    print("=" * 100)

    for regime in ["Low", "Med", "High"]:
        regime_data = data[data["atr_regime"] == regime]
        print(f"\n--- ATR Regime: {regime} (N={len(regime_data):,} hourly obs) ---")
        header = f"{'Hour':>4} {'Session':<10} {'N':>7} {'Mean':>8} {'Median':>8} {'Std':>8} {'Up%':>6} {'T-stat':>7}"
        print(header)
        print("-" * 80)

        for hour in range(24):
            subset = regime_data[regime_data["hour_utc"] == hour]
            if len(subset) < 30:
                continue

            n = len(subset)
            returns = subset["return_pts"].values
            mean_r = np.mean(returns)
            median_r = np.median(returns)
            std_r = np.std(returns, ddof=1)
            up_pct = (returns > 0).sum() / n * 100
            t_stat = mean_r / (std_r / np.sqrt(n)) if std_r > 0 else 0
            session = get_session(hour)

            print(f"{hour:>4} {session:<10} {n:>7,} {mean_r:>+8.3f} {median_r:>+8.3f} {std_r:>8.3f} {up_pct:>5.1f}% {t_stat:>+7.2f}")


def print_expr_if_trading(data: pd.DataFrame) -> None:
    """Compute ExpR if you buy at hour open and sell at hour close, using cost model."""
    print("\n" + "=" * 100)
    print("EXPECTED R-MULTIPLE: BUY AT HOUR OPEN, SELL AT HOUR CLOSE")
    print(f"  Cost model: {SPEC.instrument}, friction={SPEC.friction_in_points:.2f} pts/RT")
    print("=" * 100)

    header = f"{'Hour':>4} {'Session':<10} {'N':>7} {'WR%':>6} {'ExpR':>8} {'Sharpe':>8} {'MaxDD_R':>8} {'Total_R':>9}"
    print(header)
    print("-" * 100)

    # For each hour, treat the hourly return as a "trade" and convert to R
    # Risk = hourly range (realistic stop would be high-low of that hour)
    # But for structural bias test, use a fixed 1-point risk
    risk_pts = 2.0  # Assume 2pt stop (~$20 risk for MGC)

    for hour in range(24):
        subset = data[data["hour_utc"] == hour]
        if len(subset) < 30:
            continue

        # PnL in R: (return_pts * point_value - friction) / (risk_pts * point_value + friction)
        pnl_dollars = subset["return_pts"].values * SPEC.point_value - SPEC.total_friction
        risk_dollars = risk_pts * SPEC.point_value + SPEC.total_friction
        pnl_r = pnl_dollars / risk_dollars

        stats = compute_strategy_metrics(pnl_r)
        if stats is None:
            continue

        session = get_session(hour)
        print(f"{hour:>4} {session:<10} {stats['n']:>7,} {stats['wr']*100:>5.1f}% {stats['expr']:>+8.4f} {stats['sharpe']:>+8.4f} {stats['maxdd']:>+8.2f} {stats['total']:>+9.2f}")


def print_best_worst_hours(data: pd.DataFrame) -> None:
    """Print the most biased hours."""
    print("\n" + "=" * 100)
    print("TOP 5 HOURS WITH STRONGEST DIRECTIONAL BIAS (by t-statistic)")
    print("=" * 100)

    results = []
    for hour in range(24):
        subset = data[data["hour_utc"] == hour]
        if len(subset) < 100:
            continue
        returns = subset["return_pts"].values
        n = len(returns)
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        t_stat = mean_r / (std_r / np.sqrt(n)) if std_r > 0 else 0
        results.append((hour, get_session(hour), n, mean_r, std_r, t_stat))

    results.sort(key=lambda x: abs(x[5]), reverse=True)

    for hour, session, n, mean_r, std_r, t_stat in results[:5]:
        direction = "BULLISH" if mean_r > 0 else "BEARISH"
        sig = "SIGNIFICANT" if abs(t_stat) > 1.96 else "not significant"
        print(f"  Hour {hour:02d} UTC ({session}): {direction} bias, mean={mean_r:+.4f} pts, t={t_stat:+.2f} ({sig}, N={n:,})")

    print("\n  Bottom 5 (weakest bias):")
    for hour, session, n, mean_r, std_r, t_stat in results[-5:]:
        print(f"  Hour {hour:02d} UTC ({session}): mean={mean_r:+.4f} pts, t={t_stat:+.2f} (N={n:,})")


def main():
    print("=" * 100)
    print("GOLD (MGC) TIME-OF-DAY RETURN ANALYSIS")
    print("=" * 100)
    print()

    bars = load_5m_bars(DB_PATH)
    atr_df = load_atr(DB_PATH)
    hourly = compute_hourly_returns(bars)
    hourly["session"] = hourly["hour_utc"].apply(get_session)

    data = merge_atr_regimes(hourly, atr_df)

    print_hourly_stats(data)
    print_session_summary(data)
    print_regime_breakdown(data)
    print_expr_if_trading(data)
    print_best_worst_hours(data)

    # Year-by-year for top hours
    print("\n" + "=" * 100)
    print("YEAR-BY-YEAR BREAKDOWN FOR ALL HOURS (mean return pts)")
    print("=" * 100)

    data["year"] = pd.to_datetime(data["date"]).dt.year
    years = sorted(data["year"].unique())

    header = f"{'Hour':>4} {'Session':<10}" + "".join(f"{y:>8}" for y in years)
    print(header)
    print("-" * (14 + 8 * len(years)))

    for hour in range(24):
        hour_data = data[data["hour_utc"] == hour]
        if len(hour_data) < 100:
            continue
        session = get_session(hour)
        row = f"{hour:>4} {session:<10}"
        for y in years:
            ydata = hour_data[hour_data["year"] == y]
            if len(ydata) > 10:
                row += f"{ydata['return_pts'].mean():>+8.3f}"
            else:
                row += f"{'---':>8}"
        print(row)

    print("\n[Done]")


if __name__ == "__main__":
    main()
