#!/usr/bin/env python3
"""
Overlay Filter Comparison: ADX, VWAP Direction, Combined.

Tests 4 overlay configurations on ORB breakout strategies via independent
walk-forwards:

  1. SIZE-ONLY (baseline)     — ORB size >= G2/G4, no overlay
  2. SIZE + ADX               — + ADX(14) >= threshold at break time
  3. SIZE + VWAP_DIR          — + break direction matches VWAP side
  4. SIZE + ADX + VWAP_DIR    — both ADX and VWAP direction must confirm

VWAP Direction Filter:
  At break time, compute cumulative VWAP from session open using bars_5m.
  Long break is CONFIRMED if close > VWAP; short if close < VWAP.
  Counter-trend breakouts are rejected.

Walk-forward: 12-month train, 1-month test steps, OOS from 2018-01-01.
Each overlay config runs its own independent walk-forward to avoid overfitting.

Grid per config:
  ORB [0900, 1000] x EM [E1, E2] x RR [1.5, 2.0, 2.5] x Size [G2, G4] x CB=1
  = 24 base combos (+ ADX threshold variants for configs 2 and 4)
"""

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from research._alt_strategy_utils import (
    compute_strategy_metrics,
    compute_walk_forward_windows,
    save_results,
)

SPEC = get_cost_spec("MGC")

# Grid dimensions
ORB_LABELS = ["0900", "1000"]
ENTRY_MODELS = ["E1", "E2"]
RR_TARGETS = [1.5, 2.0, 2.5]
SIZE_FILTERS = {"G2": 2.0, "G4": 4.0}
ADX_THRESHOLDS = [20, 25, 30, 35]

REGIME_BOUNDARY = date(2025, 1, 1)


# ---------------------------------------------------------------------------
# ADX computation (copied from analyze_adx_filter.py)
# ---------------------------------------------------------------------------

def compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> np.ndarray:
    """Compute Wilder's ADX from high/low/close arrays."""
    n = len(highs)
    adx = np.full(n, np.nan)
    if n < 2 * period + 1:
        return adx

    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    atr = np.zeros(n)
    smooth_plus = np.zeros(n)
    smooth_minus = np.zeros(n)

    atr[period] = tr[1:period + 1].mean()
    smooth_plus[period] = plus_dm[1:period + 1].mean()
    smooth_minus[period] = minus_dm[1:period + 1].mean()

    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        smooth_plus[i] = (smooth_plus[i - 1] * (period - 1) + plus_dm[i]) / period
        smooth_minus[i] = (smooth_minus[i - 1] * (period - 1) + minus_dm[i]) / period

    dx = np.full(n, np.nan)
    for i in range(period, n):
        if atr[i] > 0:
            plus_di = 100 * smooth_plus[i] / atr[i]
            minus_di = 100 * smooth_minus[i] / atr[i]
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di - minus_di) / di_sum

    seed_start = period
    seed_end = 2 * period
    valid_dx = dx[seed_start:seed_end]
    valid_dx = valid_dx[~np.isnan(valid_dx)]
    if len(valid_dx) == 0:
        return adx

    adx[2 * period] = valid_dx.mean()
    for i in range(2 * period + 1, n):
        if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


# ---------------------------------------------------------------------------
# VWAP computation (adapted from analyze_vwap_pullback.py, works on any bars)
# ---------------------------------------------------------------------------

def compute_vwap(bars: pd.DataFrame) -> np.ndarray:
    """Compute cumulative VWAP from bars (works on 1m or 5m).

    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    """
    tp = (bars["high"].values + bars["low"].values + bars["close"].values) / 3.0
    vol = bars["volume"].values.astype(float)
    vol = np.where(vol > 0, vol, 1.0)  # avoid zero-volume bars
    cum_tp_vol = np.cumsum(tp * vol)
    cum_vol = np.cumsum(vol)
    return cum_tp_vol / cum_vol


# ---------------------------------------------------------------------------
# Data loading — bulk pre-computation
# ---------------------------------------------------------------------------

def _load_bars_5m_with_warmup(db_path: Path, trading_day: date) -> pd.DataFrame:
    """Load 5m bars for current + 3 prior calendar days (ADX warmup)."""
    prev_day = trading_day - timedelta(days=3)
    start_utc, _ = compute_trading_day_utc_range(prev_day)
    _, end_utc = compute_trading_day_utc_range(trading_day)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_5m
            WHERE symbol = 'MGC'
              AND ts_utc >= ? AND ts_utc < ?
            ORDER BY ts_utc
        """, [start_utc, end_utc]).fetchdf()
    finally:
        con.close()
    return df


def _get_adx_at_time(bars_5m: pd.DataFrame, break_ts: pd.Timestamp,
                     period: int = 14) -> float | None:
    """Compute ADX(14) at a specific timestamp from 5m bars."""
    if bars_5m.empty or len(bars_5m) < 2 * period + 1:
        return None

    ts_col = bars_5m["ts_utc"]
    if break_ts.tzinfo is None:
        break_ts_cmp = break_ts
    else:
        if ts_col.dt.tz is not None:
            break_ts_cmp = break_ts.tz_convert(ts_col.dt.tz)
        else:
            break_ts_cmp = break_ts.tz_localize(None)

    mask = ts_col <= break_ts_cmp
    if mask.sum() < 2 * period + 1:
        return None

    subset = bars_5m[mask]
    adx_values = compute_adx(
        subset["high"].values, subset["low"].values, subset["close"].values, period
    )
    last_adx = adx_values[-1]
    return float(last_adx) if not np.isnan(last_adx) else None


def _get_vwap_at_time(bars_5m: pd.DataFrame, break_ts: pd.Timestamp,
                      session_start_utc: pd.Timestamp) -> float | None:
    """Compute cumulative VWAP at break_ts using bars from session_start_utc.

    Returns VWAP value at the break bar, or None if insufficient data.
    """
    if bars_5m.empty:
        return None

    ts_col = bars_5m["ts_utc"]

    # Normalize timezone for comparisons
    if session_start_utc.tzinfo is None:
        ss_cmp = session_start_utc
    else:
        if ts_col.dt.tz is not None:
            ss_cmp = session_start_utc.tz_convert(ts_col.dt.tz)
        else:
            ss_cmp = session_start_utc.tz_localize(None)

    if break_ts.tzinfo is None:
        bt_cmp = break_ts
    else:
        if ts_col.dt.tz is not None:
            bt_cmp = break_ts.tz_convert(ts_col.dt.tz)
        else:
            bt_cmp = break_ts.tz_localize(None)

    # Session bars: from session open to break time
    session_mask = (ts_col >= ss_cmp) & (ts_col <= bt_cmp)
    session_bars = bars_5m[session_mask]

    if len(session_bars) < 2:
        return None

    vwap_arr = compute_vwap(session_bars)
    return float(vwap_arr[-1])


def _get_close_at_time(bars_5m: pd.DataFrame, break_ts: pd.Timestamp) -> float | None:
    """Get the close price of the bar at or just before break_ts."""
    if bars_5m.empty:
        return None

    ts_col = bars_5m["ts_utc"]
    if break_ts.tzinfo is None:
        bt_cmp = break_ts
    else:
        if ts_col.dt.tz is not None:
            bt_cmp = break_ts.tz_convert(ts_col.dt.tz)
        else:
            bt_cmp = break_ts.tz_localize(None)

    mask = ts_col <= bt_cmp
    if mask.sum() == 0:
        return None
    return float(bars_5m[mask].iloc[-1]["close"])


def _session_open_utc(trading_day: date, orb_label: str) -> pd.Timestamp:
    """Return session open time in UTC for a given trading day and ORB label.

    Session open = 23:00 UTC previous calendar day (the trading day boundary).
    This is the start of data accumulation for VWAP.
    """
    # Trading day boundary is 23:00 UTC previous calendar day
    prev_cal = trading_day - timedelta(days=1)
    return pd.Timestamp(prev_cal.isoformat() + "T23:00:00", tz="UTC")


def load_all_overlay_data(db_path: Path, start: date, end: date) -> pd.DataFrame:
    """Load outcomes + daily_features + ADX + VWAP for all (day, session) pairs.

    Returns a single DataFrame with columns:
      trading_day, orb_label, entry_model, rr_target, confirm_bars, pnl_r, outcome,
      orb_size, break_dir, adx_at_break, vwap_at_break, close_at_break, vwap_aligned
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        features = con.execute("""
            SELECT trading_day,
                   orb_0900_break_ts, orb_0900_size, orb_0900_break_dir,
                   orb_1000_break_ts, orb_1000_size, orb_1000_break_dir
            FROM daily_features
            WHERE symbol = 'MGC' AND orb_minutes = 5
              AND trading_day BETWEEN ? AND ?
            ORDER BY trading_day
        """, [start, end]).fetchdf()

        outcomes = con.execute("""
            SELECT trading_day, orb_label, entry_model, rr_target,
                   confirm_bars, pnl_r, outcome
            FROM orb_outcomes
            WHERE symbol = 'MGC'
              AND trading_day BETWEEN ? AND ?
        """, [start, end]).fetchdf()
    finally:
        con.close()

    if features.empty or outcomes.empty:
        return pd.DataFrame()

    # Pre-compute ADX, VWAP, close at break for each (day, session) pair
    adx_map = {}
    vwap_map = {}
    close_map = {}
    size_map = {}
    break_dir_map = {}
    total = len(features)

    for idx, (_, row) in enumerate(features.iterrows()):
        if idx % 200 == 0:
            print(f"    Computing overlays: day {idx+1}/{total}...")

        td = row["trading_day"]
        if hasattr(td, "date") and callable(td.date):
            td_date = td.date()
        elif isinstance(td, str):
            td_date = date.fromisoformat(td)
        else:
            td_date = td

        # Load bars_5m with warmup (shared for ADX + VWAP)
        bars_5m = _load_bars_5m_with_warmup(db_path, td_date)
        if bars_5m.empty:
            continue

        for orb_label in ORB_LABELS:
            break_ts = row[f"orb_{orb_label}_break_ts"]
            orb_size = row[f"orb_{orb_label}_size"]
            break_dir = row[f"orb_{orb_label}_break_dir"]

            if pd.isna(break_ts):
                continue

            key = (str(td_date), orb_label)

            # Size + break_dir
            if not pd.isna(orb_size):
                size_map[key] = orb_size
            if not pd.isna(break_dir):
                break_dir_map[key] = break_dir

            # ADX at break time
            adx_val = _get_adx_at_time(bars_5m, break_ts)
            if adx_val is not None:
                adx_map[key] = adx_val

            # VWAP at break time (session open = 23:00 UTC prev day)
            session_open = _session_open_utc(td_date, orb_label)
            vwap_val = _get_vwap_at_time(bars_5m, break_ts, session_open)
            if vwap_val is not None:
                vwap_map[key] = vwap_val

            # Close at break bar
            close_val = _get_close_at_time(bars_5m, break_ts)
            if close_val is not None:
                close_map[key] = close_val

    print(f"    ADX: {len(adx_map)} pairs, VWAP: {len(vwap_map)} pairs, "
          f"Close: {len(close_map)} pairs")

    # Merge into outcomes
    outcomes["trading_day_str"] = outcomes["trading_day"].astype(str).str[:10]

    outcomes["orb_size"] = outcomes.apply(
        lambda r: size_map.get((r["trading_day_str"], r["orb_label"])), axis=1
    )
    outcomes["break_dir"] = outcomes.apply(
        lambda r: break_dir_map.get((r["trading_day_str"], r["orb_label"])), axis=1
    )
    outcomes["adx_at_break"] = outcomes.apply(
        lambda r: adx_map.get((r["trading_day_str"], r["orb_label"])), axis=1
    )
    outcomes["vwap_at_break"] = outcomes.apply(
        lambda r: vwap_map.get((r["trading_day_str"], r["orb_label"])), axis=1
    )
    outcomes["close_at_break"] = outcomes.apply(
        lambda r: close_map.get((r["trading_day_str"], r["orb_label"])), axis=1
    )

    # Compute VWAP alignment: long confirmed if close > VWAP, short if close < VWAP
    def _vwap_aligned(row):
        bd = row["break_dir"]
        vwap = row["vwap_at_break"]
        close = row["close_at_break"]
        if pd.isna(bd) or pd.isna(vwap) or pd.isna(close):
            return None
        if bd == "long" and close > vwap:
            return True
        if bd == "short" and close < vwap:
            return True
        return False

    outcomes["vwap_aligned"] = outcomes.apply(_vwap_aligned, axis=1)

    outcomes["trading_day_date"] = pd.to_datetime(outcomes["trading_day"]).dt.date

    return outcomes


# ---------------------------------------------------------------------------
# Walk-forward engine — runs one overlay configuration
# ---------------------------------------------------------------------------

def _run_single_walk_forward(
    data: pd.DataFrame,
    config_name: str,
    use_adx: bool,
    use_vwap: bool,
    windows: list[dict],
) -> dict:
    """Run walk-forward for a single overlay config.

    Returns dict with window_results, oos_pnls, oos_dates.
    """
    window_results = []
    oos_pnls = []
    oos_dates = []

    # Filter: must have ADX if using ADX, must have vwap_aligned if using VWAP
    valid = data.copy()
    if use_adx:
        valid = valid.dropna(subset=["adx_at_break"])
    if use_vwap:
        valid = valid.dropna(subset=["vwap_aligned"])

    for w in windows:
        train_mask = (
            (valid["trading_day_date"] >= w["train_start"])
            & (valid["trading_day_date"] <= w["train_end"])
        )
        test_mask = (
            (valid["trading_day_date"] >= w["test_start"])
            & (valid["trading_day_date"] <= w["test_end"])
        )

        train_data = valid[train_mask]
        test_data = valid[test_mask]

        if train_data.empty:
            continue

        best_combo = None
        best_sharpe = -999.0

        for orb_label in ORB_LABELS:
            ol_train = train_data[train_data["orb_label"] == orb_label]
            for em in ENTRY_MODELS:
                em_train = ol_train[ol_train["entry_model"] == em]
                for rr in RR_TARGETS:
                    rr_train = em_train[em_train["rr_target"] == rr]
                    cb_train = rr_train[rr_train["confirm_bars"] == 1]
                    if cb_train.empty:
                        continue

                    for sf_name, sf_min in SIZE_FILTERS.items():
                        sf_train = cb_train[cb_train["orb_size"] >= sf_min]

                        # Apply VWAP filter (binary, always on for VWAP configs)
                        if use_vwap:
                            sf_train = sf_train[sf_train["vwap_aligned"] == True]  # noqa: E712

                        if not use_adx:
                            # No ADX: just evaluate the size (+ optional VWAP) combo
                            if len(sf_train) < 10:
                                continue
                            stats = compute_strategy_metrics(sf_train["pnl_r"].dropna().values)
                            if stats and stats["sharpe"] > best_sharpe:
                                best_sharpe = stats["sharpe"]
                                best_combo = (orb_label, em, rr, sf_name, None)
                        else:
                            # ADX: test each threshold
                            for adx_thresh in ADX_THRESHOLDS:
                                filtered = sf_train[sf_train["adx_at_break"] >= adx_thresh]
                                if len(filtered) < 10:
                                    continue
                                stats = compute_strategy_metrics(filtered["pnl_r"].dropna().values)
                                if stats and stats["sharpe"] > best_sharpe:
                                    best_sharpe = stats["sharpe"]
                                    best_combo = (orb_label, em, rr, sf_name, adx_thresh)

        if best_combo is None:
            continue

        orb_label, em, rr, sf_name, adx_thresh = best_combo

        # Apply to OOS
        oos_base = test_data[
            (test_data["orb_label"] == orb_label)
            & (test_data["entry_model"] == em)
            & (test_data["rr_target"] == rr)
            & (test_data["confirm_bars"] == 1)
        ]
        sf_min = SIZE_FILTERS[sf_name]
        oos_sized = oos_base[oos_base["orb_size"] >= sf_min]

        if use_vwap:
            oos_sized = oos_sized[oos_sized["vwap_aligned"] == True]  # noqa: E712

        if use_adx and adx_thresh is not None:
            oos_final = oos_sized[oos_sized["adx_at_break"] >= adx_thresh]
        else:
            oos_final = oos_sized

        if oos_final.empty:
            continue

        final_pnls = oos_final["pnl_r"].dropna().values
        oos_stats = compute_strategy_metrics(final_pnls)

        oos_pnls.extend(final_pnls)
        oos_dates.extend(oos_final.dropna(subset=["pnl_r"])["trading_day_date"].values)

        combo_parts = [orb_label, em, f"RR{rr}", sf_name]
        if adx_thresh is not None:
            combo_parts.append(f"ADX{adx_thresh}")
        combo_label = "_".join(combo_parts)

        window_results.append({
            "test_start": str(w["test_start"]),
            "test_end": str(w["test_end"]),
            "selected": combo_label,
            "train_sharpe": best_sharpe,
            "oos_stats": oos_stats,
            "adx_thresh": adx_thresh,
        })

    return {
        "config": config_name,
        "window_results": window_results,
        "oos_pnls": np.array(oos_pnls) if oos_pnls else np.array([]),
        "oos_dates": np.array(oos_dates) if oos_dates else np.array([]),
    }


# ---------------------------------------------------------------------------
# Session-level breakdown
# ---------------------------------------------------------------------------

def _compute_per_session_stats(data: pd.DataFrame, config_name: str,
                               use_adx: bool, use_vwap: bool,
                               windows: list[dict]) -> dict:
    """Run walk-forward per session and return per-session combined OOS stats."""
    session_stats = {}
    for orb_label in ORB_LABELS:
        session_data = data[data["orb_label"] == orb_label]
        result = _run_single_walk_forward(
            session_data, f"{config_name}_{orb_label}",
            use_adx, use_vwap, windows
        )
        if len(result["oos_pnls"]) > 0:
            session_stats[orb_label] = compute_strategy_metrics(result["oos_pnls"])
        else:
            session_stats[orb_label] = None
    return session_stats


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_overlay_comparison(
    db_path: Path,
    train_months: int = 12,
    test_start: date = date(2018, 1, 1),
    test_end: date = date(2026, 2, 1),
) -> dict:
    """Run all 4 overlay walk-forwards and compare."""
    full_start = date(2016, 1, 1)

    print("Phase 1: Loading all overlay data (outcomes + ADX + VWAP)...")
    data = load_all_overlay_data(db_path, full_start, test_end)
    if data.empty:
        print("  No data loaded")
        return {}

    # Report data coverage
    has_adx = data["adx_at_break"].notna().sum()
    has_vwap = data["vwap_aligned"].notna().sum()
    vwap_aligned_count = (data["vwap_aligned"] == True).sum()  # noqa: E712
    vwap_rejected_count = (data["vwap_aligned"] == False).sum()  # noqa: E712
    print(f"  Total outcomes: {len(data)}")
    print(f"  With ADX: {has_adx} ({has_adx/len(data):.0%})")
    print(f"  With VWAP direction: {has_vwap} ({has_vwap/len(data):.0%})")
    print(f"  VWAP aligned: {vwap_aligned_count}, rejected: {vwap_rejected_count} "
          f"({vwap_rejected_count/(vwap_aligned_count+vwap_rejected_count):.0%} rejection rate)")
    print()

    windows = compute_walk_forward_windows(test_start, test_end, train_months)
    print(f"Phase 2: Running 4 independent walk-forwards ({len(windows)} windows each)...")
    print()

    configs = [
        ("SIZE_ONLY",        False, False),
        ("SIZE_ADX",         True,  False),
        ("SIZE_VWAP",        False, True),
        ("SIZE_ADX_VWAP",    True,  True),
    ]

    results = {}
    for config_name, use_adx, use_vwap in configs:
        print(f"  Running {config_name}...")
        wf_result = _run_single_walk_forward(data, config_name, use_adx, use_vwap, windows)
        n_windows = len(wf_result["window_results"])
        n_oos = len(wf_result["oos_pnls"])
        print(f"    {n_windows} windows, {n_oos} OOS trades")
        results[config_name] = wf_result

    # Per-session breakdown
    print()
    print("Phase 3: Per-session breakdown...")
    session_results = {}
    for config_name, use_adx, use_vwap in configs:
        session_results[config_name] = _compute_per_session_stats(
            data, config_name, use_adx, use_vwap, windows
        )

    # Compute combined metrics + regime split
    combined = {}
    regime_split = {}
    for config_name in [c[0] for c in configs]:
        r = results[config_name]
        if len(r["oos_pnls"]) > 0:
            combined[config_name] = compute_strategy_metrics(r["oos_pnls"])
            # Regime split
            dates = r["oos_dates"]
            pnls = r["oos_pnls"]
            low_mask = dates < np.datetime64(REGIME_BOUNDARY)
            high_mask = ~low_mask
            regime_split[config_name] = {
                "low_vol_2018_2024": compute_strategy_metrics(pnls[low_mask]) if low_mask.sum() > 0 else None,
                "high_vol_2025_2026": compute_strategy_metrics(pnls[high_mask]) if high_mask.sum() > 0 else None,
            }
        else:
            combined[config_name] = None
            regime_split[config_name] = {"low_vol_2018_2024": None, "high_vol_2025_2026": None}

    # Selection frequency
    selection_freq = {}
    for config_name in [c[0] for c in configs]:
        r = results[config_name]
        adx_selected = sum(1 for w in r["window_results"] if w["adx_thresh"] is not None)
        no_adx = len(r["window_results"]) - adx_selected
        selection_freq[config_name] = {
            "total_windows": len(r["window_results"]),
            "adx_selected": adx_selected,
            "no_adx_selected": no_adx,
        }

    return {
        "combined": combined,
        "regime_split": regime_split,
        "session_results": session_results,
        "selection_freq": selection_freq,
        "window_details": {k: v["window_results"] for k, v in results.items()},
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(stats: dict | None) -> str:
    """Format stats dict as a compact string."""
    if stats is None:
        return "No data"
    return (f"N={stats['n']:>4}, WR={stats['wr']:>5.0%}, ExpR={stats['expr']:>+7.3f}, "
            f"Sharpe={stats['sharpe']:>+6.3f}, MaxDD={stats['maxdd']:>+6.1f}R, "
            f"Total={stats['total']:>+7.1f}R")


def print_report(results: dict) -> None:
    """Print the comparison report."""
    combined = results["combined"]
    regime_split = results["regime_split"]
    session_results = results["session_results"]
    selection_freq = results["selection_freq"]

    sep = "=" * 80
    print()
    print(sep)
    print("OVERLAY FILTER COMPARISON RESULTS")
    print(sep)

    # Combined OOS
    print()
    print("COMBINED OOS RESULTS:")
    config_order = ["SIZE_ONLY", "SIZE_ADX", "SIZE_VWAP", "SIZE_ADX_VWAP"]
    for cfg in config_order:
        label = f"  {cfg:<20s}"
        print(f"{label} {_fmt(combined.get(cfg))}")

    # Uplift vs SIZE_ONLY
    baseline = combined.get("SIZE_ONLY")
    if baseline:
        print()
        print("UPLIFT vs SIZE_ONLY:")
        for cfg in ["SIZE_ADX", "SIZE_VWAP", "SIZE_ADX_VWAP"]:
            c = combined.get(cfg)
            if c and baseline:
                expr_delta = c["expr"] - baseline["expr"]
                sharpe_delta = c["sharpe"] - baseline["sharpe"]
                print(f"  {cfg:<20s} ExpR {expr_delta:>+7.3f}, Sharpe {sharpe_delta:>+7.3f}")
            else:
                print(f"  {cfg:<20s} N/A")

    # Regime split
    print()
    print("REGIME SPLIT:")
    for period_label in ["low_vol_2018_2024", "high_vol_2025_2026"]:
        nice_label = "2018-2024 (low vol)" if "low" in period_label else "2025-2026 (high vol)"
        print(f"  {nice_label}:")
        for cfg in config_order:
            rs = regime_split.get(cfg, {})
            stats = rs.get(period_label)
            print(f"    {cfg:<20s} {_fmt(stats)}")

    # Selection frequency
    print()
    print("WALK-FORWARD SELECTION FREQUENCY:")
    for cfg in config_order:
        sf = selection_freq.get(cfg, {})
        tw = sf.get("total_windows", 0)
        adx_sel = sf.get("adx_selected", 0)
        print(f"  {cfg:<20s} {tw} windows total"
              + (f", ADX selected in {adx_sel}/{tw}" if adx_sel > 0 else ""))

    # Per-session breakdown
    print()
    print("PER-SESSION BREAKDOWN:")
    for orb_label in ORB_LABELS:
        print(f"  {orb_label}:")
        for cfg in config_order:
            stats = session_results.get(cfg, {}).get(orb_label)
            print(f"    {cfg:<20s} {_fmt(stats)}")

    # GO/NO-GO for each overlay
    print()
    print("GO/NO-GO EVALUATION:")
    for cfg in ["SIZE_ADX", "SIZE_VWAP", "SIZE_ADX_VWAP"]:
        c = combined.get(cfg)
        b = baseline
        if c is None or b is None:
            print(f"  {cfg}: NO-GO (insufficient data)")
            continue

        checks = {
            "OOS ExpR > 0": c["expr"] > 0,
            "OOS N > 100": c["n"] > 100,
            "OOS Sharpe > baseline": c["sharpe"] > b["sharpe"],
            "ExpR uplift > 0": c["expr"] > b["expr"],
        }
        rs = regime_split.get(cfg, {})
        lv = rs.get("low_vol_2018_2024")
        if lv:
            checks["Low-vol ExpR > 0"] = lv["expr"] > 0

        all_pass = all(checks.values())
        verdict = "GO" if all_pass else "NO-GO"
        detail = ", ".join(f"{'PASS' if v else 'FAIL'}:{k}" for k, v in checks.items())
        print(f"  {cfg}: {verdict} [{detail}]")

    print()
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overlay Filter Comparison")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sep = "=" * 80
    print(sep)
    print("OVERLAY FILTER COMPARISON: ADX, VWAP Direction, Combined")
    print(sep)
    print()
    print("Configs: SIZE_ONLY | SIZE+ADX | SIZE+VWAP_DIR | SIZE+ADX+VWAP_DIR")
    print(f"Grid: {len(ORB_LABELS)} ORBs x {len(ENTRY_MODELS)} EMs x {len(RR_TARGETS)} RR x "
          f"{len(SIZE_FILTERS)} sizes x CB=1 = 24 base combos")
    print(f"ADX thresholds: {ADX_THRESHOLDS}")
    print(f"Train: {args.train_months} months, OOS from 2018-01-01")
    print()

    results = run_overlay_comparison(args.db_path, args.train_months)

    if results:
        print_report(results)
        if args.output:
            save_results(results, args.output)
    else:
        print("No results produced.")

    print("DONE")
    print(sep)


if __name__ == "__main__":
    main()
