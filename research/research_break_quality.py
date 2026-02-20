#!/usr/bin/env python3
"""
Break Quality Research — post-break candle behavior as entry/exit signals.

Investigates whether the bar-level action around the breakout predicts
trade success. Answers questions like:
  - If the entry bar engulfs back inside the ORB, should I exit immediately?
  - Does break strength (how far outside ORB the confirm bar closes) predict outcome?
  - Does immediate follow-through (next bar continues in break direction) matter?
  - How quickly does a "re-entry into ORB" kill the trade?

CONDITIONS TESTED:

  PRE-ENTRY (available before E1 entry, can be used as don't-enter filters):
    C1 - Break distance:  How far outside the ORB the confirm bar closed,
                          measured in R (multiples of ORB size).
    C2 - Confirm bar wick: Ratio of wick back toward ORB vs body.
                          High wick ratio = rejection, price tried to come back.
    C3 - Break speed:     Minutes from ORB close to confirm bar.
                          Fast break = momentum. Slow break = grind.

  POST-ENTRY (observable after E1 fill, used as early-exit triggers):
    C4 - Entry bar engulfing: Entry bar closes back inside ORB.
                          The specific question the user asked.
    C5 - Entry bar direction: Entry bar close vs open — does it continue
                          in the break direction or reverse?
    C6 - Quick reversal:  Within first N bars after entry, ANY close back inside ORB.

For each condition, we split outcomes into two groups and compare avgR, WR, totR.

Usage:
  python research/research_break_quality.py
  python research/research_break_quality.py --db-path C:/db/gold.db
  python research/research_break_quality.py --instrument MGC --sessions 0900,1000
"""

import argparse
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from pipeline.log import get_logger
logger = get_logger(__name__)


def _bh_reject(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean mask of rejections."""
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    below = sorted_p <= thresholds
    if not np.any(below):
        return np.zeros(m, dtype=bool)
    max_k = int(np.where(below)[0][-1])
    reject = np.zeros(m, dtype=bool)
    reject[order[: max_k + 1]] = True
    return reject

BRISBANE_TZ = ZoneInfo("Australia/Brisbane")

# =========================================================================
# Data loading
# =========================================================================

def load_break_context(con, instrument: str, session: str) -> pd.DataFrame:
    """Load outcomes joined with ORB levels from daily_features.

    orb_outcomes has entry/exit/pnl but NOT ORB levels or break direction.
    daily_features has orb_{session}_high/low/break_dir/break_ts.
    We join them to get the full picture.
    """
    query = f"""
        SELECT
            oo.trading_day,
            df.orb_{session}_high   AS orb_high,
            df.orb_{session}_low    AS orb_low,
            df.orb_{session}_size   AS orb_size,
            df.orb_{session}_break_dir AS break_dir,
            df.orb_{session}_break_ts  AS break_ts,
            oo.entry_ts,
            oo.exit_ts,
            oo.entry_price,
            oo.stop_price,
            oo.target_price,
            oo.pnl_r,
            oo.outcome,
            oo.entry_model,
            oo.rr_target,
            oo.confirm_bars
        FROM orb_outcomes oo
        JOIN daily_features df
            ON oo.symbol = df.symbol
            AND oo.trading_day = df.trading_day
            AND oo.orb_minutes = df.orb_minutes
        WHERE oo.symbol = ?
          AND oo.orb_label = ?
          AND oo.entry_model = 'E1'
          AND oo.rr_target = 2.0
          AND oo.confirm_bars = 2
          AND oo.pnl_r IS NOT NULL
          AND oo.orb_minutes = 5
          AND df.orb_minutes = 5
          AND df.orb_{session}_break_dir IS NOT NULL
        ORDER BY oo.trading_day
    """
    return con.execute(query, [instrument, session]).fetchdf()


def load_bars_for_day(con, instrument: str, trading_day, orb_label: str) -> pd.DataFrame:
    """Load 1m bars for a specific trading day."""
    # Trading day boundaries: 09:00 Brisbane today -> 09:00 Brisbane tomorrow
    # = 23:00 UTC previous day -> 23:00 UTC today
    td = trading_day if isinstance(trading_day, date) else trading_day.date()
    bris_start = datetime(td.year, td.month, td.day, 9, 0, 0, tzinfo=BRISBANE_TZ)
    bris_end = bris_start + timedelta(hours=24)
    utc_start = bris_start.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    utc_end = bris_end.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

    bars = con.execute("""
        SELECT ts_utc, open, high, low, close, volume
        FROM bars_1m
        WHERE symbol = ?
          AND ts_utc >= ?
          AND ts_utc < ?
        ORDER BY ts_utc
    """, [instrument, utc_start, utc_end]).fetchdf()

    return bars


# =========================================================================
# Condition extractors
# =========================================================================

def extract_conditions(bars: pd.DataFrame, row: dict) -> dict | None:
    """Extract all break-quality conditions for a single trade.

    Args:
        bars: 1m bars for this trading day
        row: orb_outcomes row (dict)

    Returns:
        dict of condition values, or None if data insufficient
    """
    orb_high = row["orb_high"]
    orb_low = row["orb_low"]
    orb_size = orb_high - orb_low
    break_dir = row["break_dir"]
    entry_ts = row["entry_ts"]

    if orb_size <= 0 or pd.isna(entry_ts):
        return None

    entry_price = row["entry_price"]
    entry_ts_pd = pd.Timestamp(entry_ts)

    # Compute bars_until_exit: how many 1m bars elapsed between entry and trade close.
    # Used to determine whether a post-bar-30 ORB return happened while trade was still open.
    exit_ts = row.get("exit_ts")
    if exit_ts is not None and not pd.isna(exit_ts):
        exit_ts_pd = pd.Timestamp(exit_ts)
        bars_until_exit = int((bars["ts_utc"] < exit_ts_pd).sum() - (bars["ts_utc"] <= entry_ts_pd).sum())
        bars_until_exit = max(bars_until_exit, 0)
    else:
        bars_until_exit = None  # unknown — treat conservatively

    # E1 entry = next bar open after confirm bar.
    # So the confirm bar is the bar IMMEDIATELY BEFORE entry_ts.
    pre_entry = bars[bars["ts_utc"] < entry_ts_pd].sort_values("ts_utc")
    if pre_entry.empty:
        return None
    cb = pre_entry.iloc[-1]  # confirm bar = last bar before entry
    confirm_ts_pd = cb["ts_utc"]

    # Get entry bar (bar at entry_ts)
    entry_bar = bars[bars["ts_utc"] == entry_ts_pd]
    if entry_bar.empty:
        return None
    eb = entry_bar.iloc[0]

    # All bars after entry (needed for extended time-based analysis)
    post_entry = bars[bars["ts_utc"] > entry_ts_pd].sort_values("ts_utc")

    result = {
        "trading_day": row["trading_day"],
        "pnl_r": row["pnl_r"],
        "outcome": row["outcome"],
        "break_dir": break_dir,
        "orb_size": orb_size,
    }

    # --- C1: Break distance (confirm bar close distance from ORB level, in R) ---
    if break_dir == "long":
        break_distance = (cb["close"] - orb_high) / orb_size
    else:
        break_distance = (orb_low - cb["close"]) / orb_size
    result["c1_break_distance_r"] = float(break_distance)

    # --- C2: Confirm bar wick ratio ---
    # Wick toward ORB / total bar range
    bar_range = cb["high"] - cb["low"]
    if bar_range > 0:
        if break_dir == "long":
            # For long: wick toward ORB = distance from low to open (or close, whichever is lower body)
            body_low = min(cb["open"], cb["close"])
            wick_back = body_low - cb["low"]
        else:
            # For short: wick toward ORB = distance from high to open (or close, whichever is higher body)
            body_high = max(cb["open"], cb["close"])
            wick_back = cb["high"] - body_high
        result["c2_wick_ratio"] = float(wick_back / bar_range)
    else:
        result["c2_wick_ratio"] = 0.0

    # --- C3: Break speed (minutes from ORB close to confirm) ---
    break_ts = row["break_ts"]
    if not pd.isna(break_ts):
        break_ts_pd = pd.Timestamp(break_ts)
        minutes_to_break = (confirm_ts_pd - break_ts_pd).total_seconds() / 60
        result["c3_break_speed_min"] = float(minutes_to_break)
    else:
        result["c3_break_speed_min"] = np.nan

    # --- C4: Entry bar engulfing (closes back inside ORB) ---
    if break_dir == "long":
        entry_bar_inside = eb["close"] <= orb_high
    else:
        entry_bar_inside = eb["close"] >= orb_low
    result["c4_entry_bar_engulf"] = bool(entry_bar_inside)

    # --- C5: Entry bar direction (continues in break direction?) ---
    if break_dir == "long":
        entry_bar_continues = eb["close"] > eb["open"]
    else:
        entry_bar_continues = eb["close"] < eb["open"]
    result["c5_entry_bar_continues"] = bool(entry_bar_continues)

    # --- C6: Close inside ORB within N bars — 1/3/5/10/15/30 ---
    for n_bars in [1, 3, 5, 10, 15, 30]:
        check_bars = post_entry.head(n_bars)
        if len(check_bars) > 0:
            if break_dir == "long":
                any_inside = (check_bars["close"] <= orb_high).any()
            else:
                any_inside = (check_bars["close"] >= orb_low).any()
            result[f"c6_reversal_{n_bars}bar"] = bool(any_inside)
        else:
            result[f"c6_reversal_{n_bars}bar"] = None

    # --- C7: First close inside ORB — which bar? (None = never returned) ---
    t_first_inside = None
    trigger_bar_close = None
    for i, (_, bar) in enumerate(post_entry.iterrows()):
        if break_dir == "long" and bar["close"] <= orb_high:
            t_first_inside = i + 1
            trigger_bar_close = float(bar["close"])
            break
        elif break_dir == "short" and bar["close"] >= orb_low:
            t_first_inside = i + 1
            trigger_bar_close = float(bar["close"])
            break
    result["c7_first_inside_bar"] = t_first_inside  # None = never closed inside

    # --- C6 exit simulation: actual pnl_r if you exited at trigger bar close ---
    # Replaces the ~0R assumption with the real exit price distribution.
    for n_bars, key in [(3, "c6_exit_pnl_r_3bar"), (5, "c6_exit_pnl_r_5bar")]:
        if t_first_inside is not None and t_first_inside <= n_bars and trigger_bar_close is not None:
            if break_dir == "long":
                exit_pnl_r = (trigger_bar_close - entry_price) / orb_size
            else:
                exit_pnl_r = (entry_price - trigger_bar_close) / orb_size
            result[key] = float(exit_pnl_r)
        else:
            result[key] = None  # C6 didn't fire within this threshold

    # --- C8: Held 30 bars clean, then returned ---
    held_outside_30 = not bool(result.get("c6_reversal_30bar", True))
    result["c8_held_outside_30"] = held_outside_30
    result["c8_held30_then_returned"] = (
        held_outside_30 and t_first_inside is not None and t_first_inside > 30
    )
    # Was the trade still OPEN at bar 30? (target/stop not yet hit)
    result["c8_open_at_bar30"] = (
        bars_until_exit is None or bars_until_exit > 30
    )
    # Did price return inside ORB WHILE the trade was still open? (the Case B problem)
    result["c8_returned_while_open"] = bool(
        held_outside_30
        and t_first_inside is not None
        and t_first_inside > 30
        and (bars_until_exit is None or t_first_inside <= bars_until_exit)
    )

    # --- C9: Maximum favorable excursion (MFE) in R at 10 and 30 bars ---
    for n_bars, key in [(10, "c9_mfe_10bar_r"), (30, "c9_mfe_30bar_r")]:
        window = post_entry.head(n_bars)
        if len(window) > 0 and not pd.isna(entry_price):
            if break_dir == "long":
                mfe = (window["high"].max() - entry_price) / orb_size
            else:
                mfe = (entry_price - window["low"].min()) / orb_size
            result[key] = float(mfe)
        else:
            result[key] = None

    # --- C10: Volume dropoff — break bar volume vs mean of first 5 post-entry bars ---
    break_bar_rows = bars[bars["ts_utc"] == confirm_ts_pd]
    if not break_bar_rows.empty and "volume" in bars.columns:
        break_vol = float(break_bar_rows.iloc[0]["volume"])
        post5_vols = post_entry.head(5)["volume"].dropna()
        if len(post5_vols) >= 3 and break_vol > 0:
            result["c10_vol_ratio"] = float(post5_vols.mean() / break_vol)
        else:
            result["c10_vol_ratio"] = None
    else:
        result["c10_vol_ratio"] = None

    # --- BONUS: Entry bar close distance from ORB (in R) ---
    if break_dir == "long":
        entry_bar_dist = (eb["close"] - orb_high) / orb_size
    else:
        entry_bar_dist = (orb_low - eb["close"]) / orb_size
    result["entry_bar_dist_r"] = float(entry_bar_dist)

    # --- C5 exit simulation: actual pnl_r if you bail when entry bar reverses ---
    # Only fires when entry bar closed against the break direction.
    if not entry_bar_continues:
        if break_dir == "long":
            result["c5_exit_pnl_r"] = float((eb["close"] - entry_price) / orb_size)
        else:
            result["c5_exit_pnl_r"] = float((entry_price - eb["close"]) / orb_size)
    else:
        result["c5_exit_pnl_r"] = None  # no bail — trade continues normally

    # --- C9 bar-30 close: actual pnl_r at bar 30's close (kill-switch exit price) ---
    # Only set if trade is still open at bar 30 (i.e., 30+ post-entry bars exist).
    # If trade resolved (stop/target) before bar 30, this is None and C9 doesn't apply.
    if len(post_entry) >= 30:
        bar30 = post_entry.iloc[29]  # bar 30 after entry (0-indexed)
        if break_dir == "long":
            result["c9_bar30_close_r"] = float((bar30["close"] - entry_price) / orb_size)
        else:
            result["c9_bar30_close_r"] = float((entry_price - bar30["close"]) / orb_size)
    else:
        result["c9_bar30_close_r"] = None  # trade resolved before bar 30

    return result


# =========================================================================
# Reporting helpers
# =========================================================================

def split_report(df: pd.DataFrame, condition_col: str, label_true: str, label_false: str,
                 header: str, min_n: int = 20) -> dict | None:
    """Split by boolean condition and report comparative stats."""
    if condition_col not in df.columns:
        return None

    mask = df[condition_col] == True  # noqa: E712
    group_true = df[mask]
    group_false = df[~mask]

    if len(group_true) < min_n or len(group_false) < min_n:
        return None

    def _grp_stats(g):
        rs = g["pnl_r"]
        n = len(rs)
        return {
            "n": n,
            "avgR": rs.mean(),
            "wr": (rs > 0).mean(),
            "totR": rs.sum(),
            "sharpe": rs.mean() / rs.std() * np.sqrt(252) if rs.std() > 0 else 0,
        }

    st = _grp_stats(group_true)
    sf = _grp_stats(group_false)
    delta_avgr = st["avgR"] - sf["avgR"]

    print(f"\n  {header}")
    print(f"    {'':>25s} {'N':>6s} {'avgR':>8s} {'WR':>7s} {'totR':>9s} {'Sharpe':>8s}")
    print(f"    {'-' * 65}")
    print(f"    {label_true:>25s} {st['n']:6d} {st['avgR']:+8.3f} {st['wr']:6.1%} {st['totR']:+9.1f} {st['sharpe']:8.2f}")
    print(f"    {label_false:>25s} {sf['n']:6d} {sf['avgR']:+8.3f} {sf['wr']:6.1%} {sf['totR']:+9.1f} {sf['sharpe']:8.2f}")
    print(f"    {'Delta':>25s} {'':>6s} {delta_avgr:+8.3f}")

    # t-test (Welch, unequal variance)
    t_stat, p_val = stats.ttest_ind(group_true["pnl_r"], group_false["pnl_r"], equal_var=False)
    sig = ""
    if abs(delta_avgr) > 0.15:
        sig = " << ACTIONABLE" if delta_avgr < -0.15 else " >> ACTIONABLE"
    elif abs(delta_avgr) > 0.08:
        sig = " ~ notable"
    p_flag = " ***" if p_val < 0.01 else (" **" if p_val < 0.05 else (" *" if p_val < 0.10 else ""))
    print(f"    Signal: {abs(delta_avgr):.3f}R separation{sig} | p={p_val:.4f}{p_flag}")

    return {"true": st, "false": sf, "delta": delta_avgr, "p_val": p_val}


def quantile_report(df: pd.DataFrame, condition_col: str, header: str,
                    cuts: list[float] | None = None, min_n: int = 15) -> dict | None:
    """Split continuous variable into quantile buckets and report stats."""
    if condition_col not in df.columns:
        return None

    valid = df[df[condition_col].notna()].copy()
    if len(valid) < min_n * 3:
        return None

    if cuts is None:
        cuts = [0.0, 0.25, 0.50, 0.75, 1.0]

    try:
        valid["bucket"] = pd.qcut(valid[condition_col], q=cuts, duplicates="drop")
    except ValueError:
        return None

    print(f"\n  {header}")
    print(f"    {'Bucket':>25s} {'N':>6s} {'avgR':>8s} {'WR':>7s} {'totR':>9s}")
    print(f"    {'-' * 55}")

    results = {}
    for bucket_label, group in valid.groupby("bucket", observed=True):
        rs = group["pnl_r"]
        n = len(rs)
        if n < min_n:
            continue
        avg = rs.mean()
        wr = (rs > 0).mean()
        tot = rs.sum()
        marker = ""
        if avg > 0.15:
            marker = "  ++"
        elif avg < -0.10:
            marker = "  --"
        print(f"    {str(bucket_label):>25s} {n:6d} {avg:+8.3f} {wr:6.1%} {tot:+9.1f}{marker}")
        results[str(bucket_label)] = {"n": n, "avgR": avg, "wr": wr, "totR": tot}

    return results


def year_breakdown(df: pd.DataFrame, condition_col: str, threshold,
                   label_on: str, label_off: str, header: str, min_n: int = 8) -> dict:
    """Show signal effectiveness year by year. threshold can be a bool (True/False groups)
    or a numeric cutoff (col < threshold = 'on' group)."""
    df = df.copy()
    df["_year"] = pd.to_datetime(df["trading_day"]).dt.year

    if isinstance(threshold, bool):
        df["_on"] = df[condition_col] == threshold
    else:
        df["_on"] = df[condition_col] < threshold

    years = sorted(df["_year"].unique())
    print(f"\n  Year-by-year: {header}")
    print(f"    {'Year':>6s} {'N_on':>5s} {'avgR_on':>9s} {'N_off':>6s} {'avgR_off':>9s} {'Delta':>8s}")
    print(f"    {'-' * 52}")

    year_results = {}
    for yr in years:
        ydf = df[df["_year"] == yr]
        grp_on = ydf[ydf["_on"]]
        grp_off = ydf[~ydf["_on"]]
        if len(grp_on) < min_n or len(grp_off) < min_n:
            continue
        avg_on = grp_on["pnl_r"].mean()
        avg_off = grp_off["pnl_r"].mean()
        delta = avg_on - avg_off
        flag = " **" if abs(delta) > 0.15 else ""
        print(f"    {yr:>6d} {len(grp_on):5d} {avg_on:+9.3f} {len(grp_off):6d} {avg_off:+9.3f} {delta:+8.3f}{flag}")
        year_results[yr] = {"n_on": len(grp_on), "avg_on": avg_on, "n_off": len(grp_off), "avg_off": avg_off, "delta": delta}

    if year_results:
        deltas = [v["delta"] for v in year_results.values()]
        pct_consistent = sum(1 for d in deltas if d > 0) / len(deltas)
        print(f"    Consistent direction: {pct_consistent:.0%} of years ({sum(1 for d in deltas if d > 0)}/{len(deltas)})")

    return year_results


def c3_sensitivity_sweep(df: pd.DataFrame, cutoffs: list[float]) -> None:
    """Test C3 break-speed filter at multiple cutoff thresholds."""
    valid = df[df["c3_break_speed_min"].notna()].copy()
    if len(valid) < 30:
        return

    print(f"\n  C3 Sensitivity Sweep (break speed cutoff):")
    print(f"    {'Cutoff':>10s} {'N_fast':>7s} {'avgR_fast':>10s} {'N_slow':>7s} {'avgR_slow':>10s} {'Delta':>8s} {'p':>7s}")
    print(f"    {'-' * 65}")

    p_vals = []
    rows = []
    for cut in cutoffs:
        fast = valid[valid["c3_break_speed_min"] <= cut]
        slow = valid[valid["c3_break_speed_min"] > cut]
        if len(fast) < 10 or len(slow) < 10:
            continue
        avg_fast = fast["pnl_r"].mean()
        avg_slow = slow["pnl_r"].mean()
        delta = avg_fast - avg_slow
        _, p = stats.ttest_ind(fast["pnl_r"], slow["pnl_r"], equal_var=False)
        p_vals.append(p)
        rows.append((cut, len(fast), avg_fast, len(slow), avg_slow, delta, p))

    if not rows:
        return

    # BH correction across all cutoffs tested
    rejects = _bh_reject(np.array(p_vals))
    for i, (cut, nf, af, ns, as_, delta, p) in enumerate(rows):
        bh = " BH-SIG" if rejects[i] else ""
        p_flag = " ***" if p < 0.01 else (" **" if p < 0.05 else (" *" if p < 0.10 else ""))
        print(f"    {f'<={cut:.0f}min':>10s} {nf:7d} {af:+10.3f} {ns:7d} {as_:+10.3f} {delta:+8.3f} {p:7.4f}{p_flag}{bh}")


def simulate_exit_rules(
    cdf: pd.DataFrame,
    session: str,
    instrument: str,
    c9_mfe_threshold: float = 0.33,
    c3_cutoff: float = 3.0,
) -> pd.Series:
    """Simulate portfolio EV under the four validated exit/filter rules.

    Rule application order (priority):
      1. C3 (1000 only): slow break -> no entry (pnl_r = 0)
      2. C5 (1000 only): entry bar reverses -> exit at entry bar close
      3. C9 (all sessions): MFE at bar 30 < threshold -> exit at bar30 close
      4. C8 (all sessions): held 30 bars clean + losing -> break-even (pnl_r = 0)

    Rules are applied sequentially; a trade modified by an earlier rule
    is excluded from later rules.
    """
    sim = cdf["pnl_r"].copy()
    modified = pd.Series(False, index=cdf.index)  # tracks already-handled trades

    baseline = sim.mean()
    n = len(sim)
    print(f"\n{'=' * 80}")
    print(f"  EXIT RULE SIMULATION — {instrument} {session}")
    print(f"{'=' * 80}")
    print(f"  Baseline: N={n}  EV={baseline:+.3f}R  totR={sim.sum():+.1f}")

    total_delta = 0.0

    # ---- C3: slow break -> skip entirely (1000 only) ----
    if session == "1000" and "c3_break_speed_min" in cdf.columns:
        mask = (~modified) & (cdf["c3_break_speed_min"] > c3_cutoff)
        n_c3 = mask.sum()
        if n_c3 > 0:
            old_ev = cdf.loc[mask, "pnl_r"].mean()
            sim[mask] = 0.0
            modified[mask] = True
            delta = (0.0 - old_ev) * n_c3 / n
            total_delta += delta
            print(f"\n  C3 slow-break skip (>{c3_cutoff}min, 1000 only):")
            print(f"    Skipped N={n_c3} ({n_c3/n:.0%})  old avgR={old_ev:+.3f}R -> 0.000R")
            print(f"    Portfolio delta: {delta:+.3f}R/trade")

    # ---- C5: entry bar reverses -> exit at entry bar close (1000 only) ----
    if session == "1000" and "c5_exit_pnl_r" in cdf.columns:
        mask = (~modified) & (cdf["c5_entry_bar_continues"] == False) & cdf["c5_exit_pnl_r"].notna()
        n_c5 = mask.sum()
        if n_c5 > 0:
            old_ev = cdf.loc[mask, "pnl_r"].mean()
            new_ev = cdf.loc[mask, "c5_exit_pnl_r"].mean()
            sim[mask] = cdf.loc[mask, "c5_exit_pnl_r"]
            modified[mask] = True
            delta = (new_ev - old_ev) * n_c5 / n
            total_delta += delta
            print(f"\n  C5 1-bar bail (entry bar reverses, 1000 only):")
            print(f"    Triggered N={n_c5} ({n_c5/n:.0%})  old avgR={old_ev:+.3f}R -> new {new_ev:+.3f}R")
            print(f"    Exit pnl_r p25={cdf.loc[mask, 'c5_exit_pnl_r'].quantile(0.25):+.3f}  "
                  f"p50={cdf.loc[mask, 'c5_exit_pnl_r'].quantile(0.50):+.3f}  "
                  f"p75={cdf.loc[mask, 'c5_exit_pnl_r'].quantile(0.75):+.3f}")
            print(f"    Portfolio delta: {delta:+.3f}R/trade")

    # ---- C9: MFE at bar 30 < threshold -> exit at bar30 close ----
    if "c9_mfe_30bar_r" in cdf.columns and "c9_bar30_close_r" in cdf.columns:
        mask = (
            (~modified)
            & cdf["c9_mfe_30bar_r"].notna()
            & (cdf["c9_mfe_30bar_r"] < c9_mfe_threshold)
            & cdf["c9_bar30_close_r"].notna()
        )
        n_c9 = mask.sum()
        if n_c9 > 0:
            old_ev = cdf.loc[mask, "pnl_r"].mean()
            new_ev = cdf.loc[mask, "c9_bar30_close_r"].mean()
            sim[mask] = cdf.loc[mask, "c9_bar30_close_r"]
            modified[mask] = True
            delta = (new_ev - old_ev) * n_c9 / n
            total_delta += delta
            pctiles = cdf.loc[mask, "c9_bar30_close_r"].quantile([0.10, 0.50, 0.90])
            print(f"\n  C9 kill-switch (MFE_30 < {c9_mfe_threshold}R -- exit at bar30 close):")
            print(f"    Triggered N={n_c9} ({n_c9/n:.0%})  old avgR={old_ev:+.3f}R -> new {new_ev:+.3f}R")
            print(f"    Bar30 close pnl_r: p10={pctiles[0.10]:+.3f}  p50={pctiles[0.50]:+.3f}  p90={pctiles[0.90]:+.3f}")
            print(f"    Trades where exit < holding: {(cdf.loc[mask, 'c9_bar30_close_r'] < cdf.loc[mask, 'pnl_r']).mean():.0%}")
            print(f"    Portfolio delta: {delta:+.3f}R/trade")

    # ---- C8: held 30 bars clean -> break-even stop at bar 30 ----
    # CORRECTED: applies to ALL trades where the return inside ORB happened while the trade
    # was still open (c8_returned_while_open), regardless of eventual outcome.
    # Also catches trades that held 30 clean, never returned inside ORB, but still lost
    # (i.e., c8_open_at_bar30 == True, pnl_r < 0, NOT returned_while_open).
    if "c8_held_outside_30" in cdf.columns and "c8_returned_while_open" in cdf.columns:
        # Case B: still open at bar 30+, price returns inside ORB -> scratched regardless of final outcome
        mask_caseb = (~modified) & (cdf["c8_returned_while_open"] == True)
        # Case A-loss: still open at bar 30, never returned inside ORB, but lost anyway
        mask_aloss = (
            (~modified)
            & (cdf["c8_held_outside_30"] == True)
            & (cdf["c8_returned_while_open"] == False)
            & (cdf["c8_open_at_bar30"] == True)
            & (cdf["pnl_r"] < 0)
        )
        mask = mask_caseb | mask_aloss
        n_c8 = mask.sum()
        n_caseb = mask_caseb.sum()
        n_aloss = mask_aloss.sum()

        if n_c8 > 0:
            old_ev = cdf.loc[mask, "pnl_r"].mean()
            caseb_winners = (mask_caseb & (cdf["pnl_r"] > 0)).sum()
            caseb_losers = (mask_caseb & (cdf["pnl_r"] <= 0)).sum()
            sim[mask] = 0.0
            modified[mask] = True
            delta = (0.0 - old_ev) * n_c8 / n
            total_delta += delta
            print(f"\n  C8 break-even stop (held 30 bars clean -> scratch at entry):")
            print(f"    Case B (returned while open): N={n_caseb}  "
                  f"({caseb_winners} winners scratched, {caseb_losers} losers saved)")
            print(f"    Case A-loss (slow grind to stop): N={n_aloss}")
            print(f"    Total triggered N={n_c8} ({n_c8/n:.0%})  old avgR={old_ev:+.3f}R -> 0.000R")
            print(f"    Portfolio delta: {delta:+.3f}R/trade")

    new_ev = sim.mean()
    print(f"\n  COMBINED RESULT:")
    print(f"    EV: {baseline:+.3f}R -> {new_ev:+.3f}R  (delta {total_delta:+.3f}R/trade, "
          f"{total_delta/abs(baseline)*100:+.0f}% vs baseline)" if baseline != 0 else
          f"    EV: {baseline:+.3f}R -> {new_ev:+.3f}R  (delta {total_delta:+.3f}R/trade)")
    verdict = "** IMPROVES PORTFOLIO" if total_delta > 0.02 else ("~ NEUTRAL" if total_delta > -0.02 else "** HURTS PORTFOLIO")
    print(f"    Verdict: {verdict}")
    print(f"    Modified trades: {modified.sum()}/{n} ({modified.mean():.0%})")

    return sim


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Break Quality Research")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--instrument", type=str, default=None,
                        help="Single instrument (default: all tradeable)")
    parser.add_argument("--sessions", type=str, default=None,
                        help="Comma-separated sessions (default: 0900,1000,1800)")
    parser.add_argument("--min-orb-size", type=float, default=4.0,
                        help="Min ORB size in R (default: G4+)")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    instruments = [args.instrument] if args.instrument else ["MGC"]
    sessions = args.sessions.split(",") if args.sessions else ["0900", "1000", "1800"]
    min_orb = args.min_orb_size

    print(f"\n{'=' * 100}")
    print(f"  BREAK QUALITY RESEARCH — Post-Break Candle Behavior")
    print(f"  DB: {db_path} | Instruments: {instruments} | Sessions: {sessions}")
    print(f"  Filter: G{int(min_orb)}+ | Entry: E1 | RR: 2.0 | CB: 2")
    print(f"{'=' * 100}")

    con = duckdb.connect(str(db_path), read_only=True)
    t_start = time.time()
    all_conditions = []

    try:
        for instrument in instruments:
            for session in sessions:
                print(f"\n{'=' * 100}")
                print(f"  {instrument} {session}")
                print(f"{'=' * 100}")

                # Load outcomes
                outcomes = load_break_context(con, instrument, session)
                if len(outcomes) == 0:
                    print(f"  No outcomes found.")
                    continue

                # Filter to G4+ (or whatever min_orb is)
                outcomes = outcomes[outcomes["orb_size"] >= min_orb].copy()
                if len(outcomes) < 30:
                    print(f"  Only {len(outcomes)} trades at G{int(min_orb)}+, skipping.")
                    continue

                print(f"  Loading bar data for {len(outcomes)} break-days...")
                t_load = time.time()

                # Extract conditions for each trade
                conditions = []
                for _, row in outcomes.iterrows():
                    bars = load_bars_for_day(con, instrument, row["trading_day"], session)
                    if len(bars) == 0:
                        continue
                    cond = extract_conditions(bars, row.to_dict())
                    if cond is not None:
                        cond["instrument"] = instrument
                        cond["session"] = session
                        conditions.append(cond)

                if len(conditions) < 30:
                    print(f"  Only {len(conditions)} valid conditions extracted, skipping.")
                    continue

                cdf = pd.DataFrame(conditions)
                all_conditions.extend(conditions)
                print(f"  {len(cdf)} trades analyzed in {time.time() - t_load:.1f}s")
                print(f"  Baseline: N={len(cdf)}, avgR={cdf['pnl_r'].mean():+.3f}, "
                      f"WR={( cdf['pnl_r'] > 0).mean():.1%}, totR={cdf['pnl_r'].sum():+.1f}")

                # ================================================
                # C4: Entry bar engulfing (THE KEY QUESTION)
                # ================================================
                split_report(cdf, "c4_entry_bar_engulf",
                             "Entry bar INSIDE ORB", "Entry bar OUTSIDE ORB",
                             "C4: Entry bar engulfs back inside ORB?")

                # ================================================
                # C5: Entry bar direction
                # ================================================
                split_report(cdf, "c5_entry_bar_continues",
                             "Entry bar CONTINUES", "Entry bar REVERSES",
                             "C5: Does entry bar continue in break direction?")

                # ================================================
                # C6: Close inside ORB at 1/3/5/10/15/30 bars
                # ================================================
                for n in [1, 3, 5, 10, 15, 30]:
                    col = f"c6_reversal_{n}bar"
                    split_report(cdf, col,
                                 f"Close inside ORB <={n}bars", f"Stays outside <={n}bars",
                                 f"C6: Any close back inside ORB within {n} bar(s)?")

                # ================================================
                # C7: First close inside ORB — time distribution
                # ================================================
                never = cdf["c7_first_inside_bar"].isna()
                returned = cdf[~never]
                stayed = cdf[never]
                if len(returned) >= 10 and len(stayed) >= 10:
                    print(f"\n  C7: When does price first close back inside ORB?")
                    print(f"    Never returned:   N={len(stayed):4d}  avgR={stayed['pnl_r'].mean():+.3f}  "
                          f"WR={( stayed['pnl_r'] > 0).mean():.1%}")
                    print(f"    Returned (any):   N={len(returned):4d}  avgR={returned['pnl_r'].mean():+.3f}  "
                          f"WR={( returned['pnl_r'] > 0).mean():.1%}")
                    print(f"    Delta (never-returned): {stayed['pnl_r'].mean() - returned['pnl_r'].mean():+.3f}")
                    print(f"\n    Outcome by when price first re-entered ORB:")
                    print(f"    {'First re-entry':>20s} {'N':>5s} {'avgR':>8s} {'WR':>7s}")
                    print(f"    {'-' * 45}")
                    buckets = [(1, 3), (4, 10), (11, 30), (31, 60), (61, 999)]
                    for lo, hi in buckets:
                        grp = cdf[cdf["c7_first_inside_bar"].between(lo, hi)]
                        if len(grp) >= 5:
                            label = f"bar {lo}-{hi}"
                            print(f"    {label:>20s} {len(grp):5d} {grp['pnl_r'].mean():+8.3f} "
                                  f"{( grp['pnl_r'] > 0).mean():6.1%}")

                # ================================================
                # C6 EXIT SIMULATION: Actual pnl_r at trigger bar close
                # ================================================
                for n_bars, col in [(3, "c6_exit_pnl_r_3bar"), (5, "c6_exit_pnl_r_5bar")]:
                    triggered = cdf[cdf[col].notna()].copy()
                    not_triggered = cdf[cdf[col].isna()].copy()
                    if len(triggered) < 10:
                        continue

                    exit_r = triggered[col]
                    hold_r = triggered["pnl_r"]  # what actually happened if you held
                    improvement = exit_r - hold_r  # positive = early exit was better

                    pct_better = (improvement > 0).mean()
                    avg_exit = exit_r.mean()
                    avg_hold = hold_r.mean()
                    avg_improvement = improvement.mean()

                    print(f"\n  C6 EXIT SIMULATION (trigger within <={n_bars} bars):")
                    print(f"    Triggered trades:     N={len(triggered)}")
                    print(f"    Not triggered:        N={len(not_triggered)}")
                    print(f"\n    If you EXIT at trigger bar close:")
                    print(f"      Avg exit pnl_r:     {avg_exit:+.3f}R")
                    print(f"      Avg hold pnl_r:     {avg_hold:+.3f}R  (actual outcome)")
                    print(f"      Avg improvement:    {avg_improvement:+.3f}R  (per triggered trade)")
                    print(f"      % trades exit beats hold: {pct_better:.0%}")

                    # Distribution of exit prices
                    pctiles = exit_r.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
                    print(f"\n    Exit pnl_r distribution (actual close price):")
                    print(f"      p10={pctiles[0.10]:+.3f}  p25={pctiles[0.25]:+.3f}  "
                          f"p50={pctiles[0.50]:+.3f}  p75={pctiles[0.75]:+.3f}  p90={pctiles[0.90]:+.3f}")

                    # Breakdown: exit < -0.5R, -0.5 to 0, 0 to +0.5, > +0.5
                    buckets = [
                        (exit_r < -0.5,  "Exit < -0.5R  (big loss, worse than stop)"),
                        ((exit_r >= -0.5) & (exit_r < 0), "Exit -0.5R to 0  (small loss)"),
                        ((exit_r >= 0) & (exit_r < 0.5),  "Exit 0 to +0.5R (small gain)"),
                        (exit_r >= 0.5,  "Exit > +0.5R  (good exit)"),
                    ]
                    print(f"\n    Exit price breakdown:")
                    for mask, label in buckets:
                        grp = triggered[mask]
                        if len(grp) > 0:
                            print(f"      {label}: N={len(grp):3d} ({len(grp)/len(triggered):.0%}), "
                                  f"avg_exit={grp[col].mean():+.3f}R, avg_hold={grp['pnl_r'].mean():+.3f}R")

                    # The real net math: does exiting help?
                    # Expected value if you always exit on trigger vs never
                    # Triggered group: avg(exit) vs avg(hold)
                    # Not-triggered group: held to outcome regardless
                    # Full portfolio EV comparison:
                    n_total = len(cdf)
                    if n_total > 0:
                        ev_always_exit = (len(triggered) * avg_exit + len(not_triggered) * not_triggered["pnl_r"].mean()) / n_total if len(not_triggered) > 0 else avg_exit
                        ev_never_exit = cdf["pnl_r"].mean()
                        print(f"\n    Portfolio EV comparison (full {n_total} trades):")
                        print(f"      EV (never exit early):   {ev_never_exit:+.3f}R")
                        print(f"      EV (always exit on C6):  {ev_always_exit:+.3f}R")
                        net = ev_always_exit - ev_never_exit
                        verdict = " ** EXIT HELPS" if net > 0.03 else (" ** EXIT HURTS" if net < -0.03 else " ~ NEUTRAL")
                        print(f"      Net effect:              {net:+.3f}R per trade{verdict}")

                # ================================================
                # C8: Held 30 clean, then returned — the key hypothesis
                # ================================================
                split_report(cdf, "c8_held_outside_30",
                             "Clean outside 30 bars", "Reversed within 30 bars",
                             "C8: Held cleanly outside ORB for 30+ bars?")

                # Within the "held 30 clean" group, split by whether it eventually returned
                held30 = cdf[cdf["c8_held_outside_30"] == True]
                if len(held30) >= 15:
                    then_returned = held30[held30["c8_held30_then_returned"] == True]
                    never_returned = held30[held30["c8_held30_then_returned"] == False]
                    print(f"\n  C8b: Within 'held 30 clean' — did it EVENTUALLY return inside?")
                    print(f"    {'':>28s} {'N':>5s} {'avgR':>8s} {'WR':>7s}")
                    print(f"    {'-' * 50}")
                    if len(then_returned) >= 5:
                        print(f"    {'Held 30 then returned inside':>28s} {len(then_returned):5d} "
                              f"{then_returned['pnl_r'].mean():+8.3f} "
                              f"{( then_returned['pnl_r'] > 0).mean():6.1%}")
                    if len(never_returned) >= 5:
                        print(f"    {'Held 30, never returned':>28s} {len(never_returned):5d} "
                              f"{never_returned['pnl_r'].mean():+8.3f} "
                              f"{( never_returned['pnl_r'] > 0).mean():6.1%}")
                    if len(then_returned) >= 5 and len(never_returned) >= 5:
                        delta = never_returned["pnl_r"].mean() - then_returned["pnl_r"].mean()
                        flag = " ** ACTIONABLE" if abs(delta) > 0.15 else ""
                        print(f"    Delta (never-returned minus returned): {delta:+.3f}{flag}")

                # C8c: Case B analysis — did the return happen while trade was still open?
                if "c8_returned_while_open" in cdf.columns:
                    caseb = cdf[cdf["c8_returned_while_open"] == True]
                    if len(caseb) >= 5:
                        caseb_win = caseb[caseb["pnl_r"] > 0]
                        caseb_loss = caseb[caseb["pnl_r"] <= 0]
                        print(f"\n  C8c: Case B — return inside ORB happened WHILE TRADE STILL OPEN")
                        print(f"    Total Case B: N={len(caseb)} ({len(caseb)/len(cdf):.0%} of all trades)")
                        print(f"    Winners (C8 would scratch): N={len(caseb_win)}, avgR={caseb_win['pnl_r'].mean():+.3f}" if len(caseb_win) > 0 else f"    Winners: N=0")
                        print(f"    Losers  (C8 would save):   N={len(caseb_loss)}, avgR={caseb_loss['pnl_r'].mean():+.3f}" if len(caseb_loss) > 0 else f"    Losers:  N=0")
                        net = (len(caseb_loss) * abs(caseb_loss["pnl_r"].mean()) - len(caseb_win) * caseb_win["pnl_r"].mean()) / len(cdf) if len(caseb) > 0 else 0
                        print(f"    Net C8 Case B impact: {net:+.3f}R/trade (positive = C8 helps on this group)")

                # ================================================
                # C9: MFE context — how far did price get before reversal?
                # ================================================
                quantile_report(cdf, "c9_mfe_10bar_r",
                                "C9: Max favorable excursion in first 10 bars (R)")
                quantile_report(cdf, "c9_mfe_30bar_r",
                                "C9: Max favorable excursion in first 30 bars (R)")

                # ================================================
                # C10: Volume dropoff after breakout
                # ================================================
                quantile_report(cdf, "c10_vol_ratio",
                                "C10: Volume ratio (post-entry mean / break bar) — dropoff = <1.0",
                                cuts=[0.0, 0.25, 0.50, 0.75, 1.0])

                # ================================================
                # C1: Break distance (quantiles)
                # ================================================
                quantile_report(cdf, "c1_break_distance_r",
                                "C1: Confirm bar break distance (in R)")

                # ================================================
                # C2: Wick ratio (quantiles)
                # ================================================
                quantile_report(cdf, "c2_wick_ratio",
                                "C2: Confirm bar wick-back ratio")

                # ================================================
                # C3: Break speed (quantiles)
                # ================================================
                quantile_report(cdf, "c3_break_speed_min",
                                "C3: Break speed (minutes to confirm)")

                # ================================================
                # BONUS: Entry bar distance from ORB
                # ================================================
                quantile_report(cdf, "entry_bar_dist_r",
                                "BONUS: Entry bar close distance from ORB (in R)")

                # ================================================
                # VALIDITY TESTING — p-values, year-by-year, BH, sensitivity
                # ================================================
                print(f"\n{'=' * 100}")
                print(f"  VALIDITY TESTING — {instrument} {session}")
                print(f"{'=' * 100}")

                # Collect all main signal p-values for BH correction
                bh_tests = []

                # --- C3: Break speed pre-entry filter ---
                c3_valid = cdf[cdf["c3_break_speed_min"].notna()].copy()
                if len(c3_valid) >= 30:
                    fast = c3_valid[c3_valid["c3_break_speed_min"] <= 3]
                    slow = c3_valid[c3_valid["c3_break_speed_min"] > 3]
                    if len(fast) >= 10 and len(slow) >= 10:
                        _, p_c3 = stats.ttest_ind(fast["pnl_r"], slow["pnl_r"], equal_var=False)
                        delta_c3 = fast["pnl_r"].mean() - slow["pnl_r"].mean()
                        bh_tests.append(("C3 speed<3min (pre-entry)", p_c3, delta_c3, "pre-entry"))
                        print(f"\n  [C3] Pre-entry: Break speed < 3 min")
                        print(f"    Fast (<=3m): N={len(fast)}, avgR={fast['pnl_r'].mean():+.3f}")
                        print(f"    Slow (>3m):  N={len(slow)}, avgR={slow['pnl_r'].mean():+.3f}")
                        print(f"    Delta: {delta_c3:+.3f}R | p={p_c3:.4f}")
                        c3_sensitivity_sweep(cdf, cutoffs=[1, 2, 3, 5, 7, 10])
                        year_breakdown(c3_valid, "c3_break_speed_min", 3.0,
                                       "Fast (<= 3m)", "Slow (> 3m)", "C3 break speed < 3 min")

                # --- C5: Entry bar direction (post-fill, 1-bar exit signal) ---
                r5 = split_report(cdf, "c5_entry_bar_continues",
                                  "Continues", "Reverses",
                                  "C5 (re-test p-value): Entry bar direction")
                if r5 is not None:
                    bh_tests.append(("C5 entry-bar direction", r5["p_val"], r5["delta"], "post-fill 1-bar"))
                    year_breakdown(cdf, "c5_entry_bar_continues", True,
                                   "Continues", "Reverses", "C5 entry bar direction (True = continues)")

                # --- C8: Held 30 bars clean (post-hoc classifier) ---
                r8 = split_report(cdf, "c8_held_outside_30",
                                  "Held 30 clean", "Reversed <30",
                                  "C8 (re-test p-value): Held 30 bars outside ORB")
                if r8 is not None:
                    bh_tests.append(("C8 held 30 bars clean", r8["p_val"], r8["delta"], "post-hoc (30bar look-ahead)"))
                    year_breakdown(cdf, "c8_held_outside_30", True,
                                   "Held 30 clean", "Reversed <30", "C8 held 30 bars clean")

                # --- C9: MFE at 30 bars (post-entry mid-trade) ---
                c9_valid = cdf[cdf["c9_mfe_30bar_r"].notna()].copy()
                if len(c9_valid) >= 40:
                    q25_mfe = c9_valid["c9_mfe_30bar_r"].quantile(0.25)
                    bottom_q = c9_valid[c9_valid["c9_mfe_30bar_r"] <= q25_mfe]
                    top_3q = c9_valid[c9_valid["c9_mfe_30bar_r"] > q25_mfe]
                    if len(bottom_q) >= 10 and len(top_3q) >= 10:
                        _, p_c9 = stats.ttest_ind(bottom_q["pnl_r"], top_3q["pnl_r"], equal_var=False)
                        delta_c9 = bottom_q["pnl_r"].mean() - top_3q["pnl_r"].mean()
                        bh_tests.append(("C9 MFE<30bar p25 (30-bar look-ahead)", p_c9, delta_c9, "post-entry 30-bar look-ahead"))
                        print(f"\n  [C9] MFE at 30 bars: bottom quartile (cutoff={q25_mfe:.3f}R)")
                        print(f"    Bottom Q (MFE <= {q25_mfe:.3f}R): N={len(bottom_q)}, avgR={bottom_q['pnl_r'].mean():+.3f}, WR={( bottom_q['pnl_r'] > 0).mean():.0%}")
                        print(f"    Top 3Q  (MFE >  {q25_mfe:.3f}R): N={len(top_3q)},  avgR={top_3q['pnl_r'].mean():+.3f}, WR={( top_3q['pnl_r'] > 0).mean():.0%}")
                        print(f"    Delta: {delta_c9:+.3f}R | p={p_c9:.4f}")
                        # Sensitivity: p50 cutoff
                        q50_mfe = c9_valid["c9_mfe_30bar_r"].quantile(0.50)
                        bottom_half = c9_valid[c9_valid["c9_mfe_30bar_r"] <= q50_mfe]
                        top_half = c9_valid[c9_valid["c9_mfe_30bar_r"] > q50_mfe]
                        if len(bottom_half) >= 10 and len(top_half) >= 10:
                            _, p50 = stats.ttest_ind(bottom_half["pnl_r"], top_half["pnl_r"], equal_var=False)
                            print(f"    Sensitivity p50 cutoff ({q50_mfe:.3f}R): delta={bottom_half['pnl_r'].mean() - top_half['pnl_r'].mean():+.3f}R | p={p50:.4f}")
                        year_breakdown(c9_valid, "c9_mfe_30bar_r", q25_mfe,
                                       f"Low MFE (<={q25_mfe:.2f}R)", "Higher MFE", "C9 MFE<30bar bottom quartile")

                # --- BH correction across all 4 signals ---
                if bh_tests:
                    p_arr = np.array([t[1] for t in bh_tests])
                    rejects = _bh_reject(p_arr)
                    print(f"\n  BH CORRECTION SUMMARY ({instrument} {session}) — {len(bh_tests)} tests, alpha=0.05:")
                    print(f"    {'Signal':>45s} {'delta':>8s} {'p_raw':>8s} {'BH-pass':>8s} {'Type'}")
                    print(f"    {'-' * 90}")
                    for i, (label, p, delta, kind) in enumerate(bh_tests):
                        bh_pass = "YES ***" if rejects[i] else "no"
                        p_flag = " **" if p < 0.05 else (" *" if p < 0.10 else "")
                        print(f"    {label:>45s} {delta:+8.3f} {p:8.4f}{p_flag}  {bh_pass:>8s}  {kind}")

                # ================================================
                # SIMULATION: combined exit rule impact on portfolio EV
                # ================================================
                simulate_exit_rules(cdf, session, instrument)

        # Save all conditions
        if all_conditions:
            out_dir = Path("research/output")
            out_dir.mkdir(parents=True, exist_ok=True)
            all_df = pd.DataFrame(all_conditions)
            all_df.to_csv(out_dir / "break_quality_conditions.csv",
                          index=False, float_format="%.4f")
            print(f"\n  CSV saved: research/output/break_quality_conditions.csv")

    finally:
        con.close()

    print(f"\n  Total: {time.time() - t_start:.1f}s")
    print(f"\n{'=' * 100}")
    print(f"  INTERPRETATION GUIDE")
    print(f"{'=' * 100}")
    print("""
  C4 (Entry bar engulfing):
    If "Entry bar INSIDE ORB" has significantly worse avgR -- exit immediately
    when entry bar closes back inside. This is the engulfing question.

  C5 (Entry bar direction):
    If "Entry bar REVERSES" has significantly worse avgR -- the break had
    no follow-through momentum. Consider exiting on reversal close.

  C6 (Quick reversal):
    Tests how quickly "coming back inside ORB" kills the trade.
    1-bar = immediate, 3-bar = within first few minutes, 5-bar = slightly delayed.

  C1 (Break distance):
    Higher distance = stronger conviction break. If bottom quartile has
    much worse avgR -- skip weak breaks (close barely outside ORB).

  C2 (Wick ratio):
    High wick = rejection. If top quartile (most wick) has worse avgR --
    rejection wicks are negative signal. Could be pre-entry filter.

  C3 (Break speed):
    Fast breaks (low minutes) = momentum. Slow breaks (high minutes) = grind.
    If speed quartiles differ -- momentum quality matters.

  ACTIONABLE thresholds:
    |delta| > 0.15R = strong signal, worth implementing
    |delta| > 0.08R = notable but needs more data
    |delta| < 0.08R = noise, don't filter
""")


if __name__ == "__main__":
    main()
