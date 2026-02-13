#!/usr/bin/env python3
"""
Honest IB Exploit Backtest -- 3-Part Strategy.

Tests the IB-aligned directional day strategy with zero lookahead:
  1. DEFENSIVE (Single Break Kill): opposed IB break -> exit at market
  2. OFFENSIVE (Target Unlock): aligned IB break -> cancel target, hold 7h
  3. AGGRESSIVE (Turtle Pyramid): aligned IB break -> add second unit

Bar-by-bar simulation. Only acts on information available at each bar.

Read-only. No DB writes.

Usage:
    python scripts/analyze_trend_holding.py --db-path C:/db/gold.db
"""

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec, to_r_multiple
from scripts._alt_strategy_utils import compute_strategy_metrics

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config -- FIXED
# ---------------------------------------------------------------------------
MIN_ORB_SIZE = 4.0
RR_TARGET = 2.0          # limbo-phase fixed target
CONFIRM_BARS = 2
HOLD_HOURS = 7
SESSIONS = ["0900", "1000"]

SESSION_UTC = {"0900": 23, "1000": 0}
MARKET_OPEN_UTC_HOUR = 23

# Best IB config per session (from directional day analysis)
SESSION_IB_CONFIG = {
    "0900": ("session", 120),   # session_120m == mktopen_120m for 0900
    "1000": ("mktopen", 120),   # mktopen_120m best separation for 1000
}

SPEC = get_cost_spec("MGC")


# ---------------------------------------------------------------------------
# IB computation
# ---------------------------------------------------------------------------

def compute_ib(ts, highs, lows, anchor_utc_hour, duration_minutes):
    """Compute IB from numpy arrays."""
    hours = np.array([t.hour for t in ts])
    minutes = np.array([t.minute for t in ts])
    anchor_idx = np.flatnonzero((hours == anchor_utc_hour) & (minutes == 0))
    if len(anchor_idx) == 0:
        return None
    ib_start = ts[anchor_idx[0]]
    ib_end = ib_start + timedelta(minutes=duration_minutes)
    ib_mask = (ts >= ib_start) & (ts < ib_end)
    if ib_mask.sum() < max(10, duration_minutes // 12):
        return None
    return {
        "ib_high": float(highs[ib_mask].max()),
        "ib_low": float(lows[ib_mask].min()),
        "ib_end": ib_end,
    }


def find_ib_break(ts, highs, lows, ib):
    """Find first IB break after ib_end. Returns (direction, break_ts, break_idx)."""
    post_idx = np.flatnonzero(ts >= ib["ib_end"])
    for i in post_idx:
        bh = highs[i] > ib["ib_high"]
        bl = lows[i] < ib["ib_low"]
        if bh and bl:
            return None, ts[i], i
        if bh:
            return "long", ts[i], i
        if bl:
            return "short", ts[i], i
    return None, None, None


# ---------------------------------------------------------------------------
# Strategy simulations
# ---------------------------------------------------------------------------

def sim_fixed_target(ts, highs, lows, closes, entry_idx, entry_price,
                     stop_price, target_price, is_long, cutoff_ts):
    """Plain fixed-target strategy (control). Returns (pnl_r, exit_reason)."""
    for i in range(entry_idx, len(ts)):
        h, lo, c = highs[i], lows[i], closes[i]
        # Stop (priority)
        if is_long and lo <= stop_price:
            return to_r_multiple(SPEC, entry_price, stop_price,
                                 stop_price - entry_price), "stop"
        if not is_long and h >= stop_price:
            return to_r_multiple(SPEC, entry_price, stop_price,
                                 entry_price - stop_price), "stop"
        # Target
        if is_long and h >= target_price:
            return to_r_multiple(SPEC, entry_price, stop_price,
                                 target_price - entry_price), "target"
        if not is_long and lo <= target_price:
            return to_r_multiple(SPEC, entry_price, stop_price,
                                 entry_price - target_price), "target"
        # Time
        if ts[i] >= cutoff_ts:
            pnl = (c - entry_price) if is_long else (entry_price - c)
            return to_r_multiple(SPEC, entry_price, stop_price, pnl), "time"
    # End of data
    c = closes[-1]
    pnl = (c - entry_price) if is_long else (entry_price - c)
    return to_r_multiple(SPEC, entry_price, stop_price, pnl), "eod"


def sim_exploit(ts, highs, lows, closes, entry_idx, entry_price,
                stop_price, target_price, is_long, cutoff_ts,
                ib_break_dir, ib_break_idx, orb_dir):
    """3-part exploit (no pyramid). Returns (pnl_r, exit_reason, phase)."""
    alignment_known = False
    alignment = None

    for i in range(entry_idx, len(ts)):
        h, lo, c = highs[i], lows[i], closes[i]

        # --- Stop always active ---
        if is_long and lo <= stop_price:
            phase = "limbo" if not alignment_known else alignment
            return (to_r_multiple(SPEC, entry_price, stop_price,
                                  stop_price - entry_price),
                    "stop", phase)
        if not is_long and h >= stop_price:
            phase = "limbo" if not alignment_known else alignment
            return (to_r_multiple(SPEC, entry_price, stop_price,
                                  entry_price - stop_price),
                    "stop", phase)

        # --- Limbo phase ---
        if not alignment_known:
            # Fixed target active
            if is_long and h >= target_price:
                return (to_r_multiple(SPEC, entry_price, stop_price,
                                      target_price - entry_price),
                        "limbo_target", "limbo")
            if not is_long and lo <= target_price:
                return (to_r_multiple(SPEC, entry_price, stop_price,
                                      entry_price - target_price),
                        "limbo_target", "limbo")

            # Check if IB breaks on this bar
            if ib_break_idx is not None and i >= ib_break_idx:
                alignment_known = True
                if ib_break_dir is None:
                    alignment = "no_break"
                elif ib_break_dir == orb_dir:
                    alignment = "aligned"
                else:
                    alignment = "opposed"

                # DEFENSIVE: opposed -> exit at market NOW
                if alignment == "opposed":
                    pnl = (c - entry_price) if is_long else (entry_price - c)
                    return (to_r_multiple(SPEC, entry_price, stop_price, pnl),
                            "opposed_kill", "opposed")
                # aligned/no_break: fall through, target removed on next bar
                continue

            # Time exit during limbo
            if ts[i] >= cutoff_ts:
                pnl = (c - entry_price) if is_long else (entry_price - c)
                return (to_r_multiple(SPEC, entry_price, stop_price, pnl),
                        "limbo_time", "limbo")
            continue

        # --- Post-alignment: aligned or no_break ---
        if alignment == "aligned":
            # OFFENSIVE: no target, hold with stop until 7h
            if ts[i] >= cutoff_ts:
                pnl = (c - entry_price) if is_long else (entry_price - c)
                return (to_r_multiple(SPEC, entry_price, stop_price, pnl),
                        "time_7h", "aligned")
            continue

        if alignment == "no_break":
            # Keep fixed target
            if is_long and h >= target_price:
                return (to_r_multiple(SPEC, entry_price, stop_price,
                                      target_price - entry_price),
                        "nobreak_target", "no_break")
            if not is_long and lo <= target_price:
                return (to_r_multiple(SPEC, entry_price, stop_price,
                                      entry_price - target_price),
                        "nobreak_target", "no_break")
            if ts[i] >= cutoff_ts:
                pnl = (c - entry_price) if is_long else (entry_price - c)
                return (to_r_multiple(SPEC, entry_price, stop_price, pnl),
                        "nobreak_time", "no_break")
            continue

    c = closes[-1]
    pnl = (c - entry_price) if is_long else (entry_price - c)
    phase = "limbo" if not alignment_known else (alignment or "unknown")
    return to_r_multiple(SPEC, entry_price, stop_price, pnl), "eod", phase


def sim_pyramid(ts, highs, lows, closes, entry_idx, entry_price,
                stop_price, target_price, is_long, cutoff_ts,
                ib_break_dir, ib_break_idx, orb_dir):
    """3-part exploit WITH pyramid on aligned break.

    Returns (total_pnl_r, exit_reason, phase, units_deployed).
    pnl_r is in units of ORIGINAL risk (1R = original stop distance).
    """
    alignment_known = False
    alignment = None
    pyramided = False
    unit2_entry = None
    original_risk = abs(entry_price - stop_price)

    for i in range(entry_idx, len(ts)):
        h, lo, c = highs[i], lows[i], closes[i]

        # --- Stop check (applies to current stop level) ---
        current_stop = stop_price  # until pyramid moves it
        if pyramided:
            current_stop = entry_price  # moved to breakeven

        if is_long and lo <= current_stop:
            pnl1 = to_r_multiple(SPEC, entry_price, stop_price,
                                 current_stop - entry_price)
            pnl2 = 0.0
            if pyramided:
                # Unit 2 also stops at original entry (its own stop)
                pnl2_pts = current_stop - unit2_entry
                pnl2 = (pnl2_pts * SPEC.point_value - SPEC.total_friction) / (
                    original_risk * SPEC.point_value + SPEC.total_friction)
            phase = "limbo" if not alignment_known else alignment
            return pnl1 + pnl2, "stop", phase, 2 if pyramided else 1

        if not is_long and h >= current_stop:
            pnl1 = to_r_multiple(SPEC, entry_price, stop_price,
                                 entry_price - current_stop)
            pnl2 = 0.0
            if pyramided:
                pnl2_pts = unit2_entry - current_stop
                pnl2 = (pnl2_pts * SPEC.point_value - SPEC.total_friction) / (
                    original_risk * SPEC.point_value + SPEC.total_friction)
            phase = "limbo" if not alignment_known else alignment
            return pnl1 + pnl2, "stop", phase, 2 if pyramided else 1

        # --- Limbo phase ---
        if not alignment_known:
            if is_long and h >= target_price:
                return (to_r_multiple(SPEC, entry_price, stop_price,
                                      target_price - entry_price),
                        "limbo_target", "limbo", 1)
            if not is_long and lo <= target_price:
                return (to_r_multiple(SPEC, entry_price, stop_price,
                                      entry_price - target_price),
                        "limbo_target", "limbo", 1)

            if ib_break_idx is not None and i >= ib_break_idx:
                alignment_known = True
                if ib_break_dir is None:
                    alignment = "no_break"
                elif ib_break_dir == orb_dir:
                    alignment = "aligned"
                    # PYRAMID: add unit 2 at this bar's close, move stop to BE
                    pyramided = True
                    unit2_entry = c
                else:
                    alignment = "opposed"
                    pnl = (c - entry_price) if is_long else (entry_price - c)
                    return (to_r_multiple(SPEC, entry_price, stop_price, pnl),
                            "opposed_kill", "opposed", 1)
                continue

            if ts[i] >= cutoff_ts:
                pnl = (c - entry_price) if is_long else (entry_price - c)
                return (to_r_multiple(SPEC, entry_price, stop_price, pnl),
                        "limbo_time", "limbo", 1)
            continue

        # --- Post-alignment ---
        if alignment == "aligned":
            if ts[i] >= cutoff_ts:
                pnl1_pts = (c - entry_price) if is_long else (entry_price - c)
                pnl1 = to_r_multiple(SPEC, entry_price, stop_price, pnl1_pts)
                pnl2 = 0.0
                if pyramided:
                    pnl2_pts = (c - unit2_entry) if is_long else (unit2_entry - c)
                    pnl2 = (pnl2_pts * SPEC.point_value - SPEC.total_friction) / (
                        original_risk * SPEC.point_value + SPEC.total_friction)
                return pnl1 + pnl2, "time_7h", "aligned", 2 if pyramided else 1
            continue

        if alignment == "no_break":
            if is_long and h >= target_price:
                return (to_r_multiple(SPEC, entry_price, stop_price,
                                      target_price - entry_price),
                        "nobreak_target", "no_break", 1)
            if not is_long and lo <= target_price:
                return (to_r_multiple(SPEC, entry_price, stop_price,
                                      entry_price - target_price),
                        "nobreak_target", "no_break", 1)
            if ts[i] >= cutoff_ts:
                pnl = (c - entry_price) if is_long else (entry_price - c)
                return (to_r_multiple(SPEC, entry_price, stop_price, pnl),
                        "nobreak_time", "no_break", 1)
            continue

    c = closes[-1]
    pnl1_pts = (c - entry_price) if is_long else (entry_price - c)
    pnl1 = to_r_multiple(SPEC, entry_price, stop_price, pnl1_pts)
    pnl2 = 0.0
    if pyramided:
        pnl2_pts = (c - unit2_entry) if is_long else (unit2_entry - c)
        pnl2 = (pnl2_pts * SPEC.point_value - SPEC.total_friction) / (
            original_risk * SPEC.point_value + SPEC.total_friction)
    phase = "limbo" if not alignment_known else (alignment or "unknown")
    return pnl1 + pnl2, "eod", phase, 2 if pyramided else 1


# ---------------------------------------------------------------------------
# Process session
# ---------------------------------------------------------------------------

def process_session(db_path, session_label, start, end):
    session_utc_hour = SESSION_UTC[session_label]
    anchor, duration = SESSION_IB_CONFIG[session_label]
    anchor_hour = session_utc_hour if anchor == "session" else MARKET_OPEN_UTC_HOUR

    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(f"""
        SELECT o.trading_day, o.entry_ts, o.entry_price, o.stop_price,
               o.target_price, o.pnl_r,
               d.orb_{session_label}_break_dir
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
        WHERE o.symbol = 'MGC' AND o.orb_minutes = 5
          AND o.orb_label = ? AND o.entry_model = 'E1'
          AND o.rr_target = ? AND o.confirm_bars = ?
          AND o.entry_ts IS NOT NULL AND o.outcome IS NOT NULL
          AND o.pnl_r IS NOT NULL AND d.orb_{session_label}_size >= ?
          AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day
    """, [session_label, RR_TARGET, CONFIRM_BARS, MIN_ORB_SIZE, start, end]).fetchdf()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)

    # Bulk-load bars
    unique_days = sorted(df["trading_day"].unique())
    bars_cache = {}
    for td in unique_days:
        s, e = compute_trading_day_utc_range(td)
        b = con.execute(
            "SELECT ts_utc, high, low, close FROM bars_1m "
            "WHERE symbol='MGC' AND ts_utc>=? AND ts_utc<? ORDER BY ts_utc",
            [s, e],
        ).fetchdf()
        if not b.empty:
            b["ts_utc"] = pd.to_datetime(b["ts_utc"], utc=True)
            ts_raw = b["ts_utc"].values.astype("datetime64[ms]")
            ts_py = np.array([pd.Timestamp(t).to_pydatetime().replace(tzinfo=None)
                              for t in ts_raw])
            bars_cache[td] = (
                ts_py, ts_raw,
                b["high"].values.astype(np.float64),
                b["low"].values.astype(np.float64),
                b["close"].values.astype(np.float64),
            )
    con.close()

    results = []
    for _, row in df.iterrows():
        td = row["trading_day"]
        if td not in bars_cache:
            continue
        ts_py, ts_raw, h_arr, l_arr, c_arr = bars_cache[td]

        entry_ts_aware = row["entry_ts"].to_pydatetime()
        entry_ts = entry_ts_aware.replace(tzinfo=None)
        entry_p = float(row["entry_price"])
        stop_p = float(row["stop_price"])
        target_p = float(row["target_price"])
        orb_dir = row[f"orb_{session_label}_break_dir"]
        is_long = orb_dir == "long"
        stored_pnl = float(row["pnl_r"])

        entry_idx = int(np.searchsorted(ts_raw, np.datetime64(entry_ts_aware, "ms")))
        if entry_idx >= len(ts_py):
            continue

        cutoff = entry_ts + timedelta(hours=HOLD_HOURS)

        # Compute IB
        ib = compute_ib(ts_py, h_arr, l_arr, anchor_hour, duration)
        if ib is None:
            continue
        ib_dir, ib_break_ts, ib_break_idx = find_ib_break(ts_py, h_arr, l_arr, ib)

        # 1. Control: fixed target (stored outcome)
        ctrl_pnl = stored_pnl

        # 2. Exploit (no pyramid)
        exploit_pnl, exploit_reason, exploit_phase = sim_exploit(
            ts_py, h_arr, l_arr, c_arr, entry_idx, entry_p, stop_p,
            target_p, is_long, cutoff, ib_dir, ib_break_idx, orb_dir)

        # 3. Pyramid
        pyr_pnl, pyr_reason, pyr_phase, pyr_units = sim_pyramid(
            ts_py, h_arr, l_arr, c_arr, entry_idx, entry_p, stop_p,
            target_p, is_long, cutoff, ib_dir, ib_break_idx, orb_dir)

        # Determine true alignment
        if ib_dir is None:
            alignment = "no_break"
        elif ib_dir == orb_dir:
            alignment = "aligned"
        else:
            alignment = "opposed"

        year = str(td.year) if hasattr(td, "year") else str(td)[:4]

        results.append({
            "td": td, "year": year, "alignment": alignment,
            "ctrl_pnl": ctrl_pnl,
            "exploit_pnl": exploit_pnl,
            "exploit_reason": exploit_reason,
            "exploit_phase": exploit_phase,
            "pyramid_pnl": pyr_pnl,
            "pyramid_reason": pyr_reason,
            "pyramid_units": pyr_units,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_row(label, m):
    return (f"  {label:22s} {m['n']:>5d} {m['wr']:>7.1%} {m['expr']:>+8.4f} "
            f"{m['sharpe']:>8.4f} {m['maxdd']:>8.2f} {m['total']:>+8.1f}")


def print_report(pdf, session_label):
    if len(pdf) == 0:
        print(f"  No trades for {session_label}")
        return

    n = len(pdf)
    anchor, dur = SESSION_IB_CONFIG[session_label]
    ib_name = f"{anchor}_{dur}m"

    print(f"\n{'=' * 90}")
    print(f"{session_label} SESSION  |  IB: {ib_name}  |  N={n}")
    print(f"E1 CB{CONFIRM_BARS} RR{RR_TARGET} G{MIN_ORB_SIZE}+  |  Limbo target={RR_TARGET}R, Hold={HOLD_HOURS}h")
    print(f"{'=' * 90}")

    # Alignment distribution
    for a in ["aligned", "opposed", "no_break"]:
        na = len(pdf[pdf["alignment"] == a])
        print(f"  {a:10s}: {na:>4d} ({na/n*100:.0f}%)", end="  ")
    print()

    # Exit reason distribution (exploit)
    print(f"\n  Exploit exit reasons:")
    for reason, count in pdf["exploit_reason"].value_counts().items():
        avg = pdf[pdf["exploit_reason"] == reason]["exploit_pnl"].mean()
        print(f"    {reason:18s}  {count:>4d}  ({count/n*100:5.1f}%)  avg={avg:+.3f}R")

    # --- Main comparison ---
    print(f"\n  {'Strategy':22s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} "
          f"{'Sharpe':>8s} {'MaxDD':>8s} {'Total':>8s}")
    print(f"  {'-'*22} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for label, col in [
        ("Fixed RR (control)", "ctrl_pnl"),
        ("Exploit (no pyramid)", "exploit_pnl"),
        ("Exploit + Pyramid", "pyramid_pnl"),
    ]:
        m = compute_strategy_metrics(pdf[col].values)
        if m:
            print(fmt_row(label, m))

    # --- By alignment ---
    aligned = pdf[pdf["alignment"] == "aligned"]
    opposed = pdf[pdf["alignment"] == "opposed"]

    if len(aligned) >= 5:
        print(f"\n  ALIGNED ONLY (N={len(aligned)}):")
        print(f"  {'Strategy':22s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} "
              f"{'Sharpe':>8s} {'MaxDD':>8s} {'Total':>8s}")
        print(f"  {'-'*22} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for label, col in [
            ("Fixed RR", "ctrl_pnl"),
            ("Exploit", "exploit_pnl"),
            ("Pyramid", "pyramid_pnl"),
        ]:
            m = compute_strategy_metrics(aligned[col].values)
            if m:
                print(fmt_row(label, m))

    if len(opposed) >= 5:
        print(f"\n  OPPOSED ONLY (N={len(opposed)}):")
        print(f"  {'Strategy':22s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} "
              f"{'Sharpe':>8s} {'MaxDD':>8s} {'Total':>8s}")
        print(f"  {'-'*22} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for label, col in [
            ("Fixed RR", "ctrl_pnl"),
            ("Exploit (kill)", "exploit_pnl"),
        ]:
            m = compute_strategy_metrics(opposed[col].values)
            if m:
                print(fmt_row(label, m))

    # --- Yearly stability ---
    print(f"\n  Yearly Sharpe comparison:")
    print(f"  {'Year':5s} {'N':>4s} {'Aln':>4s} {'Opp':>4s} "
          f"{'Fixed':>8s} {'Exploit':>8s} {'Pyramid':>8s} {'Best':>8s}")
    print(f"  {'-'*5} {'-'*4} {'-'*4} {'-'*4} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for year in sorted(pdf["year"].unique()):
        ydf = pdf[pdf["year"] == year]
        if len(ydf) < 3:
            continue
        na = len(ydf[ydf["alignment"] == "aligned"])
        no = len(ydf[ydf["alignment"] == "opposed"])
        mf = compute_strategy_metrics(ydf["ctrl_pnl"].values)
        me = compute_strategy_metrics(ydf["exploit_pnl"].values)
        mp = compute_strategy_metrics(ydf["pyramid_pnl"].values)
        if mf and me and mp:
            sharpes = {"fixed": mf["sharpe"], "exploit": me["sharpe"],
                       "pyramid": mp["sharpe"]}
            best = max(sharpes, key=sharpes.get)
            print(f"  {year:5s} {len(ydf):>4d} {na:>4d} {no:>4d} "
                  f"{mf['sharpe']:>8.3f} {me['sharpe']:>8.3f} "
                  f"{mp['sharpe']:>8.3f} {best:>8s}")

    # Total row
    mf = compute_strategy_metrics(pdf["ctrl_pnl"].values)
    me = compute_strategy_metrics(pdf["exploit_pnl"].values)
    mp = compute_strategy_metrics(pdf["pyramid_pnl"].values)
    if mf and me and mp:
        print(f"  {'TOTAL':5s} {n:>4d} {len(aligned):>4d} {len(opposed):>4d} "
              f"{mf['sharpe']:>8.3f} {me['sharpe']:>8.3f} {mp['sharpe']:>8.3f}")

    # --- Pyramid unit stats ---
    pyr_aligned = pdf[(pdf["alignment"] == "aligned") & (pdf["pyramid_units"] == 2)]
    if len(pyr_aligned) > 0:
        print(f"\n  Pyramid stats (aligned, 2-unit trades): {len(pyr_aligned)} trades")
        m = compute_strategy_metrics(pyr_aligned["pyramid_pnl"].values)
        if m:
            print(f"    WR={m['wr']:.1%}  ExpR={m['expr']:+.3f}R  "
                  f"Total={m['total']:+.1f}R  MaxDD={m['maxdd']:.2f}R")


def run_checks(pdf, session_label):
    print(f"\n  Integrity checks ({session_label}):")
    ok = total = 0

    # Conservation: every trade has an exit
    total += 1
    valid_reasons = {"stop", "target", "time", "eod",
                     "limbo_target", "limbo_time", "opposed_kill",
                     "time_7h", "nobreak_target", "nobreak_time"}
    all_valid = pdf["exploit_reason"].isin(valid_reasons).all()
    if all_valid:
        print(f"    [PASS] all exit reasons valid")
        ok += 1
    else:
        bad = pdf[~pdf["exploit_reason"].isin(valid_reasons)]["exploit_reason"].unique()
        print(f"    [FAIL] unexpected exit reasons: {bad}")

    # Opposed kill: all opposed trades should exit via opposed_kill or limbo exits
    total += 1
    opposed = pdf[pdf["alignment"] == "opposed"]
    if len(opposed) > 0:
        opp_exits = opposed["exploit_reason"].value_counts()
        # Opposed trades exit via: opposed_kill, or limbo_* (resolved before IB break)
        limbo_or_kill = opposed["exploit_reason"].isin(
            {"opposed_kill", "limbo_target", "limbo_time", "stop"}).all()
        if limbo_or_kill:
            print(f"    [PASS] opposed trades: all exit via kill or limbo "
                  f"({len(opposed)} trades)")
            ok += 1
        else:
            print(f"    [WARN] opposed exits: {dict(opp_exits)}")

    # Exploit ExpR >= Control ExpR for aligned trades (target unlock should help)
    total += 1
    aligned = pdf[pdf["alignment"] == "aligned"]
    if len(aligned) >= 10:
        mc = compute_strategy_metrics(aligned["ctrl_pnl"].values)
        me = compute_strategy_metrics(aligned["exploit_pnl"].values)
        if mc and me:
            # Not a strict invariant, but expected
            print(f"    [INFO] aligned: ctrl ExpR={mc['expr']:+.3f} "
                  f"exploit ExpR={me['expr']:+.3f} "
                  f"({'exploit wins' if me['expr'] > mc['expr'] else 'ctrl wins'})")
            ok += 1

    print(f"    {ok}/{total} passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(db_path, start, end):
    print("Honest IB Exploit Backtest -- 3-Part Strategy")
    print(f"Date range: {start} to {end}")
    print(f"Cost model: MGC ($10/pt, $8.40 RT friction)")
    print(f"Params: E1 CB{CONFIRM_BARS} RR{RR_TARGET} G{MIN_ORB_SIZE}+ Hold={HOLD_HOURS}h")
    print(f"IB configs: {dict((s, f'{a}_{d}m') for s,(a,d) in SESSION_IB_CONFIG.items())}")

    for session in SESSIONS:
        print(f"\nProcessing {session} session...")
        t0 = time.time()
        pdf = process_session(db_path, session, start, end)
        elapsed = time.time() - t0
        print(f"  {len(pdf)} trades in {elapsed:.1f}s")
        print_report(pdf, session)
        run_checks(pdf, session)

    print(f"\n{'#' * 90}")
    print("VERDICT CRITERIA")
    print(f"{'#' * 90}")
    print("- Exploit Sharpe > Fixed Sharpe = strategy works honestly")
    print("- Exploit should win on TotalR (captures fat tails)")
    print("- Pyramid should amplify aligned winners but increase variance")
    print("- If Exploit LOSES to Fixed: IB signal is real but not exploitable")
    print("- Check yearly: must work across regimes, not just 2025")


def main():
    parser = argparse.ArgumentParser(description="Honest IB Exploit Backtest")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 2, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    args = parser.parse_args()
    run(args.db_path, args.start, args.end)


if __name__ == "__main__":
    main()
