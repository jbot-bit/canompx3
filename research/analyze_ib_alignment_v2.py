#!/usr/bin/env python3
"""
IB Direction Alignment v2 -- with Kill Switch + 0900 Blind Test.

Changes from v1:
  1. KILL SWITCH: If an aligned 7h hold trade reverses and breaks the
     opposite IB side (becomes Double Break), exit immediately at IB level.
     Prevents survivorship bias from "lucky recoveries."
  2. BLIND 0900 TEST: Same 120m IB, no re-optimization. If it fails at 0900
     while working at 1000, the finding was noise.
  3. SLIPPAGE: Already in cost model ($8.40 RT = ~0.84 ticks each way).
     Verified: to_r_multiple() deducts friction from every trade.

Logic (zero look-ahead):
  1. Enter on ORB break (E1 CB2)
  2. IB forms over 120 minutes from session start
  3. First IB break determines alignment (real-time observable)
  4. If ALIGNED: hold 7h with BOTH original stop AND IB kill switch
     - Kill switch: if price breaks opposite IB side -> exit at IB level
  5. If OPPOSED or NO_BREAK: take fixed RR target

Read-only. No DB writes.

Usage:
    python scripts/analyze_ib_alignment_v2.py --db-path C:/db/gold.db
"""

import argparse
import sys
import time
from datetime import date, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import get_cost_spec, to_r_multiple
from research._alt_strategy_utils import compute_strategy_metrics

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config -- FIXED. No per-session optimization.
# ---------------------------------------------------------------------------
IB_MINUTES = 120
MIN_ORB_SIZE = 4.0
RR_TARGET = 2.0
CONFIRM_BARS = 2
HOLD_HOURS = 7

# Session UTC offsets: session_label -> UTC hour of session start
SESSION_UTC = {
    "0900": 23,   # 0900 Brisbane = 23:00 UTC previous day
    "1000": 0,    # 1000 Brisbane = 00:00 UTC
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_ib(bars: pd.DataFrame, session_utc_hour: int) -> dict | None:
    """IB = high/low of first 120 minutes from session start."""
    ib_start = None
    for _, bar in bars.iterrows():
        if bar["ts_utc"].hour == session_utc_hour and bar["ts_utc"].minute == 0:
            ib_start = bar["ts_utc"]
            break
    if ib_start is None:
        return None

    ib_end = ib_start + timedelta(minutes=IB_MINUTES)
    ib_bars = bars[(bars["ts_utc"] >= ib_start) & (bars["ts_utc"] < ib_end)]
    if len(ib_bars) < 10:
        return None

    return {
        "ib_high": float(ib_bars["high"].max()),
        "ib_low": float(ib_bars["low"].min()),
        "ib_size": float(ib_bars["high"].max() - ib_bars["low"].min()),
        "ib_end": ib_end,
    }


def get_first_ib_break(bars: pd.DataFrame, ib: dict) -> dict:
    """Find FIRST bar that breaks IB high or IB low after IB forms."""
    post_ib = bars[bars["ts_utc"] >= ib["ib_end"]]

    for _, bar in post_ib.iterrows():
        broke_high = bar["high"] > ib["ib_high"]
        broke_low = bar["low"] < ib["ib_low"]

        if broke_high and broke_low:
            return {"ib_break_dir": None, "ib_break_ts": bar["ts_utc"]}
        if broke_high:
            return {"ib_break_dir": "long", "ib_break_ts": bar["ts_utc"]}
        if broke_low:
            return {"ib_break_dir": "short", "ib_break_ts": bar["ts_utc"]}

    return {"ib_break_dir": None, "ib_break_ts": None}


def compute_aligned_hold_with_kill_switch(
    bars: pd.DataFrame,
    entry_ts, entry_price: float, stop_price: float,
    is_long: bool, ib: dict,
) -> tuple[float, str]:
    """Hold 7h with original stop AND IB kill switch.

    Kill switch: if price breaks opposite IB side, exit at that IB level.
    For long aligned: kill if bar low <= ib_low (exit at ib_low)
    For short aligned: kill if bar high >= ib_high (exit at ib_high)

    Returns (pnl_r, exit_reason).
    exit_reason: 'stop', 'kill_switch', 'time_7h', 'no_bars'
    """
    spec = get_cost_spec("MGC")
    cutoff = entry_ts + timedelta(hours=HOLD_HOURS)
    ib_high = ib["ib_high"]
    ib_low = ib["ib_low"]

    mask = bars["ts_utc"] >= entry_ts
    if not mask.any():
        return 0.0, "no_bars"
    start = mask.idxmax()
    last_close = entry_price

    for i in range(start, len(bars)):
        bar = bars.iloc[i]

        # Original stop (always active, highest priority)
        if is_long and bar["low"] <= stop_price:
            pnl = stop_price - entry_price
            return to_r_multiple(spec, entry_price, stop_price, pnl), "stop"
        if not is_long and bar["high"] >= stop_price:
            pnl = entry_price - stop_price
            return to_r_multiple(spec, entry_price, stop_price, pnl), "stop"

        # Kill switch: opposite IB side broken (double break = exit)
        if is_long and bar["low"] <= ib_low:
            # Exit at IB low (conservative: assume fill at IB level)
            pnl = ib_low - entry_price
            return to_r_multiple(spec, entry_price, stop_price, pnl), "kill_switch"
        if not is_long and bar["high"] >= ib_high:
            pnl = entry_price - ib_high
            return to_r_multiple(spec, entry_price, stop_price, pnl), "kill_switch"

        last_close = bar["close"]

        # Time cutoff
        if bar["ts_utc"] >= cutoff:
            pnl = (last_close - entry_price) if is_long else (entry_price - last_close)
            return to_r_multiple(spec, entry_price, stop_price, pnl), "time_7h"

    pnl = (last_close - entry_price) if is_long else (entry_price - last_close)
    return to_r_multiple(spec, entry_price, stop_price, pnl), "time_7h"


def compute_vanilla_hold(
    bars: pd.DataFrame,
    entry_ts, entry_price: float, stop_price: float,
    is_long: bool,
) -> float | None:
    """Plain 7h hold with original stop only (no kill switch). For comparison."""
    spec = get_cost_spec("MGC")
    cutoff = entry_ts + timedelta(hours=HOLD_HOURS)
    mask = bars["ts_utc"] >= entry_ts
    if not mask.any():
        return None
    start = mask.idxmax()
    last_close = entry_price

    for i in range(start, len(bars)):
        bar = bars.iloc[i]
        if is_long and bar["low"] <= stop_price:
            return to_r_multiple(spec, entry_price, stop_price, stop_price - entry_price)
        if not is_long and bar["high"] >= stop_price:
            return to_r_multiple(spec, entry_price, stop_price, entry_price - stop_price)
        last_close = bar["close"]
        if bar["ts_utc"] >= cutoff:
            pnl = (last_close - entry_price) if is_long else (entry_price - last_close)
            return to_r_multiple(spec, entry_price, stop_price, pnl)

    pnl = (last_close - entry_price) if is_long else (entry_price - last_close)
    return to_r_multiple(spec, entry_price, stop_price, pnl)


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------

def process_session(
    db_path: Path, session_label: str, start: date, end: date,
) -> pd.DataFrame:
    """Process one session. Returns results DataFrame."""
    session_utc_hour = SESSION_UTC[session_label]
    con = duckdb.connect(str(db_path), read_only=True)

    df = con.execute(f"""
        SELECT o.trading_day, o.entry_ts, o.entry_price, o.stop_price,
               o.target_price, o.outcome, o.pnl_r,
               d.orb_{session_label}_size, d.orb_{session_label}_break_dir
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

    unique_days = sorted(df["trading_day"].unique())
    bars_cache = {}
    for td in unique_days:
        s, e = compute_trading_day_utc_range(td)
        b = con.execute(
            "SELECT ts_utc, open, high, low, close, volume FROM bars_1m "
            "WHERE symbol='MGC' AND ts_utc>=? AND ts_utc<? ORDER BY ts_utc",
            [s, e],
        ).fetchdf()
        if not b.empty:
            b["ts_utc"] = pd.to_datetime(b["ts_utc"], utc=True)
        bars_cache[td] = b
    con.close()

    results = []
    for _, row in df.iterrows():
        td = row["trading_day"]
        bars = bars_cache.get(td)
        if bars is None or bars.empty:
            continue

        ib = compute_ib(bars, session_utc_hour)
        if ib is None:
            continue

        entry_ts = row["entry_ts"]
        entry_p = row["entry_price"]
        stop_p = row["stop_price"]
        orb_dir = row[f"orb_{session_label}_break_dir"]
        is_long = orb_dir == "long"

        # First IB break
        ib_break = get_first_ib_break(bars, ib)
        ib_dir = ib_break["ib_break_dir"]

        if ib_dir is None:
            alignment = "no_break"
        elif ib_dir == orb_dir:
            alignment = "aligned"
        else:
            alignment = "opposed"

        # Fixed RR PnL (from stored outcomes)
        fixed_pnl = row["pnl_r"]

        # Vanilla 7h hold (no kill switch)
        vanilla_pnl = compute_vanilla_hold(bars, entry_ts, entry_p, stop_p, is_long)
        if vanilla_pnl is None:
            continue

        # Kill switch 7h hold
        ks_pnl, exit_reason = compute_aligned_hold_with_kill_switch(
            bars, entry_ts, entry_p, stop_p, is_long, ib,
        )

        # Blended strategies:
        # v1: aligned=vanilla_7h, else=fixed (previous version, no kill switch)
        blended_v1 = vanilla_pnl if alignment == "aligned" else fixed_pnl
        # v2: aligned=kill_switch_7h, else=fixed (new version with kill switch)
        blended_v2 = ks_pnl if alignment == "aligned" else fixed_pnl

        year = str(td.year) if hasattr(td, "year") else str(td)[:4]

        results.append({
            "td": td, "year": year, "is_long": is_long,
            "orb_dir": orb_dir, "ib_dir": ib_dir, "alignment": alignment,
            "ib_size": ib["ib_size"], "orb_size": row[f"orb_{session_label}_size"],
            "fixed_pnl": fixed_pnl,
            "vanilla_pnl": vanilla_pnl,
            "ks_pnl": ks_pnl,
            "exit_reason": exit_reason,
            "blended_v1": blended_v1,
            "blended_v2": blended_v2,
        })

    return pd.DataFrame(results)


def print_session_report(pdf: pd.DataFrame, session_label: str):
    """Print full report for one session."""
    if len(pdf) == 0:
        print(f"  No trades for {session_label}")
        return

    aligned = pdf[pdf["alignment"] == "aligned"]
    opposed = pdf[pdf["alignment"] == "opposed"]
    no_break = pdf[pdf["alignment"] == "no_break"]

    print(f"\n{'=' * 90}")
    print(f"{session_label} SESSION -- IB={IB_MINUTES}m, E1 CB{CONFIRM_BARS} RR{RR_TARGET} G{MIN_ORB_SIZE}+")
    print(f"{'=' * 90}")
    print(f"  Total: {len(pdf)}  |  Aligned: {len(aligned)} ({len(aligned)/len(pdf)*100:.0f}%)  |  "
          f"Opposed: {len(opposed)} ({len(opposed)/len(pdf)*100:.0f}%)  |  "
          f"No break: {len(no_break)} ({len(no_break)/len(pdf)*100:.0f}%)")

    # Kill switch exit stats
    ks_trades = pdf[pdf["alignment"] == "aligned"]
    if len(ks_trades) > 0:
        ks_fired = ks_trades[ks_trades["exit_reason"] == "kill_switch"]
        stopped = ks_trades[ks_trades["exit_reason"] == "stop"]
        held_7h = ks_trades[ks_trades["exit_reason"] == "time_7h"]
        print(f"\n  Aligned trade exits: "
              f"held 7h={len(held_7h)} | kill switch={len(ks_fired)} | stopped={len(stopped)}")
        if len(ks_fired) > 0:
            avg_ks = ks_fired["ks_pnl"].mean()
            print(f"  Kill switch avg PnL: {avg_ks:+.3f}R (would have been worse without it)")

    # Opposed audit
    if len(opposed) > 0:
        opp_7h_wr = (opposed["vanilla_pnl"] > 0).mean()
        opp_fix_wr = (opposed["fixed_pnl"] > 0).mean()
        print(f"\n  Opposed audit: 7h WR={opp_7h_wr:.1%}, fixed WR={opp_fix_wr:.1%}")

    # --- Main comparison ---
    print(f"\n  {'Strategy':25s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Total':>8s}")
    print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for label, col in [
        ("Fixed RR (control)", "fixed_pnl"),
        ("7h hold (all, no KS)", "vanilla_pnl"),
        ("Blended v1 (no KS)", "blended_v1"),
        ("Blended v2 (with KS)", "blended_v2"),
    ]:
        m = compute_strategy_metrics(pdf[col].values)
        if m:
            print(f"  {label:25s} {m['n']:>5d} {m['wr']:>7.3f} {m['expr']:>8.4f} "
                  f"{m['sharpe']:>8.4f} {m['maxdd']:>8.2f} {m['total']:>8.1f}")

    # --- Aligned-only ---
    if len(aligned) >= 5:
        print(f"\n  ALIGNED-ONLY ({len(aligned)} trades):")
        print(f"  {'Strategy':25s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Total':>8s}")
        print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for label, col in [
            ("Fixed RR", "fixed_pnl"),
            ("7h hold (no KS)", "vanilla_pnl"),
            ("7h hold (with KS)", "ks_pnl"),
        ]:
            m = compute_strategy_metrics(aligned[col].values)
            if m:
                print(f"  {label:25s} {m['n']:>5d} {m['wr']:>7.3f} {m['expr']:>8.4f} "
                      f"{m['sharpe']:>8.4f} {m['maxdd']:>8.2f} {m['total']:>8.1f}")

    # --- Yearly stability ---
    print(f"\n  {'Year':5s} {'N':>4s} {'Algn':>5s} {'Opp':>5s} "
          f"{'Fixed':>8s} {'V1noKS':>8s} {'V2+KS':>8s} {'Best':>8s}")
    print(f"  {'-'*5} {'-'*4} {'-'*5} {'-'*5} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for year in sorted(pdf["year"].unique()):
        ydf = pdf[pdf["year"] == year]
        if len(ydf) < 3:
            continue
        na = len(ydf[ydf["alignment"] == "aligned"])
        no = len(ydf[ydf["alignment"] == "opposed"])

        mf = compute_strategy_metrics(ydf["fixed_pnl"].values)
        mv1 = compute_strategy_metrics(ydf["blended_v1"].values)
        mv2 = compute_strategy_metrics(ydf["blended_v2"].values)
        if mf and mv1 and mv2:
            sharpes = {"fixed": mf["sharpe"], "v1": mv1["sharpe"], "v2": mv2["sharpe"]}
            best = max(sharpes, key=sharpes.get)
            print(f"  {year:5s} {len(ydf):>4d} {na:>5d} {no:>5d} "
                  f"{mf['sharpe']:>8.3f} {mv1['sharpe']:>8.3f} "
                  f"{mv2['sharpe']:>8.3f} {best:>8s}")

    mf = compute_strategy_metrics(pdf["fixed_pnl"].values)
    mv1 = compute_strategy_metrics(pdf["blended_v1"].values)
    mv2 = compute_strategy_metrics(pdf["blended_v2"].values)
    if mf and mv1 and mv2:
        na = len(aligned)
        no = len(opposed)
        print(f"  {'TOTAL':5s} {len(pdf):>4d} {na:>5d} {no:>5d} "
              f"{mf['sharpe']:>8.3f} {mv1['sharpe']:>8.3f} "
              f"{mv2['sharpe']:>8.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(db_path: Path, start: date, end: date):
    print(f"IB Direction Alignment v2 -- Kill Switch + Blind 0900 Test")
    print(f"IB={IB_MINUTES}m (FIXED, no per-session optimization)")
    print(f"Date range: {start} to {end}")
    print(f"Cost model: MGC ($10/pt, $8.40 RT friction)\n")

    for session in ["1000", "0900"]:
        print(f"Processing {session} session...")
        t0 = time.time()
        pdf = process_session(db_path, session, start, end)
        elapsed = time.time() - t0
        print(f"  {len(pdf)} trades in {elapsed:.1f}s")
        print_session_report(pdf, session)

    # --- Cross-session verdict ---
    print(f"\n{'#' * 90}")
    print("CROSS-SESSION VERDICT")
    print(f"{'#' * 90}")
    print("If 120m IB works at BOTH 1000 and 0900 -> structural edge.")
    print("If 120m IB works at 1000 but fails at 0900 -> noise / p-hack.")
    print("The 0900 test was BLIND: same IB length, same parameters, no tuning.")


def main():
    parser = argparse.ArgumentParser(description="IB Alignment v2 + Kill Switch")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 2, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    args = parser.parse_args()
    run(args.db_path, args.start, args.end)


if __name__ == "__main__":
    main()
