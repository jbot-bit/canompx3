#!/usr/bin/env python3
"""
IB Direction Alignment -- stripped to the one honest signal.

Logic (zero look-ahead):
  1. Enter trade on ORB break (1000 E1 CB2, as usual)
  2. IB forms over first 60 minutes of session (00:00-01:00 UTC)
  3. After IB complete, wait for FIRST IB break (price exceeds IB high or low)
  4. IB break direction is known the instant it happens (real-time observable)
  5. If IB break direction == ORB break direction -> ALIGNED -> hold 7h
     If IB break direction != ORB break direction -> OPPOSED -> take fixed RR
     If IB never breaks (no_break) -> take fixed RR

No single/double classification. No volume filter. One signal.

Decision point: the moment IB first breaks (could be minute 61, could be minute 200).
The trade is already live (entered at ~minute 7). This is a HOLD vs EXIT decision.

Read-only. No DB writes.

Usage:
    python scripts/analyze_ib_direction_alignment.py --db-path C:/db/gold.db
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
# Config
# ---------------------------------------------------------------------------
IB_MINUTES_OPTIONS = [30, 60, 90, 120]
MIN_ORB_SIZE = 4.0
RR_TARGET = 2.0
CONFIRM_BARS = 2
HOLD_HOURS = 7


def compute_ib(bars: pd.DataFrame, ib_minutes: int) -> dict | None:
    """IB = high/low of first N minutes starting at 00:00 UTC."""
    ib_start = None
    for _, bar in bars.iterrows():
        if bar["ts_utc"].hour == 0 and bar["ts_utc"].minute == 0:
            ib_start = bar["ts_utc"]
            break
    if ib_start is None:
        return None

    ib_end = ib_start + timedelta(minutes=ib_minutes)
    ib_bars = bars[(bars["ts_utc"] >= ib_start) & (bars["ts_utc"] < ib_end)]
    if len(ib_bars) < max(3, ib_minutes // 10):
        return None

    return {
        "ib_high": float(ib_bars["high"].max()),
        "ib_low": float(ib_bars["low"].min()),
        "ib_size": float(ib_bars["high"].max() - ib_bars["low"].min()),
        "ib_end": ib_end,
    }


def get_first_ib_break(bars: pd.DataFrame, ib: dict) -> dict:
    """Find the FIRST bar that breaks above IB high or below IB low.

    Scans from IB end forward. Returns:
      ib_break_dir: 'long' | 'short' | None
      ib_break_ts: timestamp of break bar (or None)
      ib_break_minutes_after_entry: how long after entry the break was known
    """
    post_ib = bars[bars["ts_utc"] >= ib["ib_end"]]

    for _, bar in post_ib.iterrows():
        broke_high = bar["high"] > ib["ib_high"]
        broke_low = bar["low"] < ib["ib_low"]

        if broke_high and broke_low:
            # Ambiguous bar -- both sides hit simultaneously.
            # Conservative: treat as no usable signal.
            return {"ib_break_dir": None, "ib_break_ts": bar["ts_utc"]}

        if broke_high:
            return {"ib_break_dir": "long", "ib_break_ts": bar["ts_utc"]}
        if broke_low:
            return {"ib_break_dir": "short", "ib_break_ts": bar["ts_utc"]}

    return {"ib_break_dir": None, "ib_break_ts": None}


def compute_pnl_with_stop(
    bars: pd.DataFrame, entry_ts, entry_price: float,
    stop_price: float, is_long: bool, cutoff_hours: int,
) -> float | None:
    """Hold with stop active until cutoff. Returns pnl_r."""
    spec = get_cost_spec("MGC")
    cutoff = entry_ts + timedelta(hours=cutoff_hours)
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
# Main
# ---------------------------------------------------------------------------

def load_data(db_path: Path, start: date, end: date):
    """Load trades and bars once (shared across all IB durations)."""
    con = duckdb.connect(str(db_path), read_only=True)

    print(f"Loading 1000 E1 CB{CONFIRM_BARS} RR{RR_TARGET} G{MIN_ORB_SIZE}+ trades...")
    df = con.execute("""
        SELECT o.trading_day, o.entry_ts, o.entry_price, o.stop_price,
               o.target_price, o.outcome, o.pnl_r,
               d.orb_1000_size, d.orb_1000_break_dir
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
        WHERE o.symbol = 'MGC' AND o.orb_minutes = 5
          AND o.orb_label = '1000' AND o.entry_model = 'E1'
          AND o.rr_target = ? AND o.confirm_bars = ?
          AND o.entry_ts IS NOT NULL AND o.outcome IS NOT NULL
          AND o.pnl_r IS NOT NULL AND d.orb_1000_size >= ?
          AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day
    """, [RR_TARGET, CONFIRM_BARS, MIN_ORB_SIZE, start, end]).fetchdf()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    print(f"  {len(df)} trades ({df['trading_day'].nunique()} days)")

    unique_days = sorted(df["trading_day"].unique())
    print(f"Loading bars for {len(unique_days)} days...")
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
    print(f"  Done\n")
    return df, bars_cache


def process_trades(df, bars_cache, ib_minutes: int) -> pd.DataFrame:
    """Classify trades for a given IB duration. Returns DataFrame."""
    results = []
    for _, row in df.iterrows():
        td = row["trading_day"]
        bars = bars_cache.get(td)
        if bars is None or bars.empty:
            continue

        ib = compute_ib(bars, ib_minutes)
        if ib is None:
            continue

        entry_ts = row["entry_ts"]
        entry_p = row["entry_price"]
        stop_p = row["stop_price"]
        orb_dir = row["orb_1000_break_dir"]
        is_long = orb_dir == "long"

        # First IB break (real-time observable, zero look-ahead)
        ib_break = get_first_ib_break(bars, ib)
        ib_dir = ib_break["ib_break_dir"]
        ib_break_ts = ib_break["ib_break_ts"]

        # Alignment
        if ib_dir is None:
            alignment = "no_break"
        elif ib_dir == orb_dir:
            alignment = "aligned"
        else:
            alignment = "opposed"

        # Minutes from entry to IB break
        if ib_break_ts is not None:
            wait_minutes = (ib_break_ts - entry_ts).total_seconds() / 60
        else:
            wait_minutes = None

        # PnL: fixed RR (from stored outcomes) and 7h hold (recomputed)
        fixed_pnl = row["pnl_r"]
        hold_pnl = compute_pnl_with_stop(bars, entry_ts, entry_p, stop_p, is_long, HOLD_HOURS)
        if hold_pnl is None:
            continue

        # Blended: aligned -> hold, opposed/no_break -> fixed
        blended_pnl = hold_pnl if alignment == "aligned" else fixed_pnl

        year = str(td.year) if hasattr(td, "year") else str(td)[:4]

        results.append({
            "td": td, "year": year, "is_long": is_long,
            "orb_dir": orb_dir, "ib_dir": ib_dir, "alignment": alignment,
            "wait_minutes": wait_minutes, "ib_size": ib["ib_size"],
            "orb_size": row["orb_1000_size"],
            "fixed_pnl": fixed_pnl, "hold_pnl": hold_pnl,
            "blended_pnl": blended_pnl,
        })

    return pd.DataFrame(results)


def run(db_path: Path, start: date, end: date):
    df, bars_cache = load_data(db_path, start, end)

    # =================================================================
    # IB DURATION SWEEP -- find best IB range
    # =================================================================
    print("=" * 90)
    print("IB DURATION SWEEP: Which IB range gives best alignment signal?")
    print("=" * 90)
    print(f"  {'IB':>5s} {'N':>5s} {'Algn':>5s} {'Opp':>5s} {'NoBr':>5s} "
          f"{'OppWR':>6s} "
          f"{'Fix_Sh':>8s} {'7h_Sh':>8s} {'Bln_Sh':>8s} {'Bln_ExpR':>9s} {'Bln_Tot':>8s}")
    print(f"  {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} "
          f"{'-'*6} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")

    best_ib = None
    best_sharpe = -999
    sweep_results = {}

    for ib_min in IB_MINUTES_OPTIONS:
        pdf = process_trades(df, bars_cache, ib_min)
        if len(pdf) < 10:
            continue
        sweep_results[ib_min] = pdf

        aligned = pdf[pdf["alignment"] == "aligned"]
        opposed = pdf[pdf["alignment"] == "opposed"]
        no_break = pdf[pdf["alignment"] == "no_break"]

        opp_wr = (opposed["hold_pnl"] > 0).mean() if len(opposed) > 0 else 0.0

        mf = compute_strategy_metrics(pdf["fixed_pnl"].values)
        m7 = compute_strategy_metrics(pdf["hold_pnl"].values)
        mb = compute_strategy_metrics(pdf["blended_pnl"].values)

        if mf and m7 and mb:
            print(f"  {ib_min:>4d}m {len(pdf):>5d} {len(aligned):>5d} {len(opposed):>5d} {len(no_break):>5d} "
                  f"{opp_wr:>5.1%} "
                  f"{mf['sharpe']:>8.4f} {m7['sharpe']:>8.4f} {mb['sharpe']:>8.4f} "
                  f"{mb['expr']:>+9.4f} {mb['total']:>8.1f}")

            if mb["sharpe"] > best_sharpe:
                best_sharpe = mb["sharpe"]
                best_ib = ib_min

    print(f"\n  Best IB: {best_ib}m (Sharpe={best_sharpe:.4f})")

    # =================================================================
    # FULL REPORT for best IB duration
    # =================================================================
    ib_minutes = best_ib
    pdf = sweep_results[ib_minutes]
    print(f"\n{'#' * 90}")
    print(f"DETAILED REPORT: IB={ib_minutes}m")
    print(f"{'#' * 90}\n")

    # ===================================================================
    # REPORT
    # ===================================================================
    aligned = pdf[pdf["alignment"] == "aligned"]
    opposed = pdf[pdf["alignment"] == "opposed"]
    no_break = pdf[pdf["alignment"] == "no_break"]

    print("=" * 90)
    print(f"IB DIRECTION ALIGNMENT (1000 E1 CB{CONFIRM_BARS} RR{RR_TARGET} G{MIN_ORB_SIZE}+)")
    print("=" * 90)
    print(f"  Aligned:  {len(aligned):>4d} ({len(aligned)/len(pdf)*100:4.1f}%)")
    print(f"  Opposed:  {len(opposed):>4d} ({len(opposed)/len(pdf)*100:4.1f}%)")
    print(f"  No break: {len(no_break):>4d} ({len(no_break)/len(pdf)*100:4.1f}%)")

    # Wait time for signal
    wait = pdf[pdf["wait_minutes"].notna()]["wait_minutes"]
    if len(wait) > 0:
        print(f"\n  Signal wait time (minutes after entry):")
        print(f"    Median: {wait.median():.0f}m  |  Mean: {wait.mean():.0f}m  |  "
              f"P25: {wait.quantile(0.25):.0f}m  |  P75: {wait.quantile(0.75):.0f}m")

    # --- Metrics by alignment ---
    print("\n" + "=" * 90)
    print("METRICS BY ALIGNMENT")
    print("=" * 90)
    print(f"  {'Group':15s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Total':>8s}")
    print(f"  {'-'*15} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for label, subset, col in [
        ("All (fixed)", pdf, "fixed_pnl"),
        ("All (7h)", pdf, "hold_pnl"),
        ("Aligned (fixed)", aligned, "fixed_pnl"),
        ("Aligned (7h)", aligned, "hold_pnl"),
        ("Opposed (fixed)", opposed, "fixed_pnl"),
        ("Opposed (7h)", opposed, "hold_pnl"),
        ("No break (fix)", no_break, "fixed_pnl"),
    ]:
        if len(subset) < 3:
            continue
        m = compute_strategy_metrics(subset[col].values)
        if m:
            print(f"  {label:15s} {m['n']:>5d} {m['wr']:>7.3f} {m['expr']:>8.4f} "
                  f"{m['sharpe']:>8.4f} {m['maxdd']:>8.2f} {m['total']:>8.1f}")

    # --- Blended comparison ---
    print("\n" + "=" * 90)
    print("BLENDED: Aligned=7h hold, Opposed/NoBreak=FixedRR")
    print("=" * 90)
    print(f"  {'Strategy':20s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Total':>8s}")
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for label, col in [
        ("Fixed RR (control)", "fixed_pnl"),
        ("7h hold (all)", "hold_pnl"),
        ("Blended (aligned)", "blended_pnl"),
    ]:
        m = compute_strategy_metrics(pdf[col].values)
        if m:
            print(f"  {label:20s} {m['n']:>5d} {m['wr']:>7.3f} {m['expr']:>8.4f} "
                  f"{m['sharpe']:>8.4f} {m['maxdd']:>8.2f} {m['total']:>8.1f}")

    # --- Variant: ONLY trade aligned days (skip opposed entirely) ---
    print("\n" + "=" * 90)
    print("VARIANT: ONLY TRADE ALIGNED DAYS (skip opposed & no_break)")
    print("=" * 90)
    if len(aligned) >= 5:
        for label, col in [
            ("Aligned fixed", "fixed_pnl"),
            ("Aligned 7h", "hold_pnl"),
        ]:
            m = compute_strategy_metrics(aligned[col].values)
            if m:
                print(f"  {label:20s} {m['n']:>5d} {m['wr']:>7.3f} {m['expr']:>8.4f} "
                      f"{m['sharpe']:>8.4f} {m['maxdd']:>8.2f} {m['total']:>8.1f}")

    # --- Yearly stability ---
    print("\n" + "=" * 90)
    print("YEARLY STABILITY")
    print("=" * 90)
    print(f"  {'Year':5s} {'N':>4s} {'Algn':>5s} {'Opp':>5s} {'NoBr':>5s} "
          f"{'Fixed':>8s} {'7h':>8s} {'Blended':>8s} {'Best':>8s}")
    print(f"  {'-'*5} {'-'*4} {'-'*5} {'-'*5} {'-'*5} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for year in sorted(pdf["year"].unique()):
        ydf = pdf[pdf["year"] == year]
        if len(ydf) < 3:
            continue
        na = len(ydf[ydf["alignment"] == "aligned"])
        no = len(ydf[ydf["alignment"] == "opposed"])
        nn = len(ydf[ydf["alignment"] == "no_break"])
        mf = compute_strategy_metrics(ydf["fixed_pnl"].values)
        m7 = compute_strategy_metrics(ydf["hold_pnl"].values)
        mb = compute_strategy_metrics(ydf["blended_pnl"].values)
        if mf and m7 and mb:
            sharpes = {"fixed": mf["sharpe"], "7h": m7["sharpe"], "blended": mb["sharpe"]}
            best = max(sharpes, key=sharpes.get)
            print(f"  {year:5s} {len(ydf):>4d} {na:>5d} {no:>5d} {nn:>5d} "
                  f"{mf['sharpe']:>8.3f} {m7['sharpe']:>8.3f} "
                  f"{mb['sharpe']:>8.3f} {best:>8s}")

    # Totals
    mf = compute_strategy_metrics(pdf["fixed_pnl"].values)
    m7 = compute_strategy_metrics(pdf["hold_pnl"].values)
    mb = compute_strategy_metrics(pdf["blended_pnl"].values)
    if mf and m7 and mb:
        na = len(aligned)
        no = len(opposed)
        nn = len(no_break)
        print(f"  {'TOTAL':5s} {len(pdf):>4d} {na:>5d} {no:>5d} {nn:>5d} "
              f"{mf['sharpe']:>8.3f} {m7['sharpe']:>8.3f} "
              f"{mb['sharpe']:>8.3f}")

    # --- Aligned-only yearly ---
    print("\n" + "=" * 90)
    print("ALIGNED-ONLY YEARLY (skip opposed & no_break entirely)")
    print("=" * 90)
    print(f"  {'Year':5s} {'N':>4s} {'Fixed':>8s} {'7h':>8s} {'Best':>8s}")
    print(f"  {'-'*5} {'-'*4} {'-'*8} {'-'*8} {'-'*8}")

    for year in sorted(aligned["year"].unique()):
        ydf = aligned[aligned["year"] == year]
        if len(ydf) < 3:
            continue
        mf = compute_strategy_metrics(ydf["fixed_pnl"].values)
        m7 = compute_strategy_metrics(ydf["hold_pnl"].values)
        if mf and m7:
            best = "7h" if m7["sharpe"] > mf["sharpe"] else "fixed"
            print(f"  {year:5s} {len(ydf):>4d} {mf['sharpe']:>8.3f} {m7['sharpe']:>8.3f} {best:>8s}")

    if len(aligned) >= 5:
        mf = compute_strategy_metrics(aligned["fixed_pnl"].values)
        m7 = compute_strategy_metrics(aligned["hold_pnl"].values)
        if mf and m7:
            print(f"  {'TOTAL':5s} {len(aligned):>4d} {mf['sharpe']:>8.3f} {m7['sharpe']:>8.3f}")

    # --- Cross-check: do opposed trades ALWAYS lose? ---
    print("\n" + "=" * 90)
    print("OPPOSED TRADES AUDIT")
    print("=" * 90)
    if len(opposed) > 0:
        wins = (opposed["hold_pnl"] > 0).sum()
        losses = (opposed["hold_pnl"] <= 0).sum()
        print(f"  Opposed 7h: {len(opposed)} trades, {wins} wins, {losses} losses")
        print(f"  Win rate: {wins/len(opposed)*100:.1f}%")
        if wins > 0:
            print(f"  WARNING: opposed trades have some wins -- not 100% loss")
        else:
            print(f"  CONFIRMED: 100% loss rate on opposed trades (mechanical)")

        # Also check fixed RR on opposed
        fx_wins = (opposed["fixed_pnl"] > 0).sum()
        print(f"  Opposed fixed: {fx_wins} wins out of {len(opposed)} "
              f"({fx_wins/len(opposed)*100:.1f}% WR)")

    print()


def main():
    parser = argparse.ArgumentParser(description="IB Direction Alignment")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 2, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    parser.add_argument("--ib-minutes", type=str, default=None,
                        help="Comma-separated IB durations (overrides default sweep)")
    args = parser.parse_args()
    if args.ib_minutes:
        global IB_MINUTES_OPTIONS
        IB_MINUTES_OPTIONS = [int(x.strip()) for x in args.ib_minutes.split(",")]
    run(args.db_path, args.start, args.end)


if __name__ == "__main__":
    main()
