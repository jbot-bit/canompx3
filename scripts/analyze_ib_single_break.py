#!/usr/bin/env python3
"""
Initial Balance (IB) Single Break Runner Detection for 1000 Session.

Framework (from trend.txt / trenddays.txt):
  - IB = high/low of first 60 minutes after 1000 Brisbane open (00:00-01:00 UTC)
  - Single Break (trend day): price breaks ONE side of IB only
  - Double Break (chop day): price breaks BOTH sides of IB
  - Volume filter: breakout bar volume > 130% of 10-bar rolling avg

Exit strategy:
  - Single Break days -> 7h hold with original stop (ride the trend)
  - Double Break days -> fixed RR target (take quick profit, don't fight chop)

For each trade in orb_outcomes (1000 E1 G4+):
  1. Compute IB from bars_1m (first 60 minutes of session)
  2. Classify day as Single Break or Double Break
  3. Apply volume filter on the actual ORB breakout bar
  4. Compare blended strategy (single=7h, double=fixed) vs pure fixed vs pure 7h

Read-only research script. No DB writes.

Usage:
    python scripts/analyze_ib_single_break.py --db-path C:/db/gold.db
    python scripts/analyze_ib_single_break.py --db-path C:/db/gold.db --ib-minutes 60
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
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
# Config
# ---------------------------------------------------------------------------
# 1000 Brisbane = 00:00 UTC
SESSION_START_UTC_HOUR = 0
SESSION_START_UTC_MINUTE = 0

DEFAULT_IB_MINUTES = 60       # Initial Balance = first 60 minutes
DEFAULT_MIN_ORB_SIZE = 4.0
DEFAULT_RR = 2.0
DEFAULT_CB = 2
HOLD_HOURS = 7                # Runner hold duration
VOL_MULTIPLIER = 1.30         # Breakout bar volume > 130% of rolling avg
VOL_LOOKBACK = 10             # Rolling average lookback bars


# ---------------------------------------------------------------------------
# IB computation
# ---------------------------------------------------------------------------

def compute_ib(bars: pd.DataFrame, ib_minutes: int) -> dict | None:
    """Compute Initial Balance (IB) from 1-minute bars.

    IB = high/low of first `ib_minutes` minutes from 00:00 UTC.
    Returns dict with ib_high, ib_low, ib_size, ib_end_ts or None if
    insufficient data.
    """
    # Filter to IB window: 00:00-01:00 UTC (1000-1100 Brisbane)
    ib_start = None
    for _, bar in bars.iterrows():
        ts = bar["ts_utc"]
        if ts.hour == SESSION_START_UTC_HOUR and ts.minute >= SESSION_START_UTC_MINUTE:
            ib_start = ts
            break

    if ib_start is None:
        return None

    ib_end = ib_start + timedelta(minutes=ib_minutes)
    ib_mask = (bars["ts_utc"] >= ib_start) & (bars["ts_utc"] < ib_end)
    ib_bars = bars[ib_mask]

    if len(ib_bars) < 5:  # need reasonable bar count
        return None

    return {
        "ib_high": float(ib_bars["high"].max()),
        "ib_low": float(ib_bars["low"].min()),
        "ib_size": float(ib_bars["high"].max() - ib_bars["low"].min()),
        "ib_start": ib_start,
        "ib_end": ib_end,
    }


def classify_day(
    bars: pd.DataFrame,
    ib: dict,
    entry_ts: pd.Timestamp,
) -> dict:
    """Classify day as Single Break or Double Break after IB forms.

    Scans all bars from IB end to end of session.
    Returns dict with:
      - day_type: 'single_bull', 'single_bear', 'double_break', 'no_break'
      - broke_high: bool
      - broke_low: bool
      - break_bar_volume: volume of the first IB break bar (for volume filter)
      - pre_break_volumes: list of volumes of 10 bars before break (for rolling avg)
    """
    ib_high = ib["ib_high"]
    ib_low = ib["ib_low"]
    ib_end = ib["ib_end"]

    # Scan post-IB bars
    post_ib = bars[bars["ts_utc"] >= ib_end].copy()
    if post_ib.empty:
        return {"day_type": "no_break", "broke_high": False, "broke_low": False,
                "break_bar_volume": None, "pre_break_volumes": []}

    broke_high = False
    broke_low = False
    first_break_bar_idx = None
    first_break_dir = None

    for idx, bar in post_ib.iterrows():
        if not broke_high and bar["high"] > ib_high:
            broke_high = True
            if first_break_bar_idx is None:
                first_break_bar_idx = idx
                first_break_dir = "high"
        if not broke_low and bar["low"] < ib_low:
            broke_low = True
            if first_break_bar_idx is None:
                first_break_bar_idx = idx
                first_break_dir = "low"

        if broke_high and broke_low:
            break  # no need to continue

    # Volume on first break bar
    break_bar_volume = None
    pre_break_volumes = []
    if first_break_bar_idx is not None:
        break_bar_volume = float(post_ib.loc[first_break_bar_idx, "volume"])
        # Get 10 bars before the break bar in the full bars df
        full_idx = bars.index.get_loc(first_break_bar_idx)
        start_vol = max(0, full_idx - VOL_LOOKBACK)
        pre_break_volumes = bars.iloc[start_vol:full_idx]["volume"].tolist()

    # Classify
    if broke_high and not broke_low:
        day_type = "single_bull"
    elif broke_low and not broke_high:
        day_type = "single_bear"
    elif broke_high and broke_low:
        day_type = "double_break"
    else:
        day_type = "no_break"

    return {
        "day_type": day_type,
        "broke_high": broke_high,
        "broke_low": broke_low,
        "break_bar_volume": break_bar_volume,
        "pre_break_volumes": pre_break_volumes,
    }


def check_volume_filter(classification: dict) -> bool:
    """Check if breakout bar volume > 130% of 10-bar rolling avg."""
    vol = classification["break_bar_volume"]
    pre_vols = classification["pre_break_volumes"]
    if vol is None or not pre_vols:
        return False
    avg_vol = sum(pre_vols) / len(pre_vols)
    if avg_vol <= 0:
        return False
    return vol > avg_vol * VOL_MULTIPLIER


def compute_7h_pnl(
    bars: pd.DataFrame,
    entry_ts: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    is_long: bool,
) -> float | None:
    """Compute 7h hold PnL with stop still active. Returns pnl_r."""
    spec = get_cost_spec("MGC")
    cutoff = entry_ts + timedelta(hours=HOLD_HOURS)

    entry_mask = bars["ts_utc"] >= entry_ts
    if not entry_mask.any():
        return None
    entry_idx = entry_mask.idxmax()

    last_close = entry_price
    for i in range(entry_idx, len(bars)):
        bar = bars.iloc[i]

        # Stop check
        if is_long and bar["low"] <= stop_price:
            pnl_pts = stop_price - entry_price
            return to_r_multiple(spec, entry_price, stop_price, pnl_pts)
        if not is_long and bar["high"] >= stop_price:
            pnl_pts = entry_price - stop_price
            return to_r_multiple(spec, entry_price, stop_price, pnl_pts)

        last_close = bar["close"]

        # Time cutoff
        if bar["ts_utc"] >= cutoff:
            pnl_pts = (last_close - entry_price) if is_long else (entry_price - last_close)
            return to_r_multiple(spec, entry_price, stop_price, pnl_pts)

    # Ran out of bars
    pnl_pts = (last_close - entry_price) if is_long else (entry_price - last_close)
    return to_r_multiple(spec, entry_price, stop_price, pnl_pts)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(db_path: Path, ib_minutes: int, min_orb_size: float,
        start: date, end: date):
    con = duckdb.connect(str(db_path), read_only=True)

    # Load 1000 E1 CB2 RR2.0 trades with G4+ filter
    print(f"Loading 1000 E1 CB{DEFAULT_CB} RR{DEFAULT_RR} G{min_orb_size}+ trades...")
    df = con.execute("""
        SELECT o.trading_day, o.entry_ts, o.entry_price, o.stop_price,
               o.target_price, o.outcome, o.pnl_r,
               d.orb_1000_size, d.orb_1000_break_dir
        FROM orb_outcomes o
        JOIN daily_features d ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
        WHERE o.symbol = 'MGC' AND o.orb_minutes = 5
          AND o.orb_label = '1000' AND o.entry_model = 'E1'
          AND o.rr_target = ? AND o.confirm_bars = ?
          AND o.entry_ts IS NOT NULL AND o.outcome IS NOT NULL
          AND o.pnl_r IS NOT NULL AND d.orb_1000_size >= ?
          AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day
    """, [DEFAULT_RR, DEFAULT_CB, min_orb_size, start, end]).fetchdf()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    print(f"  {len(df)} trades loaded ({df['trading_day'].nunique()} unique days)")

    # Load bars for all trading days
    unique_days = sorted(df["trading_day"].unique())
    print(f"Loading bars for {len(unique_days)} trading days...")
    t0 = time.time()
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
    print(f"  Bars loaded in {time.time() - t0:.1f}s")

    # Process each trade
    print(f"Classifying {len(df)} trades (IB={ib_minutes}m)...")
    t0 = time.time()

    results = []
    for _, row in df.iterrows():
        td = row["trading_day"]
        bars = bars_cache.get(td)
        if bars is None or bars.empty:
            continue

        # Compute IB
        ib = compute_ib(bars, ib_minutes)
        if ib is None:
            continue

        entry_ts = row["entry_ts"]
        entry_p = row["entry_price"]
        stop_p = row["stop_price"]
        is_long = row["orb_1000_break_dir"] == "long"
        fixed_pnl_r = row["pnl_r"]

        # Classify day
        cl = classify_day(bars, ib, entry_ts)
        vol_ok = check_volume_filter(cl)

        # Compute 7h hold PnL
        hold_pnl_r = compute_7h_pnl(bars, entry_ts, entry_p, stop_p, is_long)
        if hold_pnl_r is None:
            continue

        # Blended: single break -> 7h hold, double break -> fixed RR
        is_single = cl["day_type"] in ("single_bull", "single_bear")
        blended_pnl_r = hold_pnl_r if is_single else fixed_pnl_r

        # Blended with volume: single + vol -> 7h, else fixed
        blended_vol_pnl_r = hold_pnl_r if (is_single and vol_ok) else fixed_pnl_r

        year = str(td.year) if hasattr(td, "year") else str(td)[:4]

        results.append({
            "td": td,
            "year": year,
            "day_type": cl["day_type"],
            "vol_ok": vol_ok,
            "is_long": is_long,
            "ib_size": ib["ib_size"],
            "orb_size": row["orb_1000_size"],
            "fixed_pnl_r": fixed_pnl_r,
            "hold_pnl_r": hold_pnl_r,
            "blended_pnl_r": blended_pnl_r,
            "blended_vol_pnl_r": blended_vol_pnl_r,
        })

    elapsed = time.time() - t0
    print(f"  Classified in {elapsed:.1f}s")

    pdf = pd.DataFrame(results)
    print(f"  {len(pdf)} trades classified\n")

    # ===================================================================
    # REPORT
    # ===================================================================

    # --- Day type distribution ---
    print("=" * 90)
    print(f"IB SINGLE BREAK ANALYSIS (1000 E1 CB{DEFAULT_CB} RR{DEFAULT_RR} G{min_orb_size}+, IB={ib_minutes}m)")
    print("=" * 90)

    for dt in ["single_bull", "single_bear", "double_break", "no_break"]:
        n = len(pdf[pdf["day_type"] == dt])
        pct = n / len(pdf) * 100 if len(pdf) > 0 else 0
        print(f"  {dt:15s}: {n:>4d} ({pct:4.1f}%)")

    single = pdf[pdf["day_type"].isin(["single_bull", "single_bear"])]
    double = pdf[pdf["day_type"] == "double_break"]
    single_vol = pdf[(pdf["day_type"].isin(["single_bull", "single_bear"])) & pdf["vol_ok"]]
    print(f"\n  Single Break rate: {len(single)/len(pdf)*100:.1f}%")
    print(f"  Double Break rate: {len(double)/len(pdf)*100:.1f}%")
    print(f"  Single + Volume:   {len(single_vol)} trades ({len(single_vol)/len(pdf)*100:.1f}%)")

    # --- Metrics by day type ---
    print("\n" + "=" * 90)
    print("METRICS BY DAY TYPE")
    print("=" * 90)
    print(f"  {'Type':15s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Total':>8s}")
    print(f"  {'-'*15} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for label, subset, col in [
        ("All (fixed)", pdf, "fixed_pnl_r"),
        ("All (7h hold)", pdf, "hold_pnl_r"),
        ("Single (fixed)", single, "fixed_pnl_r"),
        ("Single (7h)", single, "hold_pnl_r"),
        ("Double (fixed)", double, "fixed_pnl_r"),
        ("Double (7h)", double, "hold_pnl_r"),
        ("Singl+Vol (7h)", single_vol, "hold_pnl_r"),
    ]:
        if len(subset) < 3:
            continue
        m = compute_strategy_metrics(subset[col].values)
        if m is None:
            continue
        print(f"  {label:15s} {m['n']:>5d} {m['wr']:>7.3f} {m['expr']:>8.4f} "
              f"{m['sharpe']:>8.4f} {m['maxdd']:>8.2f} {m['total']:>8.1f}")

    # --- Blended strategy comparison ---
    print("\n" + "=" * 90)
    print("BLENDED STRATEGY: Single=7h, Double=FixedRR")
    print("=" * 90)
    print(f"  {'Strategy':20s} {'N':>5s} {'WR':>7s} {'ExpR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Total':>8s}")
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for label, col in [
        ("Fixed RR (control)", "fixed_pnl_r"),
        ("7h hold (all)", "hold_pnl_r"),
        ("Blended (IB only)", "blended_pnl_r"),
        ("Blended (IB+Vol)", "blended_vol_pnl_r"),
    ]:
        m = compute_strategy_metrics(pdf[col].values)
        if m is None:
            continue
        print(f"  {label:20s} {m['n']:>5d} {m['wr']:>7.3f} {m['expr']:>8.4f} "
              f"{m['sharpe']:>8.4f} {m['maxdd']:>8.2f} {m['total']:>8.1f}")

    # --- Yearly stability of blended ---
    print("\n" + "=" * 90)
    print("YEARLY STABILITY: Blended (Single=7h, Double=Fixed)")
    print("=" * 90)
    print(f"  {'Year':5s} {'N':>4s} {'Single':>7s} {'Double':>7s} {'Fixed':>8s} {'7h':>8s} {'Blended':>8s} {'Blend+V':>8s} {'Best':>8s}")
    print(f"  {'-'*5} {'-'*4} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for year in sorted(pdf["year"].unique()):
        ydf = pdf[pdf["year"] == year]
        if len(ydf) < 3:
            continue
        n_single = len(ydf[ydf["day_type"].isin(["single_bull", "single_bear"])])
        n_double = len(ydf[ydf["day_type"] == "double_break"])

        mf = compute_strategy_metrics(ydf["fixed_pnl_r"].values)
        m7 = compute_strategy_metrics(ydf["hold_pnl_r"].values)
        mb = compute_strategy_metrics(ydf["blended_pnl_r"].values)
        mv = compute_strategy_metrics(ydf["blended_vol_pnl_r"].values)

        if mf and m7 and mb and mv:
            sharpes = {"fixed": mf["sharpe"], "7h": m7["sharpe"],
                       "blended": mb["sharpe"], "blend+v": mv["sharpe"]}
            best = max(sharpes, key=sharpes.get)
            print(f"  {year:5s} {len(ydf):>4d} {n_single:>7d} {n_double:>7d} "
                  f"{mf['sharpe']:>8.3f} {m7['sharpe']:>8.3f} "
                  f"{mb['sharpe']:>8.3f} {mv['sharpe']:>8.3f} {best:>8s}")

    # Overall
    mf = compute_strategy_metrics(pdf["fixed_pnl_r"].values)
    m7 = compute_strategy_metrics(pdf["hold_pnl_r"].values)
    mb = compute_strategy_metrics(pdf["blended_pnl_r"].values)
    mv = compute_strategy_metrics(pdf["blended_vol_pnl_r"].values)
    if mf and m7 and mb and mv:
        print(f"  {'TOTAL':5s} {len(pdf):>4d} {len(single):>7d} {len(double):>7d} "
              f"{mf['sharpe']:>8.3f} {m7['sharpe']:>8.3f} "
              f"{mb['sharpe']:>8.3f} {mv['sharpe']:>8.3f}")

    # --- IB size vs runner rate ---
    print("\n" + "=" * 90)
    print("IB SIZE AND DAY TYPE")
    print("=" * 90)
    for ib_tier in [3, 5, 7, 10, 15]:
        subset = pdf[pdf["ib_size"] >= ib_tier]
        if len(subset) < 5:
            continue
        single_rate = len(subset[subset["day_type"].isin(["single_bull", "single_bear"])]) / len(subset) * 100
        runner_rate = (subset["hold_pnl_r"] > 1.0).mean() * 100
        avg_7h = subset["hold_pnl_r"].mean()
        print(f"  IB>={ib_tier:>2d}pt: N={len(subset):>4d}, "
              f"single rate={single_rate:.0f}%, "
              f"runner rate={runner_rate:.0f}%, "
              f"avg 7h R={avg_7h:+.3f}")

    # --- Direction analysis ---
    print("\n" + "=" * 90)
    print("SINGLE BREAK DIRECTION ALIGNMENT WITH ORB BREAK")
    print("=" * 90)
    for dt in ["single_bull", "single_bear"]:
        sub = pdf[pdf["day_type"] == dt]
        if len(sub) < 3:
            continue
        aligned = sub[
            ((dt == "single_bull") & sub["is_long"]) |
            ((dt == "single_bear") & ~sub["is_long"])
        ]
        opposed = sub[~sub.index.isin(aligned.index)]

        print(f"\n  {dt}:")
        for label, ss in [("Aligned", aligned), ("Opposed", opposed)]:
            if len(ss) < 2:
                print(f"    {label:10s}: N={len(ss)} (too few)")
                continue
            m = compute_strategy_metrics(ss["hold_pnl_r"].values)
            if m:
                print(f"    {label:10s}: N={m['n']:>4d}, "
                      f"ExpR={m['expr']:+.4f}, Sharpe={m['sharpe']:.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="IB Single Break Runner Detection")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--ib-minutes", type=int, default=DEFAULT_IB_MINUTES)
    parser.add_argument("--min-orb-size", type=float, default=DEFAULT_MIN_ORB_SIZE)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 2, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    args = parser.parse_args()
    run(args.db_path, args.ib_minutes, args.min_orb_size, args.start, args.end)


if __name__ == "__main__":
    main()
