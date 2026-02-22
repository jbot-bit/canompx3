#!/usr/bin/env python3
"""
Profile what makes a 1000 session runner day.

Institutional trader thinking: don't decide at entry. Check in at
15m, 30m, 60m, 120m. If the trade is winning, switch to runner mode
(trailing stop or extended hold). If losing, keep fixed target or kill.

This script profiles EVERY 1000 E1 G4+ trade to find:
1. What does the trade look like at checkpoint N minutes in?
2. Does early momentum predict runner vs chop?
3. Does 0900 alignment help?
4. Does ORB size predict runners?

Read-only. No DB writes.

Usage:
    python scripts/profile_1000_runners.py --db-path C:/db/gold.db
"""

import argparse
import sys
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

from pipeline.cost_model import get_cost_spec, to_r_multiple
from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.paths import GOLD_DB_PATH
from research._alt_strategy_utils import compute_strategy_metrics

sys.stdout.reconfigure(line_buffering=True)

CHECKPOINTS = [15, 30, 60, 120]
THRESHOLDS = [0.0, 0.3, 0.5, 0.75, 1.0]

def run(db_path: Path, start: date, end: date):
    spec = get_cost_spec("MGC")
    con = duckdb.connect(str(db_path), read_only=True)

    # Load 1000 E1 G4+ trades (all RR/CB combos pooled)
    df = con.execute("""
        SELECT o.trading_day, o.entry_ts, o.entry_price, o.stop_price,
               o.target_price, o.outcome, o.pnl_r, o.rr_target, o.confirm_bars,
               d.orb_1000_size, d.orb_1000_break_dir,
               d.orb_0900_size, d.orb_0900_break_dir
        FROM orb_outcomes o
        JOIN daily_features d ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
        WHERE o.symbol = 'MGC' AND o.orb_minutes = 5
          AND o.orb_label = '1000' AND o.entry_model = 'E1'
          AND o.rr_target = 2.0 AND o.confirm_bars = 2
          AND o.entry_ts IS NOT NULL AND o.outcome IS NOT NULL
          AND o.pnl_r IS NOT NULL AND d.orb_1000_size >= 4
          AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day
    """, [start, end]).fetchdf()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    print(f"Trades: {len(df)} ({start} to {end})")

    # Load bars
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

    # Profile each trade
    profiles = []
    for _, row in df.iterrows():
        td = row["trading_day"]
        bars = bars_cache.get(td)
        if bars is None or bars.empty:
            continue

        entry_ts = row["entry_ts"]
        entry_p = row["entry_price"]
        stop_p = row["stop_price"]
        is_long = row["orb_1000_break_dir"] == "long"
        risk_pts = abs(entry_p - stop_p)
        if risk_pts <= 0:
            continue

        mask = bars["ts_utc"] >= entry_ts
        if not mask.any():
            continue
        entry_idx = mask.idxmax()

        profile = {
            "td": td,
            "year": str(td.year) if hasattr(td, "year") else str(td)[:4],
            "entry": entry_p,
            "stop": stop_p,
            "is_long": is_long,
            "orb_size": row["orb_1000_size"],
            "orb_0900_size": row["orb_0900_size"],
            "orb_0900_dir": row["orb_0900_break_dir"],
            "fixed_rr_pnl": row["pnl_r"],
        }

        cutoff_7h = entry_ts + timedelta(hours=7)
        mfe_pts = 0.0
        last_close = entry_p
        stopped = False

        for i in range(entry_idx, len(bars)):
            bar = bars.iloc[i]
            minutes_in = int((bar["ts_utc"] - entry_ts).total_seconds() / 60)

            # MFE tracking
            if is_long:
                excursion = bar["high"] - entry_p
            else:
                excursion = entry_p - bar["low"]
            mfe_pts = max(mfe_pts, excursion)

            # Current unrealized
            if is_long:
                current_pts = bar["close"] - entry_p
            else:
                current_pts = entry_p - bar["close"]

            # Stop check
            if is_long and bar["low"] <= stop_p:
                stopped = True
                for cp in CHECKPOINTS:
                    key = f"r_at_{cp}m"
                    if key not in profile:
                        profile[key] = -1.0
                        profile[f"mfe_at_{cp}m"] = mfe_pts / risk_pts
                break
            if not is_long and bar["high"] >= stop_p:
                stopped = True
                for cp in CHECKPOINTS:
                    key = f"r_at_{cp}m"
                    if key not in profile:
                        profile[key] = -1.0
                        profile[f"mfe_at_{cp}m"] = mfe_pts / risk_pts
                break

            last_close = bar["close"]

            # Record checkpoints
            for cp in CHECKPOINTS:
                key = f"r_at_{cp}m"
                if key not in profile and minutes_in >= cp:
                    profile[key] = current_pts / risk_pts
                    profile[f"mfe_at_{cp}m"] = mfe_pts / risk_pts

            # 7h cutoff
            if bar["ts_utc"] >= cutoff_7h:
                pnl = (last_close - entry_p) if is_long else (entry_p - last_close)
                profile["r_7h"] = to_r_multiple(spec, entry_p, stop_p, pnl)
                break

        if stopped:
            pnl = (stop_p - entry_p) if is_long else (entry_p - stop_p)
            profile["r_7h"] = to_r_multiple(spec, entry_p, stop_p, pnl)

        if "r_7h" not in profile:
            pnl = (last_close - entry_p) if is_long else (entry_p - last_close)
            profile["r_7h"] = to_r_multiple(spec, entry_p, stop_p, pnl)

        profiles.append(profile)

    pdf = pd.DataFrame(profiles)
    print(f"Profiled {len(pdf)} trades")

    # --- Analysis ---
    pdf["is_runner"] = pdf["r_7h"] > 1.0
    pdf["is_big_runner"] = pdf["r_7h"] > 3.0
    runners = pdf[pdf["is_runner"]]
    non_runners = pdf[~pdf["is_runner"]]

    print()
    print("=" * 85)
    print("WHAT MAKES A RUNNER? (1000 E1 CB2 RR2.0 G4+)")
    print("=" * 85)
    print(f"Runners (7h > +1R): {len(runners)} ({len(runners)/len(pdf)*100:.0f}%)")
    print(f"Non-runners:        {len(non_runners)} ({len(non_runners)/len(pdf)*100:.0f}%)")

    # Checkpoint analysis
    for cp in CHECKPOINTS:
        col = f"r_at_{cp}m"
        mfe_col = f"mfe_at_{cp}m"
        if col not in pdf.columns:
            continue
        r_run = runners[col].dropna()
        r_non = non_runners[col].dropna()
        if len(r_run) == 0 or len(r_non) == 0:
            continue

        print(f"\n--- At {cp} minutes ---")
        print(f"  Runners:     median R = {r_run.median():+.3f}, mean = {r_run.mean():+.3f}")
        print(f"  Non-runners: median R = {r_non.median():+.3f}, mean = {r_non.mean():+.3f}")

        if mfe_col in pdf.columns:
            mfe_run = runners[mfe_col].dropna()
            mfe_non = non_runners[mfe_col].dropna()
            if len(mfe_run) > 0 and len(mfe_non) > 0:
                print(f"  Runners MFE:     median = {mfe_run.median():+.3f}")
                print(f"  Non-runners MFE: median = {mfe_non.median():+.3f}")

        # Adaptive exit: if R >= threshold at checkpoint, hold 7h. Else take fixed RR.
        print(f"\n  ADAPTIVE EXIT: check at {cp}m, hold 7h if winning, else fixed RR")
        for threshold in THRESHOLDS:
            hold_mask = pdf[col] >= threshold
            hold = pdf[hold_mask]
            take_fixed = pdf[~hold_mask]

            if len(hold) < 3 or len(take_fixed) < 3:
                continue

            # Blended result: runners get 7h pnl, others get fixed_rr pnl
            blended = pd.concat([
                hold["r_7h"],
                take_fixed["fixed_rr_pnl"],
            ])
            arr = blended.values
            m = compute_strategy_metrics(arr)
            if m is None:
                continue

            hold_pct = len(hold) / len(pdf) * 100
            print(f"    R>={threshold:+.1f}: hold {len(hold)} ({hold_pct:.0f}%), "
                  f"ExpR={m['expr']:+.4f}  Sharpe={m['sharpe']:.4f}  "
                  f"Total={m['total']:+.1f}")

    # 0900 alignment
    print()
    print("=" * 85)
    print("0900 MOMENTUM ALIGNMENT")
    print("=" * 85)
    pdf["same_dir"] = pdf.apply(
        lambda r: (
            r["orb_0900_dir"] == ("long" if r["is_long"] else "short")
            if pd.notna(r.get("orb_0900_dir"))
            else False
        ),
        axis=1,
    )
    for label, mask in [("Aligned", pdf["same_dir"]), ("Opposed", ~pdf["same_dir"])]:
        subset = pdf[mask]
        if len(subset) < 5:
            continue
        avg_7h = subset["r_7h"].mean()
        runner_rate = subset["is_runner"].mean() * 100
        print(f"  {label:10s}: N={len(subset):>4}, avg 7h R={avg_7h:+.3f}, "
              f"runner rate={runner_rate:.0f}%")

    # ORB size tiers
    print()
    print("=" * 85)
    print("ORB SIZE AND RUNNERS")
    print("=" * 85)
    for g in [4, 6, 8, 10, 12]:
        subset = pdf[pdf["orb_size"] >= g]
        if len(subset) < 5:
            continue
        avg_7h = subset["r_7h"].mean()
        runner_rate = subset["is_runner"].mean() * 100
        avg_fixed = subset["fixed_rr_pnl"].mean()
        print(f"  G{g:>2}+: N={len(subset):>4}, "
              f"fixed ExpR={avg_fixed:+.4f}, 7h ExpR={avg_7h:+.4f}, "
              f"runner rate={runner_rate:.0f}%")

    # Year breakdown of adaptive strategy
    print()
    print("=" * 85)
    print("BEST ADAPTIVE STRATEGY -- YEARLY STABILITY")
    print("=" * 85)
    # Use 30m checkpoint, R>=0.3 as the candidate
    best_cp = 30
    best_thresh = 0.3
    col = f"r_at_{best_cp}m"
    if col in pdf.columns:
        hold_mask = pdf[col] >= best_thresh
        blended_pnl = pd.Series(
            np.where(hold_mask, pdf["r_7h"], pdf["fixed_rr_pnl"]),
            index=pdf.index,
        )
        pdf["adaptive_pnl"] = blended_pnl

        print(f"Rule: At {best_cp}m, if R >= {best_thresh:+.1f} -> hold 7h, else fixed RR")
        print()
        print(f"  {'Year':5s} {'N':>4s} {'FixedRR':>8s} {'7h_only':>8s} {'Adaptive':>8s} {'Best':>8s}")
        print(f"  {'-'*5} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for year in sorted(pdf["year"].unique()):
            ypdf = pdf[pdf["year"] == year]
            if len(ypdf) < 3:
                continue
            mf = compute_strategy_metrics(ypdf["fixed_rr_pnl"].values)
            m7 = compute_strategy_metrics(ypdf["r_7h"].values)
            ma = compute_strategy_metrics(ypdf["adaptive_pnl"].values)
            if mf and m7 and ma:
                sharpes = {"fixed": mf["sharpe"], "7h": m7["sharpe"], "adaptive": ma["sharpe"]}
                best = max(sharpes, key=sharpes.get)
                print(f"  {year:5s} {len(ypdf):>4d} {mf['sharpe']:>8.3f} {m7['sharpe']:>8.3f} "
                      f"{ma['sharpe']:>8.3f} {best:>8s}")

        # Overall
        mf = compute_strategy_metrics(pdf["fixed_rr_pnl"].values)
        m7 = compute_strategy_metrics(pdf["r_7h"].values)
        ma = compute_strategy_metrics(pdf["adaptive_pnl"].values)
        if mf and m7 and ma:
            print(f"  {'TOTAL':5s} {len(pdf):>4d} {mf['sharpe']:>8.3f} {m7['sharpe']:>8.3f} "
                  f"{ma['sharpe']:>8.3f}")
            print()
            print(f"  Fixed RR: ExpR={mf['expr']:+.4f} Sharpe={mf['sharpe']:.4f} Total={mf['total']:+.1f}")
            print(f"  7h only:  ExpR={m7['expr']:+.4f} Sharpe={m7['sharpe']:.4f} Total={m7['total']:+.1f}")
            print(f"  Adaptive: ExpR={ma['expr']:+.4f} Sharpe={ma['sharpe']:.4f} Total={ma['total']:+.1f}")

def main():
    parser = argparse.ArgumentParser(description="Profile 1000 runner days")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 2, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    args = parser.parse_args()
    run(args.db_path, args.start, args.end)

if __name__ == "__main__":
    main()
