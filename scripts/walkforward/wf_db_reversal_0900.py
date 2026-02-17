#!/usr/bin/env python3
"""Walk-forward validation of 0900 double-break reversal (ORB-size stop).

Entry: opposite direction after double-break on 0900 ORB.
Stop: opposite ORB level (risk = ORB size).
Target: RR * risk.
Grid: G3/G4/G5/G6 size filters x RR 1.0-3.0.
Walk-forward: 12m train, 1m test step.
"""

import sys
from datetime import date
from dateutil.relativedelta import relativedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from collections import Counter
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.stdout.reconfigure(line_buffering=True)

from pipeline.cost_model import get_cost_spec, to_r_multiple
from research._alt_strategy_utils import (
    load_daily_features, load_bars_for_day, resolve_bar_outcome,
)

SPEC = get_cost_spec("MGC")
UTC = ZoneInfo("UTC")

def compute_all_reversal_outcomes(db_path, features, orb_label="0900"):
    """Pre-compute all double-break reversal outcomes with ORB-size stop."""
    col_db = f"orb_{orb_label}_double_break"
    col_dir = f"orb_{orb_label}_break_dir"
    col_high = f"orb_{orb_label}_high"
    col_low = f"orb_{orb_label}_low"
    col_size = f"orb_{orb_label}_size"
    col_bts = f"orb_{orb_label}_break_ts"

    RRS = [1.0, 1.5, 2.0, 2.5, 3.0]
    db_days = features[features[col_db] == True].copy()
    print(f"  {len(db_days)} double-break days at {orb_label}")

    outcomes = []
    processed = 0

    for _, row in db_days.iterrows():
        td = row["trading_day"]
        if hasattr(td, "date") and callable(td.date):
            td = td.date()
        elif isinstance(td, str):
            td = date.fromisoformat(td)

        bdir = row[col_dir]
        oh, ol = row[col_high], row[col_low]
        sz, bts = row[col_size], row[col_bts]

        if pd.isna(bdir) or pd.isna(oh) or pd.isna(ol) or pd.isna(bts) or pd.isna(sz):
            continue
        if sz < 1.0:
            continue

        bars = load_bars_for_day(db_path, td)
        if bars.empty:
            continue

        processed += 1
        if processed % 200 == 0:
            print(f"    processed {processed} days...")

        bts_utc = bts.astimezone(UTC) if hasattr(bts, "astimezone") else bts

        # Reversal: enter opposite side of ORB, stop = first-break ORB level
        if bdir == "long":
            ep, sp, d = ol, oh, "short"
        elif bdir == "short":
            ep, sp, d = oh, ol, "long"
        else:
            continue

        risk = abs(ep - sp)
        if risk < SPEC.min_risk_floor_points:
            continue

        # Find second break bar
        eidx = None
        for i in range(len(bars)):
            b = bars.iloc[i]
            bt = b["ts_utc"]
            if hasattr(bt, "astimezone"):
                bt = bt.astimezone(UTC)
            if bt <= bts_utc:
                continue
            if d == "short" and b["low"] <= ep:
                eidx = i
                break
            elif d == "long" and b["high"] >= ep:
                eidx = i
                break

        if eidx is None:
            continue

        for rr in RRS:
            reward = rr * risk
            tp = ep - reward if d == "short" else ep + reward

            res = resolve_bar_outcome(bars, ep, sp, tp, d, eidx + 1)
            if res is None:
                lc = bars.iloc[-1]["close"]
                pp = (ep - lc) if d == "short" else (lc - ep)
                pr = to_r_multiple(SPEC, ep, sp, pp)
                oc = "eod"
            else:
                pr = to_r_multiple(SPEC, ep, sp, res["pnl_points"])
                oc = res["outcome"]

            outcomes.append({
                "td": td, "sz": sz, "rr": rr, "pnl_r": pr, "oc": oc,
            })

    return pd.DataFrame(outcomes)

def run_walk_forward(df, train_months=12):
    """Run walk-forward on pre-computed outcomes."""
    FILTERS = {"G3": 3.0, "G4": 4.0, "G5": 5.0, "G6": 6.0}
    RRS = [1.0, 1.5, 2.0, 2.5, 3.0]

    test_start = date(2018, 1, 1)
    test_end = date(2026, 1, 1)
    current = test_start

    windows = []
    oos_trades = []
    selections = []

    while current <= test_end:
        train_s = current - relativedelta(months=train_months)
        train_e = current - relativedelta(days=1)
        test_e = (current + relativedelta(months=1)) - relativedelta(days=1)

        train = df[(df["td"] >= train_s) & (df["td"] <= train_e)]
        test = df[(df["td"] >= current) & (df["td"] <= test_e)]

        # Find best (filter, rr) on training data by Sharpe
        best_combo = None
        best_sharpe = -999

        for fn, ft in FILTERS.items():
            for rr in RRS:
                sub = train[(train["sz"] >= ft) & (train["rr"] == rr)]
                traded = sub[sub["oc"].isin(["win", "loss"])]
                if len(traded) < 15:
                    continue
                pnls = traded["pnl_r"].values
                mu = pnls.mean()
                std = pnls.std()
                sh = mu / std if std > 0 else 0
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_combo = (fn, ft, rr, len(traded), mu, sh)

        if best_combo is None:
            current += relativedelta(months=1)
            continue

        fn, ft, rr, train_n, train_expr, train_sh = best_combo

        # Apply to OOS
        oos = test[(test["sz"] >= ft) & (test["rr"] == rr)]
        oos_traded = oos[oos["oc"].isin(["win", "loss"])]

        if len(oos_traded) > 0:
            oos_pnls = oos_traded["pnl_r"].values
            oos_expr = oos_pnls.mean()
            oos_total = oos_pnls.sum()
            oos_wr = (oos_traded["oc"] == "win").mean()
            oos_n = len(oos_traded)
        else:
            oos_expr, oos_total, oos_wr, oos_n = 0, 0, 0, 0

        selections.append(f"{fn}_RR{rr}")
        oos_trades.extend(oos_traded["pnl_r"].tolist())

        windows.append({
            "test": str(current)[:7],
            "selected": f"{fn}_RR{rr}",
            "train_n": train_n,
            "train_sh": round(train_sh, 3),
            "oos_n": oos_n,
            "oos_wr": round(oos_wr, 3) if oos_n > 0 else None,
            "oos_expr": round(oos_expr, 3) if oos_n > 0 else None,
            "oos_total": round(oos_total, 2),
        })

        current += relativedelta(months=1)

    return windows, oos_trades, selections

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Walk-forward: 0900 DB reversal")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--train-months", type=int, default=12)
    args = parser.parse_args()

    db_path = args.db_path or Path("C:/db/gold.db")

    print("Loading features...")
    feat = load_daily_features(db_path, date(2016, 1, 1), date(2026, 2, 1))
    print(f"  {len(feat)} features loaded")

    print("Computing reversal outcomes (this takes a few minutes)...")
    df = compute_all_reversal_outcomes(db_path, feat, "0900")
    print(f"  {len(df)} total outcomes, {df['td'].nunique()} unique days")

    print(f"\nRunning walk-forward ({args.train_months}m train)...")
    windows, oos_trades, selections = run_walk_forward(df, args.train_months)

    # Print results
    sep = "=" * 80
    print(f"\n{sep}")
    print("WALK-FORWARD: 0900 DOUBLE-BREAK REVERSAL (ORB-SIZE STOP)")
    print(sep)
    print(f"Train: {args.train_months}m rolling | Filters: G3/G4/G5/G6 | RR: 1.0-3.0")
    print(f"Windows: {len(windows)} | OOS trades: {len(oos_trades)}")
    print()

    hdr = f"{'Test':>8s} {'Selected':>12s} {'TrN':>5s} {'TrSh':>6s} {'OOS_N':>5s} {'WR':>5s} {'ExpR':>8s} {'TotR':>8s}"
    print(hdr)
    for w in windows:
        wr_s = f"{w['oos_wr']:.0%}" if w["oos_wr"] is not None else "-"
        expr_s = f"{w['oos_expr']:+.3f}" if w["oos_expr"] is not None else "-"
        print(f"{w['test']:>8s} {w['selected']:>12s} {w['train_n']:>5d} "
              f"{w['train_sh']:>+6.3f} {w['oos_n']:>5d} {wr_s:>5s} "
              f"{expr_s:>8s} {w['oos_total']:>+8.2f}")

    # Combined OOS
    if oos_trades:
        arr = np.array(oos_trades)
        mu = arr.mean()
        std = arr.std()
        sh = mu / std if std > 0 else 0
        wr = (arr > 0).mean()
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        maxdd = (peak - cum).max()
        print(f"\nCOMBINED OOS: N={len(arr)}, WR={wr:.0%}, "
              f"ExpR={mu:+.4f}, Sharpe={sh:+.4f}, "
              f"Total={arr.sum():+.1f}R, MaxDD={maxdd:.1f}R")

    # Selection stability
    sel_counts = Counter(selections)
    print("\nSELECTION STABILITY:")
    for combo, count in sel_counts.most_common():
        pct = count / len(windows) * 100
        print(f"  {combo}: {count}/{len(windows)} windows ({pct:.0f}%)")

    # Year-by-year OOS
    print("\nYEAR-BY-YEAR OOS:")
    yr_data = {}
    for w in windows:
        yr = w["test"][:4]
        if yr not in yr_data:
            yr_data[yr] = {"n": 0, "total": 0.0}
        yr_data[yr]["n"] += w["oos_n"]
        yr_data[yr]["total"] += w["oos_total"]
    for yr in sorted(yr_data):
        d = yr_data[yr]
        expr = d["total"] / d["n"] if d["n"] > 0 else 0
        print(f"  {yr}: N={d['n']:3d}, Total={d['total']:+6.1f}R, AvgExpR={expr:+.3f}")

    print(f"\n{sep}")
    print("DONE")

if __name__ == "__main__":
    main()
