#!/usr/bin/env python3
"""
Rolling walk-forward analysis for 0900 ORB strategies.

Uses rolling 6/12/18 month training windows, tests on the NEXT month.
Zero lookahead. Honest regime-aware analysis.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
from pipeline.paths import GOLD_DB_PATH


def load_0900_data():
    """Load all 0900 outcomes joined with daily_features."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    df = con.execute("""
        SELECT oo.trading_day, oo.entry_model, oo.rr_target, oo.confirm_bars, oo.pnl_r,
               df.orb_0900_size
        FROM orb_outcomes oo
        JOIN daily_features df
          ON df.symbol = oo.symbol AND df.trading_day = oo.trading_day AND df.orb_minutes = oo.orb_minutes
        WHERE oo.symbol = 'MGC' AND oo.orb_label = '0900' AND oo.pnl_r IS NOT NULL
        ORDER BY oo.trading_day
    """).fetchdf()
    con.close()

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["orb_size"] = df["orb_0900_size"]
    return df


def compute_stats(pnls):
    """Compute trading stats from array of R-multiples."""
    n = len(pnls)
    if n == 0:
        return None
    wr = (pnls > 0).sum() / n
    expr = pnls.mean()
    std = pnls.std()
    sharpe = expr / std if std > 0 else 0
    cumul = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumul)
    maxdd = (cumul - peak).min()
    total = pnls.sum()
    return {"n": n, "wr": wr, "expr": expr, "sharpe": sharpe, "maxdd": maxdd, "total": total}


def find_best_combo(data, min_trades=10, metric="sharpe"):
    """Find best (em, rr, cb) from data. Lower min_trades for rolling windows."""
    best_val = -999
    best_combo = None
    best_stats = None

    for em in ["E1", "E3"]:
        em_data = data[data["entry_model"] == em]
        for rr in sorted(data["rr_target"].unique()):
            for cb in sorted(data["confirm_bars"].unique()):
                mask = (em_data["rr_target"] == rr) & (em_data["confirm_bars"] == cb)
                trades = em_data[mask]
                if len(trades) < min_trades:
                    continue
                stats = compute_stats(trades["pnl_r"].values)
                if stats and stats[metric] > best_val:
                    best_val = stats[metric]
                    best_combo = (em, rr, cb)
                    best_stats = stats

    return best_combo, best_stats


def em_label(em):
    return "E1" if em == "E1" else "E3"


def main():
    df = load_0900_data()

    sep = "=" * 80
    print(sep)
    print("ROLLING WALK-FORWARD: 0900 ORB STRATEGIES")
    print(sep)
    print()
    print("Method: Rolling training window, test on NEXT month")
    print("  - Train on past N months of data")
    print("  - Select best Sharpe with min 10 trades in window")
    print("  - Apply selected strategy to next month (unseen)")
    print("  - Roll forward month by month")
    print("  - NO lookahead: test month data never seen during selection")
    print()

    # =========================================================================
    # SECTION 1: Regime reality check
    # =========================================================================
    print(sep)
    print("REGIME CHECK: Monthly G4/G5/G6 eligible days at 0900")
    print(sep)
    print()

    df["ym"] = df["trading_day"].dt.to_period("M")
    months = sorted(df["ym"].unique())

    # Show eligible days per month for last 24 months
    recent = [m for m in months if m >= pd.Period("2024-01")]
    print(f"  Month      G4days  G5days  G6days  TotalTrades(G4+)")
    for m in recent:
        m_data = df[df["ym"] == m]
        # Unique trading days with each filter
        g4_days = len(m_data[m_data["orb_size"] >= 4.0]["trading_day"].unique())
        g5_days = len(m_data[m_data["orb_size"] >= 5.0]["trading_day"].unique())
        g6_days = len(m_data[m_data["orb_size"] >= 6.0]["trading_day"].unique())
        # Total trades (any E1/E3 combo) on G4+ days
        g4_trades = len(m_data[(m_data["orb_size"] >= 4.0) & (m_data["entry_model"] == "E1")])
        print(f"  {str(m):10s}  {g4_days:6d}  {g5_days:6d}  {g6_days:6d}  {g4_trades:6d}")

    print()

    # =========================================================================
    # SECTION 2: Rolling walk-forward for each filter
    # =========================================================================
    for filt_name, filt_thresh in [("G4", 4.0), ("G5", 5.0), ("G6", 6.0), ("NO_FILTER", 0.0)]:
        for train_months in [6, 12, 18]:
            print(sep)
            print(f"ROLLING {train_months}M TRAIN -> 1M TEST | Filter: {filt_name}")
            print(sep)
            print()

            eligible = df[df["orb_size"] >= filt_thresh].copy() if filt_thresh > 0 else df.copy()

            # Test months: 2024-07 through 2025-12 (gives 6-18 months of training before)
            test_start = pd.Period("2024-07") if train_months <= 12 else pd.Period("2024-07")
            test_months = [m for m in months if m >= test_start and m <= pd.Period("2026-01")]

            oos_pnls_all = []
            results = []

            for test_m in test_months:
                # Training window: past N months
                train_end = test_m - 1  # month before test
                train_start_m = test_m - train_months
                train_data = eligible[
                    (eligible["ym"] >= train_start_m) & (eligible["ym"] <= train_end)
                ]
                test_data = eligible[eligible["ym"] == test_m]

                if len(train_data) == 0 or len(test_data) == 0:
                    continue

                combo, train_stats = find_best_combo(train_data, min_trades=10)

                if combo is None:
                    results.append({
                        "test_m": str(test_m), "combo": "NONE",
                        "train_n": len(train_data), "oos_n": 0, "oos_total": 0,
                    })
                    continue

                em, rr, cb = combo
                mask = (test_data["entry_model"] == em) & \
                       (test_data["rr_target"] == rr) & \
                       (test_data["confirm_bars"] == cb)
                oos_pnls = test_data[mask]["pnl_r"].values

                oos_total = oos_pnls.sum() if len(oos_pnls) > 0 else 0
                oos_pnls_all.extend(oos_pnls)

                results.append({
                    "test_m": str(test_m),
                    "combo": f"{em_label(em)} RR{rr} CB{cb}",
                    "train_n": train_stats["n"],
                    "train_sharpe": train_stats["sharpe"],
                    "oos_n": len(oos_pnls),
                    "oos_total": oos_total,
                })

            # Print results
            print(f"  {'Month':10s}  {'Selected':22s}  TrainN  TrainSh  OOS_N  OOS_R")
            for r in results:
                if r["combo"] == "NONE":
                    print(f"  {r['test_m']:10s}  {'(no qualifying strat)':22s}  {r['train_n']:6d}")
                else:
                    print(f"  {r['test_m']:10s}  {r['combo']:22s}  {r['train_n']:6d}  {r.get('train_sharpe', 0):+6.3f}  {r['oos_n']:5d}  {r['oos_total']:+5.1f}R")

            # Summary
            if oos_pnls_all:
                combined = compute_stats(np.array(oos_pnls_all))
                months_profitable = sum(1 for r in results if r.get("oos_total", 0) > 0)
                months_with_trades = sum(1 for r in results if r.get("oos_n", 0) > 0)
                combos_used = set(r["combo"] for r in results if r["combo"] != "NONE")

                print()
                print(f"  SUMMARY: {combined['n']} OOS trades, ExpR={combined['expr']:+.3f}, Total={combined['total']:+.1f}R, WR={combined['wr']:.0%}, MaxDD={combined['maxdd']:+.1f}R")
                print(f"  Profitable months: {months_profitable}/{months_with_trades}")
                print(f"  Distinct strategies selected: {len(combos_used)} -> {combos_used}")
            else:
                print(f"\n  NO OOS trades generated (filter too strict for this window)")

            print()

    # =========================================================================
    # SECTION 3: The honest question - does 0900 have edge WITHOUT parameter selection?
    # =========================================================================
    print(sep)
    print("FAMILY-LEVEL EDGE: ALL E1 variants averaged (no cherry-picking)")
    print(sep)
    print()
    print("If the 0900 E1 family has edge, the average across ALL RR/CB combos")
    print("should be positive. This removes parameter selection bias entirely.")
    print()

    for filt_name, filt_thresh in [("G4", 4.0), ("G5", 5.0), ("G6", 6.0), ("NO_FILTER", 0.0)]:
        eligible = df[df["orb_size"] >= filt_thresh] if filt_thresh > 0 else df
        e1_data = eligible[eligible["entry_model"] == "E1"]

        # By half-year
        for period_label, start, end in [
            ("2022H1", "2022-01", "2022-06"),
            ("2022H2", "2022-07", "2022-12"),
            ("2023H1", "2023-01", "2023-06"),
            ("2023H2", "2023-07", "2023-12"),
            ("2024H1", "2024-01", "2024-06"),
            ("2024H2", "2024-07", "2024-12"),
            ("2025H1", "2025-01", "2025-06"),
            ("2025H2", "2025-07", "2025-12"),
            ("2026", "2026-01", "2026-12"),
        ]:
            mask = (e1_data["trading_day"] >= start) & (e1_data["trading_day"] < end)
            period_data = e1_data[mask]

            if len(period_data) == 0:
                continue

            # Average ExpR across all RR/CB combos
            combo_exprs = []
            for rr in period_data["rr_target"].unique():
                for cb in period_data["confirm_bars"].unique():
                    m = (period_data["rr_target"] == rr) & (period_data["confirm_bars"] == cb)
                    trades = period_data[m]
                    if len(trades) > 0:
                        combo_exprs.append({"expr": trades["pnl_r"].mean(), "n": len(trades)})

            if combo_exprs:
                avg_expr = np.mean([c["expr"] for c in combo_exprs])
                pct_pos = sum(1 for c in combo_exprs if c["expr"] > 0) / len(combo_exprs) * 100
                avg_n = np.mean([c["n"] for c in combo_exprs])
                print(f"  {filt_name:10s}  {period_label:8s}  {len(combo_exprs):2d} combos, avg N/combo={avg_n:.0f}, avg ExpR={avg_expr:+.3f}, {pct_pos:.0f}% positive")

        print()

    print(sep)
    print("DONE")
    print(sep)


if __name__ == "__main__":
    main()
