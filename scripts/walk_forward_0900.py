#!/usr/bin/env python3
"""
Walk-forward analysis for 0900 ORB strategies.
Zero lookahead: train on past years, test on unseen year.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec, stress_test_costs

spec = get_cost_spec("MGC")


def load_0900_data():
    """Load all 0900 outcomes joined with daily_features for filter eligibility."""
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

    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
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
    return {
        "n": n, "wr": wr, "expr": expr, "sharpe": sharpe,
        "maxdd": maxdd, "total": total,
    }


def find_best_strategy(data, metric="sharpe", min_trades=30):
    """Find best (em, rr, cb) combo by metric. Returns (combo, stats) or (None, None)."""
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
                if stats is None:
                    continue
                val = stats[metric]
                if val > best_val:
                    best_val = val
                    best_combo = (em, rr, cb)
                    best_stats = stats

    return best_combo, best_stats


def find_worst_strategy(data, metric="sharpe", min_trades=30):
    """Find worst combo by metric."""
    worst_val = 999
    worst_combo = None
    worst_stats = None

    for em in ["E1", "E3"]:
        em_data = data[data["entry_model"] == em]
        for rr in sorted(data["rr_target"].unique()):
            for cb in sorted(data["confirm_bars"].unique()):
                mask = (em_data["rr_target"] == rr) & (em_data["confirm_bars"] == cb)
                trades = em_data[mask]
                if len(trades) < min_trades:
                    continue
                stats = compute_stats(trades["pnl_r"].values)
                if stats is None:
                    continue
                val = stats[metric]
                if val < worst_val:
                    worst_val = val
                    worst_combo = (em, rr, cb)
                    worst_stats = stats

    return worst_combo, worst_stats


def get_oos_trades(data, combo):
    """Get OOS trades for a specific (em, rr, cb) combo."""
    em, rr, cb = combo
    mask = (data["entry_model"] == em) & \
           (data["rr_target"] == rr) & \
           (data["confirm_bars"] == cb)
    return data[mask]["pnl_r"].values


def em_label(em):
    return "E1" if em == "E1" else "E3"


def main():
    df = load_0900_data()

    FILTERS = {"G4": 4.0, "G5": 5.0, "G6": 6.0}
    test_years = [2023, 2024, 2025]
    train_start = 2022  # Skip 2021 (structurally different)

    sep = "=" * 80

    print(sep)
    print("WALK-FORWARD ANALYSIS: 0900 ORB STRATEGIES")
    print(sep)
    print()
    print("Method: Anchored expanding window, zero lookahead")
    print("  Train: 2022 up to (not including) test year")
    print("  Test:  single calendar year of UNSEEN data")
    print("  Select: best Sharpe from training period ONLY")
    print("  Min 30 training trades to qualify")
    print("  E1 and E3 entry models (E2 removed)")
    print("  2021 excluded from training (structurally different)")
    print()

    # =========================================================================
    # SECTION 1: Walk-forward by filter
    # =========================================================================
    for filt_name, filt_thresh in FILTERS.items():
        print(f"--- FILTER: {filt_name} (ORB >= {filt_thresh}pt) ---")
        print()

        eligible = df[df["orb_size"] >= filt_thresh].copy()
        oos_all_pnls = []
        oos_results = []

        for test_year in test_years:
            train_years = list(range(train_start, test_year))
            train_data = eligible[eligible["year"].isin(train_years)]
            test_data = eligible[eligible["year"] == test_year]

            combo, train_stats = find_best_strategy(train_data)

            if combo is None:
                print(f"  Test {test_year} (train {train_years}): NO qualifying strategy")
                continue

            oos_pnls = get_oos_trades(test_data, combo)

            if len(oos_pnls) == 0:
                print(f"  Test {test_year}: Selected {em_label(combo[0])} RR{combo[1]} CB{combo[2]} -- 0 OOS trades")
                continue

            oos_stats = compute_stats(oos_pnls)

            print(f"  Test {test_year} (train {train_years}):")
            print(f"    Selected: {em_label(combo[0])} RR{combo[1]} CB{combo[2]} {filt_name}")
            print(f"    Train:    N={train_stats['n']}, Sharpe={train_stats['sharpe']:.3f}, ExpR={train_stats['expr']:+.3f}")
            print(f"    OOS:      N={oos_stats['n']}, WR={oos_stats['wr']:.0%}, ExpR={oos_stats['expr']:+.3f}, Sharpe={oos_stats['sharpe']:.3f}, MaxDD={oos_stats['maxdd']:+.1f}R, Total={oos_stats['total']:+.1f}R")

            oos_all_pnls.extend(oos_pnls)
            oos_results.append({
                "test_year": test_year,
                "combo": f"{em_label(combo[0])} RR{combo[1]} CB{combo[2]}",
                "train_sharpe": train_stats["sharpe"],
                "oos_stats": oos_stats,
            })

        # Combined OOS summary
        if oos_all_pnls:
            combined = compute_stats(np.array(oos_all_pnls))
            combos_picked = [r["combo"] for r in oos_results]
            stable = len(set(combos_picked)) == 1
            avg_train_sharpe = np.mean([r["train_sharpe"] for r in oos_results])

            print(f"  COMBINED OOS: N={combined['n']}, WR={combined['wr']:.0%}, ExpR={combined['expr']:+.3f}, Sharpe={combined['sharpe']:.3f}, MaxDD={combined['maxdd']:+.1f}R, Total={combined['total']:+.1f}R")
            if stable:
                print(f"  Selection stability: STABLE (same pick every year: {combos_picked[0]})")
            else:
                print(f"  Selection stability: CHANGED across years: {combos_picked}")
            print(f"  Sharpe decay: train avg {avg_train_sharpe:.3f} -> OOS {combined['sharpe']:.3f} ({(combined['sharpe'] / avg_train_sharpe - 1) * 100:+.0f}%)")

            # Every year profitable?
            all_profitable = all(r["oos_stats"]["total"] > 0 for r in oos_results)
            print(f"  Every test year profitable: {'YES' if all_profitable else 'NO'}")

        print()

    # =========================================================================
    # SECTION 2: Honesty checks (G6 only -- our recommended filter)
    # =========================================================================
    print(sep)
    print("HONESTY CHECKS (G6 filter)")
    print(sep)
    print()

    eligible_g6 = df[df["orb_size"] >= 6.0].copy()

    # 2a. Random selection baseline
    print("--- RANDOM SELECTION: pick random qualifying strategy each year (1000 sims) ---")
    np.random.seed(42)
    n_sims = 1000
    sim_totals = []

    for _ in range(n_sims):
        sim_pnls = []
        for test_year in test_years:
            train_years = list(range(train_start, test_year))
            train_data = eligible_g6[eligible_g6["year"].isin(train_years)]
            test_data = eligible_g6[eligible_g6["year"] == test_year]

            combos = []
            for em in ["E1", "E3"]:
                em_data = train_data[train_data["entry_model"] == em]
                for rr in train_data["rr_target"].unique():
                    for cb in train_data["confirm_bars"].unique():
                        mask = (em_data["rr_target"] == rr) & (em_data["confirm_bars"] == cb)
                        if len(em_data[mask]) >= 30:
                            combos.append((em, rr, cb))

            if not combos:
                continue
            combo = combos[np.random.randint(len(combos))]
            sim_pnls.extend(get_oos_trades(test_data, combo))

        if sim_pnls:
            sim_totals.append(sum(sim_pnls))

    sim_totals = np.array(sim_totals)
    print(f"  Total R: median={np.median(sim_totals):+.1f}, mean={np.mean(sim_totals):+.1f}")
    print(f"  Range: p5={np.percentile(sim_totals, 5):+.1f}, p25={np.percentile(sim_totals, 25):+.1f}, p75={np.percentile(sim_totals, 75):+.1f}, p95={np.percentile(sim_totals, 95):+.1f}")
    pct_profitable = (sim_totals > 0).sum() / n_sims * 100
    print(f"  Profitable sims: {pct_profitable:.0f}%")
    print()

    # 2b. Adversarial: worst qualifying strategy from training
    print("--- ADVERSARIAL: always pick WORST Sharpe from training ---")
    adv_pnls = []
    for test_year in test_years:
        train_years = list(range(train_start, test_year))
        train_data = eligible_g6[eligible_g6["year"].isin(train_years)]
        test_data = eligible_g6[eligible_g6["year"] == test_year]

        combo, _ = find_worst_strategy(train_data)
        if combo:
            pnls = get_oos_trades(test_data, combo)
            adv_pnls.extend(pnls)
            print(f"  {test_year}: Picked {em_label(combo[0])} RR{combo[1]} CB{combo[2]}, OOS N={len(pnls)}, Total={sum(pnls):+.1f}R")

    if adv_pnls:
        adv_stats = compute_stats(np.array(adv_pnls))
        print(f"  COMBINED: N={adv_stats['n']}, ExpR={adv_stats['expr']:+.3f}, Total={adv_stats['total']:+.1f}R")
    print()

    # 2c. Stress test: inflate costs 1.5x (already baked into pnl_r via cost model,
    # but let's check: what fraction of wins become losses under 50% higher friction?)
    print("--- STRESS TEST: How sensitive are wins to cost inflation? ---")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    # Get actual entry/stop distances for 0900 G6 trades
    # The pnl_r already includes friction. We need to see how thin the margin is.
    # Approach: count wins with pnl_r between 0 and 0.15R (barely profitable)
    thin_wins = eligible_g6[(eligible_g6["pnl_r"] > 0) & (eligible_g6["pnl_r"] < 0.15)]
    all_wins = eligible_g6[eligible_g6["pnl_r"] > 0]
    print(f"  G6 wins with pnl_r < 0.15R (thin margin): {len(thin_wins)} of {len(all_wins)} wins ({len(thin_wins)/max(len(all_wins),1)*100:.1f}%)")
    print(f"  These trades could flip to losses under higher real-world costs.")

    # What if we add 0.084R of extra friction to every trade (equivalent to doubling costs)?
    # friction = $8.40. At median G6 ORB of ~9pt, risk = $98.40, extra friction = $8.40 = 0.085R
    extra_friction_r = 0.085  # approximate
    adjusted = eligible_g6.copy()
    adjusted["pnl_r_stressed"] = adjusted["pnl_r"] - extra_friction_r

    for test_year in test_years:
        train_years = list(range(train_start, test_year))
        train_data = adjusted[adjusted["year"].isin(train_years)]
        test_data = adjusted[adjusted["year"] == test_year]

        # Use SAME selection as unstressed (best Sharpe on normal pnl_r)
        normal_train = eligible_g6[eligible_g6["year"].isin(train_years)]
        combo, _ = find_best_strategy(normal_train)

        if combo:
            em, rr, cb = combo
            mask = (test_data["entry_model"] == em) & \
                   (test_data["rr_target"] == rr) & \
                   (test_data["confirm_bars"] == cb)
            stressed_pnls = test_data[mask]["pnl_r_stressed"].values
            normal_pnls = test_data[mask]["pnl_r"].values
            if len(stressed_pnls) > 0:
                normal_stats = compute_stats(normal_pnls)
                stress_stats = compute_stats(stressed_pnls)
                print(f"  {test_year} {em_label(combo[0])} RR{combo[1]} CB{combo[2]}: Normal ExpR={normal_stats['expr']:+.3f}, Stressed ExpR={stress_stats['expr']:+.3f}, Stressed Total={stress_stats['total']:+.1f}R")

    con.close()
    print()

    # =========================================================================
    # SECTION 3: Does the FAMILY have edge, or just specific params?
    # =========================================================================
    print(sep)
    print("FAMILY-LEVEL CHECK: Average across ALL 0900 G6 E1 variants")
    print(sep)
    print()
    print("If the family has edge, the AVERAGE of all variants should be positive OOS.")
    print("If only cherry-picked params work, the average will be near zero.")
    print()

    for test_year in test_years:
        train_years = list(range(train_start, test_year))
        test_data = eligible_g6[eligible_g6["year"] == test_year]

        # All E1 variants in test year
        e1_test = test_data[test_data["entry_model"] == "E1"]
        if len(e1_test) > 0:
            # Average ExpR across all RR/CB combos
            combo_exprs = []
            for rr in e1_test["rr_target"].unique():
                for cb in e1_test["confirm_bars"].unique():
                    mask = (e1_test["rr_target"] == rr) & (e1_test["confirm_bars"] == cb)
                    trades = e1_test[mask]
                    if len(trades) > 0:
                        combo_exprs.append(trades["pnl_r"].mean())

            avg_expr = np.mean(combo_exprs) if combo_exprs else 0
            pct_positive = sum(1 for e in combo_exprs if e > 0) / max(len(combo_exprs), 1) * 100
            print(f"  {test_year}: {len(combo_exprs)} E1 variants, avg ExpR={avg_expr:+.3f}, {pct_positive:.0f}% positive")

    print()
    print(sep)
    print("DONE")
    print(sep)


if __name__ == "__main__":
    main()
