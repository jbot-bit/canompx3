"""
Find strategies matching Profit Factor + Annualized Sharpe criteria.

Recomputes metrics from raw orb_outcomes for a 2-year window,
restricted to structural liquidity sessions (0900, 1000, 1800).

Usage:
    python scripts/find_pf_strategy.py
    python scripts/find_pf_strategy.py --db C:/db/gold.db
    python scripts/find_pf_strategy.py --min-pf 1.3 --max-pf 2.5
"""

import sys
import math
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, ENTRY_MODELS
from trading_app.outcome_builder import RR_TARGETS, CONFIRM_BARS_OPTIONS
from trading_app.strategy_discovery import (
    compute_metrics,
    make_strategy_id,
    _load_daily_features,
    _build_filter_day_sets,
)

# --- Configuration ---
TARGET_SESSIONS = ["0900", "1000", "1800"]
START_DATE = date(2024, 2, 12)
END_DATE = date(2026, 2, 4)
YEARS_SPAN = 2.0  # for trades_per_year calc

MIN_PF = 1.5
MAX_PF = 2.0
MIN_SHANN = 0.8
MAX_SHANN = 1.5
MAX_WR = 0.75
MIN_TRADES = 30

def compute_profit_factor(outcomes: list[dict]) -> float | None:
    """Profit factor = gross_wins / gross_losses."""
    gross_wins = sum(o["pnl_r"] for o in outcomes if o["pnl_r"] is not None and o["pnl_r"] > 0)
    gross_losses = abs(sum(o["pnl_r"] for o in outcomes if o["pnl_r"] is not None and o["pnl_r"] < 0))
    if gross_losses == 0:
        return None  # infinite or undefined
    return gross_wins / gross_losses

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Find strategies by PF + Sharpe")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--instrument", default="MGC")
    parser.add_argument("--min-pf", type=float, default=MIN_PF)
    parser.add_argument("--max-pf", type=float, default=MAX_PF)
    parser.add_argument("--min-shann", type=float, default=MIN_SHANN)
    parser.add_argument("--max-shann", type=float, default=MAX_SHANN)
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES)
    parser.add_argument("--max-wr", type=float, default=MAX_WR)
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    instrument = args.instrument
    orb_minutes = 5
    print(f"Database: {db_path}")
    print(f"Window: {START_DATE} to {END_DATE} (~{YEARS_SPAN} years)")
    print(f"Sessions: {TARGET_SESSIONS}")
    print(f"Criteria: PF [{args.min_pf}, {args.max_pf}], "
          f"ShANN [{args.min_shann}, {args.max_shann}], "
          f"WR <= {args.max_wr}, N >= {args.min_trades}")
    print()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Step 1: Load daily features for the window
        print("Loading daily features...")
        features = _load_daily_features(con, instrument, orb_minutes, START_DATE, END_DATE)
        print(f"  {len(features)} rows")

        # Step 2: Build filter day sets
        print("Building filter day sets...")
        filter_days = _build_filter_day_sets(features, TARGET_SESSIONS, ALL_FILTERS)

        # Step 3: Bulk load outcomes for target sessions
        print("Loading outcomes...")
        outcomes_by_key = {}
        for orb_label in TARGET_SESSIONS:
            for em in ENTRY_MODELS:
                rows = con.execute(
                    """SELECT trading_day, rr_target, confirm_bars,
                              outcome, pnl_r, mae_r, mfe_r,
                              entry_price, stop_price
                       FROM orb_outcomes
                       WHERE symbol = ? AND orb_minutes = ?
                         AND orb_label = ? AND entry_model = ?
                         AND outcome IS NOT NULL
                         AND trading_day >= ? AND trading_day <= ?
                       ORDER BY trading_day""",
                    [instrument, orb_minutes, orb_label, em, START_DATE, END_DATE],
                ).fetchall()

                for r in rows:
                    key = (orb_label, em, r[1], r[2])
                    if key not in outcomes_by_key:
                        outcomes_by_key[key] = []
                    outcomes_by_key[key].append({
                        "trading_day": r[0],
                        "outcome": r[3],
                        "pnl_r": r[4],
                        "mae_r": r[5],
                        "mfe_r": r[6],
                        "entry_price": r[7],
                        "stop_price": r[8],
                    })

        total_outcomes = sum(len(v) for v in outcomes_by_key.values())
        print(f"  {total_outcomes} outcome rows loaded")

        # Step 4: Grid search with criteria filtering
        print("Scanning grid...")
        candidates = []
        combos_checked = 0

        for filter_key in ALL_FILTERS:
            for orb_label in TARGET_SESSIONS:
                matching_days = filter_days.get((filter_key, orb_label), set())
                if not matching_days:
                    continue

                for em in ENTRY_MODELS:
                    for rr_target in RR_TARGETS:
                        for cb in CONFIRM_BARS_OPTIONS:
                            if em == "E3" and cb > 1:
                                continue
                            combos_checked += 1

                            all_outcomes = outcomes_by_key.get(
                                (orb_label, em, rr_target, cb), []
                            )
                            outcomes = [
                                o for o in all_outcomes
                                if o["trading_day"] in matching_days
                            ]

                            if len(outcomes) < args.min_trades:
                                continue

                            metrics = compute_metrics(outcomes)
                            n = metrics["sample_size"]
                            if n < args.min_trades:
                                continue

                            wr = metrics["win_rate"]
                            if wr is None or wr > args.max_wr:
                                continue

                            # Compute profit factor
                            pf = compute_profit_factor(outcomes)
                            if pf is None or pf < args.min_pf or pf > args.max_pf:
                                continue

                            # Annualized Sharpe (use actual 2-year span)
                            sharpe = metrics["sharpe_ratio"]
                            if sharpe is None:
                                continue
                            trades_per_year = n / YEARS_SPAN
                            shann = sharpe * math.sqrt(trades_per_year)
                            if shann < args.min_shann or shann > args.max_shann:
                                continue

                            strategy_id = make_strategy_id(
                                instrument, orb_label, em, rr_target, cb, filter_key,
                            )

                            candidates.append({
                                "strategy_id": strategy_id,
                                "session": orb_label,
                                "em": em,
                                "rr": rr_target,
                                "cb": cb,
                                "filter": filter_key,
                                "n": n,
                                "wr": wr,
                                "pf": round(pf, 3),
                                "shann": round(shann, 3),
                                "expr": metrics["expectancy_r"],
                                "maxdd": metrics["max_drawdown_r"],
                                "sharpe_pt": sharpe,
                                "tpy": round(trades_per_year, 1),
                            })

        # Step 5: Sort and display
        candidates.sort(key=lambda x: x["expr"], reverse=True)

        print(f"\nChecked {combos_checked} combos, found {len(candidates)} matching\n")

        if not candidates:
            print("No strategies matched all criteria.")
            # Show near-misses: relax PF to [1.2, 2.5] and ShANN to [0.5, 2.0]
            print("\nRelaxing criteria for near-misses...")
            # Re-scan would be expensive, skip for now
            return

        # Header
        print(f"{'Strategy ID':<42} {'Sess':>4} {'EM':>2} {'RR':>4} {'CB':>2} "
              f"{'Filter':<12} {'N':>4} {'WR':>5} {'PF':>5} {'ShANN':>6} "
              f"{'ExpR':>6} {'MaxDD':>6} {'T/Yr':>5}")
        print("-" * 120)

        for c in candidates:
            print(f"{c['strategy_id']:<42} {c['session']:>4} {c['em']:>2} "
                  f"{c['rr']:>4.1f} {c['cb']:>2} {c['filter']:<12} "
                  f"{c['n']:>4} {c['wr']:>5.1%} {c['pf']:>5.2f} "
                  f"{c['shann']:>6.3f} {c['expr']:>6.4f} "
                  f"{c['maxdd']:>6.2f} {c['tpy']:>5.1f}")

        # Summary stats
        print(f"\n--- Summary ---")
        print(f"Total matches: {len(candidates)}")
        sessions = set(c["session"] for c in candidates)
        for s in sorted(sessions):
            sc = [c for c in candidates if c["session"] == s]
            print(f"  {s}: {len(sc)} strategies")
        ems = set(c["em"] for c in candidates)
        for em in sorted(ems):
            ec = [c for c in candidates if c["em"] == em]
            print(f"  {em}: {len(ec)} strategies")

    finally:
        con.close()

if __name__ == "__main__":
    main()
