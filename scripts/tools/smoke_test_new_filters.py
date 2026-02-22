#!/usr/bin/env python3
"""
Smoke test: validate InsideDayFilter and EaseDayFilter against real data.

Decision gates (per user instructions):
  - If OOS Sharpe <= baseline G4 -> kill filter immediately
  - If sample < 50 -> kill filter immediately
  - If stability score (rolling) worse -> kill filter immediately

Run ONLY on 0900/1000, G4+, E1 first. If no uplift, don't waste compute
on full grid.

Read-only. No DB writes.

Usage:
    python scripts/smoke_test_new_filters.py --db-path C:/db/gold.db
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, InsideDayFilter, EaseDayFilter
from trading_app.strategy_discovery import (
    _load_daily_features,
    _enrich_inside_day_tags,
    _enrich_ease_tags,
    _build_filter_day_sets,
    _load_outcomes_bulk,
    compute_metrics,
)
from trading_app.outcome_builder import RR_TARGETS, CONFIRM_BARS_OPTIONS

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config: narrow scope per user instructions
# ---------------------------------------------------------------------------
TEST_ORBS = ["0900", "1000"]
TEST_EMS = ["E1"]
TEST_RRS = [1.5, 2.0, 2.5, 3.0]
TEST_CBS = [1, 2, 3]

# Filters to compare: baseline vs new
COMPARE_FILTERS = {
    "NO_FILTER": ALL_FILTERS["NO_FILTER"],
    "ORB_G4": ALL_FILTERS["ORB_G4"],
    "ORB_G5": ALL_FILTERS["ORB_G5"],
    "ORB_G6": ALL_FILTERS["ORB_G6"],
    # Compound: context + G4 (intersection)
    "INSIDE_DAY": ALL_FILTERS["INSIDE_DAY"],
    "EASE_DAY": ALL_FILTERS["EASE_DAY"],
}

# Decision thresholds
MIN_SAMPLE = 50
MIN_SHARPE_UPLIFT = 0.0  # must beat G4 baseline

def run_smoke_test(db_path: Path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # --- Load & enrich ---
        print("Loading daily features...")
        features = _load_daily_features(con, "MGC", 5, None, None)
        print(f"  {len(features)} rows")

        # Sorting guard
        for i in range(1, len(features)):
            assert features[i]["trading_day"] >= features[i - 1]["trading_day"], (
                "Features not sorted by trading_day"
            )

        _enrich_inside_day_tags(features)
        _enrich_ease_tags(features)

        # --- Enrichment stats ---
        inside_true = sum(1 for f in features if f.get("inside_day") is True)
        inside_false = sum(1 for f in features if f.get("inside_day") is False)
        inside_none = sum(1 for f in features if f.get("inside_day") is None)
        ease_pos = sum(1 for f in features if (f.get("prior_day_ease") or 0) > 0)
        ease_neg = sum(1 for f in features if (f.get("prior_day_ease") or 0) < 0)
        ease_none = sum(1 for f in features if f.get("prior_day_ease") is None)

        print(f"  Inside days: {inside_true} true, {inside_false} false, {inside_none} null")
        print(f"  Ease tags:   {ease_pos} positive, {ease_neg} negative, {ease_none} null")
        print(f"  Inside day rate: {inside_true/(inside_true+inside_false)*100:.1f}%"
              if (inside_true + inside_false) > 0 else "  Inside day rate: N/A")
        print()

        # --- Eligible days ---
        filter_days = _build_filter_day_sets(features, TEST_ORBS, COMPARE_FILTERS)

        print("Eligible days per (filter, orb):")
        for orb in TEST_ORBS:
            for fk in COMPARE_FILTERS:
                days = filter_days.get((fk, orb), set())
                print(f"  {fk:15s} {orb}: {len(days):>5} days")
            print()

        # FIX5 invariant: context filter days must be subset of NO_FILTER days
        for orb in TEST_ORBS:
            nf = filter_days.get(("NO_FILTER", orb), set())
            for ctx_fk in ["INSIDE_DAY", "EASE_DAY"]:
                ctx = filter_days.get((ctx_fk, orb), set())
                overflow = ctx - nf
                if overflow:
                    print(f"  *** FIX5 VIOLATION: {ctx_fk}/{orb} has {len(overflow)} days outside NO_FILTER!")
                    return
            print(f"  FIX5 OK: {orb} context filters are subsets of NO_FILTER")
        print()

        # --- Load outcomes ---
        print("Loading outcomes for 0900/1000, E1...")
        outcomes = _load_outcomes_bulk(con, "MGC", 5, TEST_ORBS, TEST_EMS)
        total_outcomes = sum(len(v) for v in outcomes.values())
        print(f"  {total_outcomes} outcome rows")
        print()

        # --- Compute metrics for each combo ---
        # Also compute compound filters: INSIDE_DAY + G4 (intersection)
        compound_filters = {}
        for orb in TEST_ORBS:
            g4_days = filter_days.get(("ORB_G4", orb), set())
            inside_days = filter_days.get(("INSIDE_DAY", orb), set())
            ease_days = filter_days.get(("EASE_DAY", orb), set())
            compound_filters[("INSIDE+G4", orb)] = inside_days & g4_days
            compound_filters[("EASE+G4", orb)] = ease_days & g4_days

        all_filter_keys = list(COMPARE_FILTERS.keys()) + ["INSIDE+G4", "EASE+G4"]

        print("=" * 85)
        print("SMOKE TEST: New Filters vs G4 Baseline")
        print("=" * 85)
        print(f"{'Filter':15s} {'ORB':5s} {'RR':4s} {'CB':3s} {'N':>5s} {'WR':>7s} "
              f"{'ExpR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'Verdict':>8s}")
        print("-" * 85)

        # Collect G4 baselines for comparison
        g4_baselines = {}  # (orb, rr, cb) -> sharpe

        for orb in TEST_ORBS:
            for rr in TEST_RRS:
                for cb in TEST_CBS:
                    all_o = outcomes.get((orb, "E1", rr, cb), [])
                    if not all_o:
                        continue

                    for fk in all_filter_keys:
                        # Get matching days
                        if fk in COMPARE_FILTERS:
                            matching = filter_days.get((fk, orb), set())
                        else:
                            matching = compound_filters.get((fk, orb), set())

                        filtered = [o for o in all_o if o["trading_day"] in matching]
                        if not filtered:
                            continue

                        m = compute_metrics(filtered)
                        if m["sample_size"] == 0:
                            continue

                        sr = m["sharpe_ratio"] if m["sharpe_ratio"] is not None else 0.0

                        # Store G4 baseline
                        if fk == "ORB_G4":
                            g4_baselines[(orb, rr, cb)] = sr

                        # Verdict for new filters
                        verdict = ""
                        if fk in ("INSIDE_DAY", "EASE_DAY", "INSIDE+G4", "EASE+G4"):
                            baseline_sr = g4_baselines.get((orb, rr, cb), 0.0)
                            if m["sample_size"] < MIN_SAMPLE:
                                verdict = "LOW_N"
                            elif sr > baseline_sr + MIN_SHARPE_UPLIFT:
                                verdict = "BEAT"
                            elif sr > baseline_sr:
                                verdict = "~SAME"
                            else:
                                verdict = "WORSE"

                        print(f"{fk:15s} {orb:5s} {rr:4.1f} {cb:3d} {m['sample_size']:>5d} "
                              f"{m['win_rate']:>7.3f} {m['expectancy_r']:>7.3f} "
                              f"{sr:>7.3f} {m['max_drawdown_r']:>7.2f} {verdict:>8s}")

            print()

        # --- Summary verdict ---
        print("=" * 85)
        print("DECISION SUMMARY")
        print("=" * 85)
        print()
        print("If ALL new filter rows show WORSE or LOW_N -> kill the filter.")
        print("If ANY show BEAT with sample >= 50 -> proceed to full grid + walk-forward.")
        print("Compound filters (INSIDE+G4, EASE+G4) are the real test --")
        print("standalone INSIDE_DAY/EASE_DAY will have worse ExpR (includes small ORBs).")

    finally:
        con.close()

def main():
    parser = argparse.ArgumentParser(description="Smoke test new filters")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    args = parser.parse_args()
    run_smoke_test(args.db_path)

if __name__ == "__main__":
    main()
