"""
Backtest ATR(20) as a regime on/off switch for ORB breakout families.

For each (session, EM, filter) family, compare:
  - BASELINE: trade every eligible day (current behavior)
  - GATED: only trade when ATR(20) > threshold

Reports family-level metrics (averaged across RR/CB variants).
Tests thresholds at 20, 25, 30, 35, 40 (coarse grid).

HONESTY NOTES:
  - Picking the "best" threshold from 6 IS optimization, even if mild.
    Do not treat the exact cutoff as precise.
  - Variant averaging uses a FIXED variant set (determined at baseline)
    to prevent survivorship bias at high thresholds.
  - ATR gating is correlated with the same 2025-2026 regime that
    produced profitable outcomes. It identifies the regime, it does
    not prove causation.

Usage:
    python scripts/backtest_atr_regime.py
    python scripts/backtest_atr_regime.py --db C:/db/gold.db
"""

import sys
import math
from pathlib import Path
from datetime import date
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(line_buffering=True)

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec
from trading_app.config import ALL_FILTERS, ENTRY_MODELS
from trading_app.outcome_builder import RR_TARGETS, CONFIRM_BARS_OPTIONS
from trading_app.strategy_discovery import (
    compute_metrics,
    _load_daily_features,
    _build_filter_day_sets,
)

# Families to test: the live/validated ones
FAMILIES = [
    {"session": "0900", "em": "E1", "filter": "ORB_G4"},
    {"session": "0900", "em": "E1", "filter": "ORB_G5"},
    {"session": "0900", "em": "E1", "filter": "ORB_G6"},
    {"session": "1000", "em": "E1", "filter": "ORB_G3"},
    {"session": "1000", "em": "E1", "filter": "ORB_G4"},
    {"session": "1800", "em": "E3", "filter": "ORB_G4"},
    {"session": "1800", "em": "E3", "filter": "ORB_G6"},
    {"session": "1800", "em": "E1", "filter": "ORB_G6"},
]

ATR_THRESHOLDS = [0, 20, 25, 30, 35, 40]  # 0 = no gate (baseline)

TARGET_SESSIONS = sorted(set(f["session"] for f in FAMILIES))


def compute_family_metrics(outcomes_list, cost_spec, years_span):
    """Compute metrics for a list of outcome dicts."""
    if not outcomes_list:
        return None
    m = compute_metrics(outcomes_list, cost_spec)
    n = m["sample_size"]
    if n == 0:
        return None
    tpy = n / years_span if years_span > 0 else 0
    sharpe = m["sharpe_ratio"]
    shann = sharpe * math.sqrt(tpy) if sharpe is not None and tpy > 0 else None
    gross_wins = sum(o["pnl_r"] for o in outcomes_list if o.get("pnl_r") is not None and o["pnl_r"] > 0)
    gross_losses = abs(sum(o["pnl_r"] for o in outcomes_list if o.get("pnl_r") is not None and o["pnl_r"] < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else None
    total_r = sum(o["pnl_r"] for o in outcomes_list if o.get("pnl_r") is not None)
    return {
        "n": n,
        "wr": m["win_rate"],
        "expr": m["expectancy_r"],
        "sharpe": sharpe,
        "shann": round(shann, 3) if shann is not None else None,
        "maxdd": m["max_drawdown_r"],
        "pf": round(pf, 3) if pf is not None else None,
        "total_r": round(total_r, 2),
        "tpy": round(tpy, 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest ATR regime gate")
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--instrument", default="MGC")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    instrument = args.instrument
    orb_minutes = 5
    cost_spec = get_cost_spec(instrument)

    print(f"Database: {db_path}")
    print(f"Instrument: {instrument}")
    print(f"ATR thresholds: {ATR_THRESHOLDS}")
    print(f"Families: {len(FAMILIES)}")
    print()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Load features with ATR
        print("Loading daily features...")
        features = _load_daily_features(con, instrument, orb_minutes, None, None)
        print(f"  {len(features)} rows")

        # Count date range for years_span
        dates = [f["trading_day"] for f in features]
        if dates:
            min_d, max_d = min(dates), max(dates)
            years_span = (max_d - min_d).days / 365.25
        else:
            years_span = 1.0
        print(f"  Date range: {min_d} to {max_d} ({years_span:.1f} years)")

        # Build filter day sets (with ATR info attached)
        print("Building filter day sets...")
        filter_days = _build_filter_day_sets(features, TARGET_SESSIONS, ALL_FILTERS)

        # Build ATR lookup: trading_day -> atr_20
        atr_by_day = {}
        for f in features:
            atr_val = f.get("atr_20")
            if atr_val is not None:
                atr_by_day[f["trading_day"]] = atr_val

        print(f"  Days with ATR data: {len(atr_by_day)}")

        # Load outcomes
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
                       ORDER BY trading_day""",
                    [instrument, orb_minutes, orb_label, em],
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
        print(f"  {sum(len(v) for v in outcomes_by_key.values())} outcome rows")

        # For each family x threshold, compute family-level metrics
        # (average across all RR/CB variants)
        print("\n" + "=" * 130)
        print(f"{'Family':<28} {'ATR>':<5} {'#V':>3} {'N':>5} {'T/Yr':>5} {'WR':>6} {'ExpR':>7} "
              f"{'Sharpe':>7} {'ShANN':>7} {'PF':>6} {'MaxDD':>7} {'TotalR':>8} {'vs Base':>8}")
        print("=" * 130)

        for family in FAMILIES:
            session = family["session"]
            em = family["em"]
            filt = family["filter"]
            family_label = f"{session}_{em}_{filt}"

            matching_days_base = filter_days.get((filt, session), set())

            baseline_total_r = None

            # HONESTY FIX: Determine valid variant set at baseline (N>=10
            # with no ATR gate). Reuse this SAME set for all thresholds
            # to prevent survivorship bias from dropping variants.
            baseline_variant_keys = []
            for rr in RR_TARGETS:
                for cb in CONFIRM_BARS_OPTIONS:
                    if em == "E3" and cb > 1:
                        continue
                    key = (session, em, rr, cb)
                    variant_outcomes = outcomes_by_key.get(key, [])
                    baseline_filtered = [o for o in variant_outcomes
                                         if o["trading_day"] in matching_days_base]
                    if len(baseline_filtered) >= 10:
                        baseline_variant_keys.append(key)

            if not baseline_variant_keys:
                print(f"{family_label:<28}   -- no variants with N>=10 at baseline --")
                print()
                continue

            for threshold in ATR_THRESHOLDS:
                # Apply ATR gate to matching days
                if threshold == 0:
                    matching_days = matching_days_base
                else:
                    matching_days = {
                        d for d in matching_days_base
                        if atr_by_day.get(d, 0) >= threshold
                    }

                if not matching_days:
                    print(f"{family_label:<28} {threshold:>4}  {'-- no eligible days --'}")
                    continue

                # Compute per-variant metrics using FIXED variant set
                variant_metrics = []
                for key in baseline_variant_keys:
                    variant_outcomes = [o for o in outcomes_by_key.get(key, [])
                                        if o["trading_day"] in matching_days]
                    if not variant_outcomes:
                        continue
                    vm = compute_family_metrics(variant_outcomes, cost_spec, years_span)
                    if vm:
                        variant_metrics.append(vm)

                if not variant_metrics:
                    print(f"{family_label:<28} {threshold:>4}  {'-- insufficient data --'}")
                    continue

                # Family average across variants
                avg_n = round(sum(v["n"] for v in variant_metrics) / len(variant_metrics))
                avg_tpy = round(sum(v["tpy"] for v in variant_metrics) / len(variant_metrics), 1)
                avg_wr = round(sum(v["wr"] for v in variant_metrics) / len(variant_metrics), 4)
                avg_expr = round(sum(v["expr"] for v in variant_metrics) / len(variant_metrics), 4)
                sharpes = [v["sharpe"] for v in variant_metrics if v["sharpe"] is not None]
                avg_sharpe = round(sum(sharpes) / len(sharpes), 4) if sharpes else None
                shanns = [v["shann"] for v in variant_metrics if v["shann"] is not None]
                avg_shann = round(sum(shanns) / len(shanns), 3) if shanns else None
                pfs = [v["pf"] for v in variant_metrics if v["pf"] is not None]
                avg_pf = round(sum(pfs) / len(pfs), 3) if pfs else None
                avg_maxdd = round(max(v["maxdd"] for v in variant_metrics), 2)
                avg_total_r = round(sum(v["total_r"] for v in variant_metrics) / len(variant_metrics), 2)

                if threshold == 0:
                    baseline_total_r = avg_total_r
                    delta_str = "baseline"
                elif baseline_total_r is not None:
                    delta = avg_total_r - baseline_total_r
                    delta_str = f"{delta:+.2f}R"
                else:
                    delta_str = "N/A"

                shann_str = f"{avg_shann:>7.3f}" if avg_shann is not None else "    N/A"
                sharpe_str = f"{avg_sharpe:>7.4f}" if avg_sharpe is not None else "    N/A"
                pf_str = f"{avg_pf:>6.3f}" if avg_pf is not None else "   N/A"

                n_variants = len(variant_metrics)
                print(f"{family_label:<28} {threshold:>4} {n_variants:>3} {avg_n:>5} {avg_tpy:>5.1f} "
                      f"{avg_wr:>5.1%} {avg_expr:>7.4f} {sharpe_str} {shann_str} "
                      f"{pf_str} {avg_maxdd:>7.2f} {avg_total_r:>8.2f} {delta_str:>8}")

            print()  # blank line between families

        # Summary: ATR distribution in the data
        print("=" * 80)
        print("ATR(20) Regime Frequency:")
        for threshold in ATR_THRESHOLDS:
            if threshold == 0:
                continue
            above = sum(1 for v in atr_by_day.values() if v >= threshold)
            pct = 100 * above / len(atr_by_day) if atr_by_day else 0
            print(f"  ATR >= {threshold}: {above} days ({pct:.1f}%)")

    finally:
        con.close()


if __name__ == "__main__":
    main()
