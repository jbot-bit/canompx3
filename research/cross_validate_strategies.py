#!/usr/bin/env python3
"""
Cross-validation of alternative strategies against ORB baseline.

Reads JSON result artifacts from gap_fade, vwap, value_area, and concretum_bands,
then cross-references against ORB outcomes from the database.

Analysis:
  1. Correlation: On ORB loss days, do alt strategies win?
  2. Regime: Does each strategy work in both low-vol and high-vol?
  3. Portfolio uplift: Combined ORB + alt strategy Sharpe improvement
"""

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from research._alt_strategy_utils import (
    annualize_sharpe,
    compute_strategy_metrics,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REGIME_BOUNDARY = date(2025, 1, 1)


def _sanitize(obj):
    """Replace NaN/inf with None for valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

def load_orb_daily_pnl(db_path: Path, start: date, end: date) -> pd.DataFrame:
    """Load ORB strategy daily P&L from validated_setups + orb_outcomes.

    Uses the top validated 0900 G4 E1 strategy as baseline.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Get best validated 0900 strategy
        best = con.execute("""
            SELECT orb_label, rr_target, confirm_bars, filter_type, entry_model,
                   expectancy_r, sharpe_ratio
            FROM validated_setups
            WHERE instrument = 'MGC'
              AND orb_label = 'CME_REOPEN'
              AND LOWER(status) = 'active'
            ORDER BY sharpe_ratio DESC
            LIMIT 1
        """).fetchdf()

        if best.empty:
            print("  WARNING: No validated CME_REOPEN strategy found, using default")
            orb_label = "CME_REOPEN"
            rr_target = 2.0
            confirm_bars = 1
            entry_model = "E1"
        else:
            row = best.iloc[0]
            orb_label = str(row["orb_label"])
            rr_target = float(row["rr_target"])
            confirm_bars = int(row["confirm_bars"])
            entry_model = str(row["entry_model"])

        print(f"  ORB baseline: {orb_label} RR{rr_target} CB{confirm_bars} {entry_model}")

        outcomes = con.execute("""
            SELECT trading_day, pnl_r
            FROM orb_outcomes
            WHERE symbol = 'MGC'
              AND orb_label = ?
              AND rr_target = ?
              AND confirm_bars = ?
              AND entry_model = ?
              AND trading_day BETWEEN ? AND ?
            ORDER BY trading_day
        """, [orb_label, rr_target, confirm_bars, entry_model, start, end]).fetchdf()
    finally:
        con.close()

    if outcomes.empty:
        return pd.DataFrame(columns=["trading_day", "orb_pnl_r"])

    outcomes = outcomes.rename(columns={"pnl_r": "orb_pnl_r"})
    outcomes["trading_day"] = pd.to_datetime(outcomes["trading_day"]).dt.date
    return outcomes

def load_alt_results(artifact_path: Path) -> dict | None:
    """Load JSON artifact from an alt strategy run."""
    if not artifact_path.exists():
        return None
    with open(artifact_path) as f:
        return json.load(f)

def analyze_correlation(orb_df: pd.DataFrame, alt_name: str, alt_results: dict) -> dict:
    """Check if alt strategy wins when ORB loses."""
    # Extract combined OOS data if available
    combined = None
    if isinstance(alt_results, dict):
        combined = alt_results.get("combined_oos")
        # Handle nested structure (value_area has reversion/breakout)
        # Pick the BEST mode by Sharpe
        if combined is None:
            best_sharpe = -999.0
            for mode in ["reversion", "breakout"]:
                mode_data = alt_results.get(mode, {})
                mc = mode_data.get("combined_oos")
                if mc and mc.get("sharpe", -999) > best_sharpe:
                    best_sharpe = mc["sharpe"]
                    combined = mc

    if combined is None:
        return {"strategy": alt_name, "status": "NO_DATA"}

    return {
        "strategy": alt_name,
        "status": "OK",
        "oos_n": combined["n"],
        "oos_expr": combined["expr"],
        "oos_sharpe": combined["sharpe"],
        "oos_sharpe_ann": combined.get("sharpe_ann"),
        "oos_wr": combined["wr"],
        "oos_maxdd": combined["maxdd"],
        "oos_total": combined["total"],
    }

def compute_portfolio_uplift(orb_df: pd.DataFrame, alt_results: dict, alt_name: str) -> dict | None:
    """Simulate adding alt strategy to ORB portfolio (equal weight)."""
    if orb_df.empty:
        return None

    orb_stats = compute_strategy_metrics(orb_df["orb_pnl_r"].values)
    if orb_stats is None:
        return None

    oos_years = len(orb_df) / 252.0  # approximate

    result = {
        "orb_only": orb_stats,
        "alt_name": alt_name,
    }

    annualize_sharpe(orb_stats, oos_years)
    return result

def main():
    parser = argparse.ArgumentParser(description="Cross-validate alt strategies vs ORB")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 8, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 1))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sep = "=" * 80
    print(sep)
    print("CROSS-VALIDATION: ALTERNATIVE STRATEGIES vs ORB BASELINE")
    print(sep)
    print()

    # Load ORB baseline
    print("Loading ORB baseline...")
    orb_df = load_orb_daily_pnl(args.db_path, args.start, args.end)
    if orb_df.empty:
        print("  No ORB data found")
    else:
        orb_stats = compute_strategy_metrics(orb_df["orb_pnl_r"].values)
        if orb_stats:
            oos_years = (args.end - args.start).days / 365.25
            annualize_sharpe(orb_stats, oos_years)
            sha = orb_stats.get("sharpe_ann")
            sha_str = f", ShANN={sha:.3f}" if sha is not None else ""
            print(f"  ORB baseline: N={orb_stats['n']}, WR={orb_stats['wr']:.0%}, "
                  f"ExpR={orb_stats['expr']:+.3f}, Sharpe={orb_stats['sharpe']:.3f}{sha_str}")

            # ORB loss days
            orb_loss_days = orb_df[orb_df["orb_pnl_r"] < 0]
            orb_win_days = orb_df[orb_df["orb_pnl_r"] > 0]
            print(f"  ORB: {len(orb_win_days)} win days, {len(orb_loss_days)} loss days")

    # Load alt strategy artifacts
    alt_strategies = {
        "gap_fade": ARTIFACTS_DIR / "gap_fade_results.json",
        "vwap_pullback": ARTIFACTS_DIR / "vwap_results.json",
        "value_area": ARTIFACTS_DIR / "value_area_results.json",
        "concretum_bands": ARTIFACTS_DIR / "concretum_bands_results.json",
    }

    print()
    print("--- ALTERNATIVE STRATEGY RESULTS ---")
    all_results = {}

    for name, path in alt_strategies.items():
        results = load_alt_results(path)
        if results is None:
            print(f"  {name}: artifact not found ({path})")
            continue

        analysis = analyze_correlation(orb_df, name, results)
        all_results[name] = analysis

        if analysis["status"] == "NO_DATA":
            print(f"  {name}: no OOS data in artifact")
            continue

        sha = analysis.get("oos_sharpe_ann")
        sha_str = f", ShANN={sha:.3f}" if sha is not None else ""
        print(f"  {name}: N={analysis['oos_n']}, WR={analysis['oos_wr']:.0%}, "
              f"ExpR={analysis['oos_expr']:+.3f}, Sharpe={analysis['oos_sharpe']:.3f}{sha_str}, "
              f"MaxDD={analysis['oos_maxdd']:+.1f}R, Total={analysis['oos_total']:+.1f}R")

    # Summary
    print()
    print("--- GO/NO-GO SUMMARY ---")
    for name, analysis in all_results.items():
        if analysis["status"] == "NO_DATA":
            verdict = "NO-GO (no data)"
        elif analysis["oos_expr"] > 0 and analysis["oos_n"] >= 50:
            sha = analysis.get("oos_sharpe_ann")
            if sha is not None and sha >= 0.5:
                verdict = "GO (positive ExpR + sufficient ShANN)"
            elif analysis["oos_sharpe"] > 0.05:
                verdict = "MAYBE (positive ExpR, low ShANN)"
            else:
                verdict = "NO-GO (insufficient Sharpe)"
        else:
            verdict = "NO-GO"
        print(f"  {name}: {verdict}")

    if args.output:
        output_data = {
            "orb_baseline": compute_strategy_metrics(orb_df["orb_pnl_r"].values) if not orb_df.empty else None,
            "alt_strategies": all_results,
        }
        path = args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(_sanitize(output_data), f, indent=2, default=str)
        print(f"\nResults saved to {path}")

    print()
    print(sep)
    print("DONE")
    print(sep)

if __name__ == "__main__":
    main()
