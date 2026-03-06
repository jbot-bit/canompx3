#!/usr/bin/env python3
"""
Automated sensitivity analysis for live portfolio strategies.

Tests parameter stability by sweeping ±1 step on RR target, confirm bars,
and adjacent filter thresholds. Answers: "If I perturb this parameter,
does the edge survive?"

RESEARCH_RULES.md requirement: ±20% parameter sweep and ±2 unit threshold
shifts for all live strategies before deployment.

Tests:
  1. RR sensitivity - sweep ±1 RR step (e.g., 2.0 → 1.5, 2.5)
  2. Confirm bars sensitivity - sweep ±1 CB step (E2 always CB1)
  3. Filter threshold sensitivity - sweep ±1 G-level (G5 → G4, G6)
  4. Cost resilience - test at 2× slippage ticks

Usage:
    python scripts/tools/sensitivity_analysis.py
    python scripts/tools/sensitivity_analysis.py --instrument MGC
    python scripts/tools/sensitivity_analysis.py --family TOKYO_OPEN_E2_ORB_G5 --instrument MGC
"""

import argparse
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import get_cost_spec  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.config import ALL_FILTERS, TRADEABLE_INSTRUMENTS  # noqa: E402
from trading_app.live_config import LIVE_PORTFOLIO, LIVE_MIN_EXPECTANCY_R  # noqa: E402

# Ordered RR targets used in outcome_builder grid
RR_STEPS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

# Ordered CB options (E1 supports 1-5; E2/E3 always CB1)
CB_STEPS = [1, 2, 3, 4, 5]

# Adjacent G-filter escalation ladder
G_LADDER = ["NO_FILTER", "ORB_G4", "ORB_G5", "ORB_G6", "ORB_G8"]

# Stability threshold: edge must retain ≥50% of baseline ExpR to be "stable"
STABILITY_THRESHOLD = 0.50

# Minimum sample size for perturbed result to be meaningful
MIN_PERTURB_N = 20


def compute_metrics_from_rows(rows: list) -> dict:
    """Compute key metrics from raw (outcome, pnl_r) tuples."""
    wins = [r for r in rows if r[0] == "win"]
    losses = [r for r in rows if r[0] == "loss"]
    n = len(wins) + len(losses)
    if n == 0:
        return {"N": 0, "WR": None, "ExpR": None, "Sharpe": None}

    wr = len(wins) / n
    pnl_rs = [r[1] for r in wins + losses]
    avg_r = sum(pnl_rs) / n

    if n > 1:
        var = sum((x - avg_r) ** 2 for x in pnl_rs) / (n - 1)
        std = var**0.5
        sharpe = avg_r / std if std > 0 else 0.0
    else:
        sharpe = 0.0

    avg_win = sum(r[1] for r in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(r[1] for r in losses) / len(losses)) if losses else 0
    expr = (wr * avg_win) - ((1 - wr) * avg_loss)

    return {"N": n, "WR": wr, "ExpR": expr, "Sharpe": sharpe}


def get_adjacent_rr(rr: float) -> list[float]:
    """Return ±1 step RR targets."""
    if rr not in RR_STEPS:
        return []
    idx = RR_STEPS.index(rr)
    result = []
    if idx > 0:
        result.append(RR_STEPS[idx - 1])
    if idx < len(RR_STEPS) - 1:
        result.append(RR_STEPS[idx + 1])
    return result


def get_adjacent_cb(cb: int, entry_model: str) -> list[int]:
    """Return ±1 step CB values. E2/E3 always CB1 - no sweep."""
    if entry_model in ("E2", "E3"):
        return []
    if cb not in CB_STEPS:
        return []
    idx = CB_STEPS.index(cb)
    result = []
    if idx > 0:
        result.append(CB_STEPS[idx - 1])
    if idx < len(CB_STEPS) - 1:
        result.append(CB_STEPS[idx + 1])
    return result


def get_adjacent_filters(filter_type: str) -> list[str]:
    """Return adjacent G-level filters for ORB size filters."""
    # Only sweep G-ladder filters (ORB_G4 → ORB_G5, etc.)
    base = filter_type.split("_NOFRI")[0].split("_NOMON")[0].split("_NOTUE")[0]
    base = base.split("_CONT")[0].split("_FAST")[0]

    if base in G_LADDER:
        idx = G_LADDER.index(base)
        result = []
        if idx > 0 and G_LADDER[idx - 1] in ALL_FILTERS:
            result.append(G_LADDER[idx - 1])
        if idx < len(G_LADDER) - 1 and G_LADDER[idx + 1] in ALL_FILTERS:
            result.append(G_LADDER[idx + 1])
        return result
    return []


def query_strategy_outcomes(
    con,
    instrument: str,
    orb_label: str,
    entry_model: str,
    filter_type: str,
    rr_target: float,
    confirm_bars: int,
    orb_minutes_list: list[int] | None = None,
) -> list[tuple]:
    """Query orb_outcomes joined with daily_features filter eligibility.

    Returns list of (outcome, pnl_r) tuples for win/loss trades.
    """
    if orb_minutes_list is None:
        orb_minutes_list = [5, 15, 30]

    # Strip aperture suffix (_O5, _O15, _O30) to get base filter name
    base_ft = filter_type
    for suffix in ("_O5", "_O15", "_O30"):
        if base_ft.endswith(suffix):
            base_ft = base_ft[: -len(suffix)]
            break
    filt = ALL_FILTERS.get(base_ft)
    if filt is None:
        return []

    # Build eligible day set from orb_minutes=5 (has all filter columns).
    # Volume (rel_vol) columns may be NULL at 15m/30m apertures, but are
    # always populated at 5m. Strategy discovery uses this same approach.
    df_rows = con.execute(
        """
        SELECT d.*
        FROM daily_features d
        WHERE d.symbol = ? AND d.orb_minutes = 5
    """,
        [instrument],
    ).fetchdf()

    if df_rows.empty:
        return []

    eligible_days = set()
    for _, row in df_rows.iterrows():
        row_dict = row.to_dict()
        if filt.matches_row(row_dict, orb_label):
            eligible_days.add(row_dict["trading_day"])

    if not eligible_days:
        return []

    day_list = list(eligible_days)

    # Query outcomes at target aperture(s), filtered to eligible days
    all_results = []
    for om in orb_minutes_list:
        day_filtered = con.execute(
            """
            SELECT o.outcome, o.pnl_r
            FROM orb_outcomes o
            WHERE o.symbol = ?
              AND o.orb_label = ?
              AND o.entry_model = ?
              AND o.rr_target = ?
              AND o.confirm_bars = ?
              AND o.orb_minutes = ?
              AND o.outcome IN ('win', 'loss')
              AND o.trading_day IN (SELECT UNNEST(?::DATE[]))
        """,
            [instrument, orb_label, entry_model, rr_target, confirm_bars, om, day_list],
        ).fetchall()

        all_results.extend(day_filtered)

    return all_results


def find_validated_variants(con, instrument: str, spec) -> list[dict]:
    """Find all validated variants matching a spec for an instrument."""
    # Match base filter_type exactly. Aperture is in orb_minutes, not filter_type.
    # Composite variants (e.g., ORB_G5_NOTUE) are separate specs, not sub-matches.
    rows = con.execute(
        """
        SELECT strategy_id, rr_target, confirm_bars, orb_minutes,
               expectancy_r, win_rate, sample_size, sharpe_ratio,
               max_drawdown_r, filter_type
        FROM validated_setups
        WHERE instrument = ?
          AND orb_label = ?
          AND entry_model = ?
          AND filter_type = ?
          AND status = 'active'
        ORDER BY expectancy_r DESC
    """,
        [instrument, spec.orb_label, spec.entry_model, spec.filter_type],
    ).fetchall()

    return [
        {
            "strategy_id": r[0],
            "rr_target": r[1],
            "confirm_bars": r[2],
            "orb_minutes": r[3],
            "ExpR": r[4],
            "WR": r[5],
            "N": r[6],
            "Sharpe": r[7],
            "MaxDD": r[8],
            "filter_type": r[9],
        }
        for r in rows
    ]


def analyze_sensitivity(con, instrument: str, variant: dict, spec) -> dict:
    """Run full sensitivity sweep for a single validated variant."""
    rr = variant["rr_target"]
    cb = variant["confirm_bars"]
    om = variant["orb_minutes"]
    ft = variant["filter_type"]
    em = spec.entry_model

    baseline = {
        "N": variant["N"],
        "ExpR": variant["ExpR"],
        "WR": variant["WR"],
        "Sharpe": variant["Sharpe"],
    }

    results = {"baseline": baseline, "sweeps": {}}

    # 1. RR sweep
    for adj_rr in get_adjacent_rr(rr):
        outcomes = query_strategy_outcomes(con, instrument, spec.orb_label, em, ft, adj_rr, cb, [om])
        metrics = compute_metrics_from_rows(outcomes)
        label = f"RR{adj_rr}"
        results["sweeps"][label] = metrics

    # 2. CB sweep (E1 only)
    for adj_cb in get_adjacent_cb(cb, em):
        outcomes = query_strategy_outcomes(con, instrument, spec.orb_label, em, ft, rr, adj_cb, [om])
        metrics = compute_metrics_from_rows(outcomes)
        label = f"CB{adj_cb}"
        results["sweeps"][label] = metrics

    # 3. Filter threshold sweep
    for adj_ft in get_adjacent_filters(ft):
        outcomes = query_strategy_outcomes(con, instrument, spec.orb_label, em, adj_ft, rr, cb, [om])
        metrics = compute_metrics_from_rows(outcomes)
        label = f"F:{adj_ft}"
        results["sweeps"][label] = metrics

    # 4. Cost resilience (2× slippage)
    cost_spec = get_cost_spec(instrument)
    stress_friction = cost_spec.total_friction * 2.0
    if baseline["ExpR"] is not None and baseline["N"] > 0:
        # Re-compute ExpR assuming higher friction (reduces R-multiples)
        # Higher friction shrinks each R by (extra_friction / avg_risk_points)
        # Approximate: fetch median_risk_points from validated_setups
        mrp_row = con.execute(
            """
            SELECT es.median_risk_points
            FROM experimental_strategies es
            WHERE es.strategy_id = ?
        """,
            [variant["strategy_id"]],
        ).fetchone()

        if mrp_row and mrp_row[0] and mrp_row[0] > 0:
            median_risk = mrp_row[0]
            extra_friction_r = (stress_friction - cost_spec.total_friction) / (median_risk * cost_spec.point_value)
            stress_expr = baseline["ExpR"] - extra_friction_r
            results["sweeps"]["2x_slippage"] = {
                "N": baseline["N"],
                "WR": baseline["WR"],
                "ExpR": stress_expr,
                "Sharpe": None,
                "note": f"ExpR adjusted by {-extra_friction_r:.4f}R",
            }

    return results


def format_delta(baseline_val, perturbed_val) -> str:
    """Format delta as percentage with stability indicator."""
    if baseline_val is None or perturbed_val is None:
        return "N/A"
    if baseline_val == 0:
        return "N/A"
    delta_pct = (perturbed_val - baseline_val) / abs(baseline_val) * 100
    sign = "+" if delta_pct >= 0 else ""
    # Stability: check if perturbed value retains baseline edge direction
    if baseline_val > 0:
        if perturbed_val <= 0:
            marker = "UNSTABLE"
        elif perturbed_val / baseline_val < STABILITY_THRESHOLD:
            marker = "WEAK"
        elif perturbed_val / baseline_val < 0.80:
            marker = "OK"
        else:
            marker = "STABLE"
    else:
        # Negative baseline (e.g., N count can't be negative, but ExpR can)
        # For negative values, "stable" means staying close to same magnitude
        marker = "N/A"
    return f"{sign}{delta_pct:.1f}% [{marker}]"


def print_sweep_results(strategy_id: str, analysis: dict):
    """Print formatted sensitivity table for one strategy."""
    bl = analysis["baseline"]
    print(f"\n  Strategy: {strategy_id}")
    print(f"  Baseline: ExpR={bl['ExpR']:+.4f}, Sharpe={bl['Sharpe']:.3f}, WR={bl['WR']:.1%}, N={bl['N']}")

    if not analysis["sweeps"]:
        print("    (no sweeps applicable)")
        return None  # Neither stable nor unstable; skipped in count

    # Header
    print(f"  {'Perturbation':<22} {'ExpR':>8} {'d. ExpR':>18} {'N':>6} {'d. N':>10} {'WR':>7}")
    print(f"  {'-' * 22} {'-' * 8} {'-' * 18} {'-' * 6} {'-' * 10} {'-' * 7}")

    all_stable = True
    for label, metrics in analysis["sweeps"].items():
        n = metrics.get("N", 0)
        expr = metrics.get("ExpR")
        wr = metrics.get("WR")

        if n < MIN_PERTURB_N:
            print(f"  {label:<22} {'N/A':>8} {'(N too small)':>18} {n:>6} {'':>10} {'':>7}")
            continue

        expr_str = f"{expr:+.4f}" if expr is not None else "N/A"
        wr_str = f"{wr:.1%}" if wr is not None else "N/A"
        delta_expr = format_delta(bl["ExpR"], expr)
        delta_n = format_delta(bl["N"], n)

        if expr is not None and bl["ExpR"] is not None:
            ratio = expr / bl["ExpR"] if bl["ExpR"] != 0 else 0
            if ratio < STABILITY_THRESHOLD:
                all_stable = False

        print(f"  {label:<22} {expr_str:>8} {delta_expr:>18} {n:>6} {delta_n:>10} {wr_str:>7}")

        note = metrics.get("note")
        if note:
            print(f"    Note: {note}")

    verdict = "PASS" if all_stable else "REVIEW"
    print(f"  Verdict: {verdict}")
    return all_stable


def main():
    parser = argparse.ArgumentParser(description="Automated sensitivity analysis for live strategies")
    parser.add_argument("--instrument", default=None, help="Instrument to analyze (default: all tradeable)")
    parser.add_argument("--family", default=None, help="Specific family_id to analyze (default: all live specs)")
    parser.add_argument("--db", default=None, help="Database path override")
    parser.add_argument("--top-n", type=int, default=3, help="Analyze top N variants per spec (default: 3)")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    instruments = [args.instrument] if args.instrument else TRADEABLE_INSTRUMENTS

    # Filter specs if family specified
    specs = LIVE_PORTFOLIO
    if args.family:
        specs = [s for s in specs if s.family_id == args.family]
        if not specs:
            print(f"ERROR: No spec found with family_id={args.family}")
            sys.exit(1)

    print("=" * 70)
    print("SENSITIVITY ANALYSIS - LIVE PORTFOLIO STRATEGIES")
    print(f"Database: {db_path}")
    print(f"Instruments: {', '.join(instruments)}")
    print(f"Specs to analyze: {len(specs)}")
    print("=" * 70)

    con = duckdb.connect(str(db_path), read_only=True)
    total_reviewed = 0
    total_unstable = 0

    try:
        for inst in instruments:
            print(f"\n{'=' * 70}")
            print(f"  {inst}")
            print(f"{'=' * 70}")

            for spec in specs:
                variants = find_validated_variants(con, inst, spec)
                if not variants:
                    continue

                print(f"\n--- {spec.family_id} ({spec.tier}) [{len(variants)} variants] ---")

                for v in variants[: args.top_n]:
                    analysis = analyze_sensitivity(con, inst, v, spec)
                    stable = print_sweep_results(v["strategy_id"], analysis)
                    total_reviewed += 1
                    if stable is False:
                        total_unstable += 1

    finally:
        con.close()

    print()
    print("=" * 70)
    if total_reviewed == 0:
        print("SENSITIVITY ANALYSIS: No strategies found to analyze")
    elif total_unstable == 0:
        print(f"SENSITIVITY ANALYSIS PASSED: {total_reviewed} strategies reviewed, all parameter-stable")
    else:
        print(f"SENSITIVITY ANALYSIS: {total_reviewed} strategies reviewed, {total_unstable} need review")
    print("=" * 70)

    sys.exit(1 if total_unstable > 0 else 0)


if __name__ == "__main__":
    main()
