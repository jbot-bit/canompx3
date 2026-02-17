"""
Rolling portfolio aggregator: load rolling window results, compute
sample-size-weighted stability, day-of-week segmentation, and
double-break regime integration.

Reads from regime_validated table (run_label='rolling_*') and produces
strategy-family-level classifications: STABLE / TRANSITIONING / DEGRADED.

Usage:
    python -m trading_app.rolling_portfolio --train-months 12 --min-weighted-score 0.6
    python -m trading_app.rolling_portfolio --train-months 12 --report
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.stdout.reconfigure(line_buffering=True)

import duckdb
import numpy as np

from pipeline.paths import GOLD_DB_PATH

# Stability thresholds (weighted score)
STABLE_THRESHOLD = 0.6
TRANSITIONING_THRESHOLD = 0.3

# Sample size for full weight in stability scoring
FULL_WEIGHT_SAMPLE = 50

# Default lookback: only use the N most recent rolling windows for scoring.
# ~2 years of monthly windows. None = use all windows.
DEFAULT_LOOKBACK_WINDOWS = 24

@dataclass
class FamilyResult:
    """Aggregated rolling results for a strategy family."""
    family_id: str
    orb_label: str
    entry_model: str
    filter_type: str
    windows_total: int
    windows_passed: int
    weighted_stability: float
    classification: str  # STABLE / TRANSITIONING / DEGRADED
    avg_expectancy_r: float
    avg_sharpe: float
    total_sample_size: int
    oos_cumulative_r: float
    double_break_degraded_windows: int
    day_of_week_stats: dict | None = None
    day_of_week_concentration: float | None = None

def make_family_id(orb_label: str, entry_model: str, filter_type: str) -> str:
    """Create a family-level identifier (ignores RR/CB params)."""
    return f"{orb_label}_{entry_model}_{filter_type}"

def load_rolling_results(
    db_path: Path,
    train_months: int,
    instrument: str = "MGC",
    run_labels: list[str] | None = None,
) -> list[dict]:
    """Load all regime_validated rows with rolling_{train_months}m_ prefix.

    If run_labels is provided, only include rows whose run_label is in the set.
    Returns list of dicts with strategy parameters and metrics.
    """
    prefix = f"rolling_{train_months}m_%"

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT rv.run_label, rv.strategy_id,
                   rv.start_date, rv.end_date,
                   rv.orb_label, rv.entry_model, rv.filter_type,
                   rv.rr_target, rv.confirm_bars,
                   rv.sample_size, rv.win_rate, rv.expectancy_r,
                   rv.sharpe_ratio, rv.max_drawdown_r,
                   rv.yearly_results
            FROM regime_validated rv
            WHERE rv.run_label LIKE ?
              AND rv.instrument = ?
              AND LOWER(rv.status) = 'active'
            ORDER BY rv.run_label, rv.strategy_id
        """, [prefix, instrument]).fetchall()

        cols = [desc[0] for desc in con.description]
        results = [dict(zip(cols, row)) for row in rows]
        if run_labels is not None:
            label_set = set(run_labels)
            results = [r for r in results if r["run_label"] in label_set]
        return results

    finally:
        con.close()

def load_all_rolling_run_labels(
    db_path: Path,
    train_months: int,
    instrument: str = "MGC",
    lookback_windows: int | None = None,
) -> list[str]:
    """Load distinct run_labels from regime_strategies for this train_months.

    If lookback_windows is provided, return only the last N labels (most recent).
    """
    prefix = f"rolling_{train_months}m_%"

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT DISTINCT run_label
            FROM regime_strategies
            WHERE run_label LIKE ?
              AND instrument = ?
            ORDER BY run_label
        """, [prefix, instrument]).fetchall()
        labels = [r[0] for r in rows]
        if lookback_windows is not None:
            labels = labels[-lookback_windows:]
        return labels
    finally:
        con.close()

def load_rolling_degraded_counts(
    db_path: Path,
    train_months: int,
    instrument: str = "MGC",
    run_labels: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Load count of auto-degraded strategies per run_label and orb_label.

    If run_labels is provided, only count degraded windows in that set.
    Returns: {family_id: count_of_degraded_windows}
    """
    prefix = f"rolling_{train_months}m_%"

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT run_label, orb_label, entry_model, filter_type, COUNT(*)
            FROM regime_strategies
            WHERE run_label LIKE ?
              AND instrument = ?
              AND validation_notes LIKE 'Auto-degraded%'
            GROUP BY run_label, orb_label, entry_model, filter_type
        """, [prefix, instrument]).fetchall()

        if run_labels is not None:
            label_set = set(run_labels)
            rows = [r for r in rows if r[0] in label_set]

        # Count degraded windows per family
        degraded_windows = defaultdict(set)
        for run_label, orb_label, em, ft, _ in rows:
            fid = make_family_id(orb_label, em, ft)
            degraded_windows[fid].add(run_label)

        return {fid: len(labels) for fid, labels in degraded_windows.items()}

    finally:
        con.close()

def _window_weight(sample_size: int) -> float:
    """Compute weight for a window based on its sample size.

    50+ trades = 1.0 weight, 20 trades = 0.4, linear scale.
    """
    return min(sample_size / FULL_WEIGHT_SAMPLE, 1.0)

def classify_stability(weighted_score: float) -> str:
    """STABLE / TRANSITIONING / DEGRADED based on weighted stability score."""
    if weighted_score >= STABLE_THRESHOLD:
        return "STABLE"
    elif weighted_score >= TRANSITIONING_THRESHOLD:
        return "TRANSITIONING"
    return "DEGRADED"

def aggregate_rolling_performance(
    validated: list[dict],
    all_run_labels: list[str],
    degraded_counts: dict[str, int],
) -> list[FamilyResult]:
    """Aggregate rolling window results at the family level.

    For each (orb_label, entry_model, filter_type) family:
    - Count windows where at least one variant passed validation
    - Compute sample-size-weighted stability score
    - Average ExpR and Sharpe across passing windows
    """
    # Group validated strategies by (family_id, run_label)
    family_windows = defaultdict(lambda: defaultdict(list))
    for row in validated:
        fid = make_family_id(row["orb_label"], row["entry_model"], row["filter_type"])
        family_windows[fid][row["run_label"]].append(row)

    total_windows = len(all_run_labels)
    if total_windows == 0:
        return []

    results = []
    for fid, windows_data in family_windows.items():
        # Parse family_id back
        parts = fid.split("_", 2)
        orb_label, entry_model, filter_type = parts[0], parts[1], parts[2]

        # For each window, pick the best variant (highest ExpR)
        passing_windows = []
        for run_label, variants in windows_data.items():
            best = max(variants, key=lambda v: v["expectancy_r"])
            passing_windows.append(best)

        # Compute weighted stability
        passing_labels = set(windows_data.keys())
        all_labels_set = set(all_run_labels)

        # Weight each window by sample size
        total_weight = 0.0
        passing_weight = 0.0
        for label in all_labels_set:
            if label in passing_labels:
                best = max(windows_data[label], key=lambda v: v["expectancy_r"])
                w = _window_weight(best["sample_size"])
                passing_weight += w
                total_weight += w
            else:
                # Non-passing window gets weight 1.0 (assume full sample would fail)
                total_weight += 1.0

        weighted_stability = passing_weight / total_weight if total_weight > 0 else 0.0

        # Aggregate metrics from passing windows
        exp_values = [pw["expectancy_r"] for pw in passing_windows]
        sharpe_values = [pw["sharpe_ratio"] for pw in passing_windows if pw.get("sharpe_ratio") is not None]
        total_sample = sum(pw["sample_size"] for pw in passing_windows)
        oos_r = sum(pw["expectancy_r"] * pw["sample_size"] for pw in passing_windows)

        results.append(FamilyResult(
            family_id=fid,
            orb_label=orb_label,
            entry_model=entry_model,
            filter_type=filter_type,
            windows_total=total_windows,
            windows_passed=len(passing_windows),
            weighted_stability=round(weighted_stability, 3),
            classification=classify_stability(weighted_stability),
            avg_expectancy_r=round(np.mean(exp_values), 4) if exp_values else 0.0,
            avg_sharpe=round(np.mean(sharpe_values), 4) if sharpe_values else 0.0,
            total_sample_size=total_sample,
            oos_cumulative_r=round(oos_r, 2),
            double_break_degraded_windows=degraded_counts.get(fid, 0),
        ))

    # Sort by weighted stability descending
    results.sort(key=lambda r: r.weighted_stability, reverse=True)
    return results

def compute_day_of_week_stats(
    db_path: Path,
    family_results: list[FamilyResult],
    train_months: int,
    instrument: str = "MGC",
) -> list[FamilyResult]:
    """Add day-of-week segmentation to STABLE/TRANSITIONING families.

    For each family, queries orb_outcomes to compute per-day stats.
    Uses DAYOFWEEK(trading_day): 0=Sun through 6=Sat in DuckDB.
    """
    DOW_LABELS = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri"}

    from trading_app.config import ALL_FILTERS

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Pre-load daily_features for filter eligibility
        df_features = con.execute("""
            SELECT trading_day, orb_0900_size, orb_1000_size, orb_1100_size,
                   orb_1800_size, orb_2300_size, orb_0030_size
            FROM daily_features
            WHERE symbol = ? AND orb_minutes = 5
        """, [instrument]).fetchdf()

        for fam in family_results:
            if fam.classification == "DEGRADED":
                continue

            # Determine eligible days based on filter
            filt = ALL_FILTERS.get(fam.filter_type)
            if filt is None:
                continue

            size_col = f"orb_{fam.orb_label}_size"

            def _to_date(td):
                """Convert numpy.datetime64/Timestamp/date to datetime.date."""
                import pandas as pd_local
                return pd_local.Timestamp(td).date()

            if fam.filter_type == "NO_FILTER":
                eligible_days = {_to_date(td) for td in df_features["trading_day"]}
            else:
                sizes = df_features[size_col].values
                tdays = df_features["trading_day"].values
                eligible_days = set()
                for td, s in zip(tdays, sizes):
                    if s is not None and not (isinstance(s, float) and np.isnan(s)):
                        if filt.matches_row({size_col: s}, fam.orb_label):
                            eligible_days.add(_to_date(td))

            # Get trade outcomes filtered to eligible days
            rows = con.execute("""
                SELECT DAYOFWEEK(oo.trading_day) as dow,
                       oo.trading_day,
                       oo.pnl_r
                FROM orb_outcomes oo
                WHERE oo.symbol = ?
                  AND oo.orb_label = ?
                  AND oo.entry_model = ?
                  AND oo.pnl_r IS NOT NULL
            """, [instrument, fam.orb_label, fam.entry_model]).fetchall()

            # Filter to eligible days only
            rows = [(dow, pnl_r) for dow, td, pnl_r in rows
                    if td in eligible_days]

            if not rows:
                continue

            # Group by day of week
            day_stats = {}
            day_total_r = {}
            for dow, pnl_r in rows:
                dow = int(dow)
                if dow not in DOW_LABELS:
                    continue
                label = DOW_LABELS[dow]
                if label not in day_stats:
                    day_stats[label] = {"wins": 0, "total": 0, "sum_r": 0.0}
                day_stats[label]["total"] += 1
                day_stats[label]["sum_r"] += pnl_r
                if pnl_r > 0:
                    day_stats[label]["wins"] += 1

            # Compute per-day metrics
            dow_result = {}
            for label, s in day_stats.items():
                if s["total"] > 0:
                    dow_result[label] = {
                        "sample_size": s["total"],
                        "win_rate": round(s["wins"] / s["total"], 3),
                        "exp_r": round(s["sum_r"] / s["total"], 4),
                        "total_r": round(s["sum_r"], 2),
                    }
                    day_total_r[label] = abs(s["sum_r"])

            fam.day_of_week_stats = dow_result

            # Compute concentration: max single-day contribution to total |R|
            total_abs_r = sum(day_total_r.values())
            if total_abs_r > 0:
                fam.day_of_week_concentration = round(
                    max(day_total_r.values()) / total_abs_r, 3
                )
            else:
                fam.day_of_week_concentration = None

        return family_results

    finally:
        con.close()

def load_rolling_validated_strategies(
    db_path: Path,
    instrument: str,
    train_months: int,
    min_weighted_score: float = STABLE_THRESHOLD,
    min_expectancy_r: float = 0.10,
    lookback_windows: int | None = DEFAULT_LOOKBACK_WINDOWS,
) -> list[dict]:
    """Load strategies from STABLE rolling families for portfolio integration.

    If lookback_windows is set, only the N most recent rolling windows are used
    for stability scoring. Default is DEFAULT_LOOKBACK_WINDOWS (~2 years).
    Pass None to use all available windows.

    Returns list of dicts matching the format expected by portfolio.py.
    """
    all_labels = load_all_rolling_run_labels(
        db_path, train_months, instrument, lookback_windows
    )
    validated = load_rolling_results(
        db_path, train_months, instrument, run_labels=all_labels
    )
    degraded = load_rolling_degraded_counts(
        db_path, train_months, instrument, run_labels=all_labels
    )

    families = aggregate_rolling_performance(validated, all_labels, degraded)

    # Get the best variant for each qualifying family from the MOST RECENT window
    if not all_labels:
        return []

    latest_label = all_labels[-1]
    results = []

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for fam in families:
            if fam.weighted_stability < min_weighted_score:
                continue
            if fam.avg_expectancy_r < min_expectancy_r:
                continue

            # Get best variant from most recent window
            rows = con.execute("""
                SELECT rv.strategy_id, rv.instrument, rv.orb_label, rv.entry_model,
                       rv.rr_target, rv.confirm_bars, rv.filter_type,
                       rv.expectancy_r, rv.win_rate, rv.sample_size,
                       rv.sharpe_ratio, rv.max_drawdown_r,
                       rs.median_risk_points, 'rolling' as source
                FROM regime_validated rv
                LEFT JOIN regime_strategies rs
                  ON rv.run_label = rs.run_label AND rv.strategy_id = rs.strategy_id
                WHERE rv.run_label = ?
                  AND rv.instrument = ?
                  AND rv.orb_label = ?
                  AND rv.entry_model = ?
                  AND rv.filter_type = ?
                  AND LOWER(rv.status) = 'active'
                ORDER BY rv.expectancy_r DESC
                LIMIT 1
            """, [latest_label, instrument, fam.orb_label,
                  fam.entry_model, fam.filter_type]).fetchall()

            if rows:
                cols = [desc[0] for desc in con.description]
                results.append(dict(zip(cols, rows[0])))

        return results
    finally:
        con.close()

def print_report(families: list[FamilyResult]) -> None:
    """Print a human-readable report of rolling evaluation results."""
    stable = [f for f in families if f.classification == "STABLE"]
    transitioning = [f for f in families if f.classification == "TRANSITIONING"]
    degraded = [f for f in families if f.classification == "DEGRADED"]

    print(f"\n{'='*70}")
    print("ROLLING PORTFOLIO REPORT")
    print(f"{'='*70}")
    print(f"\nTotal families: {len(families)}")
    print(f"  STABLE: {len(stable)}")
    print(f"  TRANSITIONING: {len(transitioning)}")
    print(f"  DEGRADED: {len(degraded)}")

    if stable:
        print(f"\n--- STABLE (weighted score >= {STABLE_THRESHOLD}) ---")
        for f in stable:
            print(f"  {f.family_id}: score={f.weighted_stability:.3f}, "
                  f"passed={f.windows_passed}/{f.windows_total}, "
                  f"ExpR={f.avg_expectancy_r:+.4f}, "
                  f"Sharpe={f.avg_sharpe:.4f}, "
                  f"N={f.total_sample_size}")
            if f.day_of_week_stats:
                for day, stats in sorted(f.day_of_week_stats.items()):
                    conc_flag = ""
                    if f.day_of_week_concentration and f.day_of_week_concentration > 0.5:
                        conc_flag = " [DAY-DEPENDENT]"
                    print(f"    {day}: WR={stats['win_rate']:.1%}, "
                          f"ExpR={stats['exp_r']:+.4f}, N={stats['sample_size']}")
                if conc_flag:
                    print(f"    ** Day concentration: "
                          f"{f.day_of_week_concentration:.1%}{conc_flag}")

    if transitioning:
        print(f"\n--- TRANSITIONING ({TRANSITIONING_THRESHOLD} <= score < {STABLE_THRESHOLD}) ---")
        for f in transitioning:
            print(f"  {f.family_id}: score={f.weighted_stability:.3f}, "
                  f"passed={f.windows_passed}/{f.windows_total}, "
                  f"ExpR={f.avg_expectancy_r:+.4f}")

    if degraded:
        print(f"\n--- DEGRADED (score < {TRANSITIONING_THRESHOLD}) ---")
        db_degraded = [f for f in degraded if f.double_break_degraded_windows > 0]
        if db_degraded:
            print(f"  Double-break auto-degraded: {len(db_degraded)}")
        # Don't print every degraded family -- just summary
        print(f"  Total: {len(degraded)} families")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate rolling window results and classify strategy stability"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--train-months", type=int, default=12,
                        help="Training window size to analyze")
    parser.add_argument("--min-weighted-score", type=float, default=STABLE_THRESHOLD,
                        help="Minimum weighted stability score for portfolio inclusion")
    parser.add_argument("--report", action="store_true",
                        help="Print detailed report")
    parser.add_argument("--lookback-windows", type=int, default=DEFAULT_LOOKBACK_WINDOWS,
                        help=f"Only use the N most recent windows for scoring "
                             f"(default: {DEFAULT_LOOKBACK_WINDOWS}). "
                             f"Use 0 for all windows.")
    parser.add_argument("--output", type=str, default=None,
                        help="Write results JSON to this path")
    args = parser.parse_args()

    db_path = GOLD_DB_PATH
    lookback = args.lookback_windows if args.lookback_windows > 0 else None

    print(f"Loading rolling results (train_months={args.train_months}, "
          f"lookback_windows={lookback or 'all'})...")
    all_labels = load_all_rolling_run_labels(
        db_path, args.train_months, args.instrument, lookback
    )
    validated = load_rolling_results(
        db_path, args.train_months, args.instrument, run_labels=all_labels
    )
    degraded_counts = load_rolling_degraded_counts(
        db_path, args.train_months, args.instrument, run_labels=all_labels
    )

    print(f"  {len(validated)} validated strategies across {len(all_labels)} windows")

    families = aggregate_rolling_performance(validated, all_labels, degraded_counts)
    print(f"  {len(families)} unique families")

    # Add day-of-week stats for non-degraded families
    families = compute_day_of_week_stats(
        db_path, families, args.train_months, args.instrument
    )

    if args.report or not args.output:
        print_report(families)

    if args.output:
        output_path = Path(args.output)
        output_data = [asdict(f) for f in families]
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults written to {output_path}")

if __name__ == "__main__":
    main()
