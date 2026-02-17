"""
Side-by-side comparison: regime run vs full-period results.

Joins regime_strategies with experimental_strategies on strategy_id to show
which strategies improve or degrade in the selected date range.

Usage:
    python -m trading_app.regime.compare --instrument MGC --run-label 2025_only
    python -m trading_app.regime.compare --instrument MGC --run-label 2025_only --min-sample 20
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.init_db import ORB_LABELS

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

def run_comparison(
    db_path: Path | None = None,
    instrument: str = "MGC",
    run_label: str = "default",
    min_sample: int = 0,
    output_path: Path | None = None,
) -> list[dict]:
    """Compare regime strategies with full-period strategies.

    Returns list of comparison dicts sorted by ExpR delta (descending).
    """
    if db_path is None:
        db_path = GOLD_DB_PATH
    if output_path is None:
        output_path = PROJECT_ROOT / "artifacts" / f"regime_{run_label}_comparison.csv"

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Check tables exist
        tables = {
            r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        if "regime_strategies" not in tables:
            print("regime_strategies table not found. Run discovery first.")
            return []
        if "experimental_strategies" not in tables:
            print("experimental_strategies table not found.")
            return []

        # Load regime strategies
        regime_rows = con.execute(
            """SELECT strategy_id, orb_label, orb_minutes, rr_target, confirm_bars,
                      entry_model, filter_type, sample_size, win_rate,
                      expectancy_r, sharpe_ratio, max_drawdown_r,
                      validation_status
               FROM regime_strategies
               WHERE run_label = ? AND instrument = ?
               AND sample_size >= ?
               ORDER BY strategy_id""",
            [run_label, instrument, min_sample],
        ).fetchall()
        regime_cols = [desc[0] for desc in con.description]
        regime_index = {}
        for row in regime_rows:
            d = dict(zip(regime_cols, row))
            regime_index[d["strategy_id"]] = d

        print(f"Loaded {len(regime_index)} regime strategies "
              f"(run_label={run_label}, min_sample={min_sample})")

        # Load full-period strategies
        full_rows = con.execute(
            """SELECT strategy_id, orb_label, orb_minutes, rr_target, confirm_bars,
                      entry_model, filter_type, sample_size, win_rate,
                      expectancy_r, sharpe_ratio, max_drawdown_r,
                      validation_status
               FROM experimental_strategies
               WHERE instrument = ?
               ORDER BY strategy_id""",
            [instrument],
        ).fetchall()
        full_cols = [desc[0] for desc in con.description]
        full_index = {}
        for row in full_rows:
            d = dict(zip(full_cols, row))
            full_index[d["strategy_id"]] = d

        print(f"Loaded {len(full_index)} full-period strategies")

        # Build comparison
        matched_ids = set(regime_index.keys()) & set(full_index.keys())
        print(f"Matched {len(matched_ids)} strategy pairs")

        comparisons = []
        for sid in sorted(matched_ids):
            r = regime_index[sid]
            f = full_index[sid]

            r_sharpe = r["sharpe_ratio"] or 0
            f_sharpe = f["sharpe_ratio"] or 0
            r_exp = r["expectancy_r"] or 0
            f_exp = f["expectancy_r"] or 0

            comparisons.append({
                "strategy_id": sid,
                "orb_label": r["orb_label"],
                "entry_model": r["entry_model"],
                "rr_target": r["rr_target"],
                "confirm_bars": r["confirm_bars"],
                "filter_type": r["filter_type"],
                # Regime metrics
                "regime_n": r["sample_size"],
                "regime_wr": r["win_rate"],
                "regime_exp_r": r_exp,
                "regime_sharpe": r_sharpe,
                "regime_maxdd": r["max_drawdown_r"],
                "regime_validated": r.get("validation_status") == "PASSED",
                # Full-period metrics
                "full_n": f["sample_size"],
                "full_wr": f["win_rate"],
                "full_exp_r": f_exp,
                "full_sharpe": f_sharpe,
                "full_maxdd": f["max_drawdown_r"],
                "full_validated": f.get("validation_status") == "PASSED",
                # Deltas
                "delta_exp_r": round(r_exp - f_exp, 4),
                "delta_sharpe": round(r_sharpe - f_sharpe, 4),
            })

        # Also flag regime-only and full-only validated
        regime_only_ids = set(regime_index.keys()) - set(full_index.keys())
        full_only_ids = set(full_index.keys()) - set(regime_index.keys())

        # Sort by ExpR delta descending
        comparisons.sort(key=lambda c: c["delta_exp_r"], reverse=True)

        # Print report
        _print_report(comparisons, run_label, regime_only_ids, full_only_ids)

        # Write CSV
        if comparisons and output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=comparisons[0].keys())
                writer.writeheader()
                writer.writerows(comparisons)
            print(f"\nCSV written: {output_path}")

        return comparisons

    finally:
        con.close()

def _print_report(comparisons: list[dict], run_label: str,
                  regime_only: set, full_only: set):
    """Print a formatted comparison report."""
    print("\n" + "=" * 80)
    print(f"REGIME COMPARISON REPORT: {run_label}")
    print("=" * 80)

    if not comparisons:
        print("  No matched strategy pairs found.")
        return

    # Per-session summary
    session_data = {}
    for c in comparisons:
        orb = c["orb_label"]
        if orb not in session_data:
            session_data[orb] = []
        session_data[orb].append(c)

    print(f"\n  {'Session':<8} {'Pairs':>6} {'Regime>Full':>12} {'Full>Regime':>12} "
          f"{'Avg ExpR+':>10} {'Avg Sharpe+':>12}")
    print("  " + "-" * 62)

    for orb_label in ORB_LABELS:
        if orb_label not in session_data:
            continue
        pairs = session_data[orb_label]
        regime_wins = sum(1 for c in pairs if c["delta_exp_r"] > 0)
        full_wins = sum(1 for c in pairs if c["delta_exp_r"] < 0)
        avg_exp = sum(c["delta_exp_r"] for c in pairs) / len(pairs)
        avg_sharpe = sum(c["delta_sharpe"] for c in pairs) / len(pairs)
        print(f"  {orb_label:<8} {len(pairs):>6} {regime_wins:>12} "
              f"{full_wins:>12} {avg_exp:>+10.4f} {avg_sharpe:>+12.4f}")

    # Top 10 regime improvements
    print("\n  Top 10 strategies by regime ExpR improvement:")
    for c in comparisons[:10]:
        val_flag = "*" if c["regime_validated"] else " "
        print(f"    {val_flag} {c['strategy_id']}: "
              f"ExpR {c['full_exp_r']:+.4f} -> {c['regime_exp_r']:+.4f} "
              f"({c['delta_exp_r']:+.4f}), "
              f"Sharpe {c['full_sharpe']:.3f} -> {c['regime_sharpe']:.3f}, "
              f"N={c['regime_n']}")

    # Validation comparison
    regime_validated = [c for c in comparisons if c["regime_validated"]]
    full_validated = [c for c in comparisons if c["full_validated"]]
    both_validated = [c for c in comparisons if c["regime_validated"] and c["full_validated"]]
    regime_new = [c for c in comparisons if c["regime_validated"] and not c["full_validated"]]
    regime_lost = [c for c in comparisons if not c["regime_validated"] and c["full_validated"]]

    print("\n  Validation summary:")
    print(f"    Regime validated: {len(regime_validated)}")
    print(f"    Full-period validated: {len(full_validated)}")
    print(f"    Both validated: {len(both_validated)}")
    print(f"    Regime-only validated (new edge): {len(regime_new)}")
    print(f"    Full-only validated (lost edge): {len(regime_lost)}")

    if regime_new:
        print("\n  New edge in regime (validated in regime, not full-period):")
        for c in regime_new[:5]:
            print(f"    {c['strategy_id']}: ExpR={c['regime_exp_r']:+.4f}, "
                  f"Sharpe={c['regime_sharpe']:.3f}, N={c['regime_n']}")

    print("\n" + "=" * 80)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare regime vs full-period strategy results"
    )
    parser.add_argument("--instrument", default="MGC")
    parser.add_argument("--run-label", required=True, help="Regime run label")
    parser.add_argument("--min-sample", type=int, default=0,
                        help="Min sample size for inclusion")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: artifacts/regime_{label}_comparison.csv)")
    args = parser.parse_args()

    run_comparison(
        instrument=args.instrument,
        run_label=args.run_label,
        min_sample=args.min_sample,
        output_path=args.output,
    )

if __name__ == "__main__":
    main()
