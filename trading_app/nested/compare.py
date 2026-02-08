"""
A/B comparison: baseline (5m ORB + 1m entry) vs nested (15m/30m ORB + 5m entry).

Reads from BOTH experimental_strategies (baseline) and nested_strategies (nested),
and outputs side-by-side comparisons per session/filter/RR.

Computes "Structural Premium" = Sharpe(nested) - Sharpe(baseline) per session.

Usage:
    python -m trading_app.nested.compare --instrument MGC
    python -m trading_app.nested.compare --instrument MGC --min-sample 50
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.init_db import ORB_LABELS

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)


def _load_strategies(con, table_name, instrument, min_sample=0):
    """Load strategies from a table, filtered by instrument and min sample."""
    rows = con.execute(
        f"""SELECT strategy_id, orb_label, orb_minutes, rr_target, confirm_bars,
                   entry_model, filter_type, sample_size, win_rate,
                   expectancy_r, sharpe_ratio, max_drawdown_r
            FROM {table_name}
            WHERE instrument = ?
            AND sample_size >= ?
            ORDER BY strategy_id""",
        [instrument, min_sample],
    ).fetchall()
    cols = [desc[0] for desc in con.description]
    return [dict(zip(cols, r)) for r in rows]


def _make_comparison_key(row):
    """Create a comparison key for matching baseline vs nested strategies."""
    return (
        row["orb_label"],
        row["entry_model"],
        row["rr_target"],
        row["confirm_bars"],
        row["filter_type"],
    )


def run_comparison(
    db_path: Path | None = None,
    instrument: str = "MGC",
    min_sample: int = 0,
    orb_minutes_list: list[int] | None = None,
) -> dict:
    """Compare baseline vs nested strategies.

    Returns dict with comparison results keyed by session.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH
    if orb_minutes_list is None:
        orb_minutes_list = [15, 30]

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Load baseline strategies (orb_minutes=5)
        print("Loading baseline strategies (5m ORB + 1m entry)...")
        baseline = _load_strategies(con, "experimental_strategies", instrument, min_sample)
        baseline_5m = [s for s in baseline if s["orb_minutes"] == 5]
        print(f"  {len(baseline_5m)} baseline strategies loaded")

        # Index baseline by comparison key
        baseline_index = {}
        for s in baseline_5m:
            key = _make_comparison_key(s)
            baseline_index[key] = s

        results = {}

        for orb_minutes in orb_minutes_list:
            print(f"\nLoading nested strategies ({orb_minutes}m ORB + 5m entry)...")

            # Check if nested_strategies table exists
            tables = {
                r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            }
            if "nested_strategies" not in tables:
                print("  nested_strategies table not found. Run discovery first.")
                continue

            nested = _load_strategies(con, "nested_strategies", instrument, min_sample)
            nested_filtered = [s for s in nested if s["orb_minutes"] == orb_minutes]
            print(f"  {len(nested_filtered)} nested strategies loaded")

            # Index nested by comparison key
            nested_index = {}
            for s in nested_filtered:
                key = _make_comparison_key(s)
                nested_index[key] = s

            # Compare: find matched pairs
            matched_keys = set(baseline_index.keys()) & set(nested_index.keys())
            print(f"  {len(matched_keys)} matched strategy pairs")

            # Session-level aggregation
            session_results = {}
            for orb_label in ORB_LABELS:
                session_pairs = []
                for key in matched_keys:
                    if key[0] != orb_label:
                        continue
                    b = baseline_index[key]
                    n = nested_index[key]
                    session_pairs.append({
                        "key": key,
                        "baseline": b,
                        "nested": n,
                        "sharpe_delta": (
                            (n["sharpe_ratio"] or 0) - (b["sharpe_ratio"] or 0)
                            if b["sharpe_ratio"] is not None and n["sharpe_ratio"] is not None
                            else None
                        ),
                        "exp_delta": (
                            (n["expectancy_r"] or 0) - (b["expectancy_r"] or 0)
                            if b["expectancy_r"] is not None and n["expectancy_r"] is not None
                            else None
                        ),
                    })

                if not session_pairs:
                    continue

                # Compute Structural Premium for this session
                sharpe_deltas = [
                    p["sharpe_delta"] for p in session_pairs
                    if p["sharpe_delta"] is not None
                ]
                exp_deltas = [
                    p["exp_delta"] for p in session_pairs
                    if p["exp_delta"] is not None
                ]

                avg_sharpe_premium = (
                    sum(sharpe_deltas) / len(sharpe_deltas) if sharpe_deltas else None
                )
                avg_exp_premium = (
                    sum(exp_deltas) / len(exp_deltas) if exp_deltas else None
                )

                # Count how many nested beat baseline
                nested_wins = sum(1 for d in sharpe_deltas if d > 0)
                baseline_wins = sum(1 for d in sharpe_deltas if d < 0)

                # Top 5 largest improvements
                top_improvements = sorted(
                    [p for p in session_pairs if p["sharpe_delta"] is not None],
                    key=lambda p: p["sharpe_delta"],
                    reverse=True,
                )[:5]

                session_results[orb_label] = {
                    "n_pairs": len(session_pairs),
                    "avg_sharpe_premium": avg_sharpe_premium,
                    "avg_exp_premium": avg_exp_premium,
                    "nested_wins": nested_wins,
                    "baseline_wins": baseline_wins,
                    "top_improvements": top_improvements,
                }

            results[orb_minutes] = session_results

        # Print report
        _print_report(results)

        return results

    finally:
        con.close()


def _print_report(results: dict):
    """Print a formatted comparison report."""
    print("\n" + "=" * 80)
    print("NESTED ORB A/B COMPARISON REPORT")
    print("=" * 80)

    for orb_minutes, session_results in sorted(results.items()):
        print(f"\n--- {orb_minutes}m ORB + 5m Entry vs 5m ORB + 1m Entry ---")

        if not session_results:
            print("  No matched strategy pairs found.")
            continue

        # Summary table
        print(f"\n  {'Session':<8} {'Pairs':>6} {'Nested>Base':>12} {'Base>Nested':>12} "
              f"{'Avg Sharpe+':>12} {'Avg ExpR+':>10}")
        print("  " + "-" * 62)

        for orb_label in ORB_LABELS:
            if orb_label not in session_results:
                continue
            sr = session_results[orb_label]
            sharpe_str = f"{sr['avg_sharpe_premium']:+.4f}" if sr["avg_sharpe_premium"] is not None else "N/A"
            exp_str = f"{sr['avg_exp_premium']:+.4f}" if sr["avg_exp_premium"] is not None else "N/A"
            print(f"  {orb_label:<8} {sr['n_pairs']:>6} {sr['nested_wins']:>12} "
                  f"{sr['baseline_wins']:>12} {sharpe_str:>12} {exp_str:>10}")

        # Top improvements per session
        print()
        for orb_label in ORB_LABELS:
            if orb_label not in session_results:
                continue
            sr = session_results[orb_label]
            if not sr["top_improvements"]:
                continue

            print(f"  Top improvements for {orb_label}:")
            for p in sr["top_improvements"]:
                key = p["key"]
                b = p["baseline"]
                n = p["nested"]
                print(
                    f"    {key[1]} RR{key[2]} CB{key[3]} {key[4]}: "
                    f"Sharpe {b['sharpe_ratio']:.3f} -> {n['sharpe_ratio']:.3f} "
                    f"(+{p['sharpe_delta']:.4f}), "
                    f"ExpR {(b['expectancy_r'] or 0):.4f} -> {(n['expectancy_r'] or 0):.4f}"
                )

    print("\n" + "=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="A/B comparison: baseline vs nested ORB strategies"
    )
    parser.add_argument("--instrument", default="MGC")
    parser.add_argument("--min-sample", type=int, default=0,
                        help="Min sample size for inclusion")
    parser.add_argument("--orb-minutes", type=int, nargs="+", default=[15, 30])
    args = parser.parse_args()

    run_comparison(
        instrument=args.instrument,
        min_sample=args.min_sample,
        orb_minutes_list=args.orb_minutes,
    )


if __name__ == "__main__":
    main()
