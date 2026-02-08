#!/usr/bin/env python3
"""
Strategy Viewer CLI — browse, filter, and export validated strategies.

Usage:
    python trading_app/view_strategies.py                         # Top 20 by ExpR
    python trading_app/view_strategies.py --top 50                # More results
    python trading_app/view_strategies.py --orb 0900              # Filter by session
    python trading_app/view_strategies.py --orb 1800 --entry E3   # Multiple filters
    python trading_app/view_strategies.py --min-expr 0.30         # ExpR threshold
    python trading_app/view_strategies.py --sort sharpe           # Sort by Sharpe
    python trading_app/view_strategies.py --output strats.csv     # Export to CSV
    python trading_app/view_strategies.py --summary               # Session summary only
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH

# Column aliases for --sort
SORT_COLUMNS = {
    "expr": "expectancy_r",
    "expectancy": "expectancy_r",
    "expectancy_r": "expectancy_r",
    "sharpe": "sharpe_ratio",
    "sharpe_ratio": "sharpe_ratio",
    "wr": "win_rate",
    "win_rate": "win_rate",
    "n": "sample_size",
    "sample_size": "sample_size",
    "maxdd": "max_drawdown_r",
    "max_drawdown_r": "max_drawdown_r",
}


def fetch_strategies(db_path: Path, orb: str | None = None,
                     entry: str | None = None, filter_type: str | None = None,
                     min_expr: float | None = None, sort_col: str = "expectancy_r",
                     direction: str | None = None,
                     limit: int = 20) -> pd.DataFrame:
    """Query validated_setups with optional filters."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Check table exists
        tables = [t[0] for t in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
        if "validated_setups" not in tables:
            return pd.DataFrame()

        where = ["status = 'active'"]
        if orb:
            where.append(f"orb_label = '{orb}'")
        if entry:
            where.append(f"entry_model = '{entry}'")
        if filter_type:
            where.append(f"filter_type = '{filter_type}'")
        if min_expr is not None:
            where.append(f"expectancy_r >= {min_expr}")
        if direction:
            # direction is stored in strategy_id or yearly_results
            # Check if there's a direction column; if not, parse from strategy_id
            where.append(f"strategy_id LIKE '%_{direction}'")

        where_clause = " AND ".join(where)

        # Sort direction: ascending for max_drawdown_r, descending for everything else
        sort_dir = "ASC" if sort_col == "max_drawdown_r" else "DESC"

        sql = f"""
            SELECT strategy_id, orb_label, entry_model, confirm_bars,
                   rr_target, filter_type, sample_size, win_rate,
                   expectancy_r, sharpe_ratio, max_drawdown_r, years_tested,
                   stress_test_passed
            FROM validated_setups
            WHERE {where_clause}
            ORDER BY {sort_col} {sort_dir}
            LIMIT {limit}
        """
        df = con.execute(sql).fetchdf()
    finally:
        con.close()

    return df


def fetch_summary(db_path: Path) -> pd.DataFrame:
    """Session-level summary of validated strategies."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = [t[0] for t in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
        if "validated_setups" not in tables:
            return pd.DataFrame()

        df = con.execute("""
            SELECT orb_label,
                   COUNT(*) as count,
                   ROUND(AVG(expectancy_r), 3) as avg_expr,
                   ROUND(MAX(expectancy_r), 3) as best_expr,
                   ROUND(AVG(sharpe_ratio), 3) as avg_sharpe,
                   ROUND(AVG(win_rate) * 100, 1) as avg_wr_pct
            FROM validated_setups
            WHERE status = 'active'
            GROUP BY orb_label
            ORDER BY count DESC
        """).fetchdf()
    finally:
        con.close()

    return df


def fetch_total_count(db_path: Path) -> int:
    """Total active validated strategies."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = [t[0] for t in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
        if "validated_setups" not in tables:
            return 0
        return con.execute(
            "SELECT COUNT(*) FROM validated_setups WHERE status = 'active'"
        ).fetchone()[0]
    finally:
        con.close()


def format_table(df: pd.DataFrame) -> str:
    """Format strategy DataFrame as aligned terminal table."""
    if df.empty:
        return "  No strategies found.\n"

    lines = []
    # Header
    header = (f" {'#':>3}  {'ORB':<5} {'EM':<3} {'CB':>2}  {'RR':>4}  "
              f"{'Filter':<12} {'N':>5}  {'WR%':>5}  {'ExpR':>6}  "
              f"{'Sharpe':>6}  {'MaxDD':>6}  {'Yrs':>3}  {'Stress':<6}")
    lines.append(header)
    lines.append(" " + "-" * (len(header) - 1))

    for i, row in df.iterrows():
        wr_pct = f"{row['win_rate'] * 100:.0f}%"
        expr = f"+{row['expectancy_r']:.2f}" if row['expectancy_r'] >= 0 else f"{row['expectancy_r']:.2f}"
        sharpe = f"+{row['sharpe_ratio']:.2f}" if row['sharpe_ratio'] and row['sharpe_ratio'] >= 0 else (
            f"{row['sharpe_ratio']:.2f}" if row['sharpe_ratio'] else "N/A")
        maxdd = f"{row['max_drawdown_r']:.1f}R" if row['max_drawdown_r'] else "N/A"
        stress = "PASS" if row['stress_test_passed'] else "FAIL"

        line = (f" {i + 1:>3}  {row['orb_label']:<5} {row['entry_model']:<3} "
                f"{row['confirm_bars']:>2}  {row['rr_target']:>4.1f}  "
                f"{row['filter_type']:<12} {row['sample_size']:>5}  {wr_pct:>5}  "
                f"{expr:>6}  {sharpe:>6}  {maxdd:>6}  "
                f"{row['years_tested']:>3}  {stress:<6}")
        lines.append(line)

    return "\n".join(lines) + "\n"


def format_summary(df: pd.DataFrame) -> str:
    """Format session summary."""
    if df.empty:
        return "  No strategies found.\n"

    lines = ["Session summary:"]
    for _, row in df.iterrows():
        lines.append(
            f"  {row['orb_label']:>4}: {row['count']:>3} strategies | "
            f"avg ExpR {row['avg_expr']:+.3f} | "
            f"best {row['best_expr']:+.3f} | "
            f"avg Sharpe {row['avg_sharpe']:+.3f} | "
            f"avg WR {row['avg_wr_pct']:.0f}%"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="View and filter validated trading strategies"
    )
    parser.add_argument("--top", type=int, default=20,
                        help="Number of strategies to show (default: 20)")
    parser.add_argument("--orb", type=str, default=None,
                        help="Filter by ORB session (e.g., 0900, 1800)")
    parser.add_argument("--entry", type=str, default=None,
                        help="Filter by entry model (E1, E2, E3)")
    parser.add_argument("--filter", type=str, default=None, dest="filter_type",
                        help="Filter by filter type (e.g., ORB_G4, ORB_G6)")
    parser.add_argument("--direction", type=str, default=None,
                        choices=["LONG", "SHORT", "BOTH"],
                        help="Filter by direction")
    parser.add_argument("--min-expr", type=float, default=None,
                        help="Minimum expectancy_r threshold")
    parser.add_argument("--sort", type=str, default="expr",
                        choices=list(SORT_COLUMNS.keys()),
                        help="Sort column (default: expr)")
    parser.add_argument("--output", type=str, default=None,
                        help="Export to CSV file")
    parser.add_argument("--summary", action="store_true",
                        help="Show session summary only")
    parser.add_argument("--db", type=str, default=None,
                        help="Database path (default: gold.db)")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    total = fetch_total_count(db_path)
    sort_col = SORT_COLUMNS[args.sort]

    if args.summary:
        print(f"\n=== Validated Strategies ({total} active) ===\n")
        summary_df = fetch_summary(db_path)
        print(format_summary(summary_df))
        return

    # Build filter description
    filters = []
    if args.orb:
        filters.append(f"ORB={args.orb}")
    if args.entry:
        filters.append(f"entry={args.entry}")
    if args.filter_type:
        filters.append(f"filter={args.filter_type}")
    if args.direction:
        filters.append(f"dir={args.direction}")
    if args.min_expr is not None:
        filters.append(f"ExpR>={args.min_expr}")
    filter_desc = f" [{', '.join(filters)}]" if filters else ""

    print(f"\n=== Validated Strategies ({total} active){filter_desc} ===")
    print(f"Showing top {args.top} by {sort_col}\n")

    df = fetch_strategies(
        db_path, orb=args.orb, entry=args.entry,
        filter_type=args.filter_type, min_expr=args.min_expr,
        sort_col=sort_col, direction=args.direction, limit=args.top,
    )
    print(format_table(df))

    # Always show summary below main table
    summary_df = fetch_summary(db_path)
    if not summary_df.empty:
        print(format_summary(summary_df))

    # CSV export
    if args.output and not df.empty:
        output_path = PROJECT_ROOT / args.output
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} strategies to {output_path}")


if __name__ == "__main__":
    main()
