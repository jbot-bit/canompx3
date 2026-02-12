#!/usr/bin/env python3
"""
Strategy Viewer CLI — browse, filter, and export validated strategies.

Shows annualized Sharpe (ShANN) by default. Use --family to see unique trades.

Usage:
    python trading_app/view_strategies.py                         # Top 20 by ExpR
    python trading_app/view_strategies.py --top 50                # More results
    python trading_app/view_strategies.py --orb 0900              # Filter by session
    python trading_app/view_strategies.py --orb 1800 --entry E3   # Multiple filters
    python trading_app/view_strategies.py --min-expr 0.30         # ExpR threshold
    python trading_app/view_strategies.py --sort sharpe_ann       # Sort by ShANN
    python trading_app/view_strategies.py --output strats.csv     # Export to CSV
    python trading_app/view_strategies.py --summary               # Session summary only
    python trading_app/view_strategies.py --family                # Unique trade families
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
    "sharpe_ann": "sharpe_ann",
    "shann": "sharpe_ann",
    "wr": "win_rate",
    "win_rate": "win_rate",
    "n": "sample_size",
    "sample_size": "sample_size",
    "maxdd": "max_drawdown_r",
    "max_drawdown_r": "max_drawdown_r",
    "tpy": "trades_per_year",
    "trades_per_year": "trades_per_year",
}

# Allowed sort columns (whitelist for SQL safety)
_VALID_SORT_COLS = set(SORT_COLUMNS.values())


def _safe_float(val) -> float | None:
    """Return float if valid, None if None/NaN/missing."""
    if val is None:
        return None
    try:
        import math
        if math.isnan(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _fmt_signed(val, fmt=".2f") -> str:
    """Format a float with +/- sign, or N/A if None/NaN."""
    v = _safe_float(val)
    if v is None:
        return "N/A"
    return f"+{v:{fmt}}" if v >= 0 else f"{v:{fmt}}"


def _has_table(con, table_name: str) -> bool:
    """Check if a table exists in the database."""
    tables = [t[0] for t in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()]
    return table_name in tables


def fetch_strategies(db_path: Path, orb: str | None = None,
                     entry: str | None = None, filter_type: str | None = None,
                     min_expr: float | None = None, sort_col: str = "expectancy_r",
                     direction: str | None = None,
                     limit: int = 20) -> pd.DataFrame:
    """Query validated_setups with optional filters. All inputs parameterized."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if not _has_table(con, "validated_setups"):
            return pd.DataFrame()

        where = ["status = 'active'"]
        params = []

        if orb:
            where.append("orb_label = ?")
            params.append(orb)
        if entry:
            where.append("entry_model = ?")
            params.append(entry)
        if filter_type:
            where.append("filter_type = ?")
            params.append(filter_type)
        if min_expr is not None:
            where.append("expectancy_r >= ?")
            params.append(min_expr)
        if direction:
            where.append("strategy_id LIKE ?")
            params.append(f"%_{direction}")

        where_clause = " AND ".join(where)

        # Whitelist sort_col to prevent SQL injection via sort parameter
        if sort_col not in _VALID_SORT_COLS:
            sort_col = "expectancy_r"
        sort_dir = "ASC" if sort_col == "max_drawdown_r" else "DESC"

        sql = f"""
            SELECT strategy_id, orb_label, entry_model, confirm_bars,
                   rr_target, filter_type, sample_size, win_rate,
                   expectancy_r, sharpe_ratio, sharpe_ann, trades_per_year,
                   max_drawdown_r, years_tested, stress_test_passed
            FROM validated_setups
            WHERE {where_clause}
            ORDER BY {sort_col} {sort_dir}
            LIMIT ?
        """
        params.append(limit)
        df = con.execute(sql, params).fetchdf()
    finally:
        con.close()

    return df


def fetch_summary(db_path: Path) -> pd.DataFrame:
    """Session-level summary of validated strategies with unique trade counts."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if not _has_table(con, "validated_setups"):
            return pd.DataFrame()

        df = con.execute("""
            SELECT orb_label,
                   COUNT(*) as count,
                   COUNT(DISTINCT orb_label || '_' || entry_model || '_' ||
                         CAST(rr_target AS TEXT) || '_' ||
                         CAST(confirm_bars AS TEXT)) as unique_trades,
                   ROUND(AVG(expectancy_r), 3) as avg_expr,
                   ROUND(MAX(expectancy_r), 3) as best_expr,
                   ROUND(AVG(sharpe_ann), 3) as avg_shann,
                   ROUND(MAX(sharpe_ann), 3) as best_shann,
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
        if not _has_table(con, "validated_setups"):
            return 0
        return con.execute(
            "SELECT COUNT(*) FROM validated_setups WHERE status = 'active'"
        ).fetchone()[0]
    finally:
        con.close()


def fetch_unique_trade_count(db_path: Path) -> int:
    """Count unique trade families (by session/EM/RR/CB identity)."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if not _has_table(con, "validated_setups"):
            return 0
        return con.execute("""
            SELECT COUNT(DISTINCT orb_label || '_' || entry_model || '_' ||
                         CAST(rr_target AS TEXT) || '_' ||
                         CAST(confirm_bars AS TEXT))
            FROM validated_setups WHERE status = 'active'
        """).fetchone()[0]
    finally:
        con.close()


def fetch_families(db_path: Path, orb: str | None = None,
                   entry: str | None = None) -> pd.DataFrame:
    """Group strategies by trade identity (session, EM, RR, CB). Parameterized."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if not _has_table(con, "validated_setups"):
            return pd.DataFrame()

        where = ["status = 'active'"]
        params = []
        if orb:
            where.append("orb_label = ?")
            params.append(orb)
        if entry:
            where.append("entry_model = ?")
            params.append(entry)
        where_clause = " AND ".join(where)

        df = con.execute(f"""
            SELECT orb_label, entry_model, rr_target, confirm_bars,
                   COUNT(*) as filter_variants,
                   FIRST(filter_type ORDER BY sharpe_ann DESC NULLS LAST) as best_filter,
                   MAX(sharpe_ann) as best_shann,
                   MAX(sample_size) as max_n,
                   ROUND(MAX(win_rate) * 100, 1) as best_wr_pct,
                   ROUND(MAX(expectancy_r), 3) as best_expr
            FROM validated_setups
            WHERE {where_clause}
            GROUP BY orb_label, entry_model, rr_target, confirm_bars
            ORDER BY best_shann DESC NULLS LAST
        """, params).fetchdf()
    finally:
        con.close()

    return df


def format_table(df: pd.DataFrame) -> str:
    """Format strategy DataFrame as aligned terminal table."""
    if df.empty:
        return "  No strategies found.\n"

    lines = []
    header = (f" {'#':>3}  {'ORB':<5} {'EM':<3} {'CB':>2}  {'RR':>4}  "
              f"{'Filter':<12} {'N':>5}  {'WR%':>5}  {'ExpR':>6}  "
              f"{'ShANN':>6}  {'T/yr':>5}  {'MaxDD':>6}  {'Yrs':>3}  {'Stress':<6}")
    lines.append(header)
    lines.append(" " + "-" * (len(header) - 1))

    for i, row in df.iterrows():
        wr_pct = f"{row['win_rate'] * 100:.0f}%"
        expr = _fmt_signed(row['expectancy_r'])
        shann = _fmt_signed(row['sharpe_ann'])
        tpy_val = _safe_float(row['trades_per_year'])
        tpy = f"{tpy_val:.0f}" if tpy_val is not None else "N/A"
        maxdd_val = _safe_float(row['max_drawdown_r'])
        maxdd = f"{maxdd_val:.1f}R" if maxdd_val is not None else "N/A"
        stress = "PASS" if row['stress_test_passed'] else "FAIL"

        line = (f" {i + 1:>3}  {row['orb_label']:<5} {row['entry_model']:<3} "
                f"{row['confirm_bars']:>2}  {row['rr_target']:>4.1f}  "
                f"{row['filter_type']:<12} {row['sample_size']:>5}  {wr_pct:>5}  "
                f"{expr:>6}  {shann:>6}  {tpy:>5}  {maxdd:>6}  "
                f"{row['years_tested']:>3}  {stress:<6}")
        lines.append(line)

    return "\n".join(lines) + "\n"


def format_families(df: pd.DataFrame) -> str:
    """Format family DataFrame as aligned terminal table."""
    if df.empty:
        return "  No families found.\n"

    lines = []
    header = (f" {'#':>3}  {'ORB':<5} {'EM':<3} {'CB':>2}  {'RR':>4}  "
              f"{'Variants':>8}  {'Best Filter':<12} {'ShANN':>6}  "
              f"{'N':>5}  {'WR%':>5}  {'ExpR':>6}")
    lines.append(header)
    lines.append(" " + "-" * (len(header) - 1))

    for i, row in df.iterrows():
        shann = _fmt_signed(row['best_shann'])
        wr_val = _safe_float(row['best_wr_pct'])
        wr = f"{wr_val:.0f}%" if wr_val is not None else "N/A"
        expr = _fmt_signed(row['best_expr'])
        best_filter = row['best_filter'] if row['best_filter'] else "N/A"

        line = (f" {i + 1:>3}  {row['orb_label']:<5} {row['entry_model']:<3} "
                f"{row['confirm_bars']:>2}  {row['rr_target']:>4.1f}  "
                f"{row['filter_variants']:>8}  {best_filter:<12} {shann:>6}  "
                f"{row['max_n']:>5}  {wr:>5}  {expr:>6}")
        lines.append(line)

    return "\n".join(lines) + "\n"


def format_summary(df: pd.DataFrame) -> str:
    """Format session summary with unique trade counts."""
    if df.empty:
        return "  No strategies found.\n"

    lines = ["Session summary:"]
    for _, row in df.iterrows():
        avg_sh = _safe_float(row['avg_shann'])
        best_sh = _safe_float(row['best_shann'])
        shann_str = f"avg ShANN {avg_sh:+.3f}" if avg_sh is not None else "avg ShANN N/A"
        best_shann_str = f"best {best_sh:+.3f}" if best_sh is not None else "best N/A"
        lines.append(
            f"  {row['orb_label']:>4}: {row['count']:>3} strategies "
            f"({row['unique_trades']} unique trades) | "
            f"avg ExpR {row['avg_expr']:+.3f} | "
            f"{shann_str} | {best_shann_str} | "
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
                        help="Filter by entry model (E1, E3)")
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
    parser.add_argument("--family", action="store_true",
                        help="Group by unique trade identity (session, EM, RR, CB)")
    parser.add_argument("--db", type=str, default=None,
                        help="Database path (default: gold.db)")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    total = fetch_total_count(db_path)
    unique = fetch_unique_trade_count(db_path)
    sort_col = SORT_COLUMNS[args.sort]

    if args.summary:
        print(f"\n=== Validated Strategies: {total} active ({unique} unique trades) ===\n")
        summary_df = fetch_summary(db_path)
        print(format_summary(summary_df))
        return

    if args.family:
        print(f"\n=== {unique} unique trades ({total} strategy variants) ===\n")
        families_df = fetch_families(db_path, orb=args.orb, entry=args.entry)
        print(format_families(families_df))

        summary_df = fetch_summary(db_path)
        if not summary_df.empty:
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

    print(f"\n=== Validated Strategies: {total} active ({unique} unique trades){filter_desc} ===")
    print(f"Showing top {args.top} by {sort_col}\n")

    df = fetch_strategies(
        db_path, orb=args.orb, entry=args.entry,
        filter_type=args.filter_type, min_expr=args.min_expr,
        sort_col=sort_col, direction=args.direction, limit=args.top,
    )
    print(format_table(df))

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
