"""Export DuckDB tables to Parquet files for read-independent analysis.

After each pipeline stage completes, export the result to Parquet.
Analysis scripts can read Parquet instead of gold.db, completely
sidestepping the single-writer lock for read-heavy workloads.

Usage:
    python pipeline/export_parquet.py                    # Export all tables
    python pipeline/export_parquet.py --table orb_outcomes
    python pipeline/export_parquet.py --output-dir gold_parquet
"""

import argparse
import sys
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH

# Default output directory (next to gold.db)
DEFAULT_OUTPUT_DIR = GOLD_DB_PATH.parent / "gold_parquet"

# Tables to export with their partition strategy
EXPORT_CONFIG = {
    "orb_outcomes": {"partition_by": ["symbol"]},
    "daily_features": {"partition_by": ["symbol"]},
    "validated_setups": {},  # Small table, no partitioning
    "edge_families": {},     # Small table, no partitioning
}


def export_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    output_dir: Path,
    partition_by: list[str] | None = None,
) -> int:
    """Export a DuckDB table to Parquet.

    Args:
        con: DuckDB connection (read-only is fine).
        table: Table name to export.
        output_dir: Directory to write Parquet files to.
        partition_by: Optional list of columns to partition by.

    Returns:
        Number of rows exported.
    """
    row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if row_count == 0:
        print(f"  {table}: 0 rows — skipping")
        return 0

    table_dir = output_dir / table
    # Clean previous export (partitioned writes can't overwrite in DuckDB 1.4.4)
    import shutil
    if table_dir.exists():
        shutil.rmtree(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    if partition_by:
        partition_cols = ", ".join(partition_by)
        con.execute(f"""
            COPY (SELECT * FROM {table})
            TO '{table_dir}' (FORMAT PARQUET, PARTITION_BY ({partition_cols}))
        """)
    else:
        out_file = table_dir / f"{table}.parquet"
        con.execute(f"""
            COPY {table} TO '{out_file}' (FORMAT PARQUET)
        """)

    print(f"  {table}: {row_count:,} rows exported")
    return row_count


def export_all(
    db_path: Path | None = None,
    output_dir: Path | None = None,
    tables: list[str] | None = None,
) -> dict[str, int]:
    """Export all configured tables to Parquet.

    Args:
        db_path: Path to DuckDB file. Defaults to GOLD_DB_PATH.
        output_dir: Output directory. Defaults to gold_parquet/.
        tables: Specific tables to export. Defaults to all configured.

    Returns:
        Dict of table name -> rows exported.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    if tables is None:
        tables = list(EXPORT_CONFIG.keys())

    results = {}
    with duckdb.connect(str(db_path), read_only=True) as con:
        from pipeline.db_config import configure_connection
        configure_connection(con)

        for table in tables:
            config = EXPORT_CONFIG.get(table, {})
            partition_by = config.get("partition_by")

            try:
                results[table] = export_table(con, table, output_dir, partition_by)
            except duckdb.CatalogException:
                print(f"  {table}: table not found — skipping")
                results[table] = 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Export DuckDB tables to Parquet")
    parser.add_argument("--db-path", type=Path, default=None,
                        help="Path to gold.db")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: gold_parquet/)")
    parser.add_argument("--table", type=str, default=None,
                        help="Export a specific table only")
    args = parser.parse_args()

    tables = [args.table] if args.table else None

    print("Exporting tables to Parquet...")
    results = export_all(
        db_path=args.db_path,
        output_dir=args.output_dir,
        tables=tables,
    )

    total = sum(results.values())
    print(f"\nTotal: {total:,} rows exported across {len(results)} tables")

    if total == 0:
        print("WARNING: No data exported. Check that tables exist and have data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
