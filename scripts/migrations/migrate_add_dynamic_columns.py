#!/usr/bin/env python3
"""Add missing ORB columns to daily_features for new session labels.

One-time schema migration. Idempotent (safe to re-run).
Does NOT populate data — run build_daily_features after this.

Usage:
    python scripts/migrate_add_dynamic_columns.py --db-path C:\\db\\gold.db
    python scripts/migrate_add_dynamic_columns.py --db-path C:\\db\\gold.db --dry-run
"""

import argparse
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.init_db import ORB_LABELS

# 9 columns per ORB label (must match _build_daily_features_ddl in init_db.py)
ORB_COLUMN_DEFS = [
    ("high", "DOUBLE"),
    ("low", "DOUBLE"),
    ("size", "DOUBLE"),
    ("break_dir", "TEXT"),
    ("break_ts", "TIMESTAMPTZ"),
    ("outcome", "TEXT"),
    ("mae_r", "DOUBLE"),
    ("mfe_r", "DOUBLE"),
    ("double_break", "BOOLEAN"),
]


def migrate(db_path: Path, dry_run: bool = False) -> int:
    """Add missing ORB columns to daily_features.

    Returns number of columns added.
    """
    con = duckdb.connect(str(db_path))

    try:
        # Get existing columns
        existing = set(
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'daily_features'"
            ).fetchall()
        )

        print(f"Existing daily_features columns: {len(existing)}")
        print(f"Expected ORB labels: {ORB_LABELS}")

        added = 0
        for label in ORB_LABELS:
            label_missing = []
            for suffix, dtype in ORB_COLUMN_DEFS:
                col_name = f"orb_{label}_{suffix}"
                if col_name not in existing:
                    label_missing.append((col_name, dtype))

            if not label_missing:
                continue

            print(f"\n  {label}: {len(label_missing)} columns missing")
            for col_name, dtype in label_missing:
                if dry_run:
                    print(f"    Would add: {col_name} {dtype}")
                else:
                    con.execute(
                        f"ALTER TABLE daily_features ADD COLUMN {col_name} {dtype}"
                    )
                    print(f"    Added: {col_name} {dtype}")
                added += 1

        if not dry_run and added > 0:
            con.commit()

        # Verify final column count
        final_count = con.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_name = 'daily_features'"
        ).fetchone()[0]

        print(f"\n{'Would add' if dry_run else 'Added'} {added} columns")
        print(f"Final column count: {final_count}")

        if added == 0:
            print("Schema already up to date.")

        return added

    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add missing ORB columns to daily_features",
    )
    parser.add_argument(
        "--db-path", type=Path, required=True, help="Path to DuckDB database",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be added",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"FATAL: Database not found: {args.db_path}")
        sys.exit(1)

    migrate(args.db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
