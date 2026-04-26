"""Soft-retire active validated rows for non-tradeable instruments.

`validated_setups.status='active'` is the deployable shelf. Instruments outside
`ACTIVE_ORB_INSTRUMENTS` are allowed for research, but they must not remain
active on that shelf.

Safe to re-run: only rows with status='active' and a non-active instrument are
updated.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH
from trading_app.validated_shelf import (
    DEPLOYMENT_SCOPE_NON_DEPLOYABLE,
    NON_ACTIVE_INSTRUMENT_RETIREMENT_REASON,
)

RETIREMENT_REASON = NON_ACTIVE_INSTRUMENT_RETIREMENT_REASON


def retire_non_active_validated(
    db_path: Path | None = None,
    dry_run: bool = False,
    con: duckdb.DuckDBPyConnection | None = None,
) -> int:
    """Soft-retire active validated rows for instruments outside the active shelf."""
    owns_con = con is None
    if owns_con:
        db_path = db_path or GOLD_DB_PATH
        con = duckdb.connect(str(db_path))

    placeholders = ", ".join("?" * len(ACTIVE_ORB_INSTRUMENTS))
    now_utc = datetime.now(UTC)

    try:
        cols = {
            row[0]
            for row in con.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'validated_setups'
                """
            ).fetchall()
        }
        if "deployment_scope" not in cols:
            con.execute("ALTER TABLE validated_setups ADD COLUMN deployment_scope VARCHAR")

        rows = con.execute(
            f"""
            SELECT instrument, COUNT(*)
            FROM validated_setups
            WHERE status = 'active'
              AND instrument NOT IN ({placeholders})
            GROUP BY instrument
            ORDER BY instrument
            """,
            list(ACTIVE_ORB_INSTRUMENTS),
        ).fetchall()
        total = sum(row[1] for row in rows)
        print(f"Found {total} active validated strategies on non-active instruments")
        for instrument, count in rows:
            print(f"  {instrument}: {count}")

        if total == 0:
            print("Nothing to retire.")
            return 0

        if dry_run:
            print("(DRY RUN — no changes written)")
            return 0

        con.execute(
            f"""
            UPDATE validated_setups
            SET status = 'retired',
                deployment_scope = ?,
                retired_at = ?,
                retirement_reason = ?
            WHERE status = 'active'
              AND instrument NOT IN ({placeholders})
            """,
            [
                DEPLOYMENT_SCOPE_NON_DEPLOYABLE,
                now_utc,
                RETIREMENT_REASON,
                *list(ACTIVE_ORB_INSTRUMENTS),
            ],
        )

        con.execute(
            f"DELETE FROM edge_families WHERE instrument NOT IN ({placeholders})",
            list(ACTIVE_ORB_INSTRUMENTS),
        )
        print(f"Retired {total} strategies and cleared non-active edge_families rows")
        return total
    finally:
        if owns_con:
            con.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Soft-retire non-active validated shelf rows")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    print(f"Database: {db_path}")
    retire_non_active_validated(db_path=db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
