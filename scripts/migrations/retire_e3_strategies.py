"""
Soft-retire all active E3 strategies in validated_setups.

E3 (retrace limit entry) has 0/50 FDR-significant strategies and no timeout
mechanism (100% fill rate = late garbage included). Soft-retire rather than
purge so historical data is preserved.

Safe to re-run (idempotent — only updates WHERE status='active').

Usage:
    python scripts/migrations/retire_e3_strategies.py
    python scripts/migrations/retire_e3_strategies.py --dry-run
    python scripts/migrations/retire_e3_strategies.py --db C:/db/gold.db
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from pipeline.paths import GOLD_DB_PATH


def retire_e3(
    db_path: Path | None = None,
    dry_run: bool = False,
    con: duckdb.DuckDBPyConnection | None = None,
) -> int:
    """Soft-retire all active E3 strategies. Returns count retired.

    Pass `con` directly for testing (in-memory DB).
    Pass `db_path` for file-based DB.
    """
    owns_con = con is None
    if owns_con:
        db_path = db_path or GOLD_DB_PATH
        con = duckdb.connect(str(db_path))

    try:
        # Count active E3
        active_count = con.execute(
            "SELECT COUNT(*) FROM validated_setups "
            "WHERE entry_model = 'E3' AND status = 'active'"
        ).fetchone()[0]

        print(f"Found {active_count} active E3 strategies")

        if active_count == 0:
            print("Nothing to retire.")
            return 0

        if dry_run:
            print("(DRY RUN — no changes written)")
            return 0

        # Retire
        now_utc = datetime.now(timezone.utc)
        con.execute(
            "UPDATE validated_setups "
            "SET status = 'RETIRED', "
            "    retired_at = ?, "
            "    retirement_reason = 'PASS2: 0/50 FDR-sig, no timeout mechanism' "
            "WHERE entry_model = 'E3' AND status = 'active'",
            [now_utc],
        )

        print(f"Retired {active_count} E3 strategies")
        return active_count

    finally:
        if owns_con:
            con.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Soft-retire active E3 strategies in validated_setups"
    )
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    print(f"Database: {db_path}")
    retire_e3(db_path=db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
