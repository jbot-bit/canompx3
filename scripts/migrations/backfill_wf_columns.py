"""
Backfill walk-forward soft gate columns on validated_setups.

Reads walkforward_results.jsonl (append-only, last entry per strategy_id wins)
and updates wf_tested, wf_passed, wf_windows on matching validated_setups rows.

Safe to re-run (idempotent — overwrites previous WF values).

Usage:
    python scripts/migrations/backfill_wf_columns.py
    python scripts/migrations/backfill_wf_columns.py --dry-run
    python scripts/migrations/backfill_wf_columns.py --db C:/db/gold.db
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from pipeline.paths import GOLD_DB_PATH

DEFAULT_JSONL = PROJECT_ROOT / "data" / "walkforward_results.jsonl"


def backfill_wf(
    db_path: Path,
    jsonl_path: Path,
    dry_run: bool = False,
) -> int:
    """Backfill WF columns from JSONL. Returns count of rows updated."""
    # Read JSONL — last entry per strategy_id wins
    wf_map: dict[str, dict] = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["strategy_id"]
            wf_map[sid] = rec

    print(f"Loaded {len(wf_map)} unique strategy_ids from JSONL "
          f"({jsonl_path.name})")

    if dry_run:
        print("(DRY RUN — no changes written)")
        return 0

    con = duckdb.connect(str(db_path))
    try:
        # Get all strategy_ids in validated_setups
        existing = {
            r[0]
            for r in con.execute(
                "SELECT strategy_id FROM validated_setups"
            ).fetchall()
        }

        updated = 0
        for sid, rec in wf_map.items():
            if sid not in existing:
                continue
            passed = rec.get("passed", False)
            n_windows = rec.get("n_valid_windows")
            con.execute(
                "UPDATE validated_setups "
                "SET wf_tested = TRUE, wf_passed = ?, wf_windows = ? "
                "WHERE strategy_id = ?",
                [passed, n_windows, sid],
            )
            updated += 1

        print(f"Updated {updated} rows in validated_setups")
        return updated
    finally:
        con.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill walk-forward columns on validated_setups"
    )
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument(
        "--jsonl", type=str, default=None,
        help="Path to walkforward_results.jsonl",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would change",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    jsonl_path = Path(args.jsonl) if args.jsonl else DEFAULT_JSONL

    if not jsonl_path.exists():
        print(f"ERROR: JSONL not found: {jsonl_path}")
        sys.exit(1)

    print(f"Database: {db_path}")
    print(f"JSONL: {jsonl_path}")
    backfill_wf(db_path, jsonl_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
