"""
Backfill trades_per_year and sharpe_ann on existing experimental_strategies rows.

Instead of re-running full discovery (689K outcome loads + volume computation),
this reads yearly_results JSON + sharpe_ratio already in the table and computes
the two new columns directly.

Usage:
    python scripts/backfill_sharpe_ann.py --db C:/db/gold.db
    python scripts/backfill_sharpe_ann.py --db C:/db/gold.db --dry-run
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import duckdb
from pipeline.paths import GOLD_DB_PATH

def backfill(db_path: Path, dry_run: bool = False) -> int:
    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            """SELECT strategy_id, sharpe_ratio, sample_size, yearly_results
               FROM experimental_strategies"""
        ).fetchall()

        print(f"Loaded {len(rows)} strategies")

        updates = []
        for strategy_id, sharpe_ratio, sample_size, yearly_json in rows:
            yearly = json.loads(yearly_json) if yearly_json else {}
            n_years = len(yearly)
            n_traded = sample_size or 0

            trades_per_year = round(n_traded / n_years, 1) if n_years > 0 else 0
            if sharpe_ratio is not None and trades_per_year > 0:
                sharpe_ann = round(sharpe_ratio * (trades_per_year ** 0.5), 4)
            else:
                sharpe_ann = None

            updates.append((trades_per_year, sharpe_ann, strategy_id))

        if dry_run:
            # Show a few examples
            for tpy, sa, sid in updates[:5]:
                print(f"  {sid}: trades_per_year={tpy}, sharpe_ann={sa}")
            print(f"  ... ({len(updates)} total, DRY RUN)")
            return len(updates)

        con.executemany(
            """UPDATE experimental_strategies
               SET trades_per_year = ?, sharpe_ann = ?
               WHERE strategy_id = ?""",
            updates,
        )
        con.commit()
        print(f"Updated {len(updates)} rows")

        # Also backfill validated_setups
        vs_rows = con.execute(
            """SELECT strategy_id, sharpe_ratio, sample_size, yearly_results
               FROM validated_setups"""
        ).fetchall()
        if vs_rows:
            vs_updates = []
            for strategy_id, sharpe_ratio, sample_size, yearly_json in vs_rows:
                yearly = json.loads(yearly_json) if yearly_json else {}
                n_years = len(yearly)
                n_traded = sample_size or 0
                trades_per_year = round(n_traded / n_years, 1) if n_years > 0 else 0
                if sharpe_ratio is not None and trades_per_year > 0:
                    sharpe_ann = round(sharpe_ratio * (trades_per_year ** 0.5), 4)
                else:
                    sharpe_ann = None
                vs_updates.append((trades_per_year, sharpe_ann, strategy_id))

            con.executemany(
                """UPDATE validated_setups
                   SET trades_per_year = ?, sharpe_ann = ?
                   WHERE strategy_id = ?""",
                vs_updates,
            )
            con.commit()
            print(f"Updated {len(vs_updates)} validated_setups rows")

        return len(updates)
    finally:
        con.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill sharpe_ann columns")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    backfill(db_path, dry_run=args.dry_run)
