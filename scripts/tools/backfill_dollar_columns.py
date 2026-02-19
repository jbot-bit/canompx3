#!/usr/bin/env python3
"""
Backfill dollar columns on existing orb_outcomes, experimental_strategies,
and validated_setups rows.

Adds risk_dollars and pnl_dollars to orb_outcomes, and dollar aggregates
to strategy tables, without requiring a full rebuild.

Usage:
    python scripts/tools/backfill_dollar_columns.py
    python scripts/tools/backfill_dollar_columns.py --db-path C:/db/gold.db
    python scripts/tools/backfill_dollar_columns.py --dry-run
"""

import argparse
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import COST_SPECS
from trading_app.db_manager import init_trading_app_schema


def backfill_dollar_columns(db_path: Path | None = None, dry_run: bool = False) -> None:
    if db_path is None:
        db_path = GOLD_DB_PATH

    # Ensure migration columns exist
    init_trading_app_schema(db_path=db_path)

    with duckdb.connect(str(db_path)) as con:
        if not dry_run:
            from pipeline.db_config import configure_connection
            configure_connection(con, writing=True)

        # ---- orb_outcomes: backfill risk_dollars and pnl_dollars ----
        for instrument, spec in COST_SPECS.items():
            pv = spec.point_value
            friction = spec.total_friction

            # Count rows needing backfill
            count = con.execute(
                "SELECT COUNT(*) FROM orb_outcomes "
                "WHERE symbol = ? AND entry_price IS NOT NULL AND risk_dollars IS NULL",
                [instrument],
            ).fetchone()[0]

            if count == 0:
                print(f"  {instrument}: orb_outcomes — 0 rows to backfill")
                continue

            print(f"  {instrument}: orb_outcomes — backfilling {count:,} rows...")

            if not dry_run:
                con.execute(f"""
                    UPDATE orb_outcomes SET
                        risk_dollars = ROUND(ABS(entry_price - stop_price) * {pv} + {friction}, 2),
                        pnl_dollars  = ROUND(pnl_r * (ABS(entry_price - stop_price) * {pv} + {friction}), 2)
                    WHERE symbol = ?
                      AND entry_price IS NOT NULL
                      AND risk_dollars IS NULL
                """, [instrument])
                con.commit()
                print(f"    Done.")

        # ---- experimental_strategies: backfill dollar aggregates ----
        for instrument, spec in COST_SPECS.items():
            pv = spec.point_value
            friction = spec.total_friction

            count = con.execute(
                "SELECT COUNT(*) FROM experimental_strategies "
                "WHERE instrument = ? AND median_risk_points IS NOT NULL AND median_risk_dollars IS NULL",
                [instrument],
            ).fetchone()[0]

            if count == 0:
                print(f"  {instrument}: experimental_strategies — 0 rows to backfill")
                continue

            print(f"  {instrument}: experimental_strategies — backfilling {count:,} rows...")

            if not dry_run:
                con.execute(f"""
                    UPDATE experimental_strategies SET
                        median_risk_dollars = ROUND(median_risk_points * {pv} + {friction}, 2),
                        avg_risk_dollars    = ROUND(avg_risk_points * {pv} + {friction}, 2),
                        avg_win_dollars     = ROUND(avg_win_r * (avg_risk_points * {pv} + {friction}), 2),
                        avg_loss_dollars    = ROUND(avg_loss_r * (avg_risk_points * {pv} + {friction}), 2)
                    WHERE instrument = ?
                      AND median_risk_points IS NOT NULL
                      AND median_risk_dollars IS NULL
                """, [instrument])
                con.commit()
                print(f"    Done.")

        # ---- validated_setups: backfill from experimental_strategies ----
        count = con.execute(
            "SELECT COUNT(*) FROM validated_setups WHERE median_risk_dollars IS NULL"
        ).fetchone()[0]

        if count > 0:
            print(f"  validated_setups — backfilling {count:,} rows from experimental_strategies...")

            if not dry_run:
                con.execute("""
                    UPDATE validated_setups vs SET
                        median_risk_dollars = es.median_risk_dollars,
                        avg_risk_dollars    = es.avg_risk_dollars,
                        avg_win_dollars     = es.avg_win_dollars,
                        avg_loss_dollars    = es.avg_loss_dollars
                    FROM experimental_strategies es
                    WHERE vs.strategy_id = es.strategy_id
                      AND vs.median_risk_dollars IS NULL
                      AND es.median_risk_dollars IS NOT NULL
                """)
                con.commit()
                print(f"    Done.")
        else:
            print(f"  validated_setups — 0 rows to backfill")

    print("\nBackfill complete.")
    if dry_run:
        print("  (DRY RUN — no data written)")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill dollar columns on existing data"
    )
    parser.add_argument("--db-path", type=Path, help="Override DB path")
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't write")
    args = parser.parse_args()

    backfill_dollar_columns(db_path=args.db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
