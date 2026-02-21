#!/usr/bin/env python3
"""
Backfill garch_forecast_vol and garch_atr_ratio into existing daily_features rows.

Uses UPDATE (not DELETE+INSERT) to avoid FK constraint issues with orb_outcomes.
Processes one instrument at a time, computing GARCH(1,1) from trailing daily closes.

Usage:
    python scripts/tools/backfill_garch.py --instrument MGC
    python scripts/tools/backfill_garch.py --instrument MGC MNQ MES M2K
"""
import argparse
import logging
import time
import duckdb
import numpy as np

from pipeline.paths import GOLD_DB_PATH
from pipeline.build_daily_features import compute_garch_forecast

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def backfill_instrument(db_path: str, instrument: str):
    """Backfill GARCH columns for one instrument via UPDATE."""
    con = duckdb.connect(db_path)

    # Load all daily closes in order (orb_minutes=5 to avoid tripling)
    rows = con.execute("""
        SELECT trading_day, daily_close, atr_20
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
        ORDER BY trading_day ASC
    """, [instrument]).fetchall()

    logger.info(f"{instrument}: {len(rows)} trading days loaded")

    updated = 0
    t0 = time.time()

    for i, (trading_day, daily_close, atr_20) in enumerate(rows):
        # Collect prior closes (no look-ahead)
        prior_closes = [r[1] for r in rows[:i] if r[1] is not None]

        garch_vol = compute_garch_forecast(prior_closes)
        if garch_vol is None:
            continue

        # Convert annualized vol to implied daily ATR-equivalent points
        last_close = prior_closes[-1] if prior_closes else None
        if atr_20 and atr_20 > 0 and last_close is not None:
            implied_daily_atr = (garch_vol / (252 ** 0.5)) * last_close
            garch_atr_ratio = round(implied_daily_atr / atr_20, 4)
        else:
            garch_atr_ratio = None

        # UPDATE all orb_minutes rows for this day (5, 15, 30)
        con.execute("""
            UPDATE daily_features
            SET garch_forecast_vol = ?, garch_atr_ratio = ?
            WHERE symbol = ? AND trading_day = ?
        """, [garch_vol, garch_atr_ratio, instrument, trading_day])
        updated += 1

        if updated % 50 == 0:
            elapsed = time.time() - t0
            rate = updated / elapsed if elapsed > 0 else 0
            remaining = (len(rows) - i) / rate if rate > 0 else 0
            logger.info(f"  {instrument}: {updated} rows updated, "
                        f"{rate:.1f} rows/s, ~{remaining:.0f}s remaining")

    con.commit()
    con.close()
    elapsed = time.time() - t0
    logger.info(f"{instrument}: {updated} rows updated in {elapsed:.1f}s")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Backfill GARCH columns")
    parser.add_argument("--instrument", nargs="+", default=["MGC"],
                        help="Instruments to backfill")
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH))
    args = parser.parse_args()

    total = 0
    for inst in args.instrument:
        total += backfill_instrument(args.db_path, inst)

    logger.info(f"Done. {total} total rows updated across {len(args.instrument)} instruments.")


if __name__ == "__main__":
    main()
