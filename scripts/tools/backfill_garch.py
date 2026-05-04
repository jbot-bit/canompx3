#!/usr/bin/env python3
"""
Backfill garch_forecast_vol, garch_atr_ratio, and garch_forecast_vol_pct
into existing daily_features rows.

Uses UPDATE (not DELETE+INSERT) to avoid FK constraint issues with orb_outcomes.
Processes one instrument at a time, computing GARCH(1,1) from trailing daily closes.

Usage:
    python scripts/tools/backfill_garch.py --instrument MGC
    python scripts/tools/backfill_garch.py --instrument MGC MNQ MES M2K
    python scripts/tools/backfill_garch.py --instrument MGC --start-date 2026-04-28
"""

import argparse
import logging
import time
from datetime import date

import duckdb

from pipeline.build_daily_features import (
    GARCH_PCT_MIN_PRIOR_VALUES,
    _prior_rank_pct,
    compute_garch_forecast,
)
from pipeline.paths import GOLD_DB_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

GARCH_PCT_LOOKBACK = 252


def backfill_instrument(db_path: str, instrument: str, start_date: date | None = None):
    """Backfill GARCH columns for one instrument via UPDATE.

    Computes ``garch_forecast_vol`` and ``garch_atr_ratio`` from trailing
    daily closes (no look-ahead), then computes ``garch_forecast_vol_pct``
    as a rolling rank over the prior ``GARCH_PCT_LOOKBACK`` GARCH values
    using the canonical ``_prior_rank_pct`` helper from ``build_daily_features``.

    ``start_date`` (optional) skips writes for trading_days strictly before
    that date — useful for incremental backfills (e.g. only fix the last
    NULL row). Reads remain full-history so prior_closes/prior_garch are
    intact even when writes are scoped.
    """
    con = duckdb.connect(db_path)

    # Load all daily closes in order (orb_minutes=5 to avoid tripling).
    # Also pull the existing GARCH value so the pct rolling rank uses
    # already-stored values for trading_days we're not rewriting.
    rows = con.execute(
        """
        SELECT trading_day, daily_close, atr_20, garch_forecast_vol
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
        ORDER BY trading_day ASC
    """,
        [instrument],
    ).fetchall()

    logger.info(f"{instrument}: {len(rows)} trading days loaded")

    # Stage GARCH values per index so pct can rank against already-computed
    # values without re-querying the DB. Initialise from existing stored
    # values; we'll overwrite as we recompute.
    garch_series: list[float | None] = [r[3] for r in rows]

    updated = 0
    t0 = time.time()

    for i, (trading_day, _daily_close, atr_20, _existing_garch) in enumerate(rows):
        # Collect prior closes (no look-ahead)
        prior_closes = [r[1] for r in rows[:i] if r[1] is not None]

        garch_vol = compute_garch_forecast(prior_closes)
        if garch_vol is None:
            # Keep stored value (likely also None for warm-up days) and skip pct.
            continue

        garch_series[i] = garch_vol

        # Convert annualized vol to implied daily ATR-equivalent points
        last_close = prior_closes[-1] if prior_closes else None
        if atr_20 and atr_20 > 0 and last_close is not None:
            implied_daily_atr = (garch_vol / (252**0.5)) * last_close
            garch_atr_ratio = round(implied_daily_atr / atr_20, 4)
        else:
            garch_atr_ratio = None

        # garch_forecast_vol_pct: rolling rank using canonical helper.
        # _prior_rank_pct expects a list of dicts with the column key — wrap
        # garch_series[i-lookback:i+1] in dicts so the helper signature holds.
        pct_rows = [{"garch_forecast_vol": v} for v in garch_series[: i + 1]]
        garch_pct = _prior_rank_pct(
            pct_rows,
            i,
            "garch_forecast_vol",
            lookback=GARCH_PCT_LOOKBACK,
            min_prior=GARCH_PCT_MIN_PRIOR_VALUES,
        )

        if start_date is not None and trading_day < start_date:
            continue

        # UPDATE all orb_minutes rows for this day (5, 15, 30)
        con.execute(
            """
            UPDATE daily_features
            SET garch_forecast_vol = ?,
                garch_atr_ratio = ?,
                garch_forecast_vol_pct = ?
            WHERE symbol = ? AND trading_day = ?
        """,
            [garch_vol, garch_atr_ratio, garch_pct, instrument, trading_day],
        )
        updated += 1

        if updated % 50 == 0:
            elapsed = time.time() - t0
            rate = updated / elapsed if elapsed > 0 else 0
            remaining = (len(rows) - i) / rate if rate > 0 else 0
            logger.info(f"  {instrument}: {updated} rows updated, {rate:.1f} rows/s, ~{remaining:.0f}s remaining")

    con.commit()
    con.close()
    elapsed = time.time() - t0
    logger.info(f"{instrument}: {updated} rows updated in {elapsed:.1f}s")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Backfill GARCH columns")
    parser.add_argument("--instrument", nargs="+", default=["MGC"], help="Instruments to backfill")
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH))
    parser.add_argument(
        "--start-date",
        default=None,
        help="ISO date (YYYY-MM-DD) — only write rows on or after this trading_day. "
        "Reads still cover full history so rolling state is intact.",
    )
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date) if args.start_date else None

    total = 0
    for inst in args.instrument:
        total += backfill_instrument(args.db_path, inst, start_date=start_date)

    logger.info(f"Done. {total} total rows updated across {len(args.instrument)} instruments.")


if __name__ == "__main__":
    main()
