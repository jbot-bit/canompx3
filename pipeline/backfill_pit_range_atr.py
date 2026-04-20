#!/usr/bin/env python3
"""Backfill and maintain daily_features.pit_range_atr.

Canonical post-pass that populates pit_range_atr in daily_features from
exchange_statistics. Serves both one-shot historical backfill and the
forward-flow hook called at the end of build_daily_features.

Formula (zero look-ahead, mirrors scripts/research/exchange_range_t2t8.py):

    pit_range_atr(T, symbol) =
        (prev_cal_date.session_high - prev_cal_date.session_low) / atr_20(T)

where prev_cal_date is the most-recent cal_date strictly less than T in
exchange_statistics for the same symbol. Monday trading_day -> Friday
prev_cal_date; day-after-holiday -> last open session.

Look-ahead audit: CME pit of cal_date X closes at 21:15 UTC X. CME_REOPEN
on Brisbane trading_day T opens at 23:00 UTC of the previous Brisbane
calendar day. Using cal_date strictly before T means the pit session
referenced closed before the Brisbane trading_day T began.

Idempotent: a single UPDATE; re-runnable with identical output. Rows
whose upstream inputs are missing remain NULL, matching PitRangeFilter
semantics (trading_app/config.py:2337-2339).

@research-source scripts/research/exchange_range_t2t8.py
@entry-models E1, E2 (at CME_REOPEN)

Usage:
    python pipeline/backfill_pit_range_atr.py --all
    python pipeline/backfill_pit_range_atr.py --instrument MNQ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.db_config import configure_connection
from pipeline.log import get_logger
from pipeline.paths import GOLD_DB_PATH

logger = get_logger(__name__)


# Minimum coverage fraction demanded of active instruments post-backfill.
# Remaining gap is structural: US holidays where pit did not open, plus
# early-history dates before an instrument's micro launched (MES micros
# launched 2019-05-03; MNQ 2019-05-06). Research file documents ~97.2%
# as the practical ceiling for the canonical 3-instrument universe.
MIN_COVERAGE = 0.90


def _scalar(con: duckdb.DuckDBPyConnection, sql: str, params: list) -> int:
    """Fetch a single scalar count, with explicit None guard for typing."""
    row = con.execute(sql, params).fetchone()
    if row is None:
        return 0
    return int(row[0])


def backfill_instrument(instrument: str, con: duckdb.DuckDBPyConnection) -> tuple[int, int, float]:
    """Compute pit_range_atr for one instrument. Returns (populated, total, pct)."""
    # Defensive reset. Leaves NULL where upstream inputs are missing, so
    # re-running on a shrunken exchange_statistics set cannot leave stale
    # numbers behind.
    con.execute(
        "UPDATE daily_features SET pit_range_atr = NULL WHERE symbol = ?",
        [instrument],
    )

    # Correlated-subquery approach. Picks the most-recent cal_date in
    # exchange_statistics strictly before each daily_features trading_day
    # for the same symbol. Denominator is atr_20 from the daily_features
    # row being updated (prior-only rolling window, no look-ahead).
    con.execute(
        """
        UPDATE daily_features AS d
        SET pit_range_atr = CASE
            WHEN d.atr_20 IS NULL OR d.atr_20 <= 0 THEN NULL
            WHEN es.session_high IS NULL OR es.session_low IS NULL THEN NULL
            ELSE (es.session_high - es.session_low) / d.atr_20
        END
        FROM exchange_statistics es
        WHERE d.symbol = ?
          AND es.symbol = d.symbol
          AND es.cal_date = (
              SELECT MAX(es2.cal_date)
              FROM exchange_statistics es2
              WHERE es2.symbol = d.symbol
                AND es2.cal_date < d.trading_day
          )
        """,
        [instrument],
    )
    con.commit()

    total = _scalar(
        con,
        "SELECT COUNT(*) FROM daily_features WHERE symbol = ?",
        [instrument],
    )
    populated = _scalar(
        con,
        "SELECT COUNT(pit_range_atr) FROM daily_features WHERE symbol = ?",
        [instrument],
    )
    pct = (populated / total) if total > 0 else 0.0
    logger.info(f"  {instrument}: {populated:,} / {total:,} rows populated ({pct:.1%})")
    return populated, total, pct


def backfill(
    instruments: list[str] | None = None,
    db_path: Path | None = None,
    min_coverage: float = MIN_COVERAGE,
) -> dict[str, tuple[int, int, float]]:
    """Backfill pit_range_atr for given instruments (default: ACTIVE_ORB_INSTRUMENTS).

    Returns per-instrument (populated, total, pct). Raises if any active
    instrument falls below `min_coverage`.
    """
    import duckdb  # lazy import, matches pipeline convention

    db_path = db_path or GOLD_DB_PATH
    instruments = instruments or list(ACTIVE_ORB_INSTRUMENTS)

    logger.info(f"Backfilling daily_features.pit_range_atr for {instruments} against {db_path}")

    con = duckdb.connect(str(db_path))
    configure_connection(con, writing=True)
    try:
        results: dict[str, tuple[int, int, float]] = {}
        for inst in instruments:
            results[inst] = backfill_instrument(inst, con)
    finally:
        con.close()

    # Fail-closed coverage gate for active instruments.
    for inst in instruments:
        if inst in ACTIVE_ORB_INSTRUMENTS:
            _, _, pct = results[inst]
            if pct < min_coverage:
                raise RuntimeError(
                    f"pit_range_atr coverage for {inst} = {pct:.1%} "
                    f"below floor {min_coverage:.0%}. "
                    f"Check exchange_statistics ingest and atr_20 population."
                )

    return results


def enrich_date_range(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    start_date,
    end_date,
) -> int:
    """Forward-flow entry point invoked by the daily feature builder.

    Enriches just the (symbol, trading_day in [start, end]) slice.
    Returns the number of rows actually populated post-update.

    Fail-open on missing upstream: if the exchange_statistics table
    does not exist or has no rows for this symbol, the enrichment is
    skipped silently. PitRangeFilter is fail-closed at eligibility
    time, so missing upstream data drops PIT_MIN-gated lanes rather
    than silently passing them.
    """
    # Guard against missing upstream table (partial pipeline state,
    # test fixtures). information_schema gives a cheap existence check
    # that works uniformly across DuckDB versions.
    table_check = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'exchange_statistics'"
    ).fetchone()
    if table_check is None or int(table_check[0]) == 0:
        return 0

    has_stats = _scalar(
        con,
        "SELECT COUNT(*) FROM exchange_statistics WHERE symbol = ?",
        [symbol],
    )
    if has_stats == 0:
        return 0

    con.execute(
        """
        UPDATE daily_features AS d
        SET pit_range_atr = CASE
            WHEN d.atr_20 IS NULL OR d.atr_20 <= 0 THEN NULL
            WHEN es.session_high IS NULL OR es.session_low IS NULL THEN NULL
            ELSE (es.session_high - es.session_low) / d.atr_20
        END
        FROM exchange_statistics es
        WHERE d.symbol = ?
          AND d.trading_day >= ?
          AND d.trading_day <= ?
          AND es.symbol = d.symbol
          AND es.cal_date = (
              SELECT MAX(es2.cal_date)
              FROM exchange_statistics es2
              WHERE es2.symbol = d.symbol
                AND es2.cal_date < d.trading_day
          )
        """,
        [symbol, start_date, end_date],
    )
    return _scalar(
        con,
        """SELECT COUNT(pit_range_atr) FROM daily_features
           WHERE symbol = ? AND trading_day >= ? AND trading_day <= ?""",
        [symbol, start_date, end_date],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill daily_features.pit_range_atr from exchange_statistics")
    parser.add_argument(
        "--instrument",
        type=str,
        help="Single instrument (MES / MGC / MNQ)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Backfill all ACTIVE_ORB_INSTRUMENTS",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=MIN_COVERAGE,
        help=f"Minimum acceptable coverage per active instrument (default {MIN_COVERAGE})",
    )
    args = parser.parse_args()

    if not args.instrument and not args.all:
        parser.error("Specify --instrument or --all")

    instruments = list(ACTIVE_ORB_INSTRUMENTS) if args.all else [args.instrument]
    results = backfill(instruments=instruments, min_coverage=args.min_coverage)

    total_rows = sum(r[1] for r in results.values())
    total_pop = sum(r[0] for r in results.values())
    overall_pct = total_pop / total_rows if total_rows > 0 else 0.0
    logger.info(
        f"Done. {total_pop:,} / {total_rows:,} rows populated overall "
        f"({overall_pct:.1%}) across {len(instruments)} instruments."
    )


if __name__ == "__main__":
    main()
