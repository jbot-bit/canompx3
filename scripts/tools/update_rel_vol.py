#!/usr/bin/env python3
"""One-shot script: update daily_features.rel_vol to minute-of-day median.

Aligns rel_vol_{label} columns with the computation used by
strategy_discovery._compute_relative_volumes() (the canonical consumer).

Safe to re-run (idempotent). Commits per (instrument, orb_minutes, session)
so partial runs leave the DB in a consistent state per session.

Usage:
    python scripts/tools/update_rel_vol.py
    python scripts/tools/update_rel_vol.py --dry-run
"""

import argparse
import statistics
import time
from zoneinfo import ZoneInfo

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.log import get_logger
from pipeline.paths import GOLD_DB_PATH

logger = get_logger(__name__)
UTC = ZoneInfo("UTC")
LOOKBACK = 20
MIN_PRIOR = 5


def _minute_key(ts):
    utc = ts.astimezone(UTC) if ts.tzinfo is not None else ts
    return (utc.year, utc.month, utc.day, utc.hour, utc.minute)


def update_rel_vol(con, instrument, orb_minutes, dry_run=False):
    """Update all rel_vol_{label} columns for one (instrument, orb_minutes)."""
    # Get session labels from columns
    cols = [
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='daily_features' AND column_name LIKE 'rel_vol_%' "
            "ORDER BY column_name"
        ).fetchall()
    ]
    sessions = [c.replace("rel_vol_", "") for c in cols]

    total_updated = 0

    for label in sessions:
        bts_col = f"orb_{label}_break_ts"
        bvol_col = f"orb_{label}_break_bar_volume"
        rv_col = f"rel_vol_{label}"

        # Verify break_ts column exists
        has_col = con.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            f"WHERE table_name='daily_features' AND column_name='{bts_col}'"
        ).fetchone()[0]
        if not has_col:
            continue

        # Load rows
        data = con.execute(
            f"SELECT trading_day, {bts_col}, {bvol_col}, {rv_col} "
            f"FROM daily_features "
            f"WHERE symbol = ? AND orb_minutes = ? "
            f"ORDER BY trading_day",
            [instrument, orb_minutes],
        ).fetchall()

        if not data:
            continue

        # Step 1: collect unique break minutes
        unique_minutes = set()
        for _td, ts, _bvol, _old_rv in data:
            if ts is not None and hasattr(ts, "hour"):
                utc = ts.astimezone(UTC) if ts.tzinfo else ts
                unique_minutes.add(utc.hour * 60 + utc.minute)

        if not unique_minutes:
            continue

        # Step 2: load bars_1m history per minute (no date filter — full history)
        minute_history = {}
        for mod in sorted(unique_minutes):
            h, m = divmod(mod, 60)
            bars = con.execute(
                "SELECT ts_utc, volume FROM bars_1m "
                "WHERE symbol = ? "
                "AND EXTRACT(HOUR FROM (ts_utc AT TIME ZONE 'UTC')) = ? "
                "AND EXTRACT(MINUTE FROM (ts_utc AT TIME ZONE 'UTC')) = ? "
                "ORDER BY ts_utc",
                [instrument, h, m],
            ).fetchall()
            minute_history[mod] = [(_minute_key(ts), vol) for ts, vol in bars]

        # Step 3: compute new rel_vol and batch updates
        updates = []  # (new_rv, instrument, trading_day, orb_minutes)
        for td, ts, bvol, old_rv in data:
            new_rv = None

            if ts is not None and bvol is not None and bvol > 0 and hasattr(ts, "hour"):
                utc = ts.astimezone(UTC) if ts.tzinfo else ts
                mod = utc.hour * 60 + utc.minute
                history = minute_history.get(mod, [])

                if history:
                    break_key = _minute_key(ts)
                    idx = None
                    for j, (k, _) in enumerate(history):
                        if k == break_key:
                            idx = j
                            break

                    if idx is not None:
                        start = max(0, idx - LOOKBACK)
                        prior_vols = [v for _, v in history[start:idx] if v > 0]

                        if len(prior_vols) >= MIN_PRIOR:
                            baseline = statistics.median(prior_vols)
                            if baseline > 0:
                                new_rv = round(int(bvol) / baseline, 4)

            if new_rv != old_rv:
                updates.append((new_rv, instrument, td, orb_minutes))

        if not updates:
            continue

        if dry_run:
            logger.info(
                "  [DRY RUN] %s O%d %s: %d rows would change",
                instrument,
                orb_minutes,
                label,
                len(updates),
            )
            total_updated += len(updates)
            continue

        # Batch update in a single transaction per session
        con.execute("BEGIN TRANSACTION")
        for new_rv, inst, td, om in updates:
            con.execute(
                f"UPDATE daily_features SET {rv_col} = ? WHERE symbol = ? AND trading_day = ? AND orb_minutes = ?",
                [new_rv, inst, td, om],
            )
        con.execute("COMMIT")
        total_updated += len(updates)
        logger.info(
            "  %s O%d %s: %d rows updated",
            instrument,
            orb_minutes,
            label,
            len(updates),
        )

    return total_updated


def main():
    parser = argparse.ArgumentParser(description="Update rel_vol to minute-of-day median")
    parser.add_argument("--dry-run", action="store_true", help="Count changes without writing")
    args = parser.parse_args()

    con = duckdb.connect(str(GOLD_DB_PATH))
    instruments = list(ACTIVE_ORB_INSTRUMENTS)
    grand_total = 0

    for inst in instruments:
        t0 = time.time()
        logger.info("=== %s ===", inst)
        for om in [5, 15, 30]:
            n = update_rel_vol(con, inst, om, dry_run=args.dry_run)
            grand_total += n
        elapsed = time.time() - t0
        logger.info("  %s done in %.1fs", inst, elapsed)

    con.close()
    logger.info("Total: %d rows %s", grand_total, "would change" if args.dry_run else "updated")


if __name__ == "__main__":
    main()
