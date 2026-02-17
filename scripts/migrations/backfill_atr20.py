"""
Backfill atr_20 column in daily_features from existing daily OHLC data.

1. ALTER TABLE to add column if it doesn't exist
2. For each (symbol, orb_minutes) group, compute True Range and ATR(20)
3. UPDATE rows in place

Safe to re-run (idempotent).

Usage:
    python scripts/backfill_atr20.py
    python scripts/backfill_atr20.py --db C:/db/gold.db
    python scripts/backfill_atr20.py --dry-run
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(line_buffering=True)

import duckdb
from pipeline.paths import GOLD_DB_PATH


def backfill_atr20(db_path: Path, dry_run: bool = False) -> int:
    """Backfill atr_20 for all rows in daily_features. Returns rows updated."""
    con = duckdb.connect(str(db_path))
    try:
        # Step 1: Ensure column exists
        cols = [c[0] for c in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'daily_features'"
        ).fetchall()]

        if "atr_20" not in cols:
            print("Adding atr_20 column to daily_features...")
            if not dry_run:
                con.execute("ALTER TABLE daily_features ADD COLUMN atr_20 DOUBLE")
            print("  Done.")
        else:
            print("atr_20 column already exists.")

        # Step 2: Load all rows grouped by (symbol, orb_minutes)
        groups = con.execute(
            "SELECT DISTINCT symbol, orb_minutes FROM daily_features ORDER BY symbol, orb_minutes"
        ).fetchall()

        total_updated = 0

        for symbol, orb_minutes in groups:
            rows = con.execute(
                """SELECT trading_day, daily_high, daily_low, daily_close
                   FROM daily_features
                   WHERE symbol = ? AND orb_minutes = ?
                   ORDER BY trading_day""",
                [symbol, orb_minutes],
            ).fetchall()

            print(f"  {symbol} orb_minutes={orb_minutes}: {len(rows)} rows")

            if not rows:
                continue

            # Compute True Range and ATR(20) for each row
            updates = []  # [(atr_20, trading_day, symbol, orb_minutes), ...]
            true_ranges = []

            for i, (td, high, low, close) in enumerate(rows):
                prev_close = rows[i - 1][3] if i > 0 else None  # daily_close of prev row

                if high is not None and low is not None:
                    hl = high - low
                    if prev_close is not None:
                        tr = max(hl, abs(high - prev_close), abs(low - prev_close))
                    else:
                        tr = hl
                    true_ranges.append(tr)
                else:
                    true_ranges.append(None)

                # ATR(20) = SMA of last 20 TRs
                lookback = [v for v in true_ranges[max(0, i - 20):i] if v is not None]
                if lookback:
                    atr_val = round(sum(lookback) / len(lookback), 4)
                else:
                    atr_val = None

                updates.append((atr_val, td, symbol, orb_minutes))

            if not dry_run:
                con.executemany(
                    """UPDATE daily_features SET atr_20 = ?
                       WHERE trading_day = ? AND symbol = ? AND orb_minutes = ?""",
                    updates,
                )
                con.commit()

            total_updated += len(updates)

        # Step 3: Verify
        null_count = con.execute(
            "SELECT COUNT(*) FROM daily_features WHERE atr_20 IS NULL"
        ).fetchone()[0]
        total_count = con.execute(
            "SELECT COUNT(*) FROM daily_features"
        ).fetchone()[0]

        print(f"\nBackfill complete: {total_updated} rows updated")
        print(f"  Total rows: {total_count}")
        print(f"  NULL atr_20: {null_count} (expected 0 unless daily OHLC is NULL)")

        if not dry_run:
            # Spot check: show ATR(20) stats
            stats = con.execute(
                """SELECT symbol,
                          MIN(atr_20) as min_atr,
                          ROUND(AVG(atr_20), 2) as avg_atr,
                          ROUND(MEDIAN(atr_20), 2) as med_atr,
                          MAX(atr_20) as max_atr,
                          COUNT(*) as n
                   FROM daily_features
                   WHERE atr_20 IS NOT NULL
                   GROUP BY symbol"""
            ).fetchall()
            print("\nATR(20) summary:")
            print(f"  {'Symbol':<8} {'Min':>8} {'Avg':>8} {'Median':>8} {'Max':>8} {'N':>6}")
            for sym, mn, avg, med, mx, n in stats:
                print(f"  {sym:<8} {mn:>8.2f} {avg:>8.2f} {med:>8.2f} {mx:>8.2f} {n:>6}")

        if dry_run:
            print("\n(DRY RUN -- no data written)")

        return total_updated

    finally:
        con.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill ATR(20) in daily_features")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    print(f"Database: {db_path}")
    backfill_atr20(db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
