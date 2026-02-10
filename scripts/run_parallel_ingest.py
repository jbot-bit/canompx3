"""
Parallel re-ingest: GC bars into gold.db.

Each year ingests into its own temp DB, then all are merged into gold.db.
Uses INSERT OR REPLACE — no gaps, no duplicates.

IMPORTANT: Year ranges overlap by 2 days to handle Brisbane trading day
boundaries (bars from Dec 31 file can belong to Jan 1 trading day).
The merge uses INSERT OR REPLACE so overlapping rows are deduplicated.

Usage:
    python scripts/run_parallel_ingest.py
"""

import subprocess
import sys
import time
import duckdb
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DB = PROJECT_ROOT / "gold.db"
PYTHON = sys.executable

# Overlap by 2 days at boundaries to handle Brisbane trading day crossover
YEAR_RANGES = [
    ("2021-01-01", "2022-01-02"),
    ("2021-12-30", "2023-01-02"),
    ("2022-12-30", "2024-01-02"),
    ("2023-12-30", "2025-01-02"),
    ("2024-12-30", "2026-12-31"),
]


def ingest_year(start: str, end: str) -> str:
    """Ingest one date range into a temp DB. Runs in subprocess."""
    year_label = start[:4]
    temp_db = PROJECT_ROOT / f"temp_{year_label}.db"
    temp_db.unlink(missing_ok=True)

    # Create schema in temp DB
    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline.init_db import init_db
    init_db(temp_db, force=True)

    # Run ingestion into temp DB
    result = subprocess.run(
        [
            PYTHON, "pipeline/ingest_dbn_daily.py",
            "--start", start,
            "--end", end,
            "--db", str(temp_db),
        ],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        return f"FAILED {year_label}: {result.stderr[-500:]}"

    # Count rows
    con = duckdb.connect(str(temp_db), read_only=True)
    count = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
    con.close()

    return f"OK {year_label}: {count:,} rows"


def merge_all():
    """Merge all temp DBs into gold.db using ATTACH."""
    print("\n=== MERGING ===")

    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline.init_db import init_db

    # Wipe everything — pipeline tables AND trading_app tables
    con = duckdb.connect(str(GOLD_DB))
    for table in ["validated_setups_archive", "validated_setups",
                   "experimental_strategies", "orb_outcomes"]:
        con.execute(f"DROP TABLE IF EXISTS {table}")
    con.close()

    # Recreate pipeline schema
    init_db(GOLD_DB, force=True)

    con = duckdb.connect(str(GOLD_DB))
    total = 0

    for start, _ in YEAR_RANGES:
        year_label = start[:4]
        temp_db = PROJECT_ROOT / f"temp_{year_label}.db"
        if not temp_db.exists():
            print(f"  SKIP {year_label}")
            continue

        alias = f"y{year_label}"
        con.execute(f"ATTACH '{temp_db}' AS {alias} (READ_ONLY)")

        # INSERT OR REPLACE handles overlap dedup
        con.execute(f"INSERT OR REPLACE INTO bars_1m SELECT * FROM {alias}.bars_1m")
        count = con.execute(f"SELECT COUNT(*) FROM {alias}.bars_1m").fetchone()[0]
        con.execute(f"DETACH {alias}")
        total += count
        print(f"  {year_label}: {count:,} rows merged")

    con.commit()

    # Verify
    actual = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
    sources = con.execute(
        "SELECT DISTINCT LEFT(source_symbol, 2) as prefix FROM bars_1m ORDER BY prefix"
    ).fetchall()
    nulls = con.execute(
        "SELECT COUNT(*) FROM bars_1m WHERE source_symbol IS NULL"
    ).fetchone()[0]
    print(f"\n  Total bars_1m:   {actual:,} (from {total:,} incl overlaps)")
    print(f"  Source prefixes: {[r[0] for r in sources]}")
    print(f"  NULL sources:    {nulls}")

    if nulls > 0:
        print("  WARNING: NULL source_symbols found!")
    prefixes = [r[0] for r in sources]
    if prefixes != ["GC"]:
        print(f"  WARNING: Expected ['GC'], got {prefixes}")

    con.close()


def build_downstream():
    """Build bars_5m and daily_features."""
    print("\n=== BUILDING bars_5m ===")
    r = subprocess.run(
        [PYTHON, "pipeline/build_bars_5m.py",
         "--instrument", "MGC", "--start", "2021-01-01", "--end", "2026-12-31"],
        cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        print("FATAL: bars_5m build failed")
        sys.exit(1)

    print("\n=== BUILDING daily_features ===")
    r = subprocess.run(
        [PYTHON, "pipeline/build_daily_features.py",
         "--instrument", "MGC", "--start", "2021-01-01", "--end", "2026-12-31"],
        cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        print("FATAL: daily_features build failed")
        sys.exit(1)


def verify():
    """Final integrity checks."""
    print("\n=== FINAL VERIFICATION ===")
    con = duckdb.connect(str(GOLD_DB), read_only=True)

    bars = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
    bars5 = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
    feats = con.execute("SELECT COUNT(*) FROM daily_features").fetchone()[0]
    nulls = con.execute(
        "SELECT COUNT(*) FROM bars_1m WHERE source_symbol IS NULL"
    ).fetchone()[0]

    # Check year coverage
    years = con.execute("""
        SELECT EXTRACT(YEAR FROM ts_utc) as yr, COUNT(*) as cnt
        FROM bars_1m WHERE symbol='MGC'
        GROUP BY yr ORDER BY yr
    """).fetchall()

    print(f"  bars_1m:        {bars:,}")
    print(f"  bars_5m:        {bars5:,}")
    print(f"  daily_features: {feats:,}")
    print(f"  NULL sources:   {nulls}")
    print(f"  Year coverage:")
    for yr, cnt in years:
        print(f"    {int(yr)}: {cnt:,} bars")

    con.close()

    # Sanity checks
    ok = True
    if bars < 1_800_000:
        print(f"  WARN: Expected >1.8M bars (GC), got {bars:,}")
        ok = False
    if nulls > 0:
        print(f"  FAIL: {nulls} NULL source_symbols")
        ok = False
    if feats < 1400:
        print(f"  WARN: Expected ~1460 daily_features rows, got {feats}")
        ok = False

    if ok:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED — review above")


def cleanup():
    """Remove temp DBs."""
    for start, _ in YEAR_RANGES:
        year = start[:4]
        for suffix in ["", ".wal", ".tmp"]:
            p = PROJECT_ROOT / f"temp_{year}.db{suffix}"
            p.unlink(missing_ok=True)


def main():
    t0 = time.time()
    print("=== PARALLEL GC RE-INGEST ===")
    print(f"  {len(YEAR_RANGES)} parallel workers (overlapping boundaries)")
    print(f"  Temp DBs per year, then merge into gold.db")
    print(f"  INSERT OR REPLACE deduplicates overlap\n")

    # Parallel ingest
    with ProcessPoolExecutor(max_workers=len(YEAR_RANGES)) as pool:
        futures = {
            pool.submit(ingest_year, start, end): start[:4]
            for start, end in YEAR_RANGES
        }
        for f in as_completed(futures):
            year = futures[f]
            try:
                print(f"  {f.result()}")
            except Exception as e:
                print(f"  FAILED {year}: {e}")

    # Merge
    merge_all()

    # Build downstream
    build_downstream()

    # Verify
    verify()

    # Cleanup
    cleanup()

    elapsed = time.time() - t0
    print(f"\n=== DONE in {elapsed/60:.1f} minutes ===")


if __name__ == "__main__":
    main()
