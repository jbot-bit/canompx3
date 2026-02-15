"""
Parallel re-ingest: GC bars into gold.db.

Each year ingests into its own temp DB, then all are merged into gold.db.
Uses INSERT OR REPLACE -- no gaps, no duplicates.

IMPORTANT: Year ranges overlap by 2 days to handle Brisbane trading day
boundaries (bars from Dec 31 file can belong to Jan 1 trading day).
The merge uses INSERT OR REPLACE so overlapping rows are deduplicated.

SAFETY: merge_bars_only() ONLY touches bars_1m. Trading tables
(validated_setups, experimental_strategies, etc.) are never dropped.
Use --force-rebuild for full schema reset (requires explicit confirmation).

Usage:
    python scripts/run_parallel_ingest.py --instrument MGC --start 2021-01-01 --end 2026-12-31
    python scripts/run_parallel_ingest.py --instrument MGC --start 2021-01-01 --end 2026-12-31 --db-path C:/db/gold.db
    python scripts/run_parallel_ingest.py --instrument MGC --force-rebuild
"""

import argparse
import os
import subprocess
import sys
import time
import duckdb
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

PYTHON = sys.executable

# Default temp directory for rebuild (NOT OneDrive-synced)
TEMP_DIR = Path(os.environ.get("REBUILD_DIR", "C:/db/rebuild"))

# Overlap by 2 days at boundaries to handle Brisbane trading day crossover
YEAR_RANGES = [
    ("2021-01-01", "2022-01-02"),
    ("2021-12-30", "2023-01-02"),
    ("2022-12-30", "2024-01-02"),
    ("2023-12-30", "2025-01-02"),
    ("2024-12-30", "2026-12-31"),
]


def ingest_year(idx: int, start: str, end: str) -> str:
    """Ingest one date range into a temp DB. Runs in subprocess."""
    year_label = f"{idx}_{start[:4]}_{end[:4]}"
    temp_db = TEMP_DIR / f"temp_{year_label}.db"
    temp_db.unlink(missing_ok=True)

    # Create schema in temp DB
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


def _find_temp_dbs() -> list[Path]:
    """Find temp DBs in TEMP_DIR matching indexed label pattern."""
    if not TEMP_DIR.exists():
        return []
    return sorted(p for p in TEMP_DIR.glob("temp_*.db")
                  if not p.name.endswith((".wal", ".tmp")))


def merge_bars_only(db_path: Path = None, temp_dbs: list[Path] = None):
    """Merge temp DBs into gold.db -- bars_1m ONLY. Never touches trading tables."""
    if db_path is None:
        db_path = GOLD_DB_PATH

    print("\n=== MERGING (bars_1m only) ===")

    con = duckdb.connect(str(db_path))
    total = 0

    if temp_dbs is None:
        temp_dbs = _find_temp_dbs()

    for temp_db in temp_dbs:
        if not temp_db.exists():
            print(f"  SKIP {temp_db.stem}")
            continue

        alias = f"y{temp_db.stem.replace('temp_', '')}"
        con.execute(f"ATTACH '{temp_db}' AS {alias} (READ_ONLY)")
        con.execute(f"INSERT OR REPLACE INTO bars_1m SELECT * FROM {alias}.bars_1m")
        count = con.execute(f"SELECT COUNT(*) FROM {alias}.bars_1m").fetchone()[0]
        con.execute(f"DETACH {alias}")
        total += count
        print(f"  {temp_db.stem}: {count:,} rows merged")

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

    con.close()


def build_downstream(instrument: str, start: str, end: str, db_path: Path):
    """Build bars_5m and daily_features."""
    env = {**os.environ, "DUCKDB_PATH": str(db_path)}

    print("\n=== BUILDING bars_5m ===")
    r = subprocess.run(
        [PYTHON, "pipeline/build_bars_5m.py",
         "--instrument", instrument, "--start", start, "--end", end],
        cwd=str(PROJECT_ROOT), env=env,
    )
    if r.returncode != 0:
        print("FATAL: bars_5m build failed")
        sys.exit(1)

    print("\n=== BUILDING daily_features ===")
    r = subprocess.run(
        [PYTHON, "pipeline/build_daily_features.py",
         "--instrument", instrument, "--start", start, "--end", end],
        cwd=str(PROJECT_ROOT), env=env,
    )
    if r.returncode != 0:
        print("FATAL: daily_features build failed")
        sys.exit(1)


def verify(instrument: str, db_path: Path):
    """Final integrity checks."""
    print("\n=== FINAL VERIFICATION ===")
    con = duckdb.connect(str(db_path), read_only=True)

    bars = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
    bars5 = con.execute("SELECT COUNT(*) FROM bars_5m").fetchone()[0]
    feats = con.execute("SELECT COUNT(*) FROM daily_features").fetchone()[0]
    nulls = con.execute(
        "SELECT COUNT(*) FROM bars_1m WHERE source_symbol IS NULL"
    ).fetchone()[0]

    # Check year coverage
    years = con.execute("""
        SELECT EXTRACT(YEAR FROM ts_utc) as yr, COUNT(*) as cnt
        FROM bars_1m WHERE symbol=?
        GROUP BY yr ORDER BY yr
    """, [instrument]).fetchall()

    print(f"  bars_1m:        {bars:,}")
    print(f"  bars_5m:        {bars5:,}")
    print(f"  daily_features: {feats:,}")
    print(f"  NULL sources:   {nulls}")
    print(f"  Year coverage ({instrument}):")
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
        print("\n  SOME CHECKS FAILED -- review above")


def cleanup():
    """Remove temp DBs from TEMP_DIR."""
    for p in TEMP_DIR.glob("temp_*.db*"):
        p.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Parallel GC re-ingest")
    parser.add_argument(
        "--instrument", type=str, default="MGC",
        help="Instrument symbol (default: MGC)",
    )
    parser.add_argument("--start", type=str, default="2021-01-01",
                        help="Start date YYYY-MM-DD (default: 2021-01-01)")
    parser.add_argument("--end", type=str, default="2026-12-31",
                        help="End date YYYY-MM-DD (default: 2026-12-31)")
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Database path (default: DUCKDB_PATH env var or paths.py)",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="DROP ALL tables including trading_app before ingest (DANGEROUS)",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH

    if args.force_rebuild:
        print("WARNING: --force-rebuild will DROP ALL tables including:")
        print("  validated_setups, experimental_strategies, orb_outcomes,")
        print("  strategy_trade_days, edge_families, validated_setups_archive")
        confirm = input("Type 'yes-destroy-everything' to confirm: ")
        if confirm != "yes-destroy-everything":
            print("Aborted.")
            sys.exit(1)
        from pipeline.init_db import init_db
        from trading_app.db_manager import init_trading_app_schema
        init_db(db_path, force=True)
        init_trading_app_schema(db_path=db_path, force=True)
        print("Schema reset complete.\n")

    # Ensure temp dir exists
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=== PARALLEL GC RE-INGEST ===")
    print(f"  Instrument:  {args.instrument}")
    print(f"  DB path:     {db_path}")
    print(f"  Temp dir:    {TEMP_DIR}")
    print(f"  {len(YEAR_RANGES)} parallel workers (overlapping boundaries)")
    print(f"  INSERT OR REPLACE deduplicates overlap\n")

    # Parallel ingest
    failures = []
    with ProcessPoolExecutor(max_workers=len(YEAR_RANGES)) as pool:
        futures = {
            pool.submit(ingest_year, idx, start, end): f"{idx}_{start[:4]}"
            for idx, (start, end) in enumerate(YEAR_RANGES)
        }
        for f in as_completed(futures):
            label = futures[f]
            try:
                result = f.result()
                print(f"  {result}")
                if result.startswith("FAILED"):
                    failures.append(label)
            except Exception as e:
                print(f"  FAILED {label}: {e}")
                failures.append(label)

    # Abort if any workers failed (2d fix)
    if failures:
        print(f"\nFATAL: {len(failures)} workers failed: {failures}")
        print("Aborting before merge. Fix failures and re-run.")
        cleanup()
        sys.exit(1)

    # Safe merge (bars_1m only)
    merge_bars_only(db_path=db_path)

    # Build downstream
    build_downstream(args.instrument, args.start, args.end, db_path)

    # Verify
    verify(args.instrument, db_path)

    # Cleanup
    cleanup()

    elapsed = time.time() - t0
    print(f"\n=== DONE in {elapsed/60:.1f} minutes ===")


if __name__ == "__main__":
    main()
