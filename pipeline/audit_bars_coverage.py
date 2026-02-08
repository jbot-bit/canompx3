#!/usr/bin/env python3
"""
Independent bars coverage audit.

Samples ~60 trading days across 4 tiers (boundary, roll, anomaly, random),
reads the raw .dbn.zst files, reproduces the ingestion filtering logic, and
compares bar counts against gold.db to detect silent drops or duplicates.

Usage:
    python pipeline/audit_bars_coverage.py                    # Default ~60 days
    python pipeline/audit_bars_coverage.py --random-count 50  # More random samples
    python pipeline/audit_bars_coverage.py --seed 42          # Reproducible
"""

import argparse
import random
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Reuse canonical pipeline logic — no copies, single source of truth
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.paths import GOLD_DB_PATH, DAILY_DBN_DIR, PROJECT_ROOT
from pipeline.ingest_dbn_mgc import (
    GC_OUTRIGHT_PATTERN,
    choose_front_contract,
    compute_trading_days,
)
from pipeline.ingest_dbn_daily import DAILY_FILE_PATTERN, discover_daily_files

try:
    import databento as db
except ImportError:
    print("FATAL: databento package not installed (pip install databento)")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BRISBANE_TZ = ZoneInfo("Australia/Brisbane")
UTC_TZ = ZoneInfo("UTC")
TRADING_DAY_START_HOUR_LOCAL = 9

# Both data directories
DATA_DIRS = [
    DAILY_DBN_DIR,                                        # 2021-2026
    PROJECT_ROOT / "DB" / "gold_db_fullsize_2016-2021",   # 2016-2021
]


# ---------------------------------------------------------------------------
# Trading day helpers (match build_daily_features logic exactly)
# ---------------------------------------------------------------------------

def _trading_day_utc_range(trading_day: date) -> tuple[datetime, datetime]:
    """Return [start, end) UTC range for a trading day (09:00 Bris boundary)."""
    start_utc = datetime(
        trading_day.year, trading_day.month, trading_day.day,
        TRADING_DAY_START_HOUR_LOCAL, 0, 0,
        tzinfo=BRISBANE_TZ,
    ).astimezone(UTC_TZ)
    end_utc = start_utc + timedelta(hours=24)
    return start_utc, end_utc


def _calendar_dates_for_trading_day(trading_day: date) -> list[date]:
    """
    A trading day spans 09:00 Bris on `trading_day` to 09:00 Bris next day.
    In UTC that is 23:00 (trading_day - 1) to 23:00 (trading_day).
    So raw files for calendar dates (trading_day - 1) and (trading_day) may
    contain bars that belong to this trading day.
    """
    return [trading_day - timedelta(days=1), trading_day]


# ---------------------------------------------------------------------------
# File discovery across both data directories
# ---------------------------------------------------------------------------

def _build_file_index() -> dict[date, Path]:
    """Build calendar_date -> file_path index across all data dirs."""
    index = {}
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue
        for fpath in data_dir.iterdir():
            match = DAILY_FILE_PATTERN.match(fpath.name)
            if not match:
                continue
            ds = match.group(1)
            file_date = date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
            index[file_date] = fpath
    return index


# ---------------------------------------------------------------------------
# Sampling tiers
# ---------------------------------------------------------------------------

def _sample_boundary(con, n: int = 10) -> list[tuple[date, str]]:
    """First and last n/2 trading days in gold.db."""
    half = n // 2
    rows = con.execute(
        "SELECT DISTINCT trading_day FROM daily_features "
        "WHERE symbol = 'MGC' ORDER BY trading_day"
    ).fetchall()
    days = [r[0] for r in rows]
    if len(days) <= n:
        return [(d, "boundary") for d in days]
    selected = days[:half] + days[-half:]
    return [(d, "boundary") for d in selected]


def _sample_roll(con, n: int = 10) -> list[tuple[date, str]]:
    """Days where source_symbol changes (futures roll dates)."""
    rows = con.execute("""
        WITH contracts AS (
            SELECT DISTINCT
                CAST((ts_utc AT TIME ZONE 'Australia/Brisbane'
                      - INTERVAL '9 hours') AS DATE) AS trading_day,
                source_symbol
            FROM bars_1m
            WHERE symbol = 'MGC'
        ),
        with_prev AS (
            SELECT trading_day, source_symbol,
                   LAG(source_symbol) OVER (ORDER BY trading_day) AS prev_symbol
            FROM contracts
        )
        SELECT trading_day FROM with_prev
        WHERE source_symbol != prev_symbol AND prev_symbol IS NOT NULL
        ORDER BY trading_day
    """).fetchall()
    roll_days = [r[0] for r in rows]
    if len(roll_days) <= n:
        return [(d, "roll") for d in roll_days]
    # Evenly space the selection
    step = len(roll_days) / n
    selected = [roll_days[int(i * step)] for i in range(n)]
    return [(d, "roll") for d in selected]


def _sample_anomaly(con, n: int = 10) -> list[tuple[date, str]]:
    """Days with lowest bar_count_1m (potential missing-data days)."""
    rows = con.execute(
        "SELECT trading_day, bar_count_1m FROM daily_features "
        "WHERE symbol = 'MGC' AND orb_minutes = 5 AND bar_count_1m IS NOT NULL "
        "ORDER BY bar_count_1m ASC LIMIT ?",
        [n],
    ).fetchall()
    return [(r[0], "anomaly") for r in rows]


def _sample_random(con, already: set[date], n: int = 30,
                   seed: int | None = None) -> list[tuple[date, str]]:
    """Uniform random from remaining trading days."""
    rows = con.execute(
        "SELECT DISTINCT trading_day FROM daily_features "
        "WHERE symbol = 'MGC' ORDER BY trading_day"
    ).fetchall()
    pool = [r[0] for r in rows if r[0] not in already]
    if not pool:
        return []
    rng = random.Random(seed)
    count = min(n, len(pool))
    selected = rng.sample(pool, count)
    return [(d, "random") for d in sorted(selected)]


def build_sample(con, random_count: int = 30,
                 seed: int | None = None) -> list[tuple[date, str]]:
    """Build the combined sample across all 4 tiers (deduplicated)."""
    seen: set[date] = set()
    sample: list[tuple[date, str]] = []

    for tier_fn, tier_args in [
        (_sample_boundary, (con,)),
        (_sample_roll, (con,)),
        (_sample_anomaly, (con,)),
    ]:
        for day, tier in tier_fn(*tier_args):
            if day not in seen:
                seen.add(day)
                sample.append((day, tier))

    for day, tier in _sample_random(con, seen, n=random_count, seed=seed):
        if day not in seen:
            seen.add(day)
            sample.append((day, tier))

    sample.sort(key=lambda x: x[0])
    return sample


# ---------------------------------------------------------------------------
# Per-day audit: read raw files, reproduce pipeline logic, count bars
# ---------------------------------------------------------------------------

def audit_day(trading_day: date, file_index: dict[date, Path]) -> dict:
    """
    Audit a single trading day.

    Returns dict with keys: trading_day, raw_count, db_count, status, detail
    """
    result = {"trading_day": trading_day, "raw_count": None}

    start_utc, end_utc = _trading_day_utc_range(trading_day)
    calendar_dates = _calendar_dates_for_trading_day(trading_day)

    # Collect raw bars from both calendar-date files
    all_dfs = []
    files_found = 0
    for cal_date in calendar_dates:
        fpath = file_index.get(cal_date)
        if fpath is None:
            continue
        files_found += 1
        try:
            store = db.DBNStore.from_file(fpath)
            df = store.to_df().reset_index()
        except Exception as e:
            result["status"] = "ERROR"
            result["detail"] = f"Failed reading {fpath.name}: {e}"
            return result

        if len(df) == 0:
            continue

        # Filter to GC outrights (same regex as pipeline)
        mask = df["symbol"].astype(str).str.match(r"^GC[FGHJKMNQUVXZ]\d{1,2}$")
        df = df[mask]
        if len(df) == 0:
            continue

        # Keep only bars within this trading day's UTC window
        ts = df["ts_event"]
        in_window = (ts >= start_utc) & (ts < end_utc)
        df = df[in_window]
        if len(df) > 0:
            all_dfs.append(df)

    if files_found == 0:
        result["status"] = "SKIP"
        result["detail"] = "No raw files found"
        return result

    if not all_dfs:
        result["raw_count"] = 0
        return result

    combined = pd.concat(all_dfs, ignore_index=True)

    # Choose front contract (same logic as pipeline)
    volumes = combined.groupby("symbol")["volume"].sum().to_dict()
    front = choose_front_contract(
        volumes, outright_pattern=GC_OUTRIGHT_PATTERN, prefix_len=2
    )
    if front is None:
        result["raw_count"] = 0
        result["detail"] = "No front contract selected"
        return result

    front_bars = combined[combined["symbol"] == front]

    # Deduplicate by ts_event (same as INSERT OR REPLACE on PK)
    front_bars = front_bars.drop_duplicates(subset=["ts_event"], keep="last")

    result["raw_count"] = len(front_bars)
    result["front_contract"] = front
    return result


def query_db_count(con, trading_day: date) -> int:
    """Count bars_1m rows for a trading day via the same UTC-range logic."""
    start_utc, end_utc = _trading_day_utc_range(trading_day)
    row = con.execute(
        "SELECT COUNT(*) FROM bars_1m "
        "WHERE symbol = 'MGC' AND ts_utc >= ? AND ts_utc < ?",
        [start_utc, end_utc],
    ).fetchone()
    return row[0] if row else 0


def query_db_source_symbols(con, trading_day: date) -> list[str]:
    """Return distinct source_symbols for a trading day (detects roll days)."""
    start_utc, end_utc = _trading_day_utc_range(trading_day)
    rows = con.execute(
        "SELECT DISTINCT source_symbol FROM bars_1m "
        "WHERE symbol = 'MGC' AND ts_utc >= ? AND ts_utc < ?",
        [start_utc, end_utc],
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audit bars_1m coverage: raw .dbn.zst vs gold.db"
    )
    parser.add_argument(
        "--random-count", type=int, default=30,
        help="Number of random-sample days (default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Database path (default: gold.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else GOLD_DB_PATH
    if not db_path.exists():
        print(f"FATAL: Database not found: {db_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build file index
    # ------------------------------------------------------------------
    print("Building raw file index...")
    file_index = _build_file_index()
    print(f"  Indexed {len(file_index)} raw .dbn.zst files across {len(DATA_DIRS)} directories")
    print()

    # ------------------------------------------------------------------
    # Build sample
    # ------------------------------------------------------------------
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        sample = build_sample(con, random_count=args.random_count, seed=args.seed)
    finally:
        con.close()

    tier_counts = defaultdict(int)
    for _, tier in sample:
        tier_counts[tier] += 1
    tier_str = " + ".join(f"{v} {k}" for k, v in sorted(tier_counts.items()))

    print("=== Bars Coverage Audit ===")
    print(f"Sampling {len(sample)} days ({tier_str})")
    print()

    # ------------------------------------------------------------------
    # Audit each day
    # ------------------------------------------------------------------
    con = duckdb.connect(str(db_path), read_only=True)
    passes = 0
    fails = 0
    warns = 0
    skips = 0
    errors = 0
    fail_details = []
    warn_details = []

    try:
        for trading_day, tier in sample:
            result = audit_day(trading_day, file_index)

            if result.get("status") == "SKIP":
                skips += 1
                print(f"[SKIP] {trading_day}: {result.get('detail', '')} ({tier})")
                continue

            if result.get("status") == "ERROR":
                errors += 1
                print(f"[ERR ] {trading_day}: {result.get('detail', '')} ({tier})")
                continue

            db_count = query_db_count(con, trading_day)
            raw_count = result["raw_count"]
            front = result.get("front_contract", "?")

            if raw_count == db_count:
                passes += 1
                print(f"[PASS] {trading_day}: raw={raw_count}, db={db_count} ({tier}) [{front}]")
            else:
                delta = db_count - raw_count
                abs_delta = abs(delta)
                sign = "+" if delta > 0 else ""

                # Check if this is a roll day (multiple source_symbols)
                # Roll days often have small deltas due to per-file contract selection
                source_symbols = query_db_source_symbols(con, trading_day)
                is_roll_day = len(source_symbols) > 1
                pct_delta = abs_delta / db_count * 100 if db_count > 0 else 0

                # Classify: small delta on roll day = WARN, large delta = FAIL
                if is_roll_day and abs_delta <= 10:
                    warns += 1
                    line = (
                        f"[WARN] {trading_day}: raw={raw_count}, db={db_count} "
                        f"({tier}) DELTA={sign}{delta} ({pct_delta:.1f}%) [roll: {', '.join(source_symbols)}]"
                    )
                    print(line)
                    warn_details.append(line)
                else:
                    fails += 1
                    line = (
                        f"[FAIL] {trading_day}: raw={raw_count}, db={db_count} "
                        f"({tier}) DELTA={sign}{delta} ({pct_delta:.1f}%) [{front}]"
                    )
                    print(line)
                    fail_details.append(line)
    finally:
        con.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    total_audited = passes + fails + warns
    print(f"Results: {passes}/{total_audited} PASS", end="")
    if warns:
        print(f", {warns} WARN (roll-day expected)", end="")
    if fails:
        print(f", {fails} FAIL", end="")
    if skips:
        print(f", {skips} SKIP", end="")
    if errors:
        print(f", {errors} ERROR", end="")
    print()

    if warn_details:
        print()
        print("Warnings (roll-day expected behavior):")
        for line in warn_details:
            print(f"  {line}")

    if fail_details:
        print()
        print("Failed days (require investigation):")
        for line in fail_details:
            print(f"  {line}")

    # Exit code: 0 if no hard failures or errors, 1 otherwise
    # (WARNs are expected and don't fail the audit)
    sys.exit(0 if fails == 0 and errors == 0 else 1)


if __name__ == "__main__":
    main()
