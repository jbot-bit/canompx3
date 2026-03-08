#!/usr/bin/env python3
"""Refresh market data: download from Databento + ingest + build features.

One command to keep data fresh. Downloads missing bars from Databento API,
ingests into gold.db, then builds 5m bars and daily features.

Does NOT trigger the slow rebuild chain (outcome_builder/discovery/validator).

Usage:
    python scripts/tools/refresh_data.py                    # All active instruments
    python scripts/tools/refresh_data.py --instrument MGC   # Single instrument
    python scripts/tools/refresh_data.py --dry-run           # Show what would download
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# pipeline.paths import triggers .env loading via python-dotenv
import duckdb  # noqa: E402

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402

# Map stored instrument -> Databento parent symbol for download.
# Derived from outright_pattern prefix in asset_configs.
DOWNLOAD_SYMBOLS: dict[str, str] = {
    "MGC": "GC",  # Full-size gold -> stored as MGC
    "MNQ": "MNQ",  # Native micro Nasdaq (post-2024)
    "MES": "MES",  # Native micro S&P (post-2024)
    "M2K": "RTY",  # Full-size Russell -> stored as M2K
}

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
ORB_APERTURES = [5, 15, 30]


def get_last_bar_date(instrument: str) -> date | None:
    """Get the latest bar date for an instrument from gold.db."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        row = con.execute(
            "SELECT MAX(ts_utc)::DATE AS last_date FROM bars_1m WHERE symbol = $1",
            [instrument],
        ).fetchone()
        return row[0] if row and row[0] else None
    finally:
        con.close()


def _run(cmd: list[str], label: str) -> bool:
    """Run a subprocess with PROJECT_ROOT as cwd. Returns True on success."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"  FAIL: {label} returned {result.returncode}")
        for line in result.stderr.strip().splitlines()[-5:]:
            print(f"    {line}")
        return False
    return True


def download_dbn(instrument: str, start: date, end: date, dry_run: bool = False) -> Path | None:
    """Download DBN file from Databento for the given date range."""
    import databento as db

    parent_symbol = DOWNLOAD_SYMBOLS.get(instrument)
    if not parent_symbol:
        print(f"  SKIP: no download symbol mapping for {instrument}")
        return None

    config = ASSET_CONFIGS[instrument]
    dbn_path = config["dbn_path"]

    # Determine output directory
    out_dir = dbn_path if dbn_path.is_dir() else dbn_path.parent

    out_file = out_dir / f"backfill-{instrument}-{start.isoformat()}-to-{end.isoformat()}.ohlcv-1m.dbn.zst"

    if dry_run:
        print(f"  WOULD download: {parent_symbol}.FUT {start} to {end}")
        print(f"  WOULD save to:  {out_file}")
        return None

    print(f"  Downloading {parent_symbol}.FUT {start} to {end} ...")

    client = db.Historical()
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[f"{parent_symbol}.FUT"],
        schema=SCHEMA,
        stype_in="parent",
        start=str(start),
        end=str(end),
    )

    data.to_file(str(out_file))
    size_mb = out_file.stat().st_size / 1_048_576
    print(f"  Saved: {out_file.name} ({size_mb:.1f} MB)")
    return out_file


def run_ingest(instrument: str) -> bool:
    """Run ingest_dbn --resume for the instrument."""
    print(f"  Ingesting {instrument} ...")
    cmd = [sys.executable, "pipeline/ingest_dbn.py", "--instrument", instrument, "--resume"]
    if not _run(cmd, "ingest"):
        return False
    print("    Ingest complete")
    return True


def run_build_steps(instrument: str, start: date, end: date) -> bool:
    """Build 5m bars + daily features (all apertures) for the date range."""
    start_str, end_str = str(start), str(end)

    # Step 1: Build 5m bars
    print(f"  Building 5m bars {start} to {end} ...")
    cmd = [
        sys.executable,
        "pipeline/build_bars_5m.py",
        "--instrument",
        instrument,
        "--start",
        start_str,
        "--end",
        end_str,
    ]
    if not _run(cmd, "build_bars_5m"):
        return False

    # Step 2: Build daily features for all apertures
    for orb_min in ORB_APERTURES:
        print(f"  Building daily features O{orb_min} {start} to {end} ...")
        cmd = [
            sys.executable,
            "pipeline/build_daily_features.py",
            "--instrument",
            instrument,
            "--start",
            start_str,
            "--end",
            end_str,
            "--orb-minutes",
            str(orb_min),
        ]
        if not _run(cmd, f"daily_features O{orb_min}"):
            return False

    print(f"    Pipeline complete for {instrument}")
    return True


def refresh_instrument(instrument: str, dry_run: bool = False) -> bool:
    """Full refresh for one instrument: download -> ingest -> build."""
    print(f"\n{'=' * 60}")
    print(f"  {instrument}")
    print(f"{'=' * 60}")

    last_date = get_last_bar_date(instrument)
    if last_date is None:
        print(f"  No bars found for {instrument} -- skip (needs manual initial load)")
        return False

    today = date.today()
    fetch_start = last_date + timedelta(days=1)

    gap_days = (today - last_date).days
    if gap_days <= 1:
        print(f"  Already up to date (last bar: {last_date})")
        return True

    print(f"  Last bar: {last_date} ({gap_days} days ago)")
    print(f"  Gap: {fetch_start} to {today}")

    # Step 1: Download
    out_file = download_dbn(instrument, fetch_start, today, dry_run=dry_run)
    if dry_run:
        return True

    if out_file is None:
        print("  FAIL: download returned nothing")
        return False

    # Step 2: Ingest
    if not run_ingest(instrument):
        return False

    # Step 3: Build 5m bars + daily features (all apertures)
    if not run_build_steps(instrument, fetch_start, today):
        return False

    print(f"  DONE: {instrument} refreshed to {today}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Refresh market data from Databento")
    parser.add_argument("--instrument", type=str, help="Single instrument (default: all active)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would download")
    args = parser.parse_args()

    # Fail-closed: require API key before attempting downloads
    if not args.dry_run and "DATABENTO_API_KEY" not in os.environ:
        print("FATAL: DATABENTO_API_KEY not found in environment or .env file")
        sys.exit(1)

    instruments = [args.instrument.upper()] if args.instrument else list(ACTIVE_ORB_INSTRUMENTS)

    for inst in instruments:
        if inst not in DOWNLOAD_SYMBOLS:
            print(f"WARNING: {inst} has no Databento download mapping -- will skip")

    print("=" * 60)
    print("  DATA REFRESH" + (" (DRY RUN)" if args.dry_run else ""))
    print("=" * 60)

    results = {}
    for inst in instruments:
        ok = refresh_instrument(inst, dry_run=args.dry_run)
        results[inst] = "OK" if ok else "FAIL"

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for inst, status in results.items():
        print(f"  {inst:<6} {status}")
    print(f"{'=' * 60}")

    if any(v == "FAIL" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
