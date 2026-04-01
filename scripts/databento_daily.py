#!/usr/bin/env python3
"""Daily Databento data refresh -- downloads yesterday's data and ingests.

Designed to run as a daily cron job. Downloads incremental data for all active
instruments and schemas, validates, and triggers the pipeline build chain.

For ohlcv-1m: delegates to the existing refresh_data.py (download + ingest +
build 5m bars + daily features). For other schemas (ohlcv-1s, tbbo, trades,
bbo-1s, statistics): downloads to data/raw/databento/ with validation.

Usage:
    python scripts/databento_daily.py                  # Full daily refresh
    python scripts/databento_daily.py --dry-run        # Show what would download
    python scripts/databento_daily.py --schema ohlcv-1s  # Only refresh 1s bars
    python scripts/databento_daily.py --days 3         # Catch up last 3 days

Cron (Brisbane UTC+10, run at 9:30 AM local = 23:30 UTC):
    30 23 * * 0-4 cd /path/to/canompx3 && python scripts/databento_daily.py >> logs/databento_daily.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "databento_daily.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("databento_daily")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET = "GLBX.MDP3"

# Databento API symbol mapping (vendor-specific format, not in ASSET_CONFIGS).
# Keys must be a superset of ACTIVE_ORB_INSTRUMENTS.
_DATABENTO_SYMBOLS: dict[str, str] = {
    "MGC": "GC.FUT",  # Full-size gold -> micro gold pipeline
    "MNQ": "MNQ.FUT",  # Native micro Nasdaq
    "MES": "MES.FUT",  # Native micro S&P (post-2024; pre-2024 used ES)
}

# Fail-closed guard: crash if active instruments not covered
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS  # noqa: E402

_missing = set(ACTIVE_ORB_INSTRUMENTS) - set(_DATABENTO_SYMBOLS)
if _missing:
    raise RuntimeError(
        f"DATABENTO_SYMBOLS missing active instruments: {_missing}. Update _DATABENTO_SYMBOLS in {__file__}."
    )

# Additional schemas to refresh daily (beyond ohlcv-1m which refresh_data.py handles).
# Standard plan ($179/mo) includes: L0 (ohlcv, stats) unlimited, L1 (tbbo, trades,
# bbo-1s) 12mo, L2/L3 (mbp-1, mbp-10) 1mo. Refreshes L0 + L1 + mbp-1 daily.
# NOTE: mbp-10 (10-level book depth) is NOT refreshed daily — large file, backfill only.
# See config/databento_config.yaml TIER 2 for one-time mbp-10 backfill specs.
DAILY_SCHEMAS = [
    {
        "schema": "ohlcv-1s",
        "instruments": {k: _DATABENTO_SYMBOLS[k] for k in ACTIVE_ORB_INSTRUMENTS},
        "description": "1-second OHLCV bars (L0 FREE)",
    },
    {
        "schema": "statistics",
        "instruments": {k: _DATABENTO_SYMBOLS[k] for k in ACTIVE_ORB_INSTRUMENTS},
        "description": "Settlement prices, OI, volume stats (L0 FREE)",
    },
    {
        "schema": "tbbo",
        "instruments": {k: _DATABENTO_SYMBOLS[k] for k in ACTIVE_ORB_INSTRUMENTS},
        "description": "Trade+quote for slippage monitoring (L1 INCLUDED)",
    },
    {
        "schema": "trades",
        "instruments": {k: _DATABENTO_SYMBOLS[k] for k in ACTIVE_ORB_INSTRUMENTS},
        "description": "Tick-by-tick trades for order flow (L1 INCLUDED)",
    },
    {
        "schema": "bbo-1s",
        "instruments": {k: _DATABENTO_SYMBOLS[k] for k in ACTIVE_ORB_INSTRUMENTS},
        "description": "1-second bid/ask for spread tracking (L1 INCLUDED)",
    },
    {
        "schema": "mbp-1",
        "instruments": {k: _DATABENTO_SYMBOLS[k] for k in ACTIVE_ORB_INSTRUMENTS},
        "description": "Top-of-book for break quality research (L2 INCLUDED 1mo)",
    },
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def refresh_ohlcv_1m(dry_run: bool = False) -> dict[str, str]:
    """Refresh ohlcv-1m via existing refresh_data.py.

    Note: refresh_data.py computes its own date range from the last bar in
    gold.db. The --days flag on this script only affects non-1m schemas.
    """
    log.info("--- ohlcv-1m refresh (via refresh_data.py) ---")

    cmd = [sys.executable, "scripts/tools/refresh_data.py"]
    if dry_run:
        cmd.append("--dry-run")

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env)

    for line in result.stdout.strip().splitlines():
        log.info(f"  {line}")
    if result.returncode != 0:
        for line in result.stderr.strip().splitlines()[-10:]:
            log.error(f"  {line}")
        return {"ohlcv-1m": "FAIL"}

    return {"ohlcv-1m": "OK"}


def refresh_schema(
    schema_spec: dict,
    fetch_start: date,
    fetch_end: date,
    dry_run: bool = False,
) -> dict[str, str]:
    """Download incremental data for a non-1m schema."""
    import databento as db

    schema = schema_spec["schema"]
    results = {}

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        log.error("DATABENTO_API_KEY not set")
        return {schema: "FAIL (no API key)"}

    client = db.Historical(api_key)
    output_base = PROJECT_ROOT / "data" / "raw" / "databento"

    for stored_as, db_symbol in schema_spec["instruments"].items():
        name = f"{stored_as}_{schema}"
        output_dir = output_base / schema / stored_as
        output_dir.mkdir(parents=True, exist_ok=True)

        out_file = output_dir / f"daily_{stored_as}_{fetch_start}_to_{fetch_end}.{schema}.dbn.zst"

        if out_file.exists() and out_file.stat().st_size > 0:
            log.info(f"  {name}: SKIP (already exists)")
            results[name] = "SKIP"
            continue

        if dry_run:
            try:
                cost = client.metadata.get_cost(
                    dataset=DATASET,
                    symbols=[db_symbol],
                    schema=schema,
                    stype_in="parent",
                    start=str(fetch_start),
                    end=str(fetch_end),
                )
                cost_str = "FREE" if cost == 0 else f"${cost:.2f}"
                log.info(f"  {name}: WOULD download {fetch_start} -> {fetch_end} ({cost_str})")
            except Exception as e:
                log.info(f"  {name}: WOULD download {fetch_start} -> {fetch_end} (cost unknown: {e})")
            results[name] = "DRY_RUN"
            continue

        log.info(f"  {name}: downloading {fetch_start} -> {fetch_end}")

        try:
            data = client.timeseries.get_range(
                dataset=DATASET,
                symbols=[db_symbol],
                schema=schema,
                stype_in="parent",
                start=str(fetch_start),
                end=str(fetch_end),
            )
            data.to_file(str(out_file))
            size_mb = out_file.stat().st_size / (1024 * 1024)

            # Quick validation
            store = db.DBNStore.from_file(str(out_file))
            df = store.to_df()
            row_count = len(df)

            log.info(f"  {name}: OK ({size_mb:.1f} MB, {row_count:,} records)")
            results[name] = f"OK ({row_count:,} records)"

        except Exception as e:
            log.error(f"  {name}: FAIL -- {e}")
            results[name] = f"FAIL ({e})"

    return results


def run_daily_refresh(
    dry_run: bool = False,
    days: int = 1,
    schema_filter: str | None = None,
):
    """Full daily refresh pipeline."""
    start_time = time.time()
    today = date.today()
    fetch_start = today - timedelta(days=days)
    fetch_end = today

    log.info("=" * 60)
    log.info(f"  DAILY DATABENTO REFRESH -- {today}")
    log.info(f"  Fetch window: {fetch_start} -> {fetch_end} ({days} day(s))")
    if dry_run:
        log.info("  MODE: DRY RUN")
    log.info("=" * 60)

    all_results = {}

    # Step 1: ohlcv-1m (via refresh_data.py)
    if schema_filter is None or schema_filter == "ohlcv-1m":
        results = refresh_ohlcv_1m(dry_run=dry_run)
        all_results.update(results)

    # Step 2: Additional schemas
    for spec in DAILY_SCHEMAS:
        if schema_filter and schema_filter != spec["schema"]:
            continue
        log.info(f"\n--- {spec['schema']} refresh ({spec['description']}) ---")
        results = refresh_schema(spec, fetch_start, fetch_end, dry_run=dry_run)
        all_results.update(results)

    # Summary
    elapsed = time.time() - start_time
    log.info(f"\n{'=' * 60}")
    log.info(f"  DAILY REFRESH SUMMARY ({elapsed:.0f}s)")
    log.info(f"{'=' * 60}")
    for name, status in all_results.items():
        log.info(f"  {name:<35} {status}")

    fail_count = sum(1 for v in all_results.values() if "FAIL" in v)
    ok_count = sum(1 for v in all_results.values() if v.startswith("OK"))
    log.info(f"\n  OK: {ok_count}  FAIL: {fail_count}  TOTAL: {len(all_results)}")
    log.info(f"{'=' * 60}")

    if fail_count > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Daily Databento data refresh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cron setup (Brisbane UTC+10, weekdays at 9:30 AM local = 23:30 UTC prev day):
  30 23 * * 0-4 cd /path/to/canompx3 && python scripts/databento_daily.py >> logs/databento_daily.log 2>&1

Note: cron DOW 0-4 = Sun-Thu UTC = Mon-Fri Brisbane (crosses midnight)
        """,
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would download")
    parser.add_argument("--days", type=int, default=1, help="Days to look back (default: 1)")
    parser.add_argument("--schema", help="Only refresh a specific schema (e.g. ohlcv-1s, statistics)")

    args = parser.parse_args()

    # Require API key unless dry run
    if not args.dry_run and not os.getenv("DATABENTO_API_KEY"):
        log.error("FATAL: DATABENTO_API_KEY not found in environment or .env")
        sys.exit(1)

    run_daily_refresh(
        dry_run=args.dry_run,
        days=args.days,
        schema_filter=args.schema,
    )


if __name__ == "__main__":
    main()
