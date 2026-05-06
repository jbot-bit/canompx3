#!/usr/bin/env python3
"""On-demand Databento L2 (deep-book) snapshot puller.

Pulls one full month of `mbp-1` (top-of-book) and `mbp-10` (10-level book
depth) for the requested instruments. NOT scheduled — run manually when a
research question needs deep-book data (e.g., spread-regime audit, slippage-
pilot cross-check, displayed-vs-executed depth at NYSE_OPEN).

Why a separate script (not part of databento_daily.py):
    L2 schemas roll on a 1-month free window under the Databento Standard
    plan. A single representative monthly sample per instrument is
    statistically sufficient for distributional/structural questions
    (spread distribution, depth profile, queue economics). Daily refresh
    would churn ~8-12 GB/month for no incremental research value.

Cost-guard:
    Every (schema, instrument) combo is verified FREE via
    client.metadata.get_cost BEFORE any billable HTTP call. Non-zero cost
    aborts that combo with a WARN. Zero additional paid spending — by code,
    not by trust.

Usage:
    # Default: previous calendar month, all active ORB instruments.
    python scripts/databento_l2_snapshot.py

    # Specific instruments
    python scripts/databento_l2_snapshot.py --instruments MGC,MES

    # Specific month (must lie within Databento's rolling 1-month free window)
    python scripts/databento_l2_snapshot.py --instruments MGC --month 2026-04

    # Dry-run (cost-probe only, no download)
    python scripts/databento_l2_snapshot.py --dry-run --instruments MES,MGC,MNQ
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from calendar import monthrange
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# Logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "databento_l2_snapshot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("databento_l2_snapshot")

# Canonical sources only — never re-encode
from pipeline.asset_configs import (  # noqa: E402
    ACTIVE_ORB_INSTRUMENTS,
    get_outright_root,
)

# Reuse the FREE-cost threshold from the daily script so divergence is impossible.
from scripts.databento_daily import _FREE_COST_THRESHOLD_USD  # noqa: E402

DATASET = "GLBX.MDP3"
L2_SCHEMAS = ("mbp-1", "mbp-10")


def _previous_calendar_month(today: date) -> tuple[date, date]:
    """Return (start, end_exclusive) for the calendar month before `today`."""
    if today.month == 1:
        prev_year, prev_month = today.year - 1, 12
    else:
        prev_year, prev_month = today.year, today.month - 1
    start = date(prev_year, prev_month, 1)
    last_day = monthrange(prev_year, prev_month)[1]
    end_exclusive = date(prev_year, prev_month, last_day) + _one_day()
    return start, end_exclusive


def _one_day():
    from datetime import timedelta

    return timedelta(days=1)


def _parse_month_arg(value: str) -> tuple[date, date]:
    """`YYYY-MM` -> (start, end_exclusive)."""
    try:
        year_s, month_s = value.split("-")
        year, month = int(year_s), int(month_s)
        start = date(year, month, 1)
        last_day = monthrange(year, month)[1]
        end_exclusive = date(year, month, last_day) + _one_day()
        return start, end_exclusive
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"--month must be YYYY-MM ({exc})") from exc


def snapshot_one(
    client,
    instrument: str,
    schema: str,
    fetch_start: date,
    fetch_end: date,
    output_base: Path,
    dry_run: bool,
) -> str:
    """Pull one (instrument, schema, window) combo. Returns status string."""
    import databento as db

    parent_symbol = get_outright_root(instrument)
    db_symbol = f"{parent_symbol}.FUT"
    name = f"{instrument}_{schema}_{fetch_start.isoformat()}"

    output_dir = output_base / schema / instrument
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"snapshot_{instrument}_{fetch_start}_to_{fetch_end}.{schema}.dbn.zst"

    if out_file.exists() and out_file.stat().st_size > 0:
        log.info(f"  {name}: SKIP (already exists: {out_file.name})")
        return "SKIP"

    # Cost-guard: probe before any billable call.
    try:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=[db_symbol],
            schema=schema,
            stype_in="parent",
            start=str(fetch_start),
            end=str(fetch_end),
        )
        is_free = cost is not None and cost <= _FREE_COST_THRESHOLD_USD
        cost_str = "FREE" if is_free else f"${cost:.4f}"
    except Exception as exc:
        log.warning(f"  {name}: COST PROBE FAIL ({exc}) — refusing to fetch")
        return f"SKIP (cost probe failed: {exc})"

    if dry_run:
        log.info(f"  {name}: WOULD download {fetch_start} -> {fetch_end} ({cost_str})")
        return "DRY_RUN"

    if not is_free:
        log.warning(
            f"  {name}: NON-FREE ({cost_str}) — refusing to fetch. "
            f"L2 free window is rolling 1mo; check --month or your plan."
        )
        return f"SKIP (non-free: {cost_str})"

    log.info(f"  {name}: downloading {fetch_start} -> {fetch_end} (cost-guard: FREE)")

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

        # Chunked validation (mbp-10 files can be 5-10 GB).
        store = db.DBNStore.from_file(str(out_file))
        file_size_gb = out_file.stat().st_size / (1024**3)
        if file_size_gb > 2.0:
            row_count = 0
            for chunk_df in store.to_df(count=1000):
                row_count += len(chunk_df)
                break  # first chunk confirms data exists
            count_str = f">={row_count:,}"
        else:
            df = store.to_df()
            count_str = f"{len(df):,}"

        log.info(f"  {name}: OK ({size_mb:.1f} MB, {count_str} records)")
        return f"OK ({count_str} records)"

    except Exception as exc:
        log.error(f"  {name}: FAIL — {exc}")
        return f"FAIL ({exc})"


def main():
    parser = argparse.ArgumentParser(
        description="On-demand Databento L2 (mbp-1, mbp-10) snapshot puller.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=",".join(ACTIVE_ORB_INSTRUMENTS),
        help=f"Comma-separated list (default: {','.join(ACTIVE_ORB_INSTRUMENTS)})",
    )
    parser.add_argument(
        "--month",
        type=_parse_month_arg,
        default=None,
        help="Calendar month YYYY-MM (default: previous calendar month)",
    )
    parser.add_argument(
        "--schemas",
        type=str,
        default=",".join(L2_SCHEMAS),
        help=f"Comma-separated schemas (default: {','.join(L2_SCHEMAS)})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Cost-probe only, no download")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("DATABENTO_API_KEY"):
        log.error("FATAL: DATABENTO_API_KEY not found in environment or .env")
        sys.exit(1)

    instruments = [s.strip().upper() for s in args.instruments.split(",") if s.strip()]
    schemas = [s.strip() for s in args.schemas.split(",") if s.strip()]

    unknown = set(instruments) - set(ACTIVE_ORB_INSTRUMENTS)
    if unknown:
        log.error(f"FATAL: unknown instruments {sorted(unknown)} (active: {sorted(ACTIVE_ORB_INSTRUMENTS)})")
        sys.exit(1)
    unknown_schemas = set(schemas) - set(L2_SCHEMAS)
    if unknown_schemas:
        log.error(f"FATAL: unknown schemas {sorted(unknown_schemas)} (allowed: {L2_SCHEMAS})")
        sys.exit(1)

    if args.month is None:
        fetch_start, fetch_end = _previous_calendar_month(date.today())
    else:
        fetch_start, fetch_end = args.month

    import databento as db

    client = db.Historical()
    output_base = PROJECT_ROOT / "data" / "raw" / "databento"

    start_time = time.time()
    log.info("=" * 60)
    log.info(f"  L2 SNAPSHOT — {fetch_start} to {fetch_end} (exclusive)")
    log.info(f"  Instruments: {instruments}")
    log.info(f"  Schemas: {schemas}")
    if args.dry_run:
        log.info("  MODE: DRY RUN")
    log.info("=" * 60)

    results: dict[str, str] = {}
    for schema in schemas:
        log.info(f"\n--- {schema} ---")
        for instrument in instruments:
            key = f"{instrument}_{schema}"
            results[key] = snapshot_one(
                client=client,
                instrument=instrument,
                schema=schema,
                fetch_start=fetch_start,
                fetch_end=fetch_end,
                output_base=output_base,
                dry_run=args.dry_run,
            )

    elapsed = time.time() - start_time
    log.info(f"\n{'=' * 60}")
    log.info(f"  SUMMARY ({elapsed:.0f}s)")
    log.info(f"{'=' * 60}")
    for k, v in results.items():
        log.info(f"  {k:<25} {v}")
    fail_count = sum(1 for v in results.values() if "FAIL" in v)
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
