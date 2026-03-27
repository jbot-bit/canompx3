#!/usr/bin/env python3
"""Databento historical data backfill -- multi-schema, chunked, resumable.

Downloads data from Databento's Historical API with cost estimation, chunked
monthly downloads, resume support, and post-download validation.

Integrates with the existing pipeline -- ohlcv-1m downloads land in the same
DB/ directory structure and can be ingested via ingest_dbn.py. Other schemas
(tbbo, trades, bbo-1s, ohlcv-1s, statistics) land in data/raw/databento/ for
research ingestion.

Usage:
    python scripts/databento_backfill.py --dry-run                # Cost estimate only
    python scripts/databento_backfill.py --priority must_have     # Download Tier 1 only
    python scripts/databento_backfill.py --priority high          # Tiers 1 + 2
    python scripts/databento_backfill.py --name mgc_1m_extension  # Single download
    python scripts/databento_backfill.py --schema ohlcv-1s        # All downloads of a schema
    python scripts/databento_backfill.py --from 2024-01-01 --to 2024-12-31  # Override dates
    python scripts/databento_backfill.py --yes                    # Skip cost confirmation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import yaml

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "databento_config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Load and validate the YAML config."""
    if not config_path.exists():
        print(f"FATAL: Config not found: {config_path}")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("dataset", "output_base", "downloads"):
        if key not in cfg:
            print(f"FATAL: Missing required config key: {key}")
            sys.exit(1)
    return cfg


# ---------------------------------------------------------------------------
# Priority filtering
# ---------------------------------------------------------------------------

PRIORITY_ORDER = {"must_have": 0, "high": 1, "nice_to_have": 2}


def filter_downloads(
    downloads: list[dict],
    priority: str | None = None,
    name: str | None = None,
    schema: str | None = None,
) -> list[dict]:
    """Filter download specs by priority, name, or schema."""
    result = downloads

    if name:
        result = [d for d in result if d["name"] == name]
        if not result:
            print(f"FATAL: No download named '{name}' in config")
            print(f"  Available: {[d['name'] for d in downloads]}")
            sys.exit(1)
        return result

    if schema:
        result = [d for d in result if d["schema"] == schema]
        if not result:
            print(f"FATAL: No downloads with schema '{schema}' in config")
            sys.exit(1)

    if priority:
        max_level = PRIORITY_ORDER.get(priority)
        if max_level is None:
            print(f"FATAL: Unknown priority '{priority}'. Use: must_have, high, nice_to_have")
            sys.exit(1)
        result = [d for d in result if PRIORITY_ORDER.get(d["priority"], 99) <= max_level]

    result.sort(key=lambda d: PRIORITY_ORDER.get(d["priority"], 99))
    return result


# ---------------------------------------------------------------------------
# Date chunking
# ---------------------------------------------------------------------------


def chunk_date_range(start: str, end: str, chunk_months: int = 6) -> list[tuple[str, str]]:
    """Split a date range into monthly chunks for manageable downloads."""
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)
    chunks = []

    current = start_dt
    while current < end_dt:
        next_year = current.year + (current.month + chunk_months - 1) // 12
        next_month = (current.month + chunk_months - 1) % 12 + 1
        chunk_end = date(next_year, next_month, 1)
        if chunk_end > end_dt:
            chunk_end = end_dt
        chunks.append((str(current), str(chunk_end)))
        current = chunk_end

    return chunks


# ---------------------------------------------------------------------------
# Manifest (resume support)
# ---------------------------------------------------------------------------


def load_manifest(manifest_dir: Path, download_name: str) -> dict:
    """Load the manifest for a download, or return empty dict."""
    path = manifest_dir / f"{download_name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"name": download_name, "chunks": {}, "completed": False}


def save_manifest(manifest_dir: Path, download_name: str, manifest: dict):
    """Save the manifest for a download."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{download_name}.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_costs(client, cfg: dict, downloads: list[dict]) -> list[dict]:
    """Get cost estimates for each download from Databento API."""
    results = []
    for dl in downloads:
        try:
            cost = client.metadata.get_cost(
                dataset=cfg["dataset"],
                symbols=[dl["symbol"]],
                schema=dl["schema"],
                stype_in=dl.get("stype_in", "parent"),
                start=dl["start"],
                end=dl["end"],
            )
            size = client.metadata.get_billable_size(
                dataset=cfg["dataset"],
                symbols=[dl["symbol"]],
                schema=dl["schema"],
                stype_in=dl.get("stype_in", "parent"),
                start=dl["start"],
                end=dl["end"],
            )
            results.append({
                **dl,
                "est_cost": cost,
                "est_size_mb": size / (1024 * 1024) if size else 0,
            })
        except Exception as e:
            results.append({**dl, "est_cost": -1, "est_size_mb": 0, "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# Download with retry
# ---------------------------------------------------------------------------


def download_chunk(
    client,
    cfg: dict,
    dl: dict,
    chunk_start: str,
    chunk_end: str,
    output_dir: Path,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> Path | None:
    """Download a single chunk with exponential backoff retry."""
    out_file = output_dir / f"{dl['name']}_{chunk_start}_to_{chunk_end}.{dl['schema']}.dbn.zst"

    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"    SKIP (exists): {out_file.name}")
        return out_file

    for attempt in range(1, max_retries + 1):
        try:
            data = client.timeseries.get_range(
                dataset=cfg["dataset"],
                symbols=[dl["symbol"]],
                schema=dl["schema"],
                stype_in=dl.get("stype_in", "parent"),
                start=chunk_start,
                end=chunk_end,
            )
            data.to_file(str(out_file))
            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"    OK: {out_file.name} ({size_mb:.1f} MB)")
            return out_file

        except Exception as e:
            delay = base_delay * (2 ** (attempt - 1))
            if attempt < max_retries:
                print(f"    RETRY {attempt}/{max_retries}: {e} (waiting {delay:.0f}s)")
                time.sleep(delay)
            else:
                print(f"    FAIL after {max_retries} attempts: {e}")
                return None


# ---------------------------------------------------------------------------
# Post-download validation
# ---------------------------------------------------------------------------


def validate_download(file_path: Path, dl: dict) -> dict:
    """Validate a downloaded DBN file. Returns validation report."""
    import databento as db

    report = {
        "file": str(file_path),
        "valid": False,
        "rows": 0,
        "first_ts": None,
        "last_ts": None,
        "schema": dl["schema"],
    }

    try:
        store = db.DBNStore.from_file(str(file_path))
        df = store.to_df()
        report["rows"] = len(df)

        if len(df) > 0:
            ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
            if hasattr(df[ts_col], "min"):
                report["first_ts"] = str(df[ts_col].min())
                report["last_ts"] = str(df[ts_col].max())

        report["valid"] = report["rows"] > 0

    except Exception as e:
        report["error"] = str(e)

    return report


# ---------------------------------------------------------------------------
# Main download orchestrator
# ---------------------------------------------------------------------------


def run_download(
    cfg: dict,
    downloads: list[dict],
    dry_run: bool = False,
    auto_yes: bool = False,
    date_from: str | None = None,
    date_to: str | None = None,
):
    """Run the full download pipeline."""
    import databento as db

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        print("FATAL: DATABENTO_API_KEY not found in environment or .env")
        sys.exit(1)

    client = db.Historical(api_key)

    # Apply date overrides
    if date_from or date_to:
        for dl in downloads:
            if date_from:
                dl["start"] = date_from
            if date_to:
                dl["end"] = date_to

    # Step 1: Cost estimation
    print("=" * 70)
    print("  DATABENTO BACKFILL -- COST ESTIMATION")
    print("=" * 70)

    estimates = estimate_costs(client, cfg, downloads)

    total_cost = 0.0
    total_size_mb = 0.0

    print(f"\n  {'Name':<30} {'Schema':<12} {'Range':<25} {'Size':>10} {'Cost':>10}")
    print("  " + "-" * 90)

    for est in estimates:
        if est.get("error"):
            print(f"  {est['name']:<30} {est['schema']:<12} ERROR: {est['error']}")
            continue
        cost = est["est_cost"]
        size = est["est_size_mb"]
        total_cost += max(cost, 0)
        total_size_mb += size
        rng = f"{est['start']} -> {est['end']}"
        cost_str = "FREE" if cost == 0 else f"${cost:.2f}"
        print(
            f"  {est['name']:<30} {est['schema']:<12} {rng:<25} {size:>8.1f}MB {cost_str:>10}"
        )

    print("  " + "-" * 90)
    cost_str = "FREE" if total_cost == 0 else f"${total_cost:.2f}"
    print(f"  {'TOTAL':<30} {'':12} {'':25} {total_size_mb:>8.1f}MB {cost_str:>10}")
    print()

    if dry_run:
        print("  [DRY RUN] No data downloaded.")
        return

    # Step 2: Cost confirmation
    threshold = cfg.get("cost_confirm_threshold", 5.0)
    if total_cost > threshold and not auto_yes:
        print(f"  Total cost ${total_cost:.2f} exceeds ${threshold:.2f} threshold.")
        response = input("  Proceed? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("  Aborted.")
            return

    # Step 3: Download each spec
    output_base = PROJECT_ROOT / cfg["output_base"]
    manifest_dir = PROJECT_ROOT / cfg.get("manifest_dir", "data/raw/databento/manifests")
    chunk_months = cfg.get("chunk_months", 6)
    max_retries = cfg.get("retry_max_attempts", 5)
    base_delay = cfg.get("retry_base_delay_seconds", 2.0)

    results = {}

    for dl in estimates:
        if dl.get("error"):
            results[dl["name"]] = "SKIP (cost error)"
            continue

        name = dl["name"]
        print(f"\n{'=' * 70}")
        print(f"  DOWNLOADING: {name}")
        print(f"  {dl['description']}")
        print(f"  {dl['symbol']} {dl['schema']} {dl['start']} -> {dl['end']}")
        print(f"{'=' * 70}")

        # Output directory: data/raw/databento/{schema}/{stored_as}/
        stored_as = dl.get("stored_as", dl["symbol"].replace(".FUT", ""))
        output_dir = output_base / dl["schema"] / stored_as
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest for resume
        manifest = load_manifest(manifest_dir, name)
        if manifest.get("completed"):
            print("  SKIP (already completed per manifest)")
            results[name] = "SKIP (completed)"
            continue

        # Chunk the date range
        chunks = chunk_date_range(dl["start"], dl["end"], chunk_months)
        print(f"  {len(chunks)} chunk(s) to download")

        all_ok = True
        chunk_files = []

        for i, (cs, ce) in enumerate(chunks, 1):
            chunk_key = f"{cs}_{ce}"

            # Resume: skip already-downloaded chunks
            existing = manifest.get("chunks", {}).get(chunk_key, {})
            if existing.get("valid"):
                saved_path = existing.get("file")
                if saved_path and Path(saved_path).exists():
                    print(f"  [{i}/{len(chunks)}] SKIP (resume): {cs} -> {ce}")
                    chunk_files.append(Path(saved_path))
                    continue

            print(f"  [{i}/{len(chunks)}] {cs} -> {ce}")

            out_file = download_chunk(
                client, cfg, dl, cs, ce, output_dir,
                max_retries=max_retries, base_delay=base_delay,
            )

            if out_file is None:
                all_ok = False
                manifest.setdefault("chunks", {})[chunk_key] = {"valid": False}
                save_manifest(manifest_dir, name, manifest)
                continue

            # Validate
            report = validate_download(out_file, dl)
            manifest.setdefault("chunks", {})[chunk_key] = {
                "file": str(out_file),
                "valid": report["valid"],
                "rows": report["rows"],
                "first_ts": report.get("first_ts"),
                "last_ts": report.get("last_ts"),
                "downloaded_at": datetime.now(UTC).isoformat(),
            }
            save_manifest(manifest_dir, name, manifest)

            if report["valid"]:
                chunk_files.append(out_file)
                print(f"    Validated: {report['rows']:,} records")
            else:
                all_ok = False
                print(f"    VALIDATION FAILED: {report}")

        if all_ok and chunk_files:
            manifest["completed"] = True
            manifest["completed_at"] = datetime.now(UTC).isoformat()
            manifest["total_files"] = len(chunk_files)
            manifest["total_rows"] = sum(
                manifest["chunks"][k].get("rows", 0) for k in manifest["chunks"]
            )
            save_manifest(manifest_dir, name, manifest)
            results[name] = f"OK ({len(chunk_files)} files)"
        elif chunk_files:
            results[name] = f"PARTIAL ({len(chunk_files)}/{len(chunks)} chunks)"
        else:
            results[name] = "FAIL"

    # Step 4: Summary
    print(f"\n{'=' * 70}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    for name, status in results.items():
        print(f"  {name:<35} {status}")
    print(f"{'=' * 70}")

    # Note any ohlcv-1m downloads that need pipeline ingestion
    ohlcv_1m_ok = [
        dl for dl in estimates
        if dl["schema"] == "ohlcv-1m" and results.get(dl["name"], "").startswith("OK")
    ]
    if ohlcv_1m_ok:
        print(f"\n  NOTE: {len(ohlcv_1m_ok)} ohlcv-1m download(s) completed.")
        print("  To ingest into gold.db, run:")
        for dl in ohlcv_1m_ok:
            stored = dl.get("stored_as", "?")
            print(f"    python pipeline/ingest_dbn.py --instrument {stored} --resume")

    if any("FAIL" in v for v in results.values()):
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Databento historical data backfill (multi-schema, chunked, resumable)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Priority tiers:
  must_have    - Free downloads, historical extensions ($0)
  high         - 1s bars (free) + GC tick data (~$27)
  nice_to_have - MNQ/MES tick data (~$242)

Examples:
  %(prog)s --dry-run                  # Show cost estimate only
  %(prog)s --priority must_have       # Download free Tier 1 only
  %(prog)s --name mgc_1m_extension    # Single download by name
  %(prog)s --schema ohlcv-1s          # All 1-second bar downloads
  %(prog)s --yes                      # Skip cost confirmation prompt
        """,
    )
    parser.add_argument("--dry-run", action="store_true", help="Show cost estimate only")
    parser.add_argument("--priority", choices=["must_have", "high", "nice_to_have"],
                        help="Download up to this priority tier")
    parser.add_argument("--name", help="Download a specific item by name")
    parser.add_argument("--schema", help="Download all items of a specific schema")
    parser.add_argument("--from", dest="date_from", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="date_to", help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip cost confirmation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config file path")

    args = parser.parse_args()

    cfg = load_config(args.config)
    downloads = filter_downloads(
        cfg["downloads"],
        priority=args.priority,
        name=args.name,
        schema=args.schema,
    )

    if not downloads:
        print("No downloads match the specified filters.")
        sys.exit(0)

    run_download(
        cfg=cfg,
        downloads=downloads,
        dry_run=args.dry_run,
        auto_yes=args.yes,
        date_from=args.date_from,
        date_to=args.date_to,
    )


if __name__ == "__main__":
    main()
