"""Export approved gold.db read snapshots for remote/GitHub consumers.

This is intentionally narrower than ``pipeline/export_parquet.py``.  It exports
only approved read-model tables and writes a manifest that MCP tools can inspect.
It does not expose live DB writes or arbitrary table selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading_app.db_access import APPROVED_SNAPSHOT_TABLES, DEFAULT_SNAPSHOT_ROOT, export_snapshot


def _parse_tables(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export approved gold.db read snapshot with manifest")
    parser.add_argument(
        "--db-path", type=Path, default=None, help="Path to gold.db; defaults to canonical GOLD_DB_PATH"
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
        help=f"Approved snapshot root (default: {DEFAULT_SNAPSHOT_ROOT})",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory under --snapshot-root")
    parser.add_argument(
        "--tables",
        default=None,
        help=f"Comma-separated approved tables; defaults to all: {', '.join(APPROVED_SNAPSHOT_TABLES)}",
    )
    parser.add_argument("--max-age-hours", type=int, default=168, help="Refuse source DB older than this many hours")
    parser.add_argument("--allow-stale", action="store_true", help="Bypass source DB age check")
    args = parser.parse_args(argv)

    try:
        manifest = export_snapshot(
            db_path=args.db_path,
            output_dir=args.output_dir,
            snapshot_root=args.snapshot_root,
            tables=_parse_tables(args.tables),
            max_age_hours=None if args.allow_stale else args.max_age_hours,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
