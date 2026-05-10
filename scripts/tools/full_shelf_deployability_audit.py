#!/usr/bin/env python3
"""Read-only full-shelf deployability audit.

This is a deployment-readiness tool, not a research validator. It replays every
candidate through canonical surfaces and fails closed on missing evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.deployability import (  # noqa: E402
    build_deployability_audit,
    render_deployability_text,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run read-only full-shelf deployability audit.")
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH), help="DuckDB path. Defaults to canonical gold.db.")
    parser.add_argument("--scope", choices=("all-active", "profile"), default="all-active")
    parser.add_argument("--profile", default="topstep_50k_mnq_auto", help="Profile id for profile/account gates.")
    parser.add_argument(
        "--instrument",
        action="append",
        choices=("ALL", "MES", "MGC", "MNQ"),
        default=None,
        help="Restrict to an instrument. Repeat for multiple. Defaults to ALL.",
    )
    parser.add_argument("--scorecard", action="store_true", help="Alias for text output with instrument summaries.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", default=None, help="Optional output path.")
    parser.add_argument(
        "--non-strict", action="store_true", help="Reserved for diagnostics; strict deployability is default."
    )
    parser.add_argument("--max-rows", type=int, default=30, help="Max strategy rows in text output.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    instruments = None
    if args.instrument and "ALL" not in args.instrument:
        instruments = set(args.instrument)
    try:
        report = build_deployability_audit(
            db_path=Path(args.db_path),
            scope=args.scope,
            profile_id=args.profile,
            strict=not args.non_strict,
            instruments=instruments,
        )
    except Exception as exc:
        payload = {"verdict": "tool_error", "error": str(exc)}
        rendered = json.dumps(payload, indent=2, sort_keys=True) if args.format == "json" else f"tool_error: {exc}"
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 3

    rendered = (
        json.dumps(report, indent=2, sort_keys=True, default=str)
        if args.format == "json"
        else render_deployability_text(report, max_rows=args.max_rows)
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8")
    else:
        print(rendered)

    return 0 if report["summary"]["deployable_candidates"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
