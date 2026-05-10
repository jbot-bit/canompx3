#!/usr/bin/env python3
"""Full-shelf deployability audit.

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
from trading_app.deployability_state import write_deployability_state  # noqa: E402


def _exit_code_for_policy(report: dict, fail_policy: str) -> int:
    if fail_policy == "report-only":
        return 0
    if fail_policy == "legacy":
        return 0 if report["summary"]["deployable_candidates"] > 0 else 2

    total = int(report["summary"].get("total_candidates") or 0)
    hard_issue_counts = report["summary"].get("hard_issue_counts") or {}
    has_hard_blockers = bool(hard_issue_counts)
    if fail_policy == "any-hard-blockers":
        return 2 if total == 0 or has_hard_blockers else 0
    if fail_policy == "profile-hard-blockers":
        if report.get("scope") != "profile":
            return 2
        return 2 if total == 0 or has_hard_blockers else 0
    raise ValueError(f"unknown fail policy: {fail_policy}")


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
    parser.add_argument(
        "--write-state",
        action="store_true",
        help="Append strategy-level rows to deployment_readiness_evaluations.",
    )
    parser.add_argument("--rebuild-id", default=None, help="Rebuild manifest id to attach to written state.")
    parser.add_argument(
        "--fail-policy",
        choices=("legacy", "report-only", "profile-hard-blockers", "any-hard-blockers"),
        default="legacy",
        help=(
            "Exit behavior. legacy preserves candidate-count behavior; "
            "profile-hard-blockers is intended for selected-profile rebuild gates."
        ),
    )
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

    if args.write_state:
        try:
            report["state_write"] = write_deployability_state(
                report,
                db_path=Path(args.db_path),
                rebuild_id=args.rebuild_id,
            )
        except Exception as exc:
            payload = {"verdict": "state_write_error", "error": str(exc)}
            rendered = (
                json.dumps(payload, indent=2, sort_keys=True) if args.format == "json" else f"state_write_error: {exc}"
            )
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

    return _exit_code_for_policy(report, args.fail_policy)


if __name__ == "__main__":
    raise SystemExit(main())
