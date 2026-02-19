#!/usr/bin/env python3
"""Low-token operator status for canompx3.

Purpose:
- Give a high-signal, compact status summary for automated checks.
- Avoid heavy operations (no full test suite, no long backtests).

Usage:
    python scripts/operator_status.py
    python scripts/operator_status.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class OperatorStatus:
    git_dirty: bool
    git_summary: str
    roadmap_todos: list[str]
    monitor_alerting_todo: bool
    outcomes_backfill_pending: bool
    key_risk: str
    best_next_step: str


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _git_summary() -> tuple[bool, str]:
    code, out, _ = _run(["git", "status", "--short"])
    if code != 0:
        return False, "git unavailable"

    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        return False, "clean"

    staged = sum(1 for ln in lines if ln[:1] not in {"?", " "})
    untracked = sum(1 for ln in lines if ln.startswith("??"))
    modified = len(lines) - untracked
    return True, f"dirty ({modified} modified, {staged} staged, {untracked} untracked)"


def _read_roadmap() -> str:
    path = PROJECT_ROOT / "ROADMAP.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_todos(roadmap: str) -> list[str]:
    todos: list[str] = []
    for line in roadmap.splitlines():
        l = line.strip()
        if "TODO" in l or "IN PROGRESS" in l:
            if l.startswith("#") or l.startswith("-"):
                todos.append(l.lstrip("- "))
    return todos[:8]


def _contains_any(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def build_status() -> OperatorStatus:
    dirty, git_sum = _git_summary()
    roadmap = _read_roadmap()
    todos = _extract_todos(roadmap)

    monitor_todo = _contains_any(
        roadmap,
        ["monitoring & alerting", "monitoring and alerting", "phase 6e"],
    )
    backfill_pending = _contains_any(
        roadmap,
        ["backfill", "2016-2020", "2016–2020", "orb_outcomes backfill"],
    )

    if monitor_todo:
        key_risk = "No complete live monitoring/alerting loop yet"
        best_next = "Implement minimal drift alerts (drawdown, win-rate divergence, regime shift)"
    elif backfill_pending:
        key_risk = "Validation horizon incomplete until outcomes backfill finishes"
        best_next = "Complete 2016-2020 orb_outcomes backfill, then rerun rolling eval"
    elif dirty:
        key_risk = "Uncommitted changes may bypass stable baseline"
        best_next = "Review and commit current changes with guardrails"
    else:
        key_risk = "No immediate high-signal risk detected"
        best_next = "Run next planned roadmap TODO"

    return OperatorStatus(
        git_dirty=dirty,
        git_summary=git_sum,
        roadmap_todos=todos,
        monitor_alerting_todo=monitor_todo,
        outcomes_backfill_pending=backfill_pending,
        key_risk=key_risk,
        best_next_step=best_next,
    )


def print_human(status: OperatorStatus) -> None:
    print("OPERATOR_STATUS")
    print(f"- Git: {status.git_summary}")
    print(f"- Monitoring TODO: {'yes' if status.monitor_alerting_todo else 'no'}")
    print(f"- Backfill pending: {'yes' if status.outcomes_backfill_pending else 'no'}")
    if status.roadmap_todos:
        print("- Top TODOs:")
        for item in status.roadmap_todos[:3]:
            print(f"  - {item}")
    print(f"- Key risk: {status.key_risk}")
    print(f"- Best next step: {status.best_next_step}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Low-token operator status")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    status = build_status()

    if args.json:
        print(json.dumps(asdict(status), ensure_ascii=False))
    else:
        print_human(status)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
