#!/usr/bin/env python3
"""Post-compaction context re-injection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.tools.claude_superpower_brief import build_brief
except Exception:  # pragma: no cover - hook fallback path
    build_brief = None

try:
    from scripts.tools.task_route_packet import read_task_route_packet
except Exception:  # pragma: no cover - hook fallback path
    read_task_route_packet = None


def _superpower_lines() -> list[str]:
    if build_brief is None:
        return []
    try:
        return build_brief(root=PROJECT_ROOT, mode="post-compact").splitlines()
    except Exception:
        return []


def _task_route_lines() -> list[str]:
    if read_task_route_packet is None:
        return []
    try:
        return read_task_route_packet(PROJECT_ROOT)
    except Exception:
        return []


def main() -> None:
    try:
        json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    task_route_lines = _task_route_lines()
    if task_route_lines:
        lines = ["POST-COMPACTION TASK ROUTE RE-INJECTION:"]
        lines.extend(task_route_lines)
        lines.append("Reminder: re-check live files before trusting compacted context.")
    else:
        lines = ["POST-COMPACTION CONTEXT RE-INJECTION:"]
        lines.extend(_superpower_lines())
        lines.extend(
            [
                "  REMINDERS:",
                "  - Canonical sources: asset_configs, dst, config, cost_model, paths — NEVER hardcode",
                "  - daily_features JOIN must include AND o.orb_minutes = d.orb_minutes",
                "  - One-way dep: pipeline/ -> trading_app/ (never reversed)",
                "  - Data first: query before reading code for data questions",
                "  - 2026 holdout is SACRED — no discovery on 2026 data",
            ]
        )

    print("\n".join(lines))
    sys.exit(0)


if __name__ == "__main__":
    main()
