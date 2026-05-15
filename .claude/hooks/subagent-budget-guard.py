#!/usr/bin/env python3
"""Subagent budget hard-stop: PreToolUse(Task) — block spawns at tier >=4.

Layer 2 of the token-burn hardening. A subagent spawn adds 30-60K tokens of
main-context growth (CLAUDE.md + matched rules + agent definition + return
summary). At Tier 4 (>=100% of cap) this triggers compaction or context
exhaustion. The hook refuses the spawn with exit 2 and a structured stderr
message; Claude sees the block, must redirect inline.

Fail-open: ANY error (malformed payload, missing transcript, helper import
failure) exits 0 (allow). The guard must never block a session it cannot
measure.

Recovery path: user runs /clear; transcript resets; tier drops to 1; spawns
allowed again. No override flag — /clear is the only path forward.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import json
import sys
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parent
_SPEC = _importlib_util.spec_from_file_location(
    "_context_state", _HOOKS_DIR / "_context_state.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_context_state = _importlib_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_context_state)


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if event.get("tool_name") != "Task":
        sys.exit(0)

    try:
        tier, pct = _context_state.current_tier(event)
    except Exception:
        sys.exit(0)

    if tier < 4:
        sys.exit(0)

    print(
        f"BLOCKED: subagent spawn refused at context tier T{tier} (CTX: {pct}%).\n"
        f"Reason: subagent spawn adds 30-60K tokens of main-context growth.\n"
        f"At >=100% context, this triggers immediate compaction or exhaustion.\n"
        f"Action: complete current task inline, then /clear and retry.\n"
        f"Override: not available — /clear is the only path forward.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(0)
