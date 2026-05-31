#!/usr/bin/env python3
"""Auto-memory-capture: Claude-visible cue on the next session start.

The model-facing half of the 3-event loop — see
`.claude/rules/auto-memory-capture.md`. Runs as a PARALLEL SIBLING to
session-start.py (different channel: stdout-JSON additionalContext vs
session-start.py's stderr text). Reads ONLY its own breadcrumb — no .claude.pid
dependency, so it is immune to any session-lock format change.

Behavior: read the pending breadcrumb written by the prior SessionEnd. If it is
present, NOT yet consumed, younger than 24h, and the signal counts are nonzero,
emit a CLAUDE-VISIBLE `additionalContext` cue (verified to survive /clear via
the Stage-0 probe seq=2 round-trip) prompting a capture JUDGEMENT, then mark the
breadcrumb consumed (one-shot). Otherwise emit nothing (silent clean start).

Fires on all sources (matcher ""), incl. clear/compact/startup/resume. Reads
both `source` (official key, confirmed by the probe) and `session_type` (in-repo
variant) fail-open. Fail-open everywhere: any error -> exit 0, no stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parent
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

import _memory_capture as mc


_CUE_TEMPLATE = (
    "Prior session accrued {summary}. Before proceeding, CONSIDER whether any "
    "durable, non-obvious lesson is worth capturing to memory/*.md (dedup "
    "against existing files; taxonomy user/feedback/project/reference). Capture "
    "nothing if it is obvious, transient, or already documented — this is a cue "
    "to JUDGE, not an instruction to write."
)


def _counts_nonzero(counts: dict) -> bool:
    if not isinstance(counts, dict):
        return False
    return mc.signal_meets_threshold(counts)


def _should_emit(crumb: dict | None) -> bool:
    if not isinstance(crumb, dict):
        return False
    if crumb.get("consumed"):
        return False
    if not mc.breadcrumb_is_fresh(crumb):
        return False
    return _counts_nonzero(crumb.get("counts", {}))


def main() -> None:
    try:
        json.load(sys.stdin)  # drain stdin; source/session_type unused for gating
    except (json.JSONDecodeError, ValueError):
        pass

    crumb = mc.read_breadcrumb()
    if not _should_emit(crumb):
        sys.exit(0)  # silent — no stdout on a clean / already-consumed start

    summary = mc.describe_signals(crumb["counts"])
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": _CUE_TEMPLATE.format(summary=summary),
                }
            }
        )
    )
    mc.mark_breadcrumb_consumed()
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # pragma: no cover - fail-open
        sys.exit(0)
