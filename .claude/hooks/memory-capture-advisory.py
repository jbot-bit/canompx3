#!/usr/bin/env python3
"""Auto-memory-capture: PreCompact warning + SessionEnd breadcrumb writer.

Dispatched on `hook_event_name` (one command wired to both events in
settings.json). Part of the 3-event capture loop — see
`.claude/rules/auto-memory-capture.md`.

  PreCompact  -> if signal met AND session not already advised, emit a
                 USER-VISIBLE `systemMessage` (the only channel PreCompact
                 supports) warning that durable findings are about to be
                 compacted, framed as a JUDGE-don't-auto-write cue.
  SessionEnd  -> always append one telemetry line; if signal met, write/refresh
                 the pending breadcrumb the next SessionStart will consume.
                 No model/user channel exists for SessionEnd — silent.

Fail-open: any error -> exit 0. stdout on exit 0 carries ONLY valid JSON
(PreCompact emit) or nothing (everything else), per the hooks contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parent
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

import _memory_capture as mc


_ADVISORY_TEMPLATE = (
    "Auto-memory-capture: this session accrued {summary}. Context is about to be "
    "compacted. Before it is, JUDGE whether any durable, non-obvious lesson is "
    "worth capturing to memory/*.md — dedup against existing files; taxonomy "
    "user/feedback/project/reference. Capture nothing if it is obvious, "
    "transient, or already documented. This is a cue to judge, not an "
    "instruction to write."
)


def _handle_precompact(event: dict) -> None:
    sid = event.get("session_id", "")
    sig = mc.gather_signals()
    if not mc.signal_meets_threshold(sig):
        return  # nothing worth warning about — silent
    if sid and mc.already_advised(sid):
        return  # dedup: warn at most once per session
    msg = _ADVISORY_TEMPLATE.format(summary=mc.describe_signals(sig))
    print(json.dumps({"systemMessage": msg}))
    if sid:
        mc.record_advised(sid)


def _handle_sessionend(event: dict) -> None:
    sid = event.get("session_id", "")
    reason = event.get("reason")
    sig = mc.gather_signals()
    met = mc.signal_meets_threshold(sig)
    mc.append_telemetry(
        {
            "ts": mc._now_iso(),
            "event": "SessionEnd",
            "reason": reason,
            "session_id": sid,
            "counts": sig,
            "signal_met": met,
        }
    )
    if met and sid:
        mc.write_breadcrumb(sid, sig)
    # No model/user channel for SessionEnd — emit nothing to stdout.


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        event = {}
    if not isinstance(event, dict):
        event = {}

    ev = event.get("hook_event_name", "")
    if ev == "PreCompact":
        _handle_precompact(event)
    elif ev == "SessionEnd":
        _handle_sessionend(event)
    # Any other event -> no-op.
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # pragma: no cover - fail-open
        sys.exit(0)
