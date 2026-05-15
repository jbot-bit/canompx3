#!/usr/bin/env python3
"""Context-budget gauge: UserPromptSubmit — inject `[CTX: N%]` tier directive.

Layer 1 of the token-burn hardening. Reads the transcript size each turn,
maps to tier 1-5 via `_context_state.current_tier`, and emits an
`additionalContext` injection telling Claude what budget tier it's in and
what behavior to adopt.

Tier 1 (<20%)   silent — no output
Tier 2 (20-60%) normal range gauge
Tier 3 (60-100%) HIGH — prefer inline, avoid subagents, finish current scope
Tier 4 (100-150%) PRE-CLEAR — no new work, no recap docs; subagents BLOCKED
Tier 5 (>=150%) CRITICAL — defensive self-archiving forbidden; user /clear now

Fail-open: any exception exits 0 silently. The gauge must never break a turn.
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


_DIRECTIVES = {
    2: "[CTX: {pct}%] normal range.",
    3: (
        "[CTX: {pct}%] HIGH. Prefer inline work; avoid spawning subagents unless "
        "task needs >5 file reads. Do not re-read files already in context. "
        "Finish current scope before starting new work."
    ),
    4: (
        "[CTX: {pct}%] PRE-CLEAR. Do NOT start new work. Do NOT write "
        "recap/handoff docs (compaction handles preservation). Subagent spawns "
        "will be BLOCKED. Land current task and /clear."
    ),
    5: (
        "[CTX: {pct}%] CRITICAL. Defensive self-archiving is FORBIDDEN — do not "
        "re-read files, do not write summary docs, do not re-confirm state. "
        "Compaction will preserve key context. Finish current tool-call and "
        "STOP. User: /clear now."
    ),
}


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    try:
        tier, pct = _context_state.current_tier(payload)
    except Exception:
        sys.exit(0)

    if tier <= 1:
        sys.exit(0)

    directive = _DIRECTIVES.get(tier)
    if not directive:
        sys.exit(0)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": directive.format(pct=pct),
        }
    }
    try:
        sys.stdout.write(json.dumps(output))
    except Exception:
        sys.exit(0)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(0)
