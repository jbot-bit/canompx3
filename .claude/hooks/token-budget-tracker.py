#!/usr/bin/env python3
"""Token-budget tracker (PostToolUse Bash).

Accumulates bash stdout+stderr bytes per Claude Code session. Emits stderr
warnings at two thresholds:
  - 500 KB (~125K tokens) → soft warning
  - 1 MB  (~250K tokens) → loud warning

Intent: catch runaway `find`, `python -c` dumps, accidentally-verbose pipeline
runs BEFORE they eat a full session. Advisory, never blocks — bash output has
already been captured into Claude's context by the time PostToolUse fires, so
blocking would be useless. This hook exists to make the cost visible, not
prevent it.

Session boundary: state file is keyed by session_id if available, falls back
to a single .token-budget-state.json for the project. Cleared on SessionStart
via reset-if-old logic (state older than 12h is discarded as "new session").
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

STATE_FILE = Path(__file__).parent / ".token-budget-state.json"
SOFT_LIMIT = 500 * 1024         # 500 KB
HARD_LIMIT = 1024 * 1024        # 1 MB
STATE_MAX_AGE_SECONDS = 12 * 3600  # 12h → new session


def _load_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"bytes": 0, "started_at": time.time(), "warned_soft": False, "warned_hard": False}
    except Exception:
        return {"bytes": 0, "started_at": time.time(), "warned_soft": False, "warned_hard": False}
    # Expire stale state (new session)
    started = state.get("started_at", 0)
    if time.time() - started > STATE_MAX_AGE_SECONDS:
        return {"bytes": 0, "started_at": time.time(), "warned_soft": False, "warned_hard": False}
    return state


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def _response_bytes(event: dict) -> int:
    """Measure bytes from bash tool_response. Robust across shapes."""
    resp = event.get("tool_response", "")
    if isinstance(resp, str):
        return len(resp.encode("utf-8", errors="replace"))
    if isinstance(resp, dict):
        total = 0
        for key in ("stdout", "stderr", "output"):
            val = resp.get(key, "")
            if isinstance(val, str):
                total += len(val.encode("utf-8", errors="replace"))
        return total
    return 0


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as exc:
        print(f"[token-budget-tracker] unexpected: {exc}", file=sys.stderr)
        sys.exit(0)

    if event.get("hook_event_name") != "PostToolUse":
        sys.exit(0)
    if event.get("tool_name") != "Bash":
        sys.exit(0)

    added = _response_bytes(event)
    if added == 0:
        sys.exit(0)

    state = _load_state()
    state["bytes"] = state.get("bytes", 0) + added

    kb = state["bytes"] // 1024
    if state["bytes"] >= HARD_LIMIT and not state.get("warned_hard"):
        print(
            f"TOKEN BUDGET: session bash output at {kb} KB (~{kb // 4}K tokens). "
            f"HARD LIMIT crossed. Consider /clear or refactor to smaller queries.",
            file=sys.stderr,
        )
        state["warned_hard"] = True
        state["warned_soft"] = True
    elif state["bytes"] >= SOFT_LIMIT and not state.get("warned_soft"):
        print(
            f"TOKEN BUDGET: session bash output at {kb} KB (~{kb // 4}K tokens). "
            f"Soft limit — prefer targeted queries over wide scans.",
            file=sys.stderr,
        )
        state["warned_soft"] = True

    _save_state(state)
    sys.exit(0)


if __name__ == "__main__":
    main()
