#!/usr/bin/env python3
"""Plan-first enforcement hook (UserPromptSubmit).

Highest-leverage pain in the feedback corpus (8 files): jumping into
implementation without presenting a plan first. This hook detects clear
implementation-intent keywords and — if no IMPLEMENTATION stage is active —
emits a reminder directive so Claude halts and presents a plan.

Advisory, not blocking (exit 0 always). Disjoint from other UserPromptSubmit
hooks: stage-awareness.py surfaces current stage state; data-first-guard.py
routes investigation vs design vs commit intent. This one fires only when
intent is implementation AND no stage permits it.

Cooldown: 10 min between firings (same directive) to avoid nagging.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

STATE_FILE = Path(__file__).parent / ".plan-first-state.json"
COOLDOWN_MIN = 10
STAGES_DIR = Path("docs/runtime/stages")
LEGACY_STAGE_FILE = Path("docs/runtime/STAGE_STATE.md")

# Implementation-intent keywords. Kept narrow on purpose: design/investigation
# words are handled by data-first-guard.py.
IMPLEMENT_INTENT = re.compile(
    r"\b("
    r"implement|build it|ship it|go ahead|do it|just do it|"
    r"write the code|code it up|add the feature|make it happen|"
    r"wire it up|hook it up|execute the plan"
    r")\b",
    re.IGNORECASE,
)

# Prompts that explicitly ASK for a plan should not trigger the reminder.
PLAN_REQUEST = re.compile(
    r"\b("
    r"plan|design|brainstorm|think about|iterate|4t|approach|"
    r"propose|options for|pros and cons|trade.?offs"
    r")\b",
    re.IGNORECASE,
)

DIRECTIVE = (
    "plan-first: no IMPLEMENTATION stage active. "
    "Present plan + files + blast radius BEFORE writing code. "
    "Skip only if this is a trivial fix, a git op, or explicit 'just do it' override."
)


def _now() -> datetime:
    return datetime.now(UTC)


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def _has_implementation_stage() -> bool:
    """Return True if any active stage file has `mode: IMPLEMENTATION`."""
    candidates: list[Path] = []
    if STAGES_DIR.is_dir():
        candidates.extend(sorted(STAGES_DIR.glob("*.md")))
    if LEGACY_STAGE_FILE.exists():
        candidates.append(LEGACY_STAGE_FILE)

    for f in candidates:
        try:
            content = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("mode:"):
                value = stripped.split(":", 1)[1].strip().strip('"').strip("'").upper()
                if value == "IMPLEMENTATION":
                    return True
                break  # mode found for this stage, but not IMPLEMENTATION
    return False


def _in_cooldown(state: dict) -> bool:
    last = state.get("last_fired_at")
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
    except (ValueError, TypeError):
        return False
    return _now() - last_dt < timedelta(minutes=COOLDOWN_MIN)


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as exc:
        print(f"[plan-first] unexpected: {exc}", file=sys.stderr)
        sys.exit(0)

    if event.get("hook_event_name", "") != "UserPromptSubmit":
        sys.exit(0)

    prompt = event.get("prompt", "") or ""
    if not prompt.strip():
        sys.exit(0)

    # Only fire on implementation-intent prompts that aren't also plan/design requests.
    if not IMPLEMENT_INTENT.search(prompt):
        sys.exit(0)
    if PLAN_REQUEST.search(prompt):
        sys.exit(0)

    # If an IMPLEMENTATION stage is live, stage-awareness.py handles context.
    if _has_implementation_stage():
        sys.exit(0)

    state = _load_state()
    if _in_cooldown(state):
        sys.exit(0)

    state["last_fired_at"] = _now().isoformat()
    _save_state(state)
    print(DIRECTIVE, file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
