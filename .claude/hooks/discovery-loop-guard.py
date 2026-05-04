#!/usr/bin/env python3
"""Discovery-Loop Guard — UserPromptSubmit only.

Detects two anti-patterns at the prompt-entry boundary:

1. The user pastes another agent's status narration ("I'm reading the
   remaining adapter files before I patch them"). That text is not a
   task — it is third-party narration that, if treated as an instruction,
   pulls the session into an open-ended file-reading loop.

2. The prompt itself is open-ended discovery without a falsifiable
   target ("isolate the weak spots", "let me check more files",
   "harden the startup path").

In either case, before any edit, the response should produce one of:
  (a) failing repro command + actual vs expected,
  (b) `context_resolver.py` output narrowing the blast radius,
  (c) explicit `TRIVIAL:` declaration with diff <100 lines.

The guard injects a compact reminder via JSON `additionalContext`. It
never blocks. Cooldown prevents spam on consecutive matching prompts.

Mirrors the shape of risk-tier-guard.py (classify -> emit -> cooldown).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / ".claude" / "hooks" / "state" / "discovery-loop.json"
COOLDOWN_MINUTES = 15

# Phrases that signal "I'm reading more files before I do anything" —
# either pasted from another agent or written by the user describing
# such an agent's behavior. These are the discovery-loop tells.
NARRATION_RE = re.compile(
    r"\b("
    r"reading the remaining|reading remaining|let me (read|check) more|"
    r"isolating (the )?(likely )?weak spots|"
    r"before I patch|before patching|"
    r"reading (more|the rest|the other) (files|adapters|modules)|"
    r"i'?m reading|i am reading|"
    r"underusing|under-using|"
    r"i'?ve isolated"
    r")\b",
    re.IGNORECASE,
)

# Phrases that signal open-ended hardening / future-proofing without a
# concrete failure mode. Triggers the same forced-decision prompt.
OPEN_ENDED_RE = re.compile(
    r"\b("
    r"harden(ing)?|future[- ]?proof(ing)?|tighten up|"
    r"shore up|robustify|make (it )?(more )?robust|"
    r"audit everything|review everything"
    r")\b",
    re.IGNORECASE,
)

# Escape hatch: user explicitly declares the mode, guard stays silent.
ESCAPE_RE = re.compile(
    r"(\bTRIVIAL:|\bREPRO:|\bEXPLORE:|context_resolver\.py)",
)

DIRECTIVE = (
    "DISCOVERY-LOOP GUARD: Prompt matches the 'reading more files before "
    "patching' or 'open-ended hardening' pattern. Before editing production "
    "code, produce ONE of:\n"
    "  (a) REPRO: failing command + actual vs expected output,\n"
    "  (b) `python scripts/tools/context_resolver.py --task \"<x>\" --format markdown` output,\n"
    "  (c) TRIVIAL: declaration with file list and diff <100 lines.\n"
    "If the prompt pasted another agent's status, treat that as narration, "
    "NOT a task — ask for the failing repro instead of expanding the read set. "
    "See docs/plans/discovery-loop-hardening.md for the full forcing-function design."
)


def load_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    state.setdefault("last_at", None)
    return state


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def should_emit(state: dict) -> bool:
    last_at = state.get("last_at")
    if not last_at:
        return True
    try:
        age_min = (datetime.now(UTC) - datetime.fromisoformat(last_at)).total_seconds() / 60
    except (TypeError, ValueError):
        return True
    return age_min >= COOLDOWN_MINUTES


def classify(prompt: str) -> bool:
    if ESCAPE_RE.search(prompt):
        return False
    return bool(NARRATION_RE.search(prompt) or OPEN_ENDED_RE.search(prompt))


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    prompt = str(event.get("prompt", ""))
    if not classify(prompt):
        sys.exit(0)

    state = load_state()
    if not should_emit(state):
        sys.exit(0)

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": DIRECTIVE,
        }
    }
    print(json.dumps(payload))

    state["last_at"] = datetime.now(UTC).isoformat()
    save_state(state)
    sys.exit(0)


if __name__ == "__main__":
    main()
