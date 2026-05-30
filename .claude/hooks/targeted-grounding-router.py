#!/usr/bin/env python3
"""Low-token grounding cue for common action/review prompts.

The user does not reliably remember to ask for tool routing, second-pass
checking, or explicit grounding. This hook injects one compact reminder when
the prompt asks for check/improve/implement/fix/review/plan style work.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_FILE = PROJECT_ROOT / ".claude" / "rules" / "targeted-grounding.md"

ACTION_INTENT = re.compile(
    r"\b(check|improve|implement|fix|build|change|edit|patch|review|audit|verify|plan|design|"
    r"hardening|cleanup|refactor|debug|investigate|diagnose|make (it|this|that) better|do it|go|"
    r"2p|second[- ]pass|two[- ]pass|double[- ]check|fresh eyes|another pass|sanity check|"
    r"red[- ]team|critique|stress[- ]test|take a look|look over|is this good|"
    r"what am i missing|poke holes|blind spots?|hold up|sense check|smell test|"
    r"thoughts on|qa|will this work|does this work|spot flaws?|find flaws?|gotchas?|risks?)\b",
    re.IGNORECASE,
)
RESOURCE_INTENT = re.compile(
    r"(^|\s)/(resource|resources|lit|literature)\b|\b(resource grounding|local literature|local corpus|"
    r"trading bible|trading bibles|canonical truth|grounding truth)\b",
    re.IGNORECASE,
)
RESEARCH_FETCH_INTENT = re.compile(
    r"\b(research|fetch|look up|lookup|search|find sources?|source this|documentation|docs|"
    r"official sources?|primary sources?|changelog|release notes?|upgrade|upgrades|fixes|"
    r"user comments?|forum|forums|reddit|github issues?|stackoverflow)\b",
    re.IGNORECASE,
)

MESSAGE = (
    "GROUNDING ROUTE: run a compact truth check first, then second-pass the plan/work for gaps, "
    "silences, bias, errors, and simpler improvements before acting."
)
RESOURCE_MESSAGE = (
    "RESOURCE ROUTE: local-PC corpus only. Run PDF/tooling + coverage checks; use extracts when covered, raw resources only if present locally; no skim/guess."
)
RESEARCH_FETCH_MESSAGE = (
    "SOURCE ROUTE: separate official/primary sources from unofficial user reports. Use official docs/changelogs/releases/code first; treat forums/issues/comments as cautionary signals unless corroborated."
)


def _load_event() -> dict[str, object]:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8-sig", errors="replace").strip()
        if not raw:
            return {}
        event = json.loads(raw)
        return event if isinstance(event, dict) else {}
    except Exception:
        return {}


def main() -> int:
    if not RULES_FILE.exists():
        return 0
    event = _load_event()
    prompt = str(event.get("prompt") or "")
    messages: list[str] = []
    if RESOURCE_INTENT.search(prompt):
        messages.append(RESOURCE_MESSAGE)
    if RESEARCH_FETCH_INTENT.search(prompt):
        messages.append(RESEARCH_FETCH_MESSAGE)
    if len(prompt.split()) >= 2 and ACTION_INTENT.search(prompt):
        messages.append(MESSAGE)
    if not messages:
        return 0
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "\n".join(messages),
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
