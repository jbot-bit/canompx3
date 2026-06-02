#!/usr/bin/env python3
"""Intent-router - UserPromptSubmit only.

Reads `.claude/rules/auto-skill-routing.md` and matches the user's prompt
against the Intent Map / CRG section. On match, injects a single
`additionalContext` line naming the recommended skill. Never blocks.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_FILE = PROJECT_ROOT / ".claude" / "rules" / "auto-skill-routing.md"
STATE_FILE = PROJECT_ROOT / ".codex" / "hooks" / "state" / "intent-router.json"
GLOBAL_COOLDOWN_S = 30
SKILL_COOLDOWN_S = 300
MAX_CONTEXT_CHARS = 100

INTENT_RULES: list[tuple[str, str, int]] = [
    (r"\b(catch( me)? up|where are we|orient me|orient)\b", "/orient", 12),
    (r"\b(session start|status|catch[- ]up)\b", "/orient", 12),
    (r"\b(what now|keep going|continue|what'?s next|next step)\b", "/next", 13),
    (r"\b(verify done|are we done|all done|finished)\b", "/verify done", 15),
    (r"\b(trade ?book|what'?s live|tonight'?s book|playbook|live (lanes|book))\b", "/trade-book", 16),
    (r"\b(fitness|decay|regime|portfolio health|regime check)\b", "/regime-check", 17),
    (
        r"\b(doesn'?t add up|something wrong|something weird|something broken|why is .{1,30} failing)\b",
        "/quant-debug",
        18,
    ),
    (r"\b(brainstorm|design this|plan this|4t|how should we (build|approach))\b", "/design", 19),
    (r"\b(past finding|history of|remind me|why did we|what did we find)\b", "/pinecone-assistant", 20),
    (r"\b(kill verdict|ruled out|graveyard|nogo|no[- ]go on)\b", "/nogo", 21),
    (r"\b(test a hypothesis|investigate (an )?edge|run research|new hypothesis)\b", "/research", 22),
    (r"\b(real[- ]?capital|bias check|before deploy|capital review|before going live)\b", "/capital-review", 23),
    (r"\b(check my work|before I commit|code review this|review this (change|pr|commit))\b", "/code-review", 24),
    (r"\b(improve a? skill|optimize a? skill|skill loop)\b", "/skill-improve", 26),
    (r"\b(where is |what calls |who imports |find the (function|class|symbol))\b", "/crg-search", 32),
    (r"\b(blast radius|impact analysis|what will (this )?break|before (editing|I edit))\b", "/crg-blast", 33),
    (r"\b(predicate lineage|contamination|what consumes (feature|column))\b", "/crg-lineage", 34),
    (r"\b(tests for |what tests cover|test coverage of)\b", "/crg-tests", 35),
    (r"\b(dead code|unused functions?|unreferenced (function|class))\b", "/crg-deadcode", 36),
]

_COMPILED: list[tuple[re.Pattern[str], str, int, int]] = [
    (re.compile(raw, re.IGNORECASE), skill, line_no, len(raw))
    for raw, skill, line_no in INTENT_RULES
]


def _rules_file_present() -> bool:
    return RULES_FILE.exists()


def load_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            return {"last_global": None, "per_skill": {}}
        state.setdefault("last_global", None)
        state.setdefault("per_skill", {})
        if not isinstance(state["per_skill"], dict):
            state["per_skill"] = {}
        return state
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return {"last_global": None, "per_skill": {}}


def save_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def _seconds_since(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return (datetime.now(UTC) - datetime.fromisoformat(iso)).total_seconds()
    except (TypeError, ValueError):
        return None


def is_under_cooldown(state: dict, skill: str) -> bool:
    global_age = _seconds_since(state.get("last_global"))
    if global_age is not None and global_age < GLOBAL_COOLDOWN_S:
        return True
    per_skill = state.get("per_skill", {})
    skill_age = _seconds_since(per_skill.get(skill))
    if skill_age is not None and skill_age < SKILL_COOLDOWN_S:
        return True
    return False


def classify(prompt: str) -> tuple[str, int] | None:
    matches: list[tuple[re.Pattern[str], str, int, int]] = []
    for pat, skill, line_no, raw_len in _COMPILED:
        if pat.search(prompt):
            matches.append((pat, skill, line_no, raw_len))
    if not matches:
        return None
    _, skill, line_no, _ = max(matches, key=lambda item: item[3])
    return skill, line_no


def _should_skip(prompt: str) -> bool:
    if not prompt:
        return True
    stripped = prompt.strip()
    if len(stripped.split()) < 2:
        return True
    if stripped.startswith("/"):
        return True
    if "/" in stripped[:30] and re.search(r"/[a-z][a-z0-9_-]+", stripped[:30]):
        return True
    return False


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        return 0

    prompt = str(event.get("prompt", ""))
    if _should_skip(prompt):
        return 0
    if not _rules_file_present():
        return 0

    result = classify(prompt)
    if result is None:
        return 0
    skill, line_no = result

    state = load_state()
    if is_under_cooldown(state, skill):
        return 0

    cue = f"Intent match -> {skill}  ·  (auto-skill-routing.md L{line_no})"
    if len(cue) > MAX_CONTEXT_CHARS:
        cue = cue[: MAX_CONTEXT_CHARS - 3] + "..."

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": cue,
        }
    }
    print(json.dumps(payload))

    now = datetime.now(UTC).isoformat()
    state["last_global"] = now
    state["per_skill"][skill] = now
    save_state(state)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException:
        sys.exit(0)
