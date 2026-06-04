#!/usr/bin/env python3
"""Intent-router — UserPromptSubmit only.

Reads `.claude/rules/auto-skill-routing.md` and matches the user's prompt
against the Intent Map / CRG section. On match, injects a single
`additionalContext` line naming the recommended skill. Never blocks.

Design:
- Deterministic regex, never LLM.
- Fail-open: any error → exit 0 silent.
- Cooldown-gated (30s global, 5min per skill) — same pattern as
  discovery-loop-guard.py:36.
- Token budget: ≤ 100 chars per fire.
- `auto-skill-routing.md` existence is checked as an on/off kill-switch
  (RULES_FILE.exists()); the INTENT_RULES table is compiled from the
  hardcoded list below, not parsed from the file at runtime. Keep
  INTENT_RULES in sync with the rules file manually when either changes.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_FILE = PROJECT_ROOT / ".claude" / "rules" / "auto-skill-routing.md"
STATE_FILE = PROJECT_ROOT / ".claude" / "hooks" / "state" / "intent-router.json"
GLOBAL_COOLDOWN_S = 30
SKILL_COOLDOWN_S = 300
MAX_CONTEXT_CHARS = 100

# Hand-curated regex map mirroring auto-skill-routing.md Intent Map and CRG
# section. The TARGET LINE numbers reference the rules file so the operator
# can verify the routing source. Keep in sync with the rules file. If the
# rules file changes structurally, update this table.
#
# Order matters only as a stable tie-breaker; longest-literal-phrase still wins.
INTENT_RULES: list[tuple[str, str, int]] = [
    # Session start, status, catch-up → /orient
    (r"\b(catch( me)? up|where are we|orient me|orient)\b", "/orient", 12),
    (r"\b(session start|status|catch[- ]up)\b", "/orient", 12),
    # Continue, next, what now, keep going → /next
    (r"\b(what now|keep going|continue|what'?s next|next step)\b", "/next", 13),
    # Done / complete / finished → /verify done
    (r"\b(verify done|are we done|all done|finished)\b", "/verify done", 15),
    # Trading book / what's live / tonight / playbook → /trade-book
    (r"\b(trade ?book|what'?s live|tonight'?s book|playbook|live (lanes|book))\b", "/trade-book", 16),
    # Health / decay / fitness / regime → /regime-check
    (r"\b(fitness|decay|regime|portfolio health|regime check)\b", "/regime-check", 17),
    # Something wrong / weird / broken → /quant-debug
    (r"\b(doesn'?t add up|something wrong|something weird|something broken|why is .{1,30} failing)\b", "/quant-debug", 18),
    # Plan / design / brainstorm / approach / 4t → /design
    (r"\b(brainstorm|design this|plan this|4t|how should we (build|approach))\b", "/design", 19),
    # Past findings / history / NO-GO / remind me → /pinecone-assistant
    (r"\b(past finding|history of|remind me|why did we|what did we find)\b", "/pinecone-assistant", 20),
    # Kill-verdict lookup / graveyard / NO-GO on X → /nogo
    (r"\b(kill verdict|ruled out|graveyard|nogo|no[- ]go on)\b", "/nogo", 21),
    # Test a hypothesis / research / investigate edge → /research
    (r"\b(test a hypothesis|investigate (an )?edge|run research|new hypothesis)\b", "/research", 22),
    # Real-capital scrutiny / bias check / before deploy → /capital-review
    (r"\b(real[- ]?capital|bias check|before deploy|capital review|before going live)\b", "/capital-review", 23),
    # Review / check my work / before I commit → /code-review
    (r"\b(check my work|before I commit|code review this|review this (change|pr|commit))\b", "/code-review", 24),
    # Improve a skill → /skill-improve
    (r"\b(improve a? skill|optimize a? skill|skill loop)\b", "/skill-improve", 26),
    # CRG: where is X / what calls X / find Y / who imports Z → /crg-search
    # Widened to natural phrasing: "where's X", "where does X live", "show me X",
    # "which file has X", "what handles X". (2026-06-04 vocab tune)
    (r"\b(where is |where'?s |where does .{1,40} (live|defined|sit)|what calls |who imports |find the (function|class|symbol|code)|show me (the |where )|which file (has|holds|defines)|what handles )\b", "/crg-search", 32),
    # CRG: blast radius / before editing / impact → /crg-blast
    # Widened: "if I change/edit X what breaks", "safe to change/touch/delete X",
    # "what depends on/relies on X". (2026-06-04 vocab tune)
    (r"\b(blast radius|impact analysis|what (will|would) (this |it )?break|before (editing|I edit)|if I (change|edit|touch|delete|remove) .{1,50} (what|break)|safe to (change|edit|touch|delete|remove)|what depends on|what relies on)\b", "/crg-blast", 33),
    # CRG: predicate lineage / contamination → /crg-lineage
    # Widened to data-flow phrasing the operator actually uses: "what feeds into X",
    # "what's feeding X", "what uses/reads/writes X", "where does X come from",
    # "upstream/downstream of X", "trace X". (2026-06-04 vocab tune)
    (r"\b(predicate lineage|contamination|what consumes (feature|column)|what(?:'?s| is| does)? feed(s|ing)? (in)?to |what (uses|reads|writes|populates|derives) |where does .{1,40} come from|(up|down)stream of|trace (the |where |how )|lineage of)\b", "/crg-lineage", 34),
    # CRG: tests for X / what tests cover Y → /crg-tests
    # Widened: "what tests X", "is X tested", "test coverage". (2026-06-04 vocab tune)
    (r"\b(tests for |what tests (cover|exercise)?|test coverage|is .{1,40} tested|covered by (a )?test)\b", "/crg-tests", 35),
    # CRG: dead code / unused functions → /crg-deadcode
    # Widened: "is X used anywhere", "anything calling X", "still used". (2026-06-04 vocab tune)
    (r"\b(dead code|unused (functions?|code|imports?)|unreferenced (function|class)|is .{1,40} (still )?used (anywhere)?|anything (calling|using) |still (used|referenced|called))\b", "/crg-deadcode", 36),
]

# Compiled at import time. Stored as (compiled_regex, skill, line_no, raw_pattern_len)
# where raw_pattern_len is len of the raw pattern string used for longest-match
# disambiguation.
_COMPILED: list[tuple[re.Pattern[str], str, int, int]] = [
    (re.compile(raw, re.IGNORECASE), skill, line_no, len(raw))
    for raw, skill, line_no in INTENT_RULES
]


def _rules_file_present() -> bool:
    """Routing is authoritative only if the rules file exists. Fail-open if not."""
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
        pass  # fail-open: cooldown best-effort only


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
    """Return (skill, line_no) for the longest-pattern match, or None."""
    matches: list[tuple[re.Pattern[str], str, int, int]] = []
    for pat, skill, line_no, raw_len in _COMPILED:
        if pat.search(prompt):
            matches.append((pat, skill, line_no, raw_len))
    if not matches:
        return None
    # Longest raw pattern wins (more specific = longer literal phrase).
    pat, skill, line_no, _ = max(matches, key=lambda m: m[3])
    return skill, line_no


def _should_skip(prompt: str) -> bool:
    """Skip injection when the prompt is too short, empty, or already a slash command."""
    if not prompt:
        return True
    stripped = prompt.strip()
    if len(stripped.split()) < 2:
        return True
    # User already chose a skill explicitly.
    if stripped.startswith("/"):
        return True
    # Slash-command anywhere in first 30 chars also counts (covers
    # "please /orient now" style invocations).
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

    cue = f"Intent match → {skill}  ·  (auto-skill-routing.md L{line_no})"
    if len(cue) > MAX_CONTEXT_CHARS:
        cue = cue[: MAX_CONTEXT_CHARS - 1] + "…"

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
    except BaseException:  # pragma: no cover - fail-open hook contract
        sys.exit(0)
