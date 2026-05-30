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

# Single source of truth for the rigor pillars (shared with the ExitPlanMode
# gate + Stop backstop). Fall back to a literal list if the module is missing so
# the cue still fires — the gate is what enforces; this is only the reminder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from _plan_rigor import RIGOR_PILLARS as _PILLARS

    _PILLAR_LABELS = ", ".join(label for label, _ in _PILLARS)
except Exception:
    _PILLAR_LABELS = (
        "no-bias/no-look-ahead, honesty (verified vs claimed), literature grounding, "
        "edge cases (NULL/empty/sparse/failure), future-proofing/hardening"
    )

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
# ANY plan/design intent: mandatory 2-pass + rigor fold-in. First drafts are
# always wrong, so this fires whenever a plan is made OR worked on — it is NOT
# gated behind "improve"/"check" (the operator must not have to remember to ask).
PLAN_INTENT = re.compile(
    r"\b(plan|planning|design|brainstorm|approach|prereg|pre[- ]?reg|hypothesis|"
    r"spec|proposal|memo|architecture|strategy for|how should we|how do we|4t)\b",
    re.IGNORECASE,
)
# "improve / tighten / harden / gaps / silences" with no concrete code object
# almost always means the plan on the table — escalate those to the PLAN route so
# "do it properly when I say it" holds even without the word "plan" in the prompt.
PLAN_IMPROVE_VERB = re.compile(
    r"\b(improve|tighten|strengthen|harden|gaps?|silences?)\b", re.IGNORECASE
)
CODE_OBJECT = re.compile(
    r"\b(function|func|method|file|test|tests|bug|query|class|module|hook|script|"
    r"regex|import|commit|diff|line|\w+\.py)\b",
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
PLAN_RIGOR_MESSAGE = (
    "PLAN ROUTE (mandatory, every plan): the first draft is wrong — do a 2nd pass before presenting. "
    "Pass 1 draft → Pass 2 self-critique for gaps/silences/bias/error/simpler path, then FOLD IN a rigor section covering: "
    + _PILLAR_LABELS
    + ". Be terse. Never present a single-pass plan — plan-rigor-gate.py will bounce it if you do."
)


PLAN_INTENT_BREADCRUMB = Path(__file__).resolve().parent / "state" / "plan-intent.json"


def _load_event() -> dict[str, object]:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8-sig", errors="replace").strip()
        if not raw:
            return {}
        event = json.loads(raw)
        return event if isinstance(event, dict) else {}
    except Exception:
        return {}


def _drop_plan_breadcrumb(session_id: str) -> None:
    """Mark that THIS prompt asked for a plan, so the Stop backstop can fire
    precisely (on known intent) instead of guessing plan-shape from prose.
    Keyed on session_id + a monotonic turn counter; the Stop hook consumes and
    clears it. Fail-open: a write error simply means the backstop won't fire."""
    try:
        PLAN_INTENT_BREADCRUMB.parent.mkdir(parents=True, exist_ok=True)
        prev = {}
        if PLAN_INTENT_BREADCRUMB.exists():
            try:
                prev = json.loads(PLAN_INTENT_BREADCRUMB.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
        turn = int(prev.get("turn", 0)) + 1 if prev.get("session_id") == session_id else 1
        PLAN_INTENT_BREADCRUMB.write_text(
            json.dumps({"session_id": session_id, "turn": turn, "pending": True}),
            encoding="utf-8",
        )
    except Exception:
        pass


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
    plan_hit = PLAN_INTENT.search(prompt)
    # bare improve/tighten/gaps verb with no code object → treat as plan work
    improve_plan = PLAN_IMPROVE_VERB.search(prompt) and not CODE_OBJECT.search(prompt)
    if plan_hit or improve_plan:
        messages.append(PLAN_RIGOR_MESSAGE)
        _drop_plan_breadcrumb(str(event.get("session_id") or ""))
    elif len(prompt.split()) >= 2 and ACTION_INTENT.search(prompt):
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
