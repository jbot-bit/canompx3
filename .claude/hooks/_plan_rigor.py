#!/usr/bin/env python3
"""Single source of truth for the plan-rigor contract.

Imported by every layer that enforces "every plan is 2-pass + carries a rigor
section": the prompt-cue router, the ExitPlanMode gate, and the Stop backstop.
Keeping the pillars in ONE place means the cue, the gate, the doc, and the
drift-parity check can never drift apart.

First drafts are always wrong. A plan that reaches the operator must show a 2nd
pass AND a rigor section covering all five pillars below.
"""

from __future__ import annotations

import re

# The five rigor pillars every plan must fold in. (label, detection-regex).
# The regex is what the gate scans the plan text for — keep it generous so a
# legitimately-worded plan is not bounced on phrasing alone (soft-block + fail-
# open keep a false positive cheap: it costs Claude a turn, never the operator).
RIGOR_PILLARS: list[tuple[str, str]] = [
    ("no-bias/no-look-ahead", r"no[- ]?look[- ]?ahead|look-?ahead|leakage|bias|contaminat|holdout|out[- ]?of[- ]?sample|oos"),
    ("honesty (verified vs claimed)", r"honest|verified vs|verified-vs|actually (ran|tested|verified)|evidence|no skim|claimed vs"),
    ("literature grounding", r"literatur|grounding|cite|citation|institutional|resources/|docs/institutional|@research-source"),
    ("edge cases", r"edge case|null|empty|sparse|failure mode|boundary|degenerate"),
    ("future-proofing/hardening", r"future[- ]?proof|harden|regression|durab|guard|backstop"),
]

# A plan must show the 2-pass discipline, not just assert it.
SECOND_PASS_MARKER = re.compile(
    r"pass\s*2|second pass|2nd pass|2-pass|two[- ]pass|self[- ]critique|re-?reviewed|on re-?read",
    re.IGNORECASE,
)

# Performative-honesty tripwire: claims rigor/verification but shows no evidence
# block. Mirrors CLAUDE.md: "performative self-review ... is worse than none."
CLAIMS_VERIFIED = re.compile(
    r"\b(verified|2-?pass done|second pass done|self-?review(ed)?|checked|audited)\b", re.IGNORECASE
)
SHOWS_EVIDENCE = re.compile(
    r"(found:|pass 2 found|tested:|i ran|output:|=>|✓|✗|actual vs expected|drift \d|\d+/\d+ (tests|pass))",
    re.IGNORECASE,
)


def missing_pillars(text: str) -> list[str]:
    """Return labels of rigor pillars absent from ``text``."""
    return [label for label, pat in RIGOR_PILLARS if not re.search(pat, text, re.IGNORECASE)]


def has_second_pass(text: str) -> bool:
    return bool(SECOND_PASS_MARKER.search(text))


def is_performative(text: str) -> bool:
    """True if the text claims verification/rigor but shows no evidence of it."""
    return bool(CLAIMS_VERIFIED.search(text)) and not bool(SHOWS_EVIDENCE.search(text))


# NOTE (2026-05-31): a prose-shape `looks_like_plan()` heuristic was tried and
# removed. Reports and plans share vocabulary ("I'll", "done", "plan", bullets),
# so keyword classification false-fired on its own author's completion report and
# could not be made robust without breaking real plans. The Stop backstop now
# keys on KNOWN intent (a per-turn breadcrumb the UserPromptSubmit router drops
# when the PLAN route fires) instead of guessing plan-shape. See
# plan-stop-backstop.py and targeted-grounding-router.py.


def audit(text: str) -> dict[str, object]:
    """Full rigor verdict for a candidate plan."""
    miss = missing_pillars(text)
    return {
        "missing_pillars": miss,
        "second_pass": has_second_pass(text),
        "performative": is_performative(text),
        "ok": not miss and has_second_pass(text) and not is_performative(text),
    }
