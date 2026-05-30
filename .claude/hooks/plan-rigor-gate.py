#!/usr/bin/env python3
"""PreToolUse:ExitPlanMode — refuse a plan that skipped rigor.

The v1 prompt cue only *reminds* Claude to 2-pass and fold in rigor. This gate
*enforces* it at the moment the plan is presented: it reads the plan text out of
``tool_input.plan``, audits it for the five rigor pillars + a 2nd-pass marker +
performative-honesty, and SOFT-BLOCKS (exit 2) if anything is missing — handing
the plan back to Claude to fix before the operator ever sees it.

Design contract (matches every other guard in this repo):
- Soft block, never hard: exit 2 returns control to Claude, costs one turn, and
  never wedges the operator.
- Fail-open: any parse error, missing field, wrong tool, or empty plan exits 0.
  The gate never blocks a plan it cannot read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from _plan_rigor import audit  # noqa: E402
except Exception:
    # Shared module unavailable -> fail open, do not block.
    sys.exit(0)


def _load_event() -> dict:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8-sig", errors="replace").strip()
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def main() -> int:
    event = _load_event()
    if event.get("tool_name") != "ExitPlanMode":
        return 0
    plan = str(event.get("tool_input", {}).get("plan") or "").strip()
    if len(plan.split()) < 30:
        # Too short to be a real plan, or empty — fail open.
        return 0

    verdict = audit(plan)
    if verdict["ok"]:
        return 0

    problems: list[str] = []
    if verdict["missing_pillars"]:
        problems.append("rigor section missing pillars: " + ", ".join(verdict["missing_pillars"]))
    if not verdict["second_pass"]:
        problems.append("no 2nd-pass shown (first drafts are always wrong — show Pass 2)")
    if verdict["performative"]:
        problems.append("claims verification but shows no evidence (performative self-review)")

    msg = (
        "PLAN RIGOR GATE — do not present yet. "
        + "; ".join(problems)
        + ". Fold in the missing rigor (no-bias/no-look-ahead, honesty with evidence, "
        "literature grounding, edge cases, future-proofing/hardening), show the 2nd pass, "
        "then re-present. This is the contract in .claude/rules/targeted-grounding.md."
    )
    # exit 2 + stderr -> soft block: returned to Claude, never to the operator.
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        # Absolute fail-open backstop.
        sys.exit(0)
