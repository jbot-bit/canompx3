#!/usr/bin/env python3
"""Decision Governor — read-only composer for the research/decision front-door.

Implements the routing + checklist defined by
``docs/governance/research_decision_governor.md`` (the *contract* doc) and its
fill-in companion ``docs/audit/templates/decision_candidate_review.md``.

WHAT THIS IS (and is NOT):
  - It is a **router and checklist composer**. Given a candidate and its
    decision class(es), it prints the sec.4 question subset that applies, the
    canonical grounding command for each, and the live values it can resolve
    *cheaply and read-only*.
  - It **re-encodes no threshold** (``institutional-rigor.md`` "no re-encoded
    canonical logic"). It owns no number, budget, t-stat, or K-bound. Where a
    grounding step is expensive (account survival, K-budget, the 14-gate live
    preflight), it prints the EXACT command to run rather than re-running it —
    a second copy of canonical execution is a second thing that drifts.
  - Cheap, read-only pointers it *can* resolve are surfaced live by importing
    the canonical surface (never re-deriving it): the sec.2 layer table and the
    Q13 open-hypothesis count via ``research_catalog_mcp_server._list_open_hypotheses``.

DESIGN INVARIANTS (mirrored by ``check_decision_governor_pointers_resolve`` in
``pipeline/check_drift.py``):
  - The governor doc + template exist.
  - The doc states no numeric threshold (the drift check guards this).
  - Every canonical pointer the governor names resolves on disk / imports.

Exit code is always 0 (read-only advisory tool). Failure to resolve a cheap
pointer degrades to a printed NOTE, never a crash — the drift check is the
fail-closed half; this tool is the human/agent-facing composer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOVERNOR_DOC = PROJECT_ROOT / "docs" / "governance" / "research_decision_governor.md"
TEMPLATE_DOC = PROJECT_ROOT / "docs" / "audit" / "templates" / "decision_candidate_review.md"

# ---------------------------------------------------------------------------
# sec.3 decision-class router. Each class maps to the sec.4 question subset it
# requires. Q12 (which class) and Q13 (anti-tunnel) apply to EVERY candidate —
# they are appended for every class, never exempted. Kept in lockstep with the
# governor doc sec.3 table; the drift check asserts the doc still carries each
# class label so a silent rename here is caught.
# ---------------------------------------------------------------------------
DECISION_CLASSES: dict[str, list[str]] = {
    "research-validation": ["Q1", "Q3", "Q4", "Q5", "Q6", "Q8", "Q9"],
    "portfolio": ["Q1", "Q7", "Q10", "Q13"],
    "account-sizing": ["Q11", "Q13"],
    "deployment-gate": ["Q11"],  # + the 14 preflight gates (L6), printed below
    "classification": ["Q2"],
}

# Q12 + Q13 always run (Q13 is the anti-tunnel guard with no exemption).
ALWAYS = ["Q12", "Q13"]

# ---------------------------------------------------------------------------
# sec.4 question set. Each question pairs with its canonical GROUNDING COMMAND.
# The command is the source of truth — the governor prints it; it does not
# re-run the expensive ones. No thresholds appear here (drift-guarded).
# ---------------------------------------------------------------------------
QUESTIONS: dict[str, tuple[str, str]] = {
    "Q1": (
        "What is the claimed edge (ExpR / Sharpe, N)?",
        "query validated_setups row / scan output for the candidate tuple",
    ),
    "Q2": (
        "Which role is it? (R1-R8: FILTER / direction / size / stop / target / entry / confluence / allocator)",
        "read docs/institutional/mechanism_priors.md sec.4 (R1-R8 taxonomy)",
    ),
    "Q3": (
        "Already killed or parked?",
        "/nogo <topic>  (research-catalog)  +  docs/STRATEGY_BLUEPRINT.md NO-GO registry",
    ),
    "Q4": (
        "Mechanism real or story?",
        "read docs/institutional/mechanism_priors.md sec.2 + the theory_citation extract",
    ),
    "Q5": (
        "Knowable strictly before entry?",
        "check the banned-lookahead list (.claude/rules/backtesting-methodology.md sec.1.1/6.3)",
    ),
    "Q6": (
        "Uses any banned lookahead field?",
        "/crg-lineage <column>   (flag break_ts / break_delay_min / double_break / mae_r / rel_vol_*)",
    ),
    "Q7": (
        "Duplicates / correlates an existing filter?",
        "apply backtesting-methodology.md RULE 7 tautology check (correlation ceiling owned there)",
    ),
    "Q8": (
        "Honest K / trial count?",
        "research-catalog estimate_k_budget  (MinBTL; Criterion 2 bound)",
    ),
    "Q9": (
        "Survives the locked criteria? (MinBTL / BH-FDR / Chordia t / WFE / N / OOS / era / cost)",
        "run trading_app/strategy_validator.py against pre_registered_criteria.md (Criteria 2,3,4,6,7,8,9)",
    ),
    "Q10": (
        "Improves PORTFOLIO EV after correlation + drawdown constraints?",
        "strategy-lab get_lane_allocation_summary  (read staleness)",
    ),
    "Q11": (
        "Passes ACCOUNT SURVIVAL at the target profile?",
        'python -c "from trading_app.account_survival import evaluate_profile_survival" '
        "(write_state=False)  OR  scripts/tools/live_readiness_report.py --profile-id <id>",
    ),
    "Q12": (
        "Which decision class(es) is this?",
        "this tool -- route first (governor doc sec.3)",
    ),
    "Q13": (
        "What higher-EV open item are we IGNORING by doing this?",
        "research-catalog list_open_hypotheses (open count, resolved live below) + live blocker list",
    ),
}

# Extra grounding the deployment-gate class needs beyond its question subset.
DEPLOYMENT_EXTRA = (
    "+ the live preflight gates (count = len(PREFLIGHT_CHECKS)): "
    "scripts/run_live_session.py --instrument <SYM> --preflight   (read-only dry-run)"
)


def _resolve_open_hypothesis_count() -> str:
    """Cheap, read-only Q13 grounding. DELEGATES to the research-catalog MCP's
    own ``_list_open_hypotheses`` (the canonical open/closed heuristic) — never
    re-counts hypotheses here, so the two can't drift. Degrades to a NOTE on any
    import/IO failure; this tool is advisory, the drift check is fail-closed."""
    tools_dir = PROJECT_ROOT / "scripts" / "tools"
    added = False
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
        added = True
    try:
        import research_catalog_mcp_server as rc  # type: ignore

        result = rc._list_open_hypotheses(limit=1)
        total = result.get("total_open_hypotheses")
        basis = result.get("status_basis", "")
        return f"{total} open hypotheses  ({basis})"
    except Exception as e:  # noqa: BLE001 — advisory tool, never crash on a read
        return (
            f"NOTE: could not resolve open-hypothesis count "
            f"({type(e).__name__}: {e}). Run `research-catalog list_open_hypotheses` directly."
        )
    finally:
        if added:
            try:
                sys.path.remove(str(tools_dir))
            except ValueError:
                pass


def _classes_for(requested: list[str]) -> list[str]:
    """Resolve requested class names, or all classes if none/'all' given."""
    if not requested or "all" in requested:
        return list(DECISION_CLASSES)
    unknown = [c for c in requested if c not in DECISION_CLASSES]
    if unknown:
        valid = ", ".join(DECISION_CLASSES)
        raise SystemExit(f"Unknown decision class(es): {unknown}. Valid: {valid}, or 'all'.")
    return requested


def _questions_for(classes: list[str]) -> list[str]:
    """Union of each class's question subset + the always-on Q12/Q13, in Q-order."""
    needed: set[str] = set(ALWAYS)
    for cls in classes:
        needed.update(DECISION_CLASSES[cls])
    return sorted(needed, key=lambda q: int(q[1:]))


def compose(candidate: str, requested_classes: list[str]) -> str:
    classes = _classes_for(requested_classes)
    questions = _questions_for(classes)
    open_hyp = _resolve_open_hypothesis_count()

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("DECISION GOVERNOR -- composed checklist (read-only)")
    lines.append("=" * 72)
    lines.append(f"Candidate : {candidate or '<unnamed>'}")
    lines.append(f"Class(es) : {', '.join(classes)}")
    lines.append("")
    lines.append("Governor : docs/governance/research_decision_governor.md")
    lines.append("Template : docs/audit/templates/decision_candidate_review.md")
    lines.append("  -> copy the template to a dated file and PASTE live command")
    lines.append("     output under each question. An asserted answer is not an answer.")
    lines.append("")
    lines.append("-" * 72)
    lines.append("REQUIRED QUESTIONS (run only your class subset; Q12 & Q13 always):")
    lines.append("-" * 72)
    for q in questions:
        text, grounding = QUESTIONS[q]
        lines.append(f"{q}. {text}")
        lines.append(f"     ground: {grounding}")
        if q == "Q11" and "deployment-gate" in classes:
            lines.append(f"     {DEPLOYMENT_EXTRA}")
        if q == "Q13":
            lines.append(f"     live  : {open_hyp}")
        lines.append("")
    lines.append("-" * 72)
    lines.append("ANTI-TUNNEL (Q13): the open-hypothesis pool above IS the")
    lines.append("opportunity set you trade against by doing this candidate. If a")
    lines.append("higher-EV open item has fewer blockers, STOP and reconsider.")
    lines.append("-" * 72)
    lines.append("")
    lines.append("VERDICT = union of layer blockers + the explicit Q13 answer.")
    lines.append("This tool COMPOSES; the gates own the thresholds. Re-verify all")
    lines.append("live state (blockers move) before acting.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compose the decision-governor checklist for a candidate (read-only).",
    )
    parser.add_argument(
        "--candidate",
        default="",
        help="One-line description of the proposed change.",
    )
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        default=[],
        help=("Decision class (repeatable): " + ", ".join(DECISION_CLASSES) + ", or 'all'. Omit to route all classes."),
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="Print the decision classes and their question subsets, then exit.",
    )
    args = parser.parse_args(argv)

    if args.list_classes:
        for cls in DECISION_CLASSES:
            print(f"{cls:20s} -> {', '.join(_questions_for([cls]))}")
        return 0

    if not GOVERNOR_DOC.exists() or not TEMPLATE_DOC.exists():
        print(
            "NOTE: governor doc and/or template missing — composing from the "
            "in-code routing table only. Restore "
            "docs/governance/research_decision_governor.md + "
            "docs/audit/templates/decision_candidate_review.md.",
            file=sys.stderr,
        )

    print(compose(args.candidate, args.classes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
