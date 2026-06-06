#!/usr/bin/env python3
"""Low-cost claim hygiene checks for PRs and result docs.

This script is intentionally lightweight:
- String/regex checks only
- No network
- No DB
- Meant for hooks/CI, not deep semantic judgment
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PR_REQUIRED_SECTIONS = (
    "## Evidence",
    "## Claims",
    "## Disconfirming Checks",
    "## Grounding",
)

RESULT_DOC_RULES = {
    "scope": (
        re.compile(
            r"(?im)^("
            r"##\s+(Scope|Scope definitions|Goal|Question|Context|Motivation|"
            r"What this audit tests|The question|Why this test|Audited claim)\b|"
            r"\*\*Scope:\*\*"
            r")"
        ),
        "missing scope / question section",
    ),
    "decision": (
        re.compile(
            r"(?im)^("
            r"##\s+(Verdict|Recommendations|Decision|Executive Verdict|"
            r"Bottom Line|Closing verdict|Closing honest verdict|"
            r"Consolidated verdict|Final stress-test verdict)\b|"
            r"\*\*Outcome:\*\*"
            r")"
        ),
        "missing verdict / decision section",
    ),
    "repro": (
        re.compile(r"(?im)^##\s+(Reproduction|Reproducibility|Outputs|Files|Audit trail|Artefact list)\b"),
        "missing reproduction / outputs section",
    ),
    "skepticism": (
        re.compile(
            r"(?im)^##\s+("
            r"Disconfirming Checks|Failure Modes|Limitations|Guardrails|Caveats|"
            r"Falsification battery|What this audit does NOT do|"
            r"Not done\b|Not done by this re-audit\b|"
            r"Fairly tested vs prematurely ruled out\b|"
            r"Where we're tunnel-visioned\b|"
            r"Non-scope / deferred|Inherited methodology caveat|"
            r"What Is Overstated|Blocker Audit|Honest self-critique of this investigation"
            r")\b"
        ),
        "missing caveat / disconfirming / limitations section",
    ),
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --- Statistical-claim literature-anchor gate (repo-wide, staged-only) ---------
#
# Enforces the institutional-rigor canon (.claude/rules/institutional-rigor.md S7
# and .claude/skills/code-review/SKILL.md Sections A/C) at COMMIT time: a doc that
# makes a Sharpe / t-stat / significance / DSR / MinBTL / edge claim must either
# cite a literature extract / criterion / executed result path OR explicitly tag
# the claim's grounding state (MEASURED / UNSUPPORTED / INFERRED). This makes the
# seven-sins doctrine ENFORCED, not advisory — future slop cannot enter main
# without being grounded.
#
# Scope is the STAGED set only (the pre-commit passes changed files) — historical
# docs are grandfathered, only new/edited claims must be grounded. No retroactive
# landmine across the ~1044 existing result docs.

# Folders whose staged .md files are subject to the stat-claim gate. Repo-wide
# across the research/results/doctrine surfaces — not pigeonholed to results/.
_STAT_CLAIM_DIRS = (
    ("docs", "audit"),
    ("docs", "institutional"),
    ("docs", "plans"),
    ("research",),
)

# A doc "makes a stat claim" if it asserts a Sharpe/significance/edge NUMBER, or
# uses the multiple-testing-correction vocabulary (DSR/MinBTL/deflated) which is
# itself a quantitative claim. Bare methodology prose ("we should compute the
# Sharpe ratio") does NOT fire — a number must be adjacent — to avoid false
# positives on design docs.
_STAT_CLAIM_PATTERN = re.compile(
    r"(?im)("
    r"\bsharpe\s*(ratio)?\s*[=:]?\s*-?\d|"  # Sharpe = 1.2 / Sharpe 1.2
    r"\bdeflated\s+sharpe\b|\bdsr\b|\bminbtl\b|"  # MHT-correction vocabulary = a claim
    r"\bt[-_ ]?stat(istic)?\s*[=:]?\s*-?\d|"  # t-stat = 3.1
    r"\bt\s*[=≥>]\s*-?\d(\.\d+)?|"  # t = 2.0 / t >= 3.79
    r"\bp[-_ ]?value\s*[=<:]\s*-?0?\.\d|"  # p-value < 0.05
    r"\bp\s*[=<]\s*0?\.\d|"  # p < 0.05
    r"\bstatistically\s+significan|"  # "statistically significant"
    r"\bedge\s*(of|=|:)?\s*\$?-?\d"  # edge = $27 / edge of 1.2
    r")"
)

# A stat claim is GROUNDED if the doc anywhere cites a literature extract, a
# criterion, an executed result path, a provenance annotation, a named canon
# author, OR tags the grounding state. Any one satisfies the gate.
_STAT_ANCHOR_PATTERN = re.compile(
    r"(?im)("
    r"docs/institutional/literature/|institutional/literature/|"  # literature extract
    r"\bLIT_[A-Za-z0-9]|"  # LIT_* extract id
    r"pre_registered_criteria|criterion\s*\d|"  # locked criterion
    r"docs/audit/results/|\.json\b|\.csv\b|\.parquet\b|"  # executed result path
    r"@research-source|@entry-models|@revalidated-for|"  # provenance annotation
    r"\b(UNSUPPORTED|MEASURED|INFERRED)\b|FROM TRAINING MEMORY|"  # grounding tag
    # named canon authors — the seven-sins anchors
    r"\b(bailey|l[oó]pez[ -]?de[ -]?prado|prado|harvey|liu|chordia|aronson|"
    r"carver|fitschen|pepelyshev|harris)\b"
    r")"
)


def _in_stat_claim_scope(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    if path.suffix.lower() != ".md":
        return False
    return any(rel.parts[: len(prefix)] == prefix for prefix in _STAT_CLAIM_DIRS)


def check_stat_claim_anchor(path: Path) -> list[str]:
    """A staged doc making a Sharpe/significance/edge claim must cite a literature
    anchor / criterion / result path or tag its grounding state. Staged-only."""
    text = _read_text(path)
    if not _STAT_CLAIM_PATTERN.search(text):
        return []
    if _STAT_ANCHOR_PATTERN.search(text):
        return []
    return [
        f"{path}: makes a Sharpe/t-stat/significance/edge claim with no literature "
        f"anchor and no grounding tag. Cite docs/institutional/literature/<extract> "
        f"or a pre_registered_criteria criterion or an executed result path, OR tag "
        f"the claim MEASURED / UNSUPPORTED / INFERRED. "
        f"See .claude/rules/institutional-rigor.md S7."
    ]


def check_pr_body(body: str) -> list[str]:
    return [f"PR body missing required section: {section}" for section in PR_REQUIRED_SECTIONS if section not in body]


def _is_result_doc(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    return rel.parts[:3] == ("docs", "audit", "results") and path.suffix.lower() == ".md"


def check_result_doc(path: Path) -> list[str]:
    text = _read_text(path)
    issues: list[str] = []
    for _, (pattern, message) in RESULT_DOC_RULES.items():
        if not pattern.search(text):
            issues.append(f"{path}: {message}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check low-cost claim hygiene requirements")
    parser.add_argument("--pr-body-env", help="Environment variable containing PR body")
    parser.add_argument("files", nargs="*", help="Files to check")
    args = parser.parse_args(argv)

    issues: list[str] = []

    if args.pr_body_env:
        body = os.environ.get(args.pr_body_env, "")
        issues.extend(check_pr_body(body))

    for raw in args.files:
        path = Path(raw)
        if not path.exists():
            continue
        if _is_result_doc(path):
            issues.extend(check_result_doc(path))
        if _in_stat_claim_scope(path):
            issues.extend(check_stat_claim_anchor(path))

    if issues:
        print("CLAIM HYGIENE CHECK FAILED")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Claim hygiene OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
