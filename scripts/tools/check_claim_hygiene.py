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

    if issues:
        print("CLAIM HYGIENE CHECK FAILED")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Claim hygiene OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
