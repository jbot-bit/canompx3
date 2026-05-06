"""Claude-side review gate for the DeepSeek/OpenCode coding agent.

Pre-commit step 0d invokes this script when the launcher exports
``OPENCODE_AGENT_ACTIVE=1``. The reviewer reads the staged diff, asks
Claude (seven-sins rubric) for a verdict, and fails the commit on BLOCK.

Exit codes:
- 0 = APPROVE or no diff to review (nothing to gate).
- 1 = BLOCK (Claude returned findings; commit aborted).
- 2 = REVIEW_UNAVAILABLE (network/parse error; user can `--no-verify`
       once with explicit acknowledgement; not silent).

Per institutional-rigor §4 the script delegates to
``trading_app.ai.claude_client.get_client`` (single source of truth for
Claude model IDs) and never instantiates ``anthropic.Anthropic`` directly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Inlined verbatim from `.claude/rules/quant-agent-identity.md` § Seven Sins
# Awareness so the script is hermetic. If the canonical file moves or its
# wording changes, update this constant in lock-step (drift check #137 will
# catch the rule-doc move; this constant is reviewed at PR time).
_SEVEN_SINS_RUBRIC = """\
Seven Sins of Quantitative Investing (review checklist):

1. Look-ahead bias — using future data as a predictor (e.g. `double_break`
   is look-ahead; LAG() without `WHERE orb_minutes = 5`).
2. Data snooping — claiming significance after testing many hypotheses
   without BH FDR correction.
3. Overfitting — high Sharpe but N < 30, or single-year passes.
4. Survivorship bias — ignoring dead instruments (MCL/SIL/M6E/MBT/M2K)
   or purged entry models (E0).
5. Storytelling bias — narrative around noise; if p > 0.05 it's an
   observation, not a finding.
6. Outlier distortion — single extreme day driving aggregates; check
   year-by-year breakdown.
7. Transaction cost illusion — ignoring spread + slippage + commission.
   Use COST_SPECS from pipeline/cost_model.py.

Plus institutional-rigor sins:
A. Re-encoding canonical logic (filters, sessions, costs, paths) instead
   of importing from the canonical source.
B. Silent failures (bare `except Exception` returning success).
C. Hardcoded research stats (p-values, N counts) without
   @research-source / @revalidated-for annotations.
"""

_SYSTEM_PROMPT = f"""\
You are a senior quant code reviewer for a futures-trading research
pipeline. Review the staged diff against the rubric below. Be strict
about canonical-source delegation, fail-closed behavior, and statistical
rigor. The agent under review is an LLM coding agent (DeepSeek), so it
may produce plausible-looking but subtly wrong code.

{_SEVEN_SINS_RUBRIC}

Return STRICT JSON only — no prose before or after — matching this schema:
{{
    "verdict": "APPROVE" | "BLOCK",
    "findings": [
        {{
            "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
            "file": "<path>",
            "line": <int or 0 if file-level>,
            "sin": "<short tag from rubric, e.g. 'A. canonical-source delegation'>",
            "description": "<one sentence>"
        }}
    ]
}}

APPROVE only when there are zero CRITICAL/HIGH findings AND no clear
canonical-source-delegation violations. Stylistic nits are LOW; emit but
do not BLOCK on them.
"""

_USER_TEMPLATE = """\
Review this staged diff. Return strict JSON per schema.

Diff:
```diff
{diff}
```
"""


def _staged_diff() -> str:
    # Inherit caller's cwd: pre-commit invokes this script from the repo
    # being committed, which may be the canompx3 worktree, a sibling
    # worktree, or a test fixture. Pinning to _PROJECT_ROOT would always
    # read the canompx3 repo's diff regardless of where the hook fired.
    result = subprocess.run(
        ["git", "diff", "--cached"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _diff_is_doc_only(diff: str) -> bool:
    """True when every changed file is *.md or under docs/."""
    files: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            files.add(line[6:])
    if not files:
        return False
    return all(f.endswith(".md") or f.startswith("docs/") for f in files)


def _print_findings(payload: dict) -> None:
    findings = payload.get("findings") or []
    if not findings:
        print("  Verdict: BLOCK (no findings field)", file=sys.stderr)
        return
    print(f"  Verdict: {payload.get('verdict', '?')}", file=sys.stderr)
    for f in findings:
        sev = f.get("severity", "?")
        path = f.get("file", "?")
        line = f.get("line", 0)
        sin = f.get("sin", "?")
        desc = f.get("description", "?")
        print(f"  [{sev}] {path}:{line} ({sin}): {desc}", file=sys.stderr)


def _call_claude(diff: str) -> tuple[str, dict | None]:
    """Returns (status, parsed). status in {'APPROVE','BLOCK','UNAVAILABLE'}."""
    try:
        from trading_app.ai.claude_client import (
            CLAUDE_REASONING_MODEL,
            get_client,
        )
    except ImportError as exc:
        print(f"claude_review_deepseek: ImportError: {exc}", file=sys.stderr)
        return ("UNAVAILABLE", None)

    try:
        client = get_client()
    except ValueError as exc:
        print(f"claude_review_deepseek: {exc}", file=sys.stderr)
        return ("UNAVAILABLE", None)

    user_msg = _USER_TEMPLATE.format(diff=diff)
    try:
        response = client.messages.create(
            model=CLAUDE_REASONING_MODEL,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:  # noqa: BLE001  — network/SDK failure path
        print(f"claude_review_deepseek: API call failed: {exc}", file=sys.stderr)
        return ("UNAVAILABLE", None)

    text = ""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "")
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.startswith("```")).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"claude_review_deepseek: parse error: {exc}", file=sys.stderr)
        print(f"  raw: {text[:400]}", file=sys.stderr)
        return ("UNAVAILABLE", None)

    verdict = parsed.get("verdict", "BLOCK")
    if verdict == "APPROVE":
        return ("APPROVE", parsed)
    return ("BLOCK", parsed)


def _mock_response(rubric_pass: bool) -> dict:
    if rubric_pass:
        return {"verdict": "APPROVE", "findings": []}
    return {
        "verdict": "BLOCK",
        "findings": [
            {
                "severity": "HIGH",
                "file": "scripts/tools/opencode-agent.ps1",
                "line": 28,
                "sin": "A. canonical-source delegation",
                "description": "Hardcoded openrouter model literal without canonical-default-fallback annotation.",
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Use a mocked Claude response (for tests).")
    mock_group = parser.add_mutually_exclusive_group()
    mock_group.add_argument("--rubric-pass", action="store_true", help="Mock: rubric passes -> APPROVE.")
    mock_group.add_argument("--rubric-fail", action="store_true", help="Mock: rubric fails -> BLOCK.")
    args = parser.parse_args()

    if os.environ.get("OPENCODE_AGENT_ACTIVE") != "1":
        # Step 0d only fires when the launcher activates the gate; this is a
        # belt-and-braces no-op for direct invocations.
        return 0

    diff = _staged_diff()
    if not diff.strip():
        return 0

    if len(diff.splitlines()) < 5:
        return 0
    if _diff_is_doc_only(diff):
        return 0

    if args.mock:
        parsed = _mock_response(rubric_pass=args.rubric_pass or not args.rubric_fail)
        verdict = parsed["verdict"]
        if verdict == "APPROVE":
            return 0
        print("[claude_review_deepseek] BLOCK — staged diff failed review:", file=sys.stderr)
        _print_findings(parsed)
        return 1

    status, parsed = _call_claude(diff)
    if status == "UNAVAILABLE":
        print(
            "[claude_review_deepseek] REVIEW_UNAVAILABLE — Claude review could "
            "not run. Re-try, or commit with explicit `--no-verify` and "
            "self-review the diff before pushing.",
            file=sys.stderr,
        )
        return 2
    if status == "APPROVE":
        print("[claude_review_deepseek] APPROVE", file=sys.stderr)
        return 0

    print("[claude_review_deepseek] BLOCK — staged diff failed review:", file=sys.stderr)
    if parsed is not None:
        _print_findings(parsed)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
