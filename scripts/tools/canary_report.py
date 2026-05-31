#!/usr/bin/env python3
"""Canary harness CAPITAL_READY verdict (deliverable #6).

Runs every Tier-1 canary and emits a single verdict:

    CAPITAL_READY      — every canary fired (every guard caught its trap); the
                         pipeline can reject every synthetic fake-edge class.
    NOT_CAPITAL_READY  — at least one guard FAILED to fire; lists each
                         (canary, guard, expected_signature) that regressed.

This is the operator-facing answer to "can canompx3 reliably kill fake edge?"
If any guard fails to fire on its trap, the blocking drift gate
(``check_canary_suite_green``) already blocks the commit; this report makes the
same fact human-legible and writes a dated audit artifact.

Exit code: 0 on CAPITAL_READY, 1 on NOT_CAPITAL_READY (CI-friendly).

Usage
-----
    python scripts/tools/canary_report.py            # verdict + table, write MD
    python scripts/tools/canary_report.py --no-write # verdict only, no artifact
    python scripts/tools/canary_report.py --json      # machine-readable verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tests.canary_suite import CanaryResult, run_canaries  # noqa: E402

CAPITAL_READY = "CAPITAL_READY"
NOT_CAPITAL_READY = "NOT_CAPITAL_READY"


def verdict(results: list[CanaryResult] | None = None) -> tuple[str, list[CanaryResult]]:
    """Return ``(verdict_str, failed_canaries)``.

    ``CAPITAL_READY`` iff every canary fired; otherwise ``NOT_CAPITAL_READY``
    with the list of canaries whose guard failed to fire.
    """
    if results is None:
        results = run_canaries()
    failed = [r for r in results if not r.fired]
    return (CAPITAL_READY if not failed else NOT_CAPITAL_READY), failed


def _render_markdown(results: list[CanaryResult], verdict_str: str) -> str:
    """Render the verdict MD with the claim-hygiene required sections.

    docs/audit/results/*.md must carry Scope / Verdict / Reproduction /
    Limitations headings (scripts/tools/check_claim_hygiene.py).
    """
    failed = [r for r in results if not r.fired]
    fired = sum(r.fired for r in results)
    lines = [
        "# Canary harness - CAPITAL_READY verdict",
        "",
        "**Date:** 2026-05-30  ",
        f"**Verdict:** `{verdict_str}`  ",
        f"**Canaries fired:** {fired}/{len(results)}  ",
        "**Producer:** `scripts/tools/canary_report.py`",
        "",
        "## Scope",
        "",
        "Negative-control / guard-efficacy check: each canary injects a known",
        "fake-edge contamination class and asserts the canonical guard meant to",
        "catch it fires. The question - *can canompx3 reliably reject fake edge?*",
        "`CAPITAL_READY` iff every guard caught its trap. Complements the null",
        "harness (`scripts/tests/test_synthetic_null.py`), which owns the",
        "noise-to-edge regime; this owns the guard-efficacy regime.",
        "",
        "## Results",
        "",
        "| Canary | Fired | Guard | Signature |",
        "|--------|-------|-------|-----------|",
    ]
    for r in results:
        mark = "PASS" if r.fired else "MISS"
        lines.append(f"| `{r.name}` | {mark} | {r.guard} | {r.signature} |")
    lines += ["", "## Verdict", ""]
    if failed:
        lines.append(f"**`{verdict_str}`** - {len(failed)} guard(s) failed to fire:")
        lines.append("")
        for r in failed:
            lines.append(f"- **{r.name}** - guard `{r.guard}` did not catch its trap. Expected: {r.signature}")
        lines.append("")
        lines.append(
            "The pipeline can no longer reject these fake-edge classes. Fix each "
            "guard before any capital decision. The blocking drift gate "
            "`check_canary_suite_green` also blocks the commit."
        )
    else:
        lines.append(
            f"**`{verdict_str}`** - all {fired} guards fired; the pipeline rejected "
            "every synthetic fake-edge trap. Negative-control evidence that canompx3 "
            "can kill fake edge on demand (cf. Aronson 2007 Ch 8-9: a correct "
            "pipeline kills naively-flagged rules at scale)."
        )
    lines += [
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/tools/canary_report.py            # this verdict + MD",
        "python scripts/tests/canary_suite.py             # the raw canary table",
        "python pipeline/check_drift.py                   # Check 192 (blocking gate)",
        "```",
        "",
        "Outputs: this file; the blocking drift check `check_canary_suite_green`;",
        "the Tier-1 suite `scripts/tests/canary_suite.py`.",
        "",
        "## Limitations",
        "",
        "- Tier-1 calls guards at their API boundary - it proves the guard FUNCTION",
        "  fires, not that every scan ROUTES through it (the meta static-scanner",
        "  `check_research_scans_call_guards` covers routing structurally; Tier-2",
        "  end-to-end injection is deferred).",
        "- Name-based guards are defeatable by renaming a post-entry field to an",
        "  `_ALWAYS_SAFE` name; the value-based T0 tautology is the load-bearing catch",
        "  (canary 7). A magnitude-only post-entry leak (e.g. `mae_r`) is out of scope.",
        "- Canary 10 checks the DSR V[SR] universe is PINNED, not that N-hat uses ONC",
        "  clustering (Amendment 3.5 ONC_PENDING).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable verdict")
    ap.add_argument("--no-write", action="store_true", help="do not write the dated MD artifact")
    args = ap.parse_args(argv)

    results = run_canaries()
    verdict_str, failed = verdict(results)

    if args.json:
        print(
            json.dumps(
                {
                    "verdict": verdict_str,
                    "fired": sum(r.fired for r in results),
                    "total": len(results),
                    "failed": [{"canary": r.name, "guard": r.guard, "expected_signature": r.signature} for r in failed],
                },
                indent=2,
            )
        )
    else:
        print(f"\nVERDICT: {verdict_str}  ({sum(r.fired for r in results)}/{len(results)} guards fired)\n")
        for r in results:
            mark = "✓" if r.fired else "✗ MISS"
            print(f"  {mark}  {r.name}")
        if failed:
            print("\nGuards that FAILED to fire (capital-blocking):")
            for r in failed:
                print(f"  - {r.name}: guard={r.guard}; expected={r.signature}")

    if not args.no_write:
        out = PROJECT_ROOT / "docs" / "audit" / "results" / "2026-05-30-canary-capital-ready-verdict.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_markdown(results, verdict_str), encoding="utf-8")
        if not args.json:
            print(f"\nWrote {out.relative_to(PROJECT_ROOT).as_posix()}")

    return 0 if verdict_str == CAPITAL_READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
