#!/usr/bin/env python
"""
M2.5 research pre-flight — bias scan a research script before you run it.

Catches look-ahead bias, inflated N, data snooping and timezone bugs BEFORE
they contaminate your findings. Takes ~10s with Lightning model.

Usage:
    python scripts/tools/m25_preflight.py research/my_script.py
    python scripts/tools/m25_preflight.py research/my_script.py --run   # scan then run
    python scripts/tools/m25_preflight.py research/my_script.py --force # skip prompt
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.tools.m25_audit import AUDIT_MODES, audit, load_api_key, read_files  # noqa: E402

# Extra instructions bolted onto bias mode specifically for research scripts
RESEARCH_BIAS_ADDENDUM = """

RESEARCH SCRIPT SPECIFIC CHECKS (these matter most):

- TIMEZONE DOUBLE-CONVERSION: Look for `df["col"] + pd.Timedelta(hours=10)` applied
  to a column already in Australia/Brisbane timezone. DuckDB fetchdf() returns
  timezone-aware datetimes; adding hours manually double-converts. This is a known
  bug pattern in this project (fixed Mar 3 2026 — check it hasn't re-appeared).

- INFLATED SAMPLE SIZE: Does the script JOIN daily_features without orb_minutes?
  That triples N and gives fake p-values. Always check the row count before/after.

- LOOK-AHEAD COLUMNS: `double_break` is look-ahead (checked post-session).
  Any column derived from full-session data used as a pre-entry filter is look-ahead.

- POST-HOC NARRATIVE: Is the script designed to confirm a hypothesis already formed
  from peeking at the data? Flag if the test seems reverse-engineered.

Output: START with either PASS or FAIL in bold, then explain findings.
"""


def preflight(script_path: str, force: bool = False, run_after: bool = False) -> int:
    """
    Scan a research script for bias. Returns 0 (pass) or 1 (findings).
    force=True skips the confirmation prompt on findings.
    run_after=True executes the script after passing.
    """
    path = Path(script_path)
    if not path.exists():
        path = PROJECT / script_path
    if not path.exists():
        print(f"ERROR: File not found: {script_path}")
        return 1

    print(f"M2.5 pre-flight: {path.name} (Lightning)...", flush=True)

    try:
        api_key = load_api_key()
    except SystemExit:
        print("WARNING: MINIMAX_API_KEY not set — pre-flight skipped (configure key to enable).")
        return 2  # Distinct from pass (0) and fail (1) — skipped, not validated

    file_content = read_files([str(path)])
    system_prompt = AUDIT_MODES["bias"] + RESEARCH_BIAS_ADDENDUM

    try:
        result = audit(file_content, system_prompt, api_key, fast=True)
    except Exception as e:
        print(f"WARNING: M2.5 pre-flight error ({e}) — proceeding anyway.")
        return 0

    # Parse result — match "PASS" or "FAIL" on the first non-empty line only,
    # optionally wrapped in bold markers. Using the first line avoids false positives
    # from words like "failure", "fails", "fallback" appearing later in the body.
    import re as _re

    first_line = next((line_text.strip() for line_text in result.splitlines() if line_text.strip()), "")
    _verdict = _re.match(r"^\*{0,2}(PASS|FAIL)\*{0,2}", first_line, _re.IGNORECASE)
    verdict = _verdict.group(1).upper() if _verdict else ""
    passed = verdict == "PASS"
    failed = verdict == "FAIL"

    print()
    print(result)
    print()

    if failed and not passed:
        print("─" * 60)
        print("⚠  M2.5 pre-flight found potential bias issues above.")
        if not force:
            try:
                answer = input("Proceed anyway? [y/N] ").strip().lower()
            except EOFError:
                answer = "n"
            if answer not in ("y", "yes"):
                print("Aborted. Review and fix before running.")
                return 1
        else:
            print("--force set — proceeding despite findings.")
    else:
        print("✓ M2.5 pre-flight passed.")

    if run_after:
        if failed and not passed:
            print("ERROR: Script failed preflight. Review findings above before running.")
            return 1
        print(f"\nRunning {path}...\n")
        proc = subprocess.run([sys.executable, str(path)], cwd=str(PROJECT))
        return proc.returncode

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M2.5 bias pre-flight for research scripts")
    parser.add_argument("script", help="Research script to scan")
    parser.add_argument("--run", action="store_true", help="Run the script after passing")
    parser.add_argument("--force", action="store_true", help="Don't prompt on findings")
    args = parser.parse_args()

    sys.exit(preflight(args.script, force=args.force, run_after=args.run))
