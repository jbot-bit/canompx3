#!/usr/bin/env python3
"""Pre-push verifier: branch staleness + scope sanity vs origin/main.

Catches the 2026-04-20 codex/live-book-reaudit class of failures:
1. Branch base is stale → merging the branch reverts commits that have
   landed on main since the branch was created. (The concrete incident:
   codex branch branched before PR #25 perf/lazy-imports-broad-sweep
   merged; cherry-picking without rebase would have reverted three
   files' lazy-import work.)
2. Branch scope balloons silently. When diff-stat crosses a threshold
   and there is no corresponding documentation (stage file, plan,
   hypothesis), it is likely creeping beyond intent.

Behaviour:
- Fetches origin (fail-closed if that fails).
- Computes merge-base(origin/main, HEAD) age in days + distance in commits.
- Computes scope: files changed, lines changed vs origin/main.
- Prints a verdict. Non-zero exit = block push.

Bypass: VERIFY_BRANCH_SCOPE_SKIP=1 git push ...
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime

# Thresholds — tuned from the codex incident
STALE_WARN_DAYS = 3
STALE_BLOCK_DAYS = 7
STALE_BLOCK_COMMITS_AHEAD = 25  # origin/main ahead of merge-base by this many → force rebase
SCOPE_WARN_FILES = 30
SCOPE_WARN_LINES = 1500
SCOPE_SCRUTINY_FILES = 60  # At this threshold, require a stage/plan doc
SCOPE_SCRUTINY_LINES = 3000


def _run(args: list[str], check: bool = True) -> str:
    result = subprocess.run(args, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"ERROR: {' '.join(args)} failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(2)
    return result.stdout.strip()


def _merge_base_age_days(merge_base_sha: str) -> float:
    iso = _run(["git", "show", "-s", "--format=%aI", merge_base_sha])
    commit_dt = datetime.fromisoformat(iso)
    return (datetime.now(UTC) - commit_dt).total_seconds() / 86400.0


def _has_scope_doc(branch_name: str) -> bool:
    """Look for a plan / stage file / hypothesis with a name resembling the branch."""
    slug_bits = set(branch_name.replace("/", "-").replace("_", "-").lower().split("-"))
    slug_bits.discard("")
    for path in ("docs/runtime/stages", "docs/plans", "docs/audit/hypotheses"):
        try:
            for entry in os.listdir(path):
                entry_bits = set(entry.replace("_", "-").replace(".", "-").lower().split("-"))
                overlap = slug_bits & entry_bits
                # Any 2+ slug words in common = plausible scope doc
                if len(overlap) >= 2:
                    return True
        except FileNotFoundError:
            continue
    return False


def main() -> int:
    if os.environ.get("VERIFY_BRANCH_SCOPE_SKIP") == "1":
        print("verify_branch_scope: SKIPPED (VERIFY_BRANCH_SCOPE_SKIP=1)")
        return 0

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch == "HEAD":
        print("verify_branch_scope: detached HEAD — skipping")
        return 0
    if branch == "main":
        return 0  # pushes to main handled by CI / reviewers, not this hook

    # Fetch — fail closed if offline
    fetch = subprocess.run(["git", "fetch", "origin", "--quiet"], capture_output=True, text=True)
    if fetch.returncode != 0:
        print("verify_branch_scope: `git fetch origin` failed — bypass with VERIFY_BRANCH_SCOPE_SKIP=1")
        print(fetch.stderr, file=sys.stderr)
        return 1

    try:
        merge_base = _run(["git", "merge-base", "origin/main", "HEAD"])
    except SystemExit:
        print("verify_branch_scope: no common ancestor with origin/main — skipping")
        return 0

    age_days = _merge_base_age_days(merge_base)
    ahead = int(_run(["git", "rev-list", "--count", f"{merge_base}..origin/main"]))

    # Scope: diff vs origin/main
    shortstat = _run(["git", "diff", "--shortstat", "origin/main..HEAD"])
    files_changed = (
        int(_run(["git", "diff", "--name-only", "origin/main..HEAD"]).count("\n") + 1)
        if _run(["git", "diff", "--name-only", "origin/main..HEAD"])
        else 0
    )
    lines_changed = 0
    if shortstat:
        import re

        m = re.search(r"(\d+)\s+insertion", shortstat)
        if m:
            lines_changed += int(m.group(1))
        m = re.search(r"(\d+)\s+deletion", shortstat)
        if m:
            lines_changed += int(m.group(1))

    print(f"verify_branch_scope: branch={branch}")
    print(f"  merge-base age: {age_days:.1f}d ({merge_base[:10]})")
    print(f"  origin/main commits ahead of merge-base: {ahead}")
    print(f"  scope vs origin/main: {files_changed} files, {lines_changed} lines")

    errors: list[str] = []
    warnings: list[str] = []

    # Staleness
    if age_days >= STALE_BLOCK_DAYS or ahead >= STALE_BLOCK_COMMITS_AHEAD:
        errors.append(
            f"STALE BASE: merge-base is {age_days:.1f}d old and origin/main is {ahead} commits "
            f"ahead. Rebasing is required to avoid silently reverting main. "
            f"Run: git fetch origin && git rebase origin/main"
        )
    elif age_days >= STALE_WARN_DAYS:
        warnings.append(f"stale base ({age_days:.1f}d / {ahead} commits ahead) — consider rebase")

    # Scope
    if files_changed >= SCOPE_SCRUTINY_FILES or lines_changed >= SCOPE_SCRUTINY_LINES:
        if not _has_scope_doc(branch):
            errors.append(
                f"UNDOCUMENTED LARGE SCOPE: {files_changed} files / {lines_changed} lines but no "
                f"matching stage/plan/hypothesis doc found. Write one under docs/runtime/stages/, "
                f"docs/plans/, or docs/audit/hypotheses/ that references the work on this branch."
            )
    elif files_changed >= SCOPE_WARN_FILES or lines_changed >= SCOPE_WARN_LINES:
        warnings.append(
            f"large scope ({files_changed} files / {lines_changed} lines) — ensure PR body discloses scope explicitly"
        )

    for w in warnings:
        print(f"  WARN: {w}")
    for e in errors:
        print(f"  ERROR: {e}", file=sys.stderr)

    if errors:
        print("")
        print("To bypass in emergency: VERIFY_BRANCH_SCOPE_SKIP=1 git push ...", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
