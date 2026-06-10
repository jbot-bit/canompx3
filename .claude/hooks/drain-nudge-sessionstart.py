#!/usr/bin/env python3
"""Drain-nudge: Claude-visible SessionStart cue when stranded work piles up.

The behavioral half of the durable-drain fix (see
`docs/runtime/stages/drain-worktrees-routine.md`). The mechanical half
(`scripts/tools/drain_worktrees.ps1`) can drain on demand, but nothing reminds
the operator/Claude THAT a drain is due — so finished commits get marooned on
feature branches and the worktree count creeps. This hook closes that gap.

Behavior: on SessionStart, cheaply count (a) local branches strictly ahead of
origin/main and (b) capital/live-suspect stashes. If either exceeds its
threshold, emit ONE Claude-visible `additionalContext` line suggesting a
dry-run drain. Advisory only — it never blocks, never writes, never pushes. It
is a JUDGE nudge, mirroring the memory-capture SessionStart cue.

Runs as a PARALLEL SIBLING to the other SessionStart hooks (additionalContext is
concatenated, not exclusive — official hooks contract). Fail-open EVERYWHERE: any
error / missing git / timeout -> exit 0, no stdout. The guard can never disturb a
session it cannot read.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Thresholds: below these, stay silent (a few ahead-branches is normal mid-work).
_AHEAD_BRANCH_THRESHOLD = 5
_CAPITAL_STASH_THRESHOLD = 3

# Stash-label tokens that hint at capital / live-routing work worth NOT dropping.
_CAPITAL_STASH_TOKENS = (
    "live", "dashboard", "start_bot", "account", "repoint", "broker",
    "journal", "orchestrat", "preflight", "23055112", "21944866",
)


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        if r.returncode != 0:
            return None
        return r.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _count_branches_ahead_of_main() -> int:
    """Local branches (excluding main) with >=1 commit not on origin/main."""
    out = _git("for-each-ref", "--format=%(refname:short)", "refs/heads")
    if not out:
        return 0
    count = 0
    for b in out.splitlines():
        b = b.strip()
        if not b or b == "main":
            continue
        ahead = _git("rev-list", "--count", f"origin/main..{b}")
        try:
            if ahead is not None and int(ahead.strip()) > 0:
                count += 1
        except ValueError:
            continue
    return count


def _count_capital_stashes() -> int:
    out = _git("stash", "list")
    if not out:
        return 0
    n = 0
    for line in out.splitlines():
        low = line.lower()
        if any(tok in low for tok in _CAPITAL_STASH_TOKENS):
            n += 1
    return n


def main() -> None:
    try:
        # Drain payload from stdin (we don't need its fields, but consume it so the
        # hook is a well-behaved SessionStart citizen).
        if not sys.stdin.isatty():
            try:
                json.load(sys.stdin)
            except (json.JSONDecodeError, ValueError):
                pass

        ahead = _count_branches_ahead_of_main()
        cap_stash = _count_capital_stashes()

        if ahead < _AHEAD_BRANCH_THRESHOLD and cap_stash < _CAPITAL_STASH_THRESHOLD:
            sys.exit(0)  # nothing notable — silent clean start

        bits = []
        if ahead >= _AHEAD_BRANCH_THRESHOLD:
            bits.append(f"{ahead} local branches ahead of origin/main")
        if cap_stash >= _CAPITAL_STASH_THRESHOLD:
            bits.append(f"{cap_stash} capital/live-suspect stashes")
        summary = "; ".join(bits)

        cue = (
            f"Stranded-work check: {summary}. CONSIDER a dry-run drain "
            "(`pwsh scripts/tools/drain_worktrees.ps1`) to see what finished work "
            "could go home — it reports DRAIN/CAPITAL/DIVERGED and pushes nothing "
            "without -Execute. Advisory only; skip if mid-flight or already known."
        )
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": cue,
            }
        }))
    except BaseException:
        sys.exit(0)


if __name__ == "__main__":
    main()
