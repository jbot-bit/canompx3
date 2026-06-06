#!/usr/bin/env python3
"""Shared hollow-worktree predicate — the SINGLE definition of "hollow".

A *hollow* worktree is one whose uncommitted diff is almost entirely DELETIONS:
the working tree has been gutted (files removed) without compensating new work.
This is the signature of an abandoned/half-cleaned tree that poisons the
stage-reaper — it looks "dirty" (so naive guards refuse to reap it) but the dirt
is destruction, not work-at-risk. The original poisoning incident (worktree
`...2026042...`) is the calibration anchor.

This predicate lives in exactly ONE module so every consumer (fleet_state.py,
stage_reaper_audit.py, project_pulse.py) asks the same question and gets the same
answer — the split-brain class this plan exists to kill. Re-deriving the
threshold in three places is exactly the bug.

Calibration (from the live fleet, 2026-06):
  - del >= 100          : a substantial gutting, not an incidental file removal
  - nondel <= 10        : almost no additions/modifications (the tree is hollow,
                          not actively being rebuilt)
  - del / total >= 0.90 : deletions dominate the diff
A rename (R status) is a MOVE, not a deletion — it counts as non-deletion, so a
big refactor that renames many files is NOT hollow (it's real work).

The real-fleet anchors this was calibrated against:
  - `...2026042` (the poisoning tree): 6403 deletions / 8 non-deletions -> HOLLOW
  - `c11-cap-x075` (real refactor): 17 deletions among real work -> NOT hollow

READ-ONLY. Pure function over a `git status --porcelain` payload; no git calls of
its own (the caller supplies the porcelain so this stays trivially testable and
side-effect-free). Fail-closed-to-NOT-hollow: any ambiguity returns False, so a
tree is never mistakenly classified hollow (which a destructive consumer might
act on). Under-flagging is safe here; over-flagging is not.
"""

from __future__ import annotations

from dataclasses import dataclass

# Calibration thresholds — the ONLY place these numbers live. Consumers import
# the predicate, never the constants, so the definition cannot drift.
HOLLOW_MIN_DELETIONS = 100
HOLLOW_MAX_NONDELETIONS = 10
HOLLOW_DELETION_RATIO = 0.90


@dataclass(frozen=True)
class HollowVerdict:
    """Explainable hollow classification — every term surfaced so an operator
    (or a test) can see WHY a tree was or wasn't flagged."""

    is_hollow: bool
    deletions: int
    non_deletions: int
    total: int
    deletion_ratio: float
    reason: str


def _classify_porcelain_line(line: str) -> str:
    """Map one `git status --porcelain` line to 'del' | 'nondel' | 'skip'.

    Porcelain format is `XY<space>path` where X = index status, Y = worktree
    status. A line is a DELETION iff either status column is 'D' AND it is not a
    rename (renames carry 'R' and represent a move, not lost work). Untracked
    ('??') and everything else (A/M/T/C/U) are non-deletions. Blank/short lines
    are skipped.
    """
    if len(line) < 3:
        return "skip"
    x, y = line[0], line[1]
    # Renames are moves, never hollowing deletions — guard them out first so a
    # 'D' that is actually the source half of a rename pair isn't miscounted.
    if x == "R" or y == "R":
        return "nondel"
    if x == "D" or y == "D":
        return "del"
    return "nondel"


def classify_hollow(porcelain: str) -> HollowVerdict:
    """Classify a worktree as hollow from its `git status --porcelain` output.

    `porcelain` is the raw multi-line string (one line per changed path). An
    empty/whitespace payload means a CLEAN tree — not hollow (there is nothing to
    reap and nothing at risk).
    """
    deletions = 0
    non_deletions = 0
    for raw in porcelain.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        kind = _classify_porcelain_line(line)
        if kind == "del":
            deletions += 1
        elif kind == "nondel":
            non_deletions += 1
        # 'skip' lines contribute to neither count

    total = deletions + non_deletions
    if total == 0:
        return HollowVerdict(
            is_hollow=False,
            deletions=0,
            non_deletions=0,
            total=0,
            deletion_ratio=0.0,
            reason="clean tree (no changes) — not hollow",
        )

    ratio = deletions / total

    # All three gates must hold. Surface the first failing gate so the verdict is
    # explainable rather than a bare boolean.
    if deletions < HOLLOW_MIN_DELETIONS:
        reason = f"deletions {deletions} < {HOLLOW_MIN_DELETIONS} (not enough gutting)"
        is_hollow_result = False
    elif non_deletions > HOLLOW_MAX_NONDELETIONS:
        reason = f"non-deletions {non_deletions} > {HOLLOW_MAX_NONDELETIONS} (real work present)"
        is_hollow_result = False
    elif ratio < HOLLOW_DELETION_RATIO:
        reason = f"deletion ratio {ratio:.2f} < {HOLLOW_DELETION_RATIO} (not deletion-dominated)"
        is_hollow_result = False
    else:
        reason = f"HOLLOW: {deletions} deletions, {non_deletions} non-deletions, ratio {ratio:.2f} — gutted tree"
        is_hollow_result = True

    return HollowVerdict(
        is_hollow=is_hollow_result,
        deletions=deletions,
        non_deletions=non_deletions,
        total=total,
        deletion_ratio=round(ratio, 4),
        reason=reason,
    )


def is_hollow(porcelain: str) -> bool:
    """Convenience boolean wrapper around :func:`classify_hollow`."""
    return classify_hollow(porcelain).is_hollow
