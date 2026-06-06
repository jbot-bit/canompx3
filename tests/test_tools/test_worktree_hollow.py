"""Calibration tests for the shared hollow-worktree predicate.

The predicate is the SINGLE definition of "hollow" consumed by fleet_state,
stage_reaper, and project_pulse. These tests pin the real-fleet anchors so the
threshold can never silently drift:
  - the 6403-deletion poisoning tree -> HOLLOW
  - a real refactor with 17 deletions among real work -> NOT hollow
"""

from __future__ import annotations

from scripts.tools._worktree_hollow import (
    HOLLOW_MIN_DELETIONS,
    classify_hollow,
    is_hollow,
)


def _porcelain_deletions(n: int) -> str:
    """n deletion lines in `git status --porcelain` form (` D path`)."""
    return "\n".join(f" D deleted/file_{i}.py" for i in range(n))


def _porcelain_mods(n: int) -> str:
    return "\n".join(f" M mod/file_{i}.py" for i in range(n))


def test_clean_tree_is_not_hollow():
    v = classify_hollow("")
    assert v.is_hollow is False
    assert v.total == 0
    assert "clean" in v.reason.lower()


def test_whitespace_only_is_not_hollow():
    assert is_hollow("\n  \n\t\n") is False


def test_poisoning_tree_anchor_is_hollow():
    # 6403 deletions / 8 non-deletions -> the original poisoning tree.
    porcelain = _porcelain_deletions(6403) + "\n" + _porcelain_mods(8)
    v = classify_hollow(porcelain)
    assert v.is_hollow is True
    assert v.deletions == 6403
    assert v.non_deletions == 8
    assert v.deletion_ratio >= 0.90


def test_real_refactor_anchor_is_not_hollow():
    # 17 deletions among substantial real work -> NOT hollow (c11-cap-x075 class).
    porcelain = _porcelain_deletions(17) + "\n" + _porcelain_mods(40)
    v = classify_hollow(porcelain)
    assert v.is_hollow is False
    assert "non-deletions" in v.reason or "ratio" in v.reason or "gutting" in v.reason


def test_just_below_min_deletions_not_hollow():
    v = classify_hollow(_porcelain_deletions(HOLLOW_MIN_DELETIONS - 1))
    assert v.is_hollow is False
    assert "not enough gutting" in v.reason


def test_at_min_deletions_pure_deletion_is_hollow():
    v = classify_hollow(_porcelain_deletions(HOLLOW_MIN_DELETIONS))
    assert v.is_hollow is True
    assert v.deletion_ratio == 1.0


def test_too_many_nondeletions_not_hollow():
    # 200 deletions but 11 real changes (> MAX_NONDELETIONS) -> real work present.
    porcelain = _porcelain_deletions(200) + "\n" + _porcelain_mods(11)
    v = classify_hollow(porcelain)
    assert v.is_hollow is False
    assert "non-deletions" in v.reason


def test_ratio_gate_blocks_mixed_tree():
    # 100 deletions, 9 mods: passes deletion+nondel gates, but check the ratio gate
    # holds when deletions barely dominate. 100/109 = 0.917 >= 0.90 -> still hollow.
    porcelain = _porcelain_deletions(100) + "\n" + _porcelain_mods(9)
    assert is_hollow(porcelain) is True


def test_renames_count_as_nondeletions():
    # A big rename-heavy refactor must NOT be hollow even with 'D' in rename pairs.
    renames = "\n".join(f"R  old_{i}.py -> new_{i}.py" for i in range(150))
    v = classify_hollow(renames)
    assert v.deletions == 0
    assert v.non_deletions == 150
    assert v.is_hollow is False


def test_untracked_files_are_nondeletions():
    porcelain = _porcelain_deletions(150) + "\n" + "\n".join(f"?? new/file_{i}.py" for i in range(20))
    v = classify_hollow(porcelain)
    # 20 untracked non-deletions > MAX_NONDELETIONS -> not hollow.
    assert v.non_deletions == 20
    assert v.is_hollow is False
