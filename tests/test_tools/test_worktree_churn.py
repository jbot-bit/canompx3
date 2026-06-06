"""Unit tests for the canonical operational-churn predicate.

Proves `is_churn_path` is the single source fleet_state delegates to, and that it
matches known churn paths (no work-at-risk) while leaving real source paths alone.
"""

from __future__ import annotations

from scripts.tools import _worktree_churn as wc


def test_known_churn_paths_match():
    for p in wc.OPERATIONAL_CHURN_PATHS:
        assert wc.is_churn_path(p) is True


def test_nested_churn_path_matches():
    # active_plan.md appears nested in a porcelain tail — substring, not equality.
    assert wc.is_churn_path("docs/runtime/active_plan.md") is True


def test_real_source_path_is_not_churn():
    assert wc.is_churn_path("pipeline/dst.py") is False
    assert wc.is_churn_path("trading_app/config.py") is False


def test_substring_containing_churn_name_is_not_churn():
    # Segment match, NOT substring: a real source path that merely CONTAINS a
    # churn entry's text must NOT be misclassified (adversarial-audit finding).
    assert wc.is_churn_path("tests/test_live_journal.db_helpers.py") is False
    assert wc.is_churn_path("docs/runtime/active_plan.md.bak") is False
    assert wc.is_churn_path("bot_state.json.backup") is False


def test_nested_churn_segment_matches():
    # A churn file under a directory IS churn (trailing /<segment> match).
    assert wc.is_churn_path("logs/live_journal.db") is True
    assert wc.is_churn_path("a/b/bot_state.json") is True


def test_windows_backslash_path_normalized():
    # Porcelain on Windows may surface backslashes — normalize before matching.
    assert wc.is_churn_path("docs\\runtime\\active_plan.md") is True


def test_empty_or_blank_is_not_churn():
    assert wc.is_churn_path("") is False
    assert wc.is_churn_path("   ") is False


def test_fleet_state_delegates_to_canonical_predicate():
    """PROVE delegation: fleet_state._count_dirty uses THIS predicate, not a copy."""
    from scripts.tools import fleet_state as fs

    # is_churn_path imported into fleet_state IS the canonical function object.
    assert fs.is_churn_path is wc.is_churn_path
