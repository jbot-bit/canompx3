"""Tests for pipeline.session_guard — chronological look-ahead ordering.

Focused on the NYSE_PREOPEN insertion (2026-05-28) and the canonical invariant
that every ORB_LABELS session is present in _SESSION_ORDER. The look-ahead
semantics are load-bearing: a session in the wrong chronological position would
silently admit future-session features as predictors (look-ahead contamination,
backtesting-methodology.md RULE 1).
"""

from pipeline.init_db import ORB_LABELS
from pipeline.session_guard import (
    _SESSION_COL_RE,
    _SESSION_ORDER,
    get_prior_sessions,
    is_feature_safe,
)


class TestSessionOrderCoversOrbLabels:
    """Canonical invariant: every dynamic ORB_LABELS session is chronologically ordered."""

    def test_every_orb_label_in_session_order(self):
        missing = set(ORB_LABELS) - set(_SESSION_ORDER)
        assert missing == set(), f"ORB_LABELS sessions absent from _SESSION_ORDER: {sorted(missing)}"

    def test_no_duplicate_sessions(self):
        assert len(_SESSION_ORDER) == len(set(_SESSION_ORDER))


class TestNysePreopenPosition:
    """NYSE_PREOPEN (09:00 ET) sits between US_DATA_830 (08:30 ET) and NYSE_OPEN (09:30 ET)."""

    def test_chronological_position(self):
        i_830 = _SESSION_ORDER.index("US_DATA_830")
        i_pre = _SESSION_ORDER.index("NYSE_PREOPEN")
        i_open = _SESSION_ORDER.index("NYSE_OPEN")
        assert i_830 < i_pre < i_open

    def test_orb_columns_match_session_regex(self):
        """Without _SESSION_ORDER membership, orb_NYSE_PREOPEN_* columns would be
        invisible to _SESSION_COL_RE and silently masked everywhere."""
        m = _SESSION_COL_RE.match("orb_NYSE_PREOPEN_size")
        assert m is not None
        assert m.group(1) == "NYSE_PREOPEN"

    def test_prior_sessions_include_earlier_only(self):
        prior = get_prior_sessions("NYSE_PREOPEN")
        assert "US_DATA_830" in prior
        assert "NYSE_OPEN" not in prior  # later session
        assert "NYSE_PREOPEN" not in prior  # not its own prior


class TestNysePreopenLookAheadSafety:
    """Look-ahead correctness for target=NYSE_PREOPEN (RULE 1.2)."""

    def test_earlier_session_orb_is_safe(self):
        assert is_feature_safe("orb_US_DATA_830_size", "NYSE_PREOPEN") is True

    def test_same_session_orb_is_safe(self):
        assert is_feature_safe("orb_NYSE_PREOPEN_size", "NYSE_PREOPEN") is True

    def test_later_session_orb_is_lookahead(self):
        """orb_NYSE_OPEN_* is a LATER session — using it for NYSE_PREOPEN is look-ahead."""
        assert is_feature_safe("orb_NYSE_OPEN_size", "NYSE_PREOPEN") is False

    def test_ny_window_is_lookahead(self):
        """The NY window (23:00-02:00 Bris) has NOT closed at NYSE_PREOPEN time."""
        assert is_feature_safe("session_ny_high", "NYSE_PREOPEN") is False

    def test_london_window_inherits_conservative_policy(self):
        """session_london_* is safe-after NYSE_OPEN in the existing config (more
        conservative than RULE 1.2's 23:00). NYSE_PREOPEN is before NYSE_OPEN, so
        it is masked — the fail-closed-safe direction (never under-masks)."""
        assert is_feature_safe("session_london_high", "NYSE_PREOPEN") is False

    def test_reverse_nyse_open_sees_nyse_preopen(self):
        """NYSE_OPEN is later, so it CAN see the earlier NYSE_PREOPEN ORB."""
        assert is_feature_safe("orb_NYSE_PREOPEN_size", "NYSE_OPEN") is True
