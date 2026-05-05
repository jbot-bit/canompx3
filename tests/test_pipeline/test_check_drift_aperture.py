"""Tests for check_aperture_hardcode_in_scoring_paths drift check.

Prevents 4th recurrence of the PR #189 class bug (hardcoded orb_minutes=5
in lane-iterating scoring paths). PR #189 → PR #231 → PR #232 → Task #4
all fixed the same fingerprint in different files.
"""

import textwrap
from pathlib import Path

from pipeline.check_drift import check_aperture_hardcode_in_scoring_paths


def _make_trading_app(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a fake trading_app/ dir with the given files."""
    ta_dir = tmp_path / "trading_app"
    ta_dir.mkdir()
    (ta_dir / "live").mkdir()
    for relpath, content in files.items():
        target = ta_dir / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(content), encoding="utf-8")
    return ta_dir


def test_flags_paper_file_with_hardcoded_orb_minutes_5(tmp_path):
    ta_dir = _make_trading_app(
        tmp_path,
        {
            "paper_foo.py": """
                def bad():
                    con.execute(
                        '''SELECT * FROM daily_features
                           WHERE symbol = ? AND orb_minutes = 5
                             AND trading_day >= ?''',
                        [instrument, start],
                    )
            """,
        },
    )
    violations = check_aperture_hardcode_in_scoring_paths(ta_dir)
    assert any("paper_foo.py" in v for v in violations), violations


def test_flags_lane_file_with_hardcoded_orb_minutes_5(tmp_path):
    ta_dir = _make_trading_app(
        tmp_path,
        {
            "lane_bar.py": """
                con.execute(
                    'SELECT atr_20 FROM daily_features WHERE symbol = ? AND orb_minutes = 5',
                    [s],
                )
            """,
        },
    )
    violations = check_aperture_hardcode_in_scoring_paths(ta_dir)
    assert any("lane_bar.py" in v for v in violations), violations


def test_flags_live_file_with_hardcoded_orb_minutes_5(tmp_path):
    """Live execution path was the auditor's CRITICAL on PR #232."""
    ta_dir = _make_trading_app(
        tmp_path,
        {
            "live/session_orchestrator.py": """
                def f(orb_minutes):
                    con.execute(
                        '''SELECT atr_20_pct FROM daily_features
                           WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL
                             AND trading_day = ?''',
                        [s, td],
                    )
            """,
        },
    )
    violations = check_aperture_hardcode_in_scoring_paths(ta_dir)
    assert any("session_orchestrator.py" in v for v in violations), violations


def test_exempts_canonical_cte_guard_annotation(tmp_path):
    ta_dir = _make_trading_app(
        tmp_path,
        {
            "paper_dedup.py": """
                # canonical-cte-guard: dedup non-aperture column atr_20 for rolling median
                con.execute(
                    '''SELECT MEDIAN(atr_20) FROM daily_features
                       WHERE symbol = ? AND orb_minutes = 5 AND atr_20 IS NOT NULL''',
                    [s],
                )
            """,
        },
    )
    violations = check_aperture_hardcode_in_scoring_paths(ta_dir)
    assert violations == [], violations


def test_exempts_session_regime_gate_annotation_within_20_lines(tmp_path):
    """lane_allocator.py:454 annotation is 16 lines above; check tolerates 20."""
    body = """
        def _session_regime_expr():
            # session-regime-gate: orb_minutes=5 is a deliberate fixed reference
            # aperture for the regime gate; cross-aperture by design.
            # The 18 lines that follow document the +630R backtest result and
            # the cross-aperture rationale; do not change without a re-audit.
            line_5 = "padding"
            line_6 = "padding"
            line_7 = "padding"
            line_8 = "padding"
            line_9 = "padding"
            line_10 = "padding"
            line_11 = "padding"
            line_12 = "padding"
            line_13 = "padding"
            line_14 = "padding"
            line_15 = "padding"
            line_16 = "padding"
            return '''SELECT * FROM daily_features
                      WHERE rr_target = 1.0 AND confirm_bars = 1 AND orb_minutes = 5'''
    """
    ta_dir = _make_trading_app(tmp_path, {"lane_allocator.py": body})
    violations = check_aperture_hardcode_in_scoring_paths(ta_dir)
    assert violations == [], violations


def test_does_not_flag_files_outside_scope(tmp_path):
    """Research scripts and unrelated trading_app files are out of scope."""
    ta_dir = _make_trading_app(
        tmp_path,
        {
            "config.py": """
                # research script context: dedup
                'SELECT * FROM daily_features WHERE orb_minutes = 5'
            """,
        },
    )
    violations = check_aperture_hardcode_in_scoring_paths(ta_dir)
    assert violations == []


def test_check_is_registered_in_CHECKS():
    from pipeline.check_drift import CHECKS

    labels = [entry[0] for entry in CHECKS]
    assert any("Aperture hardcode" in label and "PR #189" in label for label in labels), (
        f"check not registered in CHECKS list: {labels}"
    )
