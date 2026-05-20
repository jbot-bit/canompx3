"""Sibling-coverage injection tests for check_cherry_pick_ranker_threshold_parity (Check #160).

Mutation-probe doctrine per
``memory/feedback_regex_alternation_sibling_coverage.md`` and
``memory/feedback_injection_test_catches_float_repr_class_bug.md``.
One dedicated test per ``InlineCopyPair.gated_constants`` element (this pair
has one constant: ``HEAVYWEIGHT_T_THRESHOLD``), plus fail-closed tests for
the structural failure modes (missing doctrine, doctrine restructured,
parse failure).
"""

from __future__ import annotations

import importlib

import pytest

from pipeline import check_drift


@pytest.fixture
def doctrine_text() -> str:
    """Minimal valid Criterion 4 section -- mirrors canonical doctrine wording."""
    return (
        "# Pre-registered criteria\n\n"
        "## Criterion 3 — BH FDR\n"
        "Some content.\n\n"
        "## Criterion 4 — Chordia t-statistic threshold\n\n"
        "**Rule:** After BH-FDR passes, compute the implied t-statistic. "
        "Require t >= 3.00 (Harvey-Liu-Zhu 2015) for strategies with "
        "strong pre-registered economic theory support. Require t >= 3.79 "
        "(Chordia et al 2018, verbatim Tier 1) for strategies without "
        "such theoretical support.\n\n"
        "## Criterion 5 — Deflated Sharpe Ratio\n"
    )


def _write_doctrine(tmp_path, text: str):
    p = tmp_path / "pre_registered_criteria.md"
    p.write_text(text, encoding="utf-8")
    return p


def _reload_ranker():
    """Reload the ranker module so monkeypatched constants take effect."""
    from scripts.research import cherry_pick_ranker

    return importlib.reload(cherry_pick_ranker)


def test_passes_when_threshold_matches_canonical(tmp_path, doctrine_text):
    """Baseline: ranker default constant 3.79 matches canonical 3.79 -- empty violations."""
    p = _write_doctrine(tmp_path, doctrine_text)
    violations = check_drift.check_cherry_pick_ranker_threshold_parity(p)
    assert violations == [], f"baseline parity must hold; got: {violations}"


def test_catches_heavyweight_t_threshold_drift(tmp_path, doctrine_text, monkeypatch):
    """SIBLING #1: HEAVYWEIGHT_T_THRESHOLD mutation -- the one gated constant."""
    p = _write_doctrine(tmp_path, doctrine_text)
    cpr = _reload_ranker()
    monkeypatch.setattr(cpr, "HEAVYWEIGHT_T_THRESHOLD", 3.80)
    violations = check_drift.check_cherry_pick_ranker_threshold_parity(p)
    assert any("HEAVYWEIGHT_T_THRESHOLD" in v for v in violations), (
        f"expected HEAVYWEIGHT_T_THRESHOLD drift violation; got: {violations}"
    )
    assert any("3.8" in v for v in violations), "violation must surface the bad value 3.80"


def test_fails_closed_when_doctrine_missing(tmp_path):
    """Missing doctrine path -- fail-closed with a clear violation, not silent pass."""
    missing = tmp_path / "does-not-exist.md"
    violations = check_drift.check_cherry_pick_ranker_threshold_parity(missing)
    assert violations, "missing doctrine must fail-closed"
    assert any("canonical doctrine missing" in v for v in violations)


def test_fails_closed_when_criterion_4_heading_drifts(tmp_path):
    """Doctrine restructured -- missing the Criterion 4 heading fails-closed."""
    p = _write_doctrine(
        tmp_path,
        "# Pre-registered criteria\n\n"
        "## Criterion 3 — BH FDR\n"
        "Some content.\n\n"
        "## Criterion 5 — Deflated Sharpe Ratio\n"
        "Skipped 4.\n",
    )
    violations = check_drift.check_cherry_pick_ranker_threshold_parity(p)
    assert violations, "doctrine without Criterion 4 must fail-closed"
    assert any("Criterion 4" in v for v in violations)


def test_fails_closed_when_threshold_phrase_drifts(tmp_path):
    """Doctrine wording for the no-theory threshold drifts -- fail-closed."""
    p = _write_doctrine(
        tmp_path,
        "# Pre-registered criteria\n\n"
        "## Criterion 4 — Chordia t-statistic threshold\n\n"
        "Some prose that omits the numeric threshold entirely.\n\n"
        "## Criterion 5 — Deflated Sharpe Ratio\n",
    )
    violations = check_drift.check_cherry_pick_ranker_threshold_parity(p)
    assert violations, "doctrine missing no-theory threshold phrase must fail-closed"
    assert any("no-theory" in v for v in violations)


def test_runs_against_real_canonical_doctrine():
    """Smoke: against the actual project doctrine, parity holds at landing."""
    # No override -- uses PROJECT_ROOT / docs / institutional / pre_registered_criteria.md
    violations = check_drift.check_cherry_pick_ranker_threshold_parity()
    assert violations == [], f"real-canonical parity must hold at landing; got: {violations}"
