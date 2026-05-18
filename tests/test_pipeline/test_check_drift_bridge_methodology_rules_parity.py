"""Sibling-coverage injection tests for check_bridge_methodology_rules_parity (Check #161).

Mutation-probe doctrine per
``memory/feedback_regex_alternation_sibling_coverage.md``. The single gated
constant is ``METHODOLOGY_RULES_APPLIED``; sibling coverage means one
mutation test per inlined rule slug, plus fail-closed structural tests.
"""

from __future__ import annotations

import importlib

import pytest

from pipeline import check_drift


def _minimal_doctrine(rule_numbers: list[int]) -> str:
    """Construct a backtesting-methodology.md text containing only the named RULE headings."""
    body = ["# Backtesting Methodology\n"]
    for n in rule_numbers:
        body.append(f"\n## RULE {n}: synthetic rule {n}\n\nBody for rule {n}.\n")
    return "".join(body)


@pytest.fixture
def doctrine_with_all_rules(tmp_path):
    """Canonical doctrine containing RULE 1, 3, 4, 9, 10 (matches bridge slugs)."""
    p = tmp_path / "backtesting-methodology.md"
    p.write_text(_minimal_doctrine([1, 3, 4, 9, 10]), encoding="utf-8")
    return p


def _reload_bridge():
    from scripts.research import fast_lane_to_heavyweight_bridge

    return importlib.reload(fast_lane_to_heavyweight_bridge)


def test_passes_when_all_slugs_have_canonical_rule(doctrine_with_all_rules):
    """Baseline: real canonical doctrine has every RULE the bridge cites."""
    violations = check_drift.check_bridge_methodology_rules_parity(
        doctrine_with_all_rules
    )
    assert violations == [], (
        f"baseline parity must hold against synthetic doctrine; got: {violations}"
    )


def test_catches_rule_1_temporal_alignment_drift(tmp_path, monkeypatch):
    """SIBLING #1: rule_1 missing from doctrine -> violation."""
    p = tmp_path / "doctrine.md"
    p.write_text(_minimal_doctrine([3, 4, 9, 10]), encoding="utf-8")  # no RULE 1
    violations = check_drift.check_bridge_methodology_rules_parity(p)
    assert any("rule_1_temporal_alignment" in v for v in violations), (
        f"expected rule_1 violation; got: {violations}"
    )


def test_catches_rule_3_is_oos_discipline_drift(tmp_path):
    """SIBLING #2: rule_3 missing -> violation."""
    p = tmp_path / "doctrine.md"
    p.write_text(_minimal_doctrine([1, 4, 9, 10]), encoding="utf-8")
    violations = check_drift.check_bridge_methodology_rules_parity(p)
    assert any("rule_3_is_oos_discipline" in v for v in violations)


def test_catches_rule_4_multi_framing_drift(tmp_path):
    """SIBLING #3: rule_4 missing -> violation."""
    p = tmp_path / "doctrine.md"
    p.write_text(_minimal_doctrine([1, 3, 9, 10]), encoding="utf-8")
    violations = check_drift.check_bridge_methodology_rules_parity(p)
    assert any("rule_4_multi_framing" in v for v in violations)


def test_catches_rule_9_canonical_layers_drift(tmp_path):
    """SIBLING #4: rule_9 missing -> violation."""
    p = tmp_path / "doctrine.md"
    p.write_text(_minimal_doctrine([1, 3, 4, 10]), encoding="utf-8")
    violations = check_drift.check_bridge_methodology_rules_parity(p)
    assert any("rule_9_canonical_layers" in v for v in violations)


def test_catches_rule_10_pre_registration_drift(tmp_path):
    """SIBLING #5: rule_10 missing -> violation."""
    p = tmp_path / "doctrine.md"
    p.write_text(_minimal_doctrine([1, 3, 4, 9]), encoding="utf-8")
    violations = check_drift.check_bridge_methodology_rules_parity(p)
    assert any("rule_10_pre_registration" in v for v in violations)


def test_catches_added_bogus_slug(doctrine_with_all_rules, monkeypatch):
    """Bridge adds a slug citing a non-existent RULE 99 -> violation."""
    bridge = _reload_bridge()
    monkeypatch.setattr(
        bridge,
        "METHODOLOGY_RULES_APPLIED",
        bridge.METHODOLOGY_RULES_APPLIED + ("rule_99_fabricated",),
    )
    violations = check_drift.check_bridge_methodology_rules_parity(
        doctrine_with_all_rules
    )
    assert any("rule_99" in v or "RULE 99" in v for v in violations)


def test_catches_malformed_slug(doctrine_with_all_rules, monkeypatch):
    """Bridge slug doesn't match rule_<N>_ shape -> violation."""
    bridge = _reload_bridge()
    monkeypatch.setattr(
        bridge,
        "METHODOLOGY_RULES_APPLIED",
        bridge.METHODOLOGY_RULES_APPLIED + ("not_a_rule_slug",),
    )
    violations = check_drift.check_bridge_methodology_rules_parity(
        doctrine_with_all_rules
    )
    assert any("does not match the canonical rule_" in v for v in violations)


def test_fails_closed_when_doctrine_missing(tmp_path):
    """Missing doctrine path -> fail-closed."""
    missing = tmp_path / "does-not-exist.md"
    violations = check_drift.check_bridge_methodology_rules_parity(missing)
    assert violations, "missing doctrine must fail-closed"
    assert any("canonical doctrine missing" in v for v in violations)


def test_fails_closed_when_no_rule_headings(tmp_path):
    """Doctrine with no RULE headings at all -> fail-closed."""
    p = tmp_path / "doctrine.md"
    p.write_text("# Some methodology doc\n\nPlain prose, no RULE headings.\n", encoding="utf-8")
    violations = check_drift.check_bridge_methodology_rules_parity(p)
    assert violations, "doctrine without RULE headings must fail-closed"
    assert any("no `## RULE" in v for v in violations)


def test_runs_against_real_canonical_doctrine():
    """Smoke: against the actual project methodology doc, parity holds at landing."""
    violations = check_drift.check_bridge_methodology_rules_parity()
    assert violations == [], (
        f"real-canonical parity must hold at landing; got: {violations}"
    )
