"""Injection tests for check_holdout_sentinel_inline_copy_parity (Check #171).

The trial-ledger writer inlines the Mode A sacred-window boundary as a
string sentinel ``HOLDOUT_SACRED_FROM_SENTINEL = "2026-01-01"``. The
canonical source is ``trading_app.holdout_policy.HOLDOUT_SACRED_FROM``
(a ``date`` object). This file mutation-probes the parity check via
monkeypatch on both sides, plus the clean-state baseline.

Class anchor: [[canonical-inline-copy-parity-bug-class]] (8th confirmed
instance, 2026-05-20 — Stage 2A.2 follow-up after audit found the
sentinel inlined without parity registration).

Background:
  memory/feedback_canonical_inline_copy_parity_bug_class.md
  scripts/research/fast_lane_trial_ledger.py (inline site)
  trading_app/holdout_policy.py (canonical source)
"""

from __future__ import annotations

from datetime import date

import pytest

from pipeline.check_drift import check_holdout_sentinel_inline_copy_parity
from scripts.research import fast_lane_trial_ledger as ledger
from trading_app import holdout_policy


# ----------------------------------------------------------------------
# Clean-state baseline
# ----------------------------------------------------------------------


def test_clean_state_passes():
    """Real module constants must agree byte-for-byte today."""
    violations = check_holdout_sentinel_inline_copy_parity()
    assert violations == [], f"unexpected parity violations on clean state: {violations}"


# ----------------------------------------------------------------------
# Injection 1: ledger-side sentinel mutated (string drift)
# ----------------------------------------------------------------------


def test_drift_ledger_sentinel_advanced_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the ledger's inlined string advances past the canonical date the
    check must produce one violation that names both values."""
    monkeypatch.setattr(ledger, "HOLDOUT_SACRED_FROM_SENTINEL", "2027-01-01")

    violations = check_holdout_sentinel_inline_copy_parity()

    assert violations, "ledger sentinel drift went undetected"
    assert len(violations) == 1
    assert "'2027-01-01'" in violations[0]
    assert "'2026-01-01'" in violations[0]
    assert "canonical-inline-copy-parity-bug-class" in violations[0]


def test_drift_ledger_sentinel_regressed_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression to a stale boundary (e.g. 2025) must also fire."""
    monkeypatch.setattr(ledger, "HOLDOUT_SACRED_FROM_SENTINEL", "2025-01-01")

    violations = check_holdout_sentinel_inline_copy_parity()

    assert violations
    assert any("'2025-01-01'" in v for v in violations)


# ----------------------------------------------------------------------
# Injection 2: canonical side mutated (date object drift)
# ----------------------------------------------------------------------


def test_drift_canonical_date_advanced_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``trading_app.holdout_policy.HOLDOUT_SACRED_FROM`` is advanced
    without amending the ledger's inlined sentinel the check must fire."""
    monkeypatch.setattr(holdout_policy, "HOLDOUT_SACRED_FROM", date(2027, 1, 1))

    violations = check_holdout_sentinel_inline_copy_parity()

    assert violations, "canonical-side advance went undetected"
    assert len(violations) == 1
    assert "'2027-01-01'" in violations[0]
    assert "Bailey-Lopez de Prado 2014" in violations[0]


# ----------------------------------------------------------------------
# Injection 3: empty-string sentinel (banner scrub class)
# ----------------------------------------------------------------------


def test_drift_empty_sentinel_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Total scrub of the sentinel to '' must fire — not silently pass."""
    monkeypatch.setattr(ledger, "HOLDOUT_SACRED_FROM_SENTINEL", "")

    violations = check_holdout_sentinel_inline_copy_parity()

    assert violations
    assert "''" in violations[0]


# ----------------------------------------------------------------------
# Meta: the bug-class anchor is in the violation message
# ----------------------------------------------------------------------


def test_violation_cites_bug_class_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every violation message must point at the feedback file so an
    operator opening the failing drift run can trace the class history."""
    monkeypatch.setattr(ledger, "HOLDOUT_SACRED_FROM_SENTINEL", "1999-01-01")
    violations = check_holdout_sentinel_inline_copy_parity()
    assert violations
    assert "feedback_canonical_inline_copy_parity_bug_class.md" in violations[0]
