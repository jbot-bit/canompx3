"""Tests for the EARLY_HOLDOUT_REDISCOVERY (EHR) probe-mode constants in
``trading_app.holdout_policy``.

Authority: ``EARLY_HOLDOUT_REDISCOVERY — Narrow Guarded PASS 2 Plan`` (2026-05-17),
Stage 1 acceptance. The first three tests enforce plan invariants 1, 2, and 5;
the fourth pins the predicate's strictness so future stages can rely on it.
"""

from __future__ import annotations

from datetime import date

import pytest

from trading_app.holdout_policy import (
    EARLY_HOLDOUT_BOUNDARY,
    EHR_MODE_LABEL,
    HOLDOUT_SACRED_FROM,
    STANDARD_MODE_LABEL,
    enforce_early_holdout_date,
    is_ehr_mode,
)


def test_ehr_boundary_constant() -> None:
    """EHR boundary is locked at 2025-01-01 per plan Stage 1 acceptance #1."""
    assert EARLY_HOLDOUT_BOUNDARY == date(2025, 1, 1)


def test_mode_a_sacred_unchanged() -> None:
    """Plan invariant #1: Mode A's HOLDOUT_SACRED_FROM is byte-unchanged.

    Stage 1 must NEVER modify the Mode A boundary. If this test ever fails,
    a downstream stage has either renamed the constant or shifted its value.
    Either is a critical Mode A integrity violation per plan top-of-doc.
    """
    assert HOLDOUT_SACRED_FROM == date(2026, 1, 1)


def test_ehr_boundary_strictly_before_mode_a() -> None:
    """EHR boundary must be strictly earlier than Mode A's sacred window.

    This relation is load-bearing: EHR's PSEUDO-OOS window is defined as
    ``EARLY_HOLDOUT_BOUNDARY <= trading_day < HOLDOUT_SACRED_FROM``. If the
    inequality ever inverts (e.g., someone bumps EHR to 2026-06-01), the
    PSEUDO-OOS window collapses to empty or, worse, OVERLAPS the sacred
    Mode A OOS window.
    """
    assert EARLY_HOLDOUT_BOUNDARY < HOLDOUT_SACRED_FROM


def test_enforce_early_holdout_rejects_post_boundary() -> None:
    """Dates strictly after the boundary must raise ValueError citing EHR mode.

    Per plan invariant #1, EHR's boundary CANNOT be tuned at runtime.
    There is no override token (deliberate divergence from
    ``enforce_holdout_date``). The returned date is consumed by callers
    as the exclusive upper bound for ``trading_day`` queries
    (``WHERE trading_day < <returned>``), so the boundary itself is the
    expected common case (parallel to Mode A's ``enforce_holdout_date``
    accepting ``HOLDOUT_SACRED_FROM`` as a return value).
    """
    with pytest.raises(ValueError, match="EARLY_HOLDOUT_REDISCOVERY"):
        enforce_early_holdout_date(date(2025, 6, 1))


def test_enforce_early_holdout_accepts_boundary_and_earlier() -> None:
    """Dates at or before the boundary pass through unchanged.

    The returned ``date`` is used by callers as the exclusive upper bound
    for ``trading_day`` (``WHERE trading_day < <returned>``), so
    ``EARLY_HOLDOUT_BOUNDARY`` itself is the canonical default — it produces
    the maximum permitted EHR-IS window (``trading_day < 2025-01-01``).
    """
    pre = date(2024, 12, 31)
    assert enforce_early_holdout_date(pre) == pre
    assert enforce_early_holdout_date(EARLY_HOLDOUT_BOUNDARY) == EARLY_HOLDOUT_BOUNDARY
    # ``None`` upgrades silently to the default EHR boundary.
    assert enforce_early_holdout_date(None) == EARLY_HOLDOUT_BOUNDARY


def test_is_ehr_mode_predicate_strict() -> None:
    """``is_ehr_mode`` is strict equality only — no normalization, no aliases.

    Soft matching would let a typo silently fall through to the STANDARD
    path and bypass the EHR guards added in Stages 3, 4, 5. All non-exact
    inputs MUST return False so the downstream gates default to STANDARD
    semantics (which preserve Mode A integrity).
    """
    assert is_ehr_mode(EHR_MODE_LABEL) is True
    assert is_ehr_mode("EARLY_HOLDOUT_REDISCOVERY") is True

    # Negative cases — every one of these must be False.
    assert is_ehr_mode(STANDARD_MODE_LABEL) is False
    assert is_ehr_mode("STANDARD") is False
    assert is_ehr_mode(None) is False
    assert is_ehr_mode("") is False
    assert is_ehr_mode("early_holdout_rediscovery") is False
    assert is_ehr_mode("EHR") is False
    assert is_ehr_mode("EARLY_HOLDOUT") is False


def test_mode_labels_distinct() -> None:
    """STANDARD and EHR labels are distinct and non-empty."""
    assert EHR_MODE_LABEL == "EARLY_HOLDOUT_REDISCOVERY"
    assert STANDARD_MODE_LABEL == "STANDARD"
    assert EHR_MODE_LABEL != STANDARD_MODE_LABEL
