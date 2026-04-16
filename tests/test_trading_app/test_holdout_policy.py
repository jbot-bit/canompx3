"""Tests for trading_app.holdout_policy — Amendment 2.7 Mode A canonical source.

These tests pin the canonical constants and the enforcement helper so that
future refactors cannot silently drift the sacred-from date or the
grandfather cutoff without failing a test.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

import pytest

from trading_app.holdout_policy import (
    HOLDOUT_GRANDFATHER_CUTOFF,
    HOLDOUT_OVERRIDE_TOKEN,
    HOLDOUT_SACRED_FROM,
    enforce_holdout_date,
)


class TestConstants:
    """Pin the canonical values. Changing these requires a new Amendment."""

    def test_sacred_from_is_2026_01_01(self):
        """Amendment 2.7 locks the sacred window start at 2026-01-01."""
        assert HOLDOUT_SACRED_FROM == date(2026, 1, 1)

    def test_grandfather_cutoff_is_2026_04_08_utc(self):
        """Amendment 2.7 commit moment, used to distinguish grandfathered
        pre-correction contamination from new violations."""
        assert HOLDOUT_GRANDFATHER_CUTOFF == datetime(2026, 4, 8, 0, 0, 0, tzinfo=UTC)

    def test_grandfather_cutoff_after_sacred_from(self):
        """Sanity: the grandfather moment must be strictly later than the
        sacred-from date, otherwise the semantic model breaks down."""
        assert HOLDOUT_GRANDFATHER_CUTOFF.date() > HOLDOUT_SACRED_FROM

    def test_grandfather_cutoff_is_tz_aware(self):
        """Must be timezone-aware so DuckDB comparisons against
        ``created_at`` TIMESTAMP WITH TIME ZONE columns are unambiguous."""
        assert HOLDOUT_GRANDFATHER_CUTOFF.tzinfo is not None


class TestEnforceHoldoutDate:
    """Validate the --holdout-date CLI-side gate."""

    def test_none_defaults_to_sacred_from(self):
        """Omitting --holdout-date is silently upgraded to the sacred default.
        This keeps Mode A safe-by-default: running discovery without the flag
        cannot leak sacred data."""
        assert enforce_holdout_date(None) == HOLDOUT_SACRED_FROM

    def test_exact_sacred_from_is_accepted(self):
        """Boundary case: --holdout-date 2026-01-01 is the canonical value
        and must be accepted unchanged."""
        assert enforce_holdout_date(date(2026, 1, 1)) == date(2026, 1, 1)

    def test_pre_sacred_date_is_accepted_unchanged(self):
        """Earlier holdout dates are permitted (e.g., auditing pre-2025 eras)."""
        assert enforce_holdout_date(date(2025, 1, 1)) == date(2025, 1, 1)

    def test_one_day_post_sacred_is_rejected(self):
        """2026-01-02 crosses the sacred window by one day — must raise."""
        with pytest.raises(ValueError, match="Mode A"):
            enforce_holdout_date(date(2026, 1, 2))

    def test_months_post_sacred_is_rejected(self):
        """A realistic Mode B attempt (e.g., 2026-04-07) must raise."""
        with pytest.raises(ValueError, match="Amendment 2.7"):
            enforce_holdout_date(date(2026, 4, 7))

    def test_far_future_post_sacred_is_rejected(self):
        """A 2027 holdout date (hypothetical new holdout) must raise under
        the current Amendment 2.7. Changing this behavior requires a new
        amendment that extends the canonical source."""
        with pytest.raises(ValueError, match="sacred holdout window"):
            enforce_holdout_date(date(2027, 1, 1))

    def test_error_message_mentions_canonical_source(self):
        """The error must point the caller at this module so fixes land in
        the single source of truth, not in scattered code."""
        with pytest.raises(ValueError, match="trading_app.holdout_policy"):
            enforce_holdout_date(date(2026, 6, 1))

    def test_error_message_suggests_the_fix(self):
        """The error must tell the user exactly what to type to resolve it."""
        with pytest.raises(ValueError, match=r"--holdout-date 2026-01-01"):
            enforce_holdout_date(date(2026, 3, 15))


class TestCanonicalModuleShape:
    """Pin the module's public API so import sites stay stable."""

    def test_all_exports_are_present(self):
        """The __all__ tuple is the public contract — consumers import only
        these names."""
        import trading_app.holdout_policy as hp

        assert hasattr(hp, "HOLDOUT_SACRED_FROM")
        assert hasattr(hp, "HOLDOUT_GRANDFATHER_CUTOFF")
        assert hasattr(hp, "HOLDOUT_OVERRIDE_TOKEN")
        assert hasattr(hp, "enforce_holdout_date")
        assert "HOLDOUT_SACRED_FROM" in hp.__all__
        assert "HOLDOUT_GRANDFATHER_CUTOFF" in hp.__all__
        assert "HOLDOUT_OVERRIDE_TOKEN" in hp.__all__
        assert "enforce_holdout_date" in hp.__all__


class TestOverrideToken:
    """Validate the HOLDOUT_OVERRIDE_TOKEN escape hatch (added 2026-04-08
    per explicit user instruction). The override allows discovery to access
    sacred-window data when the correct token is supplied — but emits a LOUD
    warning and the resulting strategies are research-provisional."""

    def test_override_token_value_is_3656(self):
        """The token value is pinned at '3656' per explicit user request.
        Changing this requires a new amendment to pre_registered_criteria.md."""
        assert HOLDOUT_OVERRIDE_TOKEN == "3656"

    def test_override_with_correct_token_allows_post_sacred(self):
        """date(2026, 3, 1) + token '3656' should pass through unchanged
        and not raise."""
        result = enforce_holdout_date(date(2026, 3, 1), override_token="3656")
        assert result == date(2026, 3, 1)

    def test_override_with_correct_token_uses_constant(self):
        """Equivalent: passing HOLDOUT_OVERRIDE_TOKEN constant by reference."""
        result = enforce_holdout_date(date(2026, 6, 15), override_token=HOLDOUT_OVERRIDE_TOKEN)
        assert result == date(2026, 6, 15)

    def test_override_with_wrong_token_still_raises(self):
        """Wrong token must NOT bypass the gate. Any string except '3656'
        is treated as no token."""
        with pytest.raises(ValueError, match="Mode A"):
            enforce_holdout_date(date(2026, 3, 1), override_token="wrong")

    def test_override_with_empty_string_still_raises(self):
        """Empty string is not a valid token."""
        with pytest.raises(ValueError, match="Mode A"):
            enforce_holdout_date(date(2026, 3, 1), override_token="")

    def test_override_with_none_token_still_raises(self):
        """Default behavior (no token) — unchanged."""
        with pytest.raises(ValueError, match="Mode A"):
            enforce_holdout_date(date(2026, 3, 1), override_token=None)

    def test_override_emits_loud_warning(self, caplog):
        """When the override is invoked, a LOUD warning must be logged with
        the override date and a clear research-provisional notice. The audit
        trail is the real defense — the token is just a speed bump."""
        with caplog.at_level(logging.WARNING, logger="trading_app.holdout_policy"):
            enforce_holdout_date(date(2026, 4, 1), override_token="3656")
        # Verify warning was logged
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 1, f"Expected at least one WARNING log, got {warnings}"
        msg = warnings[0].getMessage()
        assert "HOLDOUT OVERRIDE INVOKED" in msg
        assert "2026-04-01" in msg
        assert "RESEARCH-PROVISIONAL" in msg

    def test_override_does_not_affect_pre_sacred(self):
        """Pre-sacred dates pass through with or without the token —
        the override only matters for post-sacred dates."""
        # No token, pre-sacred
        assert enforce_holdout_date(date(2025, 6, 1)) == date(2025, 6, 1)
        # With token, pre-sacred
        assert enforce_holdout_date(date(2025, 6, 1), override_token="3656") == date(2025, 6, 1)

    def test_override_does_not_affect_none(self):
        """None still defaults to HOLDOUT_SACRED_FROM regardless of token.
        The override only matters for explicit post-sacred dates."""
        assert enforce_holdout_date(None) == HOLDOUT_SACRED_FROM
        assert enforce_holdout_date(None, override_token="3656") == HOLDOUT_SACRED_FROM
        assert enforce_holdout_date(None, override_token="wrong") == HOLDOUT_SACRED_FROM

    def test_override_does_not_affect_exact_sacred(self):
        """date == HOLDOUT_SACRED_FROM passes through (it's the canonical
        boundary, not a violation)."""
        assert enforce_holdout_date(date(2026, 1, 1)) == date(2026, 1, 1)
        assert enforce_holdout_date(date(2026, 1, 1), override_token="3656") == date(2026, 1, 1)
