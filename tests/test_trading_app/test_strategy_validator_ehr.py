"""Tests for EHR Stage 3 — validator hard-fail on the EHR verdict ceiling.

EARLY_HOLDOUT_REDISCOVERY PASS 2 Stage 3 (2026-05-17). The validator's
promotion path must:

- propagate validation_mode verbatim through experimental→validated promotion
  (plan invariant #4 — EHR rows identifiable end-to-end), and
- hard-cap an EHR survivor's status to 'RESEARCH_PROVISIONAL' so it can NEVER
  be 'active' (plan invariant #5 — EHR rows research-provisional only, never
  deployable; the ACTIVE_VALIDATED_VIEW `WHERE LOWER(status)='active'`
  predicate then excludes them from every deployable surface by construction).

The promotion decision is encapsulated in the canonical helper
`_derive_ehr_promotion_fields` (single source of truth, acceptance #11) plus the
status-override branch at the INSERT site. These tests exercise that contract
directly + a DB round-trip confirming the columns persist, rather than mocking
the entire run_validation pipeline.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from trading_app.db_manager import init_trading_app_schema
from trading_app.holdout_policy import (
    EARLY_HOLDOUT_BOUNDARY,
    EHR_MODE_LABEL,
    HOLDOUT_SACRED_FROM,
    STANDARD_MODE_LABEL,
)
from trading_app.strategy_validator import (
    RESEARCH_PROVISIONAL_STATUS,
    _derive_ehr_promotion_fields,
)


@pytest.fixture
def db_path(tmp_path):
    """Temp DuckDB with base pipeline schema (daily_features) + trading-app schema."""
    path = tmp_path / "test_validator_ehr.db"
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE daily_features (
            trading_day DATE NOT NULL,
            symbol TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL,
            bar_count_1m INTEGER,
            PRIMARY KEY (symbol, trading_day, orb_minutes)
        )
        """
    )
    con.commit()
    con.close()
    init_trading_app_schema(db_path=path)
    return path


def _row_status_for(ehr_fields: dict, instrument: str, active_instruments) -> str:
    """Mirror the validator's status-override branch (strategy_validator.py INSERT
    site) so the test asserts the exact production decision logic — consumes the
    is_research_provisional flag, not a string comparison."""
    if ehr_fields["is_research_provisional"]:
        return RESEARCH_PROVISIONAL_STATUS
    return "active" if instrument in active_instruments else "retired"


# ---------- 1. STANDARD-row regression ----------


def test_standard_row_derives_active_status_and_null_ehr_fields():
    """A row with no validation_mode (pre-Stage-2 grandfathered) is STANDARD:
    verdict_ceiling/pseudo_oos_* NULL, and status follows the existing
    active/retired logic — bit-for-bit identical to pre-Stage-3 behavior."""
    fields = _derive_ehr_promotion_fields({"instrument": "MNQ"})
    assert fields["validation_mode"] == STANDARD_MODE_LABEL
    assert fields["verdict_ceiling"] is None
    assert fields["pseudo_oos_window_start"] is None
    assert fields["pseudo_oos_window_end"] is None
    # Active instrument → 'active'; non-active → 'retired'. Unchanged path.
    assert _row_status_for(fields, "MNQ", {"MNQ", "MES", "MGC"}) == "active"
    assert _row_status_for(fields, "GC", {"MNQ", "MES", "MGC"}) == "retired"


# ---------- 2. EHR-row label propagation ----------


def test_ehr_row_preserves_validation_mode_verbatim():
    """An EHR experimental row keeps validation_mode='EARLY_HOLDOUT_REDISCOVERY'
    through the promotion field-derivation (invariant #4)."""
    fields = _derive_ehr_promotion_fields({"validation_mode": EHR_MODE_LABEL})
    assert fields["validation_mode"] == EHR_MODE_LABEL == "EARLY_HOLDOUT_REDISCOVERY"


# ---------- 3. EHR-row status hard-cap ----------


def test_ehr_row_status_hard_capped_to_research_provisional():
    """Invariant #5: an EHR row's status is RESEARCH_PROVISIONAL, NOT 'active',
    even for an active ORB instrument that would otherwise promote to 'active'."""
    fields = _derive_ehr_promotion_fields({"validation_mode": EHR_MODE_LABEL})
    # MNQ is an active ORB instrument — STANDARD would give 'active'. EHR must NOT.
    status = _row_status_for(fields, "MNQ", {"MNQ", "MES", "MGC"})
    assert status == "RESEARCH_PROVISIONAL"
    assert status != "active"


# ---------- 4. EHR-row verdict_ceiling + pseudo_oos window ----------


def test_ehr_row_verdict_ceiling_and_canonical_pseudo_oos_window():
    """EHR row carries verdict_ceiling='RESEARCH_PROVISIONAL' and the pseudo-OOS
    window populated with the CANONICAL boundary dates from holdout_policy
    (never inlined literals)."""
    fields = _derive_ehr_promotion_fields({"validation_mode": EHR_MODE_LABEL})
    assert fields["verdict_ceiling"] == "RESEARCH_PROVISIONAL"
    assert fields["pseudo_oos_window_start"] == EARLY_HOLDOUT_BOUNDARY == date(2025, 1, 1)
    assert fields["pseudo_oos_window_end"] == HOLDOUT_SACRED_FROM == date(2026, 1, 1)


# ---------- 5. cumulative_search_count round-trip ----------


def test_cumulative_search_count_round_trips_byte_exact():
    """cumulative_search_count propagates verbatim from the experimental row
    (Bailey-Lopez de Prado 2014 disclosure); None for STANDARD."""
    ehr = _derive_ehr_promotion_fields({"validation_mode": EHR_MODE_LABEL, "cumulative_search_count": 342})
    assert ehr["cumulative_search_count"] == 342

    standard = _derive_ehr_promotion_fields({"validation_mode": STANDARD_MODE_LABEL})
    assert standard["cumulative_search_count"] is None


# ---------- 6. Integration: RESEARCH_PROVISIONAL excluded from ACTIVE view ----------


def test_research_provisional_row_excluded_from_active_validated_view(db_path):
    """Invariant #5 self-enforcement: a validated_setups row with
    status='RESEARCH_PROVISIONAL' must NOT appear in the ACTIVE_VALIDATED_VIEW
    (WHERE LOWER(status)='active'). This is what makes EHR survivors
    non-deployable by construction — the whole point of the hard-cap."""
    from trading_app.db_manager import ACTIVE_VALIDATED_VIEW

    con = duckdb.connect(str(db_path))
    # One active STANDARD row, one RESEARCH_PROVISIONAL EHR row.
    con.execute(
        """
        INSERT INTO validated_setups
            (strategy_id, instrument, orb_label, orb_minutes, rr_target,
             confirm_bars, entry_model, filter_type, sample_size, win_rate,
             expectancy_r, years_tested, all_years_positive, stress_test_passed,
             status, validation_mode)
        VALUES
            ('std_active', 'MNQ', 'EUROPE_FLOW', 5, 1.5, 1, 'E2', 'NONE',
             100, 0.55, 0.25, 5, TRUE, TRUE, 'active', 'STANDARD'),
            ('ehr_prov', 'MNQ', 'EUROPE_FLOW', 5, 1.5, 1, 'E2', 'NONE',
             100, 0.55, 0.25, 5, TRUE, TRUE, 'RESEARCH_PROVISIONAL',
             'EARLY_HOLDOUT_REDISCOVERY')
        """
    )
    con.commit()
    visible = {r[0] for r in con.execute(f"SELECT strategy_id FROM {ACTIVE_VALIDATED_VIEW}").fetchall()}
    con.close()

    assert "std_active" in visible, "STANDARD active row should be visible in ACTIVE view"
    assert "ehr_prov" not in visible, (
        "RESEARCH_PROVISIONAL EHR row must be EXCLUDED from ACTIVE_VALIDATED_VIEW "
        "(invariant #5 self-enforcement via WHERE LOWER(status)='active')"
    )
