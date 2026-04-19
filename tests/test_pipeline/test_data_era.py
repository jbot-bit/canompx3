"""Tests for `pipeline.data_era` — canonical PARENT/MICRO classification.

Phase 3a foundation module: provides the helpers that Stage 3b/3c/3d will
consume to distinguish real-micro from parent-proxy data in bars_1m and
orb_outcomes. Zero consumers at Phase 3a — this is a foundation-only stage.

Tests cover:
- All 13 configured instruments (active + dead + parent + research-only)
- Both classification paths (source_symbol for bars_1m, trading_day for
  orb_outcomes)
- Phase 2 corruption detection (micro instrument with parent-shaped
  source_symbol → PARENT)
- Every fail-closed branch (unknown instrument, non-canonical pattern,
  null/empty source_symbol, non-micro micro_launch_day call, non-micro
  era_for_trading_day call)
"""

from datetime import date

import pytest

from pipeline.asset_configs import ASSET_CONFIGS
from pipeline.data_era import (
    DataEra,
    era_for_source_symbol,
    era_for_trading_day,
    is_micro,
    micro_launch_day,
    parent_for,
)


# =============================================================================
# Coverage matrix — updated whenever ASSET_CONFIGS gains a new instrument
# =============================================================================

# Every configured instrument + its expected parent_symbol.
# Lives in the test file (not the prod code) so a drift between config and
# test fails the suite loudly, not silently.
EXPECTED_PARENT = {
    # Active micros — post-Phase-2 real micro data, parent preserved for history
    "MGC": "GC",
    "MNQ": "NQ",
    "MES": "ES",
    # Dead micros — use parent data source, cost model at micro specs
    "M2K": "RTY",
    "MBT": "BTC",
    "M6E": "6E",
    "MCL": "CL",
    "SIL": "SI",
    # Parents and research-only — no parent relationship
    "NQ": None,
    "ES": None,
    "GC": None,
    "2YY": None,
    "ZT": None,
}


class TestIsMicro:
    """is_micro() = "has REAL micro contract data in bars_1m" (not just "has parent relationship").

    Semantic: dead micros (M2K/MBT/MCL/M6E/SIL) use PARENT data in bars_1m
    despite being micro contracts at the exchange. Their outright_pattern
    points to parent contracts (RTY/BTC/CL/6E/SI), so they are NOT real-micro
    for era classification purposes. Only active micros post-Phase-2 qualify.
    """

    def test_active_micros_are_real_micros(self):
        """MGC/MNQ/MES post-Phase-2 have real micro data → is_micro=True."""
        for inst in ("MGC", "MNQ", "MES"):
            assert is_micro(inst) is True, f"{inst} should be real micro"

    def test_dead_micros_use_parent_data(self):
        """M2K/MBT/etc are micro contracts but use parent DATA → is_micro=False."""
        for inst in ("M2K", "MBT", "M6E", "MCL", "SIL"):
            assert is_micro(inst) is False, f"{inst} is a dead micro — uses parent data, not real micro"

    def test_parents_are_not_micros(self):
        for inst in ("NQ", "ES", "GC"):
            assert is_micro(inst) is False, f"{inst} is a parent, not micro"

    def test_research_only_not_micros(self):
        for inst in ("2YY", "ZT"):
            assert is_micro(inst) is False, f"{inst} is research-only native"

    def test_case_insensitive(self):
        assert is_micro("mgc") is True
        assert is_micro("nq") is False
        assert is_micro("m2k") is False  # dead micro — parent data

    def test_unknown_instrument_raises(self):
        with pytest.raises(ValueError, match="Unknown instrument"):
            is_micro("FAKE")

    def test_coverage_matches_asset_configs(self):
        """EXPECTED_PARENT must cover every entry in ASSET_CONFIGS — no drift."""
        configured = set(ASSET_CONFIGS.keys())
        expected = set(EXPECTED_PARENT.keys())
        missing = configured - expected
        extra = expected - configured
        assert not missing, f"ASSET_CONFIGS has instruments not in test: {missing}"
        assert not extra, f"test EXPECTED_PARENT has instruments not in config: {extra}"


class TestParentFor:
    """parent_for reads the new parent_symbol field on ASSET_CONFIGS."""

    def test_resolves_all_configured(self):
        for inst, expected in EXPECTED_PARENT.items():
            assert parent_for(inst) == expected, f"parent_for({inst}) should be {expected!r}"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown instrument"):
            parent_for("FAKE")


class TestMicroLaunchDay:
    """micro_launch_day returns minimum_start_date for micros, raises otherwise."""

    def test_active_micros_return_minimum_start_date(self):
        assert micro_launch_day("MGC") == ASSET_CONFIGS["MGC"]["minimum_start_date"]
        assert micro_launch_day("MNQ") == ASSET_CONFIGS["MNQ"]["minimum_start_date"]
        assert micro_launch_day("MES") == ASSET_CONFIGS["MES"]["minimum_start_date"]

    def test_mgc_post_phase_2_date(self):
        """Pin the canonical backfilled MGC micro launch: 2022-06-13."""
        assert micro_launch_day("MGC") == date(2022, 6, 13)

    def test_mnq_mes_post_phase_2_date(self):
        """Pin the canonical post-Phase-2 MNQ/MES launch: 2019-05-06."""
        assert micro_launch_day("MNQ") == date(2019, 5, 6)
        assert micro_launch_day("MES") == date(2019, 5, 6)

    def test_non_micro_raises(self):
        """Non-micros: parents, research-only, AND dead micros (parent data)."""
        for inst in ("NQ", "ES", "GC", "2YY", "ZT", "M2K", "MBT", "M6E", "MCL", "SIL"):
            with pytest.raises(ValueError, match="not a"):
                micro_launch_day(inst)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown instrument"):
            micro_launch_day("FAKE")


class TestEraForSourceSymbol:
    """era_for_source_symbol classifies a bars_1m row by source_symbol."""

    def test_active_micro_with_own_source_is_micro(self):
        """Post-Phase-2 clean: MNQ rows with source='MNQH4' → MICRO."""
        assert era_for_source_symbol("MNQ", "MNQH4") == "MICRO"
        assert era_for_source_symbol("MES", "MESZ4") == "MICRO"
        assert era_for_source_symbol("MGC", "MGCG25") == "MICRO"

    def test_parent_with_own_source_is_parent(self):
        """NQ/ES/GC with their own contracts → PARENT (they ARE parents)."""
        assert era_for_source_symbol("NQ", "NQH4") == "PARENT"
        assert era_for_source_symbol("ES", "ESZ3") == "PARENT"
        assert era_for_source_symbol("GC", "GCQ5") == "PARENT"

    def test_phase_2_corruption_detection(self):
        """CRITICAL REGRESSION GUARD: micro instrument with parent-shaped source.

        Pre-Phase-2 bars_1m had `symbol='MNQ', source_symbol='NQH4'` — the
        exact corruption Phase 2 fixed. The helper must correctly classify
        this as PARENT so Stage 3d's drift check can catch any residual.
        """
        assert era_for_source_symbol("MNQ", "NQH4") == "PARENT"
        assert era_for_source_symbol("MES", "ESM3") == "PARENT"
        assert era_for_source_symbol("MGC", "GCZ2") == "PARENT"

    def test_dead_micros_use_parent_source(self):
        """M2K rows have source=RTYH4 (RTY parent) — PARENT data source."""
        assert era_for_source_symbol("M2K", "RTYH4") == "PARENT"
        assert era_for_source_symbol("MBT", "BTCH4") == "PARENT"
        assert era_for_source_symbol("MCL", "CLZ4") == "PARENT"
        assert era_for_source_symbol("M6E", "6EH4") == "PARENT"
        assert era_for_source_symbol("SIL", "SIZ4") == "PARENT"

    def test_unrelated_source_raises(self):
        """MNQ with source='ESH4' → neither MNQ nor NQ pattern → ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            era_for_source_symbol("MNQ", "ESH4")

    def test_null_source_raises(self):
        with pytest.raises(ValueError, match="source_symbol"):
            era_for_source_symbol("MNQ", None)  # type: ignore[arg-type]

    def test_empty_source_raises(self):
        with pytest.raises(ValueError, match="source_symbol"):
            era_for_source_symbol("MNQ", "")

    def test_unknown_instrument_raises(self):
        with pytest.raises(ValueError, match="Unknown instrument"):
            era_for_source_symbol("FAKE", "FAKEH4")


class TestEraForTradingDay:
    """era_for_trading_day classifies orb_outcomes rows by date vs launch."""

    def test_day_of_launch_is_micro(self):
        """trading_day == micro_launch_day → MICRO (inclusive boundary)."""
        launch = micro_launch_day("MGC")
        assert era_for_trading_day("MGC", launch) == "MICRO"

    def test_day_after_launch_is_micro(self):
        launch = micro_launch_day("MNQ")
        one_day_after = date(launch.year, launch.month, launch.day + 1)
        assert era_for_trading_day("MNQ", one_day_after) == "MICRO"

    def test_day_before_launch_is_parent(self):
        """trading_day == launch - 1 → PARENT (launch is inclusive)."""
        launch = micro_launch_day("MGC")
        # launch is 2022-06-13, so 2022-06-12 is before
        one_day_before = date(2022, 6, 12)
        assert launch == date(2022, 6, 13)  # sanity check
        assert era_for_trading_day("MGC", one_day_before) == "PARENT"

    def test_far_pre_launch_is_parent(self):
        """Historical date long before any micro launch → PARENT."""
        assert era_for_trading_day("MNQ", date(2015, 1, 1)) == "PARENT"
        assert era_for_trading_day("MES", date(2015, 1, 1)) == "PARENT"
        assert era_for_trading_day("MGC", date(2015, 1, 1)) == "PARENT"

    def test_non_micro_raises(self):
        """era_for_trading_day only defined for REAL micros. Dead micros (M2K
        etc) use parent data so the date boundary is meaningless — callers
        must is_micro() first."""
        for inst in ("NQ", "ES", "GC", "2YY", "ZT", "M2K", "MBT", "M6E", "MCL", "SIL"):
            with pytest.raises(ValueError, match="not a"):
                era_for_trading_day(inst, date(2025, 1, 1))

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown instrument"):
            era_for_trading_day("FAKE", date(2025, 1, 1))


class TestDataEraType:
    """DataEra is a Literal type for static guarantees."""

    def test_era_values_are_exactly_parent_or_micro(self):
        """Helper must return exactly 'PARENT' or 'MICRO', never anything else."""
        result = era_for_source_symbol("MNQ", "MNQH4")
        assert result in ("PARENT", "MICRO")
        # Type check hint
        _: DataEra = result  # should type-check without error
