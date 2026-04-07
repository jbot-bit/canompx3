"""Tests for trading_app.eligibility.builder — the THIN ADAPTER.

Post-canonical-filter-self-description refactor: builder.py is now a thin
adapter over the canonical StrategyFilter.describe() method. This file
tests the adapter's behaviour, NOT filter logic — filter semantics live
in tests/test_trading_app/test_config.py.

Adapter responsibilities under test:
1. parse_strategy_id — extract dimensions from strategy ID strings
2. ALL_FILTERS lookup — raise on unknown filter_type, walk composites
3. Status mapping — translate AtomDescription → ConditionStatus mechanically
4. validated_for → NOT_APPLICABLE_INSTRUMENT override
5. last_revalidated > 180 days → STALE_VALIDATION override
6. error_message aggregation into report.build_errors
7. Calendar overlay (HALF_SIZE → PASS with size_multiplier=0.5)
8. ATR velocity overlay (canonical delegation, monitored sessions only)
9. Freshness detection
10. Overall status derivation
11. Build errors on infrastructure failures (DB connect)

The new contract is:
  filter_type → ALL_FILTERS lookup → walk composite → call leaf .describe()
  → AtomDescription[] → adapter mechanically maps each to ConditionRecord

Zero re-encoded logic in the adapter. Filter classes own their own
metadata via ClassVar (VALIDATED_FOR, LAST_REVALIDATED, CONFIDENCE_TIER).
"""

from __future__ import annotations

from datetime import date

import pytest

from trading_app.eligibility import build_eligibility_report
from trading_app.eligibility.builder import parse_strategy_id
from trading_app.eligibility.types import (
    ConditionCategory,
    ConditionRecord,
    ConditionStatus,
    EligibilityReport,
    FreshnessStatus,
    OverallStatus,
    ResolvesAt,
)


# ==========================================================================
# Fixtures
# ==========================================================================


def _fresh_row(symbol: str = "MNQ", **extra) -> dict:
    """Build a fresh feature row for trading_day 2026-04-07.

    Includes all common feature columns so tests can target individual
    fields without worrying about unrelated DATA_MISSING noise.
    """
    row = {
        "trading_day": date(2026, 4, 7),
        "symbol": symbol,
        "pit_range_atr": 0.15,
        "prev_day_range": 200.0,
        "atr_20": 150.0,
        "gap_open_points": 5.0,
        "overnight_range": 80.0,
        "overnight_range_pct": 60.0,
        "atr_20_pct": 75.0,
        "cross_atr_MES_pct": 75.0,
        "cross_atr_MGC_pct": 75.0,
        "atr_vel_regime": "Stable",
        "orb_CME_REOPEN_compression_tier": "Compressed",
        "orb_TOKYO_OPEN_compression_tier": "Compressed",
        "day_of_week": 2,  # Wednesday
        "is_nfp_day": False,
        "is_opex_day": False,
        "is_friday": False,
    }
    row.update(extra)
    return row


# ==========================================================================
# parse_strategy_id
# ==========================================================================


class TestParseStrategyId:
    def test_standard_id(self):
        dims = parse_strategy_id("MNQ_NYSE_CLOSE_E2_RR2.0_CB1_COST_LT10")
        assert dims["instrument"] == "MNQ"
        assert dims["orb_label"] == "NYSE_CLOSE"
        assert dims["entry_model"] == "E2"
        assert dims["rr_target"] == 2.0
        assert dims["confirm_bars"] == 1
        assert dims["filter_type"] == "COST_LT10"
        assert dims["orb_minutes"] == 5

    def test_deployed_mgc_lane(self):
        dims = parse_strategy_id("MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6")
        assert dims["instrument"] == "MGC"
        assert dims["orb_label"] == "CME_REOPEN"
        assert dims["filter_type"] == "ORB_G6"

    def test_aperture_suffix(self):
        dims = parse_strategy_id("MNQ_TOKYO_OPEN_E2_RR2.0_CB1_COST_LT10_O15")
        assert dims["orb_minutes"] == 15
        assert dims["filter_type"] == "COST_LT10"

    def test_real_composite_filter(self):
        dims = parse_strategy_id("MNQ_NYSE_CLOSE_E1_RR2.0_CB1_ORB_G5_FAST5")
        assert dims["filter_type"] == "ORB_G5_FAST5"

    def test_unknown_instrument_raises(self):
        with pytest.raises(ValueError):
            parse_strategy_id("XYZ_NYSE_CLOSE_E2_RR2.0_CB1_COST_LT10")

    def test_missing_entry_model_raises(self):
        with pytest.raises(ValueError):
            parse_strategy_id("MNQ_NYSE_CLOSE_RR2.0_CB1_COST_LT10")


# ==========================================================================
# Adapter rejects unknown filter_type
# ==========================================================================


class TestUnknownFilterType:
    def test_unknown_filter_raises_value_error(self):
        """The new adapter looks filter_type up in ALL_FILTERS. An unknown
        filter is a caller bug, not a data issue — surface it loudly."""
        with pytest.raises(ValueError, match="filter_type"):
            build_eligibility_report(
                strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_TOTALLY_FAKE",
                trading_day=date(2026, 4, 7),
                feature_row=_fresh_row(),
            )


# ==========================================================================
# Status mapping: PASS / FAIL / PENDING / DATA_MISSING
# ==========================================================================


class TestStatusPASS:
    def test_pit_min_pass(self):
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.0_CB1_PIT_MIN",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC", pit_range_atr=0.15),
        )
        pit = [c for c in report.conditions if c.source_filter == "PIT_MIN"]
        assert len(pit) == 1
        assert pit[0].status == ConditionStatus.PASS
        assert pit[0].observed_value == 0.15


class TestStatusFAIL:
    def test_pit_min_fail(self):
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.0_CB1_PIT_MIN",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC", pit_range_atr=0.05),
        )
        pit = [c for c in report.conditions if c.source_filter == "PIT_MIN"][0]
        assert pit.status == ConditionStatus.FAIL


class TestStatusPENDING:
    def test_intra_session_orb_size_pending(self):
        """ORB_G6 is intra-session — pre-session brief has no orb_size yet
        so the condition is PENDING (not DATA_MISSING)."""
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC"),
        )
        orb = [c for c in report.conditions if c.source_filter == "ORB_G6"][0]
        assert orb.status == ConditionStatus.PENDING
        assert orb.category == ConditionCategory.INTRA_SESSION

    def test_direction_filter_pending_not_not_applicable(self):
        """DIR_LONG pre-break: direction is unknown but the filter DOES
        apply — adapter must map to PENDING, not the legacy
        NOT_APPLICABLE_DIRECTION (that was a parallel-model bug).
        """
        report = build_eligibility_report(
            strategy_id="MNQ_TOKYO_OPEN_E2_RR2.0_CB1_DIR_LONG",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        dir_cond = [c for c in report.conditions if c.source_filter == "DIR_LONG"][0]
        assert dir_cond.status == ConditionStatus.PENDING
        assert dir_cond.category == ConditionCategory.DIRECTIONAL


class TestStatusDATAMISSING:
    def test_pit_min_null_is_data_missing_not_fail(self):
        """NULL pit_range_atr must surface as DATA_MISSING, not FAIL."""
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.0_CB1_PIT_MIN",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC", pit_range_atr=None),
        )
        pit = [c for c in report.conditions if c.source_filter == "PIT_MIN"][0]
        assert pit.status == ConditionStatus.DATA_MISSING

    def test_pit_min_nan_is_data_missing(self):
        """NaN pit_range_atr (pandas semantics) must surface as DATA_MISSING."""
        nan = float("nan")
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.0_CB1_PIT_MIN",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC", pit_range_atr=nan),
        )
        pit = [c for c in report.conditions if c.source_filter == "PIT_MIN"][0]
        assert pit.status == ConditionStatus.DATA_MISSING

    def test_pdr_missing_atr_is_data_missing(self):
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_PDR_R080",
            trading_day=date(2026, 4, 7),
            feature_row={
                "trading_day": date(2026, 4, 7),
                "symbol": "MNQ",
                "prev_day_range": 200.0,
                "atr_20": None,
            },
        )
        pdr = [c for c in report.conditions if c.source_filter == "PDR_R080"][0]
        # PDR_R080 was last revalidated 2026-04-02; trading_day=2026-04-07 is
        # within 180 days, so STALE_VALIDATION does not trigger here.
        assert pdr.status == ConditionStatus.DATA_MISSING


# ==========================================================================
# NOT_APPLICABLE_ENTRY_MODEL: E2 look-ahead exclusions
# ==========================================================================


class TestNotApplicableEntryModel:
    def test_cont_e2_is_not_applicable_entry_model(self):
        """BRK_CONT atom inside ORB_G5_CONT composite must be marked
        NOT_APPLICABLE_ENTRY_MODEL when the strategy is E2 (look-ahead:
        bar-close direction unknown at E2 stop-market entry).
        """
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_ORB_G5_CONT",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        cont = [c for c in report.conditions if c.source_filter == "BRK_CONT"][0]
        assert cont.status == ConditionStatus.NOT_APPLICABLE_ENTRY_MODEL

    def test_fast_e2_is_not_applicable_entry_model(self):
        """BRK_FAST5 atom inside ORB_G5_FAST5 composite must be marked
        NOT_APPLICABLE_ENTRY_MODEL when the strategy is E2 (look-ahead:
        break delay unknown at E2 entry placement). The latent bug from
        pre-refactor self-review was that this fell through to PENDING.
        """
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_ORB_G5_FAST5",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        fast = [c for c in report.conditions if c.source_filter == "BRK_FAST5"][0]
        assert fast.status == ConditionStatus.NOT_APPLICABLE_ENTRY_MODEL

    def test_cont_e1_is_pending_not_excluded(self):
        """E1 strategies CAN use BRK_CONT — it must be PENDING (intra-session),
        not NOT_APPLICABLE."""
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E1_RR2.0_CB1_ORB_G5_CONT",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        cont = [c for c in report.conditions if c.source_filter == "BRK_CONT"][0]
        assert cont.status == ConditionStatus.PENDING


# ==========================================================================
# NOT_APPLICABLE_INSTRUMENT: validated_for restrictions
# ==========================================================================


class TestNotApplicableInstrument:
    def test_break_speed_on_unvalidated_lane_is_not_applicable_instrument(self):
        """BRK_FAST5 has VALIDATED_FOR = MNQ at 5 sessions + MGC CME_REOPEN.
        Used at MES NYSE_CLOSE → must surface as NOT_APPLICABLE_INSTRUMENT.
        """
        # Use E1 so the E2 look-ahead exclusion doesn't pre-empt the
        # validated_for check.
        report = build_eligibility_report(
            strategy_id="MES_NYSE_CLOSE_E1_RR2.0_CB1_ORB_G5_FAST5",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MES"),
        )
        fast = [c for c in report.conditions if c.source_filter == "BRK_FAST5"][0]
        assert fast.status == ConditionStatus.NOT_APPLICABLE_INSTRUMENT

    def test_break_speed_on_validated_lane_resolves_normally(self):
        """MNQ NYSE_CLOSE IS in BRK_FAST5.VALIDATED_FOR → no
        NOT_APPLICABLE_INSTRUMENT override; the atom resolves normally.
        Use E1 with break_delay_min present to get a real PASS.
        """
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E1_RR2.0_CB1_ORB_G5_FAST5",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(
                symbol="MNQ",
                **{
                    "orb_NYSE_CLOSE_size": 8.0,
                    "orb_NYSE_CLOSE_break_delay_min": 3.0,
                },
            ),
        )
        fast = [c for c in report.conditions if c.source_filter == "BRK_FAST5"][0]
        assert fast.status == ConditionStatus.PASS

    def test_gap_filter_on_unvalidated_lane_is_not_applicable_instrument(self):
        """GAP_R005 has VALIDATED_FOR = ((MGC, CME_REOPEN),). Used at
        MNQ NYSE_CLOSE → NOT_APPLICABLE_INSTRUMENT."""
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_GAP_R005",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        gap = [c for c in report.conditions if c.source_filter == "GAP_R005"][0]
        assert gap.status == ConditionStatus.NOT_APPLICABLE_INSTRUMENT

    def test_pit_min_on_validated_lane_passes(self):
        """PIT_MIN VALIDATED_FOR includes (MNQ, CME_REOPEN) — must NOT
        be marked NOT_APPLICABLE_INSTRUMENT."""
        report = build_eligibility_report(
            strategy_id="MNQ_CME_REOPEN_E2_RR2.0_CB1_PIT_MIN",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ", pit_range_atr=0.15),
        )
        pit = [c for c in report.conditions if c.source_filter == "PIT_MIN"][0]
        assert pit.status == ConditionStatus.PASS


# ==========================================================================
# STALE_VALIDATION: last_revalidated > 180 days override
# ==========================================================================


class TestStaleValidation:
    def test_pit_min_stale_after_180_days(self):
        """PIT_MIN.LAST_REVALIDATED = 2026-04-04. trading_day far in
        future → adapter overrides PASS to STALE_VALIDATION.
        """
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.0_CB1_PIT_MIN",
            trading_day=date(2027, 1, 1),  # > 180 days after 2026-04-04
            feature_row={
                "trading_day": date(2027, 1, 1),
                "symbol": "MGC",
                "pit_range_atr": 0.15,
            },
        )
        pit = [c for c in report.conditions if c.source_filter == "PIT_MIN"][0]
        assert pit.status == ConditionStatus.STALE_VALIDATION

    def test_pit_min_fresh_within_180_days(self):
        """Same filter, recent trading day → still PASS (not stale)."""
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.0_CB1_PIT_MIN",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC", pit_range_atr=0.15),
        )
        pit = [c for c in report.conditions if c.source_filter == "PIT_MIN"][0]
        assert pit.status == ConditionStatus.PASS


# ==========================================================================
# Composite decomposition: walk yields each leaf as a separate condition
# ==========================================================================


class TestCompositeDecomposition:
    def test_orb_g5_fast5_produces_two_conditions(self):
        """ORB_G5_FAST5 = CompositeFilter(OrbSize, BreakSpeed). Walk
        must yield two conditions tagged ORB_G5 and BRK_FAST5.
        """
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E1_RR2.0_CB1_ORB_G5_FAST5",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        sources = {c.source_filter for c in report.conditions}
        assert "ORB_G5" in sources
        assert "BRK_FAST5" in sources

    def test_orb_g5_cont_produces_two_conditions(self):
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E1_RR2.0_CB1_ORB_G5_CONT",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        sources = {c.source_filter for c in report.conditions}
        assert "ORB_G5" in sources
        assert "BRK_CONT" in sources

    def test_dow_composite_produces_two_conditions(self):
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E1_RR2.0_CB1_ORB_G5_NOFRI",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        sources = {c.source_filter for c in report.conditions}
        assert "ORB_G5" in sources
        assert "DOW_NOFRI" in sources


# ==========================================================================
# ATR velocity overlay (canonical delegation)
# ==========================================================================


class TestATRVelocityOverlay:
    def test_warmup_failopen_passes(self):
        """Canonical fail-open: missing compression tier → PASS (warm-up).
        Pre-refactor parallel model wrongly mapped this to FAIL."""
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            trading_day=date(2026, 4, 7),
            feature_row={
                "trading_day": date(2026, 4, 7),
                "symbol": "MGC",
                "atr_vel_regime": None,  # warm-up
                "orb_CME_REOPEN_compression_tier": None,
            },
        )
        atr_vel = [c for c in report.conditions if c.source_filter == "atr_velocity"]
        assert len(atr_vel) == 1
        assert atr_vel[0].status == ConditionStatus.PASS

    def test_contracting_neutral_fails(self):
        """Contracting + Neutral compression = SKIP per canonical filter."""
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            trading_day=date(2026, 4, 7),
            feature_row={
                "trading_day": date(2026, 4, 7),
                "symbol": "MGC",
                "atr_vel_regime": "Contracting",
                "orb_CME_REOPEN_compression_tier": "Neutral",
            },
        )
        atr_vel = [c for c in report.conditions if c.source_filter == "atr_velocity"][0]
        assert atr_vel.status == ConditionStatus.FAIL

    def test_contracting_expanded_passes(self):
        """Contracting + Expanded compression = ALLOWED per canonical filter."""
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            trading_day=date(2026, 4, 7),
            feature_row={
                "trading_day": date(2026, 4, 7),
                "symbol": "MGC",
                "atr_vel_regime": "Contracting",
                "orb_CME_REOPEN_compression_tier": "Expanded",
            },
        )
        atr_vel = [c for c in report.conditions if c.source_filter == "atr_velocity"][0]
        assert atr_vel.status == ConditionStatus.PASS

    def test_stable_always_passes(self):
        """Stable vel_regime → PASS regardless of compression tier."""
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            trading_day=date(2026, 4, 7),
            feature_row={
                "trading_day": date(2026, 4, 7),
                "symbol": "MGC",
                "atr_vel_regime": "Stable",
                "orb_CME_REOPEN_compression_tier": "Compressed",
            },
        )
        atr_vel = [c for c in report.conditions if c.source_filter == "atr_velocity"][0]
        assert atr_vel.status == ConditionStatus.PASS

    def test_non_monitored_session_skipped(self):
        """Sessions outside ATRVelocityFilter.apply_to_sessions → adapter
        does NOT add the atr_velocity overlay condition (it's not
        applicable). Old behavior preserved."""
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_NO_FILTER",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        atr_vel = [c for c in report.conditions if c.source_filter == "atr_velocity"]
        assert len(atr_vel) == 0


# ==========================================================================
# Calendar overlay (HALF_SIZE → PASS with size_multiplier=0.5)
# ==========================================================================


class TestCalendarOverlay:
    def test_calendar_condition_always_added(self):
        """Calendar overlay must always appear in the report."""
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_NO_FILTER",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        cal = [c for c in report.conditions if c.source_filter == "calendar"]
        assert len(cal) == 1

    def test_rules_not_loaded_when_file_missing(self):
        """If calendar_cascade_rules.json is missing → RULES_NOT_LOADED."""
        from trading_app.eligibility.builder import _calendar_rules_loaded

        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_NO_FILTER",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        cal = [c for c in report.conditions if c.source_filter == "calendar"][0]
        if not _calendar_rules_loaded():
            assert cal.status == ConditionStatus.RULES_NOT_LOADED
        else:
            # Rules loaded — must not be silently NEUTRAL
            assert cal.status in (
                ConditionStatus.PASS,
                ConditionStatus.FAIL,
                ConditionStatus.DATA_MISSING,
            )

    def test_halfsize_size_multiplier_type_contract(self):
        """Type contract: a calendar HALF_SIZE record passes the eligibility
        gate with size_multiplier=0.5 and is_blocking=False."""
        record = ConditionRecord(
            name="calendar action",
            category=ConditionCategory.OVERLAY,
            status=ConditionStatus.PASS,
            resolves_at=ResolvesAt.STARTUP,
            source_filter="calendar",
            observed_value="HALF_SIZE",
            size_multiplier=0.5,
        )
        assert record.status == ConditionStatus.PASS
        assert record.size_multiplier == 0.5
        assert record.is_blocking is False

    def test_effective_size_multiplier_compounds(self):
        """EligibilityReport.effective_size_multiplier = product of all
        PASSING conditions' size_multiplier values."""
        conds = (
            ConditionRecord(
                name="passes full",
                category=ConditionCategory.PRE_SESSION,
                status=ConditionStatus.PASS,
                resolves_at=ResolvesAt.STARTUP,
                source_filter="A",
                size_multiplier=1.0,
            ),
            ConditionRecord(
                name="passes half",
                category=ConditionCategory.OVERLAY,
                status=ConditionStatus.PASS,
                resolves_at=ResolvesAt.STARTUP,
                source_filter="calendar",
                size_multiplier=0.5,
            ),
        )
        report = EligibilityReport(
            strategy_id="X_Y_E2_RR2.0_CB1_NO_FILTER",
            instrument="X",
            session="Y",
            entry_model="E2",
            trading_day=date(2026, 4, 7),
            as_of_timestamp=None,
            freshness_status=FreshnessStatus.FRESH,
            conditions=conds,
            overall_status=OverallStatus.ELIGIBLE,
        )
        assert report.effective_size_multiplier == 0.5


# ==========================================================================
# error_message aggregation into report.build_errors
# ==========================================================================


class TestErrorMessageAggregation:
    def test_pdr_type_mismatch_appears_in_build_errors(self):
        """PrevDayRangeNormFilter explicitly catches type mismatches and
        sets AtomDescription.error_message. Adapter aggregates non-None
        error_messages into report.build_errors."""
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_PDR_R080",
            trading_day=date(2026, 4, 7),
            feature_row={
                "trading_day": date(2026, 4, 7),
                "symbol": "MNQ",
                "prev_day_range": "bad_string",  # wrong type
                "atr_20": 150.0,
            },
        )
        pdr = [c for c in report.conditions if c.source_filter == "PDR_R080"][0]
        assert pdr.status == ConditionStatus.DATA_MISSING
        assert any("PDR" in e and "type mismatch" in e for e in report.build_errors)

    def test_gap_type_mismatch_appears_in_build_errors(self):
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.0_CB1_GAP_R005",
            trading_day=date(2026, 4, 7),
            feature_row={
                "trading_day": date(2026, 4, 7),
                "symbol": "MGC",
                "gap_open_points": "bad",
                "atr_20": 150.0,
            },
        )
        gap = [c for c in report.conditions if c.source_filter == "GAP_R005"][0]
        assert gap.status == ConditionStatus.DATA_MISSING
        assert any("GAP" in e and "type mismatch" in e for e in report.build_errors)

    def test_pdr_zero_atr_appears_in_build_errors(self):
        """atr_20 <= 0 is treated as DATA_MISSING with an error_message."""
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_PDR_R080",
            trading_day=date(2026, 4, 7),
            feature_row={
                "trading_day": date(2026, 4, 7),
                "symbol": "MNQ",
                "prev_day_range": 200.0,
                "atr_20": 0.0,
            },
        )
        pdr = [c for c in report.conditions if c.source_filter == "PDR_R080"][0]
        assert pdr.status == ConditionStatus.DATA_MISSING
        assert any("PDR" in e and "atr_20" in e for e in report.build_errors)


# ==========================================================================
# Freshness detection
# ==========================================================================


class TestFreshness:
    def test_fresh_row(self):
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC"),
        )
        assert report.freshness_status == FreshnessStatus.FRESH

    def test_prior_day_row(self):
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            trading_day=date(2026, 4, 7),
            feature_row={"trading_day": date(2026, 4, 6), "symbol": "MGC"},
        )
        assert report.freshness_status == FreshnessStatus.PRIOR_DAY

    def test_stale_row(self):
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            trading_day=date(2026, 4, 7),
            feature_row={"trading_day": date(2026, 4, 4), "symbol": "MGC"},
        )
        assert report.freshness_status == FreshnessStatus.STALE

    def test_no_data(self):
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            trading_day=date(2026, 4, 7),
            feature_row=None,
        )
        assert report.freshness_status == FreshnessStatus.NO_DATA
        assert report.overall_status == OverallStatus.DATA_MISSING


# ==========================================================================
# Overall status derivation
# ==========================================================================


class TestOverallStatus:
    def test_eligible_when_no_filter_no_pending(self):
        """NO_FILTER strategy → no atoms from filter, only overlays.
        If overlays don't FAIL, overall is ELIGIBLE or NEEDS_LIVE_DATA."""
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_NO_FILTER",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )
        assert report.overall_status in (
            OverallStatus.ELIGIBLE,
            OverallStatus.NEEDS_LIVE_DATA,
        )

    def test_needs_live_data_when_intra_session_pending(self):
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC"),
        )
        assert report.overall_status == OverallStatus.NEEDS_LIVE_DATA

    def test_data_missing_overrides_eligibility(self):
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.0_CB1_PIT_MIN",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC", pit_range_atr=None),
        )
        assert report.overall_status in (
            OverallStatus.DATA_MISSING,
            OverallStatus.NEEDS_LIVE_DATA,
        )

    def test_ineligible_when_pre_session_fail(self):
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.0_CB1_PIT_MIN",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC", pit_range_atr=0.05),
        )
        assert report.overall_status == OverallStatus.INELIGIBLE


# ==========================================================================
# Error robustness: builder must never raise on bad/missing data
# ==========================================================================


class TestBuilderRobustness:
    def test_missing_feature_row_returns_no_data(self):
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_COST_LT10",
            trading_day=date(2026, 4, 7),
            feature_row=None,
        )
        assert report.overall_status == OverallStatus.DATA_MISSING
        assert report.freshness_status == FreshnessStatus.NO_DATA

    def test_empty_feature_row_returns_data_missing_or_pending(self):
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_PDR_R080",
            trading_day=date(2026, 4, 7),
            feature_row={"trading_day": date(2026, 4, 7), "symbol": "MNQ"},
        )
        pdr = [c for c in report.conditions if c.source_filter == "PDR_R080"]
        assert len(pdr) == 1
        assert pdr[0].status == ConditionStatus.DATA_MISSING

    def test_malformed_strategy_id_raises(self):
        """Parsing errors are caller bugs, not data issues — they raise."""
        with pytest.raises(ValueError):
            build_eligibility_report(
                strategy_id="not_a_strategy_id",
                trading_day=date(2026, 4, 7),
                feature_row={},
            )

    def test_db_connection_failure_populates_build_errors(self):
        """Infrastructure failure (DB connect) appends to build_errors."""
        from pathlib import Path

        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_NO_FILTER",
            trading_day=date(2026, 4, 7),
            db_path=Path("/nonexistent/path/to/nowhere.db"),
        )
        assert len(report.build_errors) >= 1
        assert any("duckdb" in e.lower() for e in report.build_errors)


# ==========================================================================
# Confidence tier propagation
# ==========================================================================


class TestConfidenceTierPropagation:
    def test_pit_min_carries_proven_tier(self):
        """PitRangeFilter.CONFIDENCE_TIER = PROVEN must reach the
        ConditionRecord."""
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.0_CB1_PIT_MIN",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC", pit_range_atr=0.15),
        )
        pit = [c for c in report.conditions if c.source_filter == "PIT_MIN"][0]
        # ConfidenceTier enum or string — test the value
        tier_str = pit.confidence_tier.value if hasattr(pit.confidence_tier, "value") else str(pit.confidence_tier)
        assert tier_str == "PROVEN"


# ==========================================================================
# Fail-closed discipline: describe() exceptions must surface visibly
#
# Regression tests for Bloomey code review findings on the
# canonical-filter-self-description refactor. The original thin-adapter
# rewrite had a silent-failure gap: when a filter's describe() raised,
# the exception was captured in build_errors but no ConditionRecord was
# produced, so overall_status still derived to ELIGIBLE. Same issue for
# ATR_VELOCITY_OVERLAY.describe() (no try/except at all, could propagate).
#
# Fix principle: infrastructure failures surface as DATA_MISSING
# conditions in the report, not just in the build_errors side-channel.
# ==========================================================================


class TestFailClosedOnDescribeException:
    """Regression: a filter's describe() raising must NOT produce ELIGIBLE."""

    def test_describe_exception_surfaces_as_data_missing_condition(self):
        """Inject a broken filter into ALL_FILTERS. Verify the adapter:
          1. Does NOT return ELIGIBLE (fail-closed)
          2. Adds a synthetic DATA_MISSING condition for the broken filter
          3. Preserves the exception detail in build_errors
        """
        from dataclasses import dataclass

        from trading_app.config import ALL_FILTERS, StrategyFilter

        @dataclass(frozen=True)
        class _BrokenFilter(StrategyFilter):
            def describe(self, row, orb_label, entry_model):
                raise RuntimeError("simulated describe failure")

        ALL_FILTERS["BROKEN_TEST_FIXTURE"] = _BrokenFilter(
            filter_type="BROKEN_TEST_FIXTURE",
            description="test fixture — raises in describe()",
        )
        try:
            report = build_eligibility_report(
                strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_BROKEN_TEST_FIXTURE",
                trading_day=date(2026, 4, 7),
                feature_row={"trading_day": date(2026, 4, 7), "symbol": "MNQ"},
            )
        finally:
            del ALL_FILTERS["BROKEN_TEST_FIXTURE"]

        # 1. Not ELIGIBLE
        assert report.overall_status != OverallStatus.ELIGIBLE

        # 2. Synthetic DATA_MISSING condition is present for the broken filter
        broken_conds = [
            c for c in report.conditions if c.source_filter == "BROKEN_TEST_FIXTURE"
        ]
        assert len(broken_conds) == 1
        assert broken_conds[0].status == ConditionStatus.DATA_MISSING

        # 3. build_errors preserves exception detail
        assert any(
            "BROKEN_TEST_FIXTURE" in e and "describe raised" in e
            for e in report.build_errors
        )
        # The error message should name the actual exception type + message
        assert any(
            "RuntimeError" in e and "simulated describe failure" in e
            for e in report.build_errors
        )

    def test_describe_exception_inside_composite_preserves_sibling_atoms(self):
        """If ONE leaf of a composite raises but the other succeeds, the
        working leaf's atoms must still appear in the report. The broken
        leaf should produce a synthetic DATA_MISSING.
        """
        from dataclasses import dataclass

        from trading_app.config import (
            ALL_FILTERS,
            CompositeFilter,
            OrbSizeFilter,
            StrategyFilter,
        )

        @dataclass(frozen=True)
        class _BrokenOverlay(StrategyFilter):
            def describe(self, row, orb_label, entry_model):
                raise RuntimeError("overlay describe failure")

        composite = CompositeFilter(
            filter_type="ORB_G5_BROKEN_OVERLAY",
            description="composite with broken overlay",
            base=OrbSizeFilter(
                filter_type="ORB_G5",
                description="ORB size >= 5 points",
                min_size=5.0,
            ),
            overlay=_BrokenOverlay(
                filter_type="BRK_OVERLAY_TEST",
                description="broken overlay leaf",
            ),
        )
        ALL_FILTERS["ORB_G5_BROKEN_OVERLAY"] = composite
        try:
            report = build_eligibility_report(
                strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_ORB_G5_BROKEN_OVERLAY",
                trading_day=date(2026, 4, 7),
                feature_row={"trading_day": date(2026, 4, 7), "symbol": "MNQ"},
            )
        finally:
            del ALL_FILTERS["ORB_G5_BROKEN_OVERLAY"]

        sources = {c.source_filter for c in report.conditions}
        # Working leaf (OrbSize) still produced its condition
        assert "ORB_G5" in sources
        # Broken leaf surfaced as a separate condition tagged with its filter_type
        assert "BRK_OVERLAY_TEST" in sources
        broken_cond = [
            c for c in report.conditions if c.source_filter == "BRK_OVERLAY_TEST"
        ][0]
        assert broken_cond.status == ConditionStatus.DATA_MISSING
        # Overall must surface the infrastructure failure
        assert report.overall_status != OverallStatus.ELIGIBLE


class TestFailClosedOnATRVelocityException:
    """Regression: ATR_VELOCITY_OVERLAY.describe() raising must surface as
    DATA_MISSING on validated lanes (not propagate uncaught).
    """

    def test_atr_velocity_describe_exception_on_validated_lane_surfaces_data_missing(
        self, monkeypatch
    ):
        """Patch ATRVelocityFilter.describe at the class level to raise.
        For MGC CME_REOPEN (a validated lane per VALIDATED_FOR), the
        adapter must:
          1. NOT propagate the exception
          2. Append the failure to build_errors
          3. Return a synthetic DATA_MISSING atr_velocity condition
        """
        from trading_app.config import ATRVelocityFilter

        def _raising_describe(self, row, orb_label, entry_model):
            raise RuntimeError("simulated atr velocity failure")

        monkeypatch.setattr(ATRVelocityFilter, "describe", _raising_describe)

        # MGC CME_REOPEN is in ATR_VELOCITY_OVERLAY.VALIDATED_FOR —
        # the adapter WILL call describe() on this lane.
        report = build_eligibility_report(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MGC"),
        )

        atr_vel = [c for c in report.conditions if c.source_filter == "atr_velocity"]
        assert len(atr_vel) == 1
        assert atr_vel[0].status == ConditionStatus.DATA_MISSING

        assert any(
            "ATR_VELOCITY_OVERLAY" in e and "describe raised" in e
            for e in report.build_errors
        )

    def test_atr_velocity_exception_on_non_validated_lane_skipped(
        self, monkeypatch
    ):
        """If ATRVelocityFilter.describe would raise, but the lane is OUTSIDE
        VALIDATED_FOR (e.g. MNQ NYSE_CLOSE), the adapter MUST NOT call
        describe() at all — pre-check on VALIDATED_FOR. The overlay is
        simply absent from the report, and no error appears in build_errors.

        Prevents polluting unrelated-lane reports with spurious DATA_MISSING
        atr_velocity conditions when the overlay doesn't apply anyway.
        """
        from trading_app.config import ATRVelocityFilter

        def _raising_describe(self, row, orb_label, entry_model):
            raise RuntimeError("should not be called on non-validated lane")

        monkeypatch.setattr(ATRVelocityFilter, "describe", _raising_describe)

        # MNQ NYSE_CLOSE is NOT in ATR_VELOCITY_OVERLAY.VALIDATED_FOR
        # (VALIDATED_FOR = ((MGC, CME_REOPEN), (MGC, TOKYO_OPEN)))
        report = build_eligibility_report(
            strategy_id="MNQ_NYSE_CLOSE_E2_RR2.0_CB1_NO_FILTER",
            trading_day=date(2026, 4, 7),
            feature_row=_fresh_row(symbol="MNQ"),
        )

        # Overlay skipped entirely — no atr_velocity condition, no error
        atr_vel = [c for c in report.conditions if c.source_filter == "atr_velocity"]
        assert len(atr_vel) == 0
        assert not any("ATR_VELOCITY" in e for e in report.build_errors)
