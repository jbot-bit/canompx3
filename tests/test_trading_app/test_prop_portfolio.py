"""Tests for trading_app.prop_portfolio — prop firm portfolio selection."""

from datetime import date

import pytest

from trading_app.portfolio import PortfolioStrategy
from trading_app.prop_profiles import (
    AccountProfile,
)
from trading_app.prop_portfolio import (
    _calendar_gate_reason,
    _compute_dd_per_contract,
    _apply_instrument_bans,
    _deduplicate_sessions,
    _format_time_ampm,
    _rank_strategies,
    _time_sort_key,
    select_for_profile,
)


def _make_strategy(**overrides) -> PortfolioStrategy:
    """Build a PortfolioStrategy with sane defaults."""
    defaults = dict(
        strategy_id="MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5",
        instrument="MGC",
        orb_label="TOKYO_OPEN",
        entry_model="E2",
        rr_target=2.0,
        confirm_bars=1,
        filter_type="ORB_G5",
        expectancy_r=0.25,
        win_rate=0.55,
        sample_size=150,
        sharpe_ratio=1.2,
        max_drawdown_r=4.0,
        median_risk_points=10.0,
        orb_minutes=5,
        stop_multiplier=1.0,
        source="baseline",
        weight=1.0,
    )
    defaults.update(overrides)
    return PortfolioStrategy(**defaults)


class TestDDPerContract:
    def test_075x_stop(self):
        dd = _compute_dd_per_contract(stop_multiplier=0.75, dd_type="eod_trailing")
        assert dd == pytest.approx(935.0)

    def test_10x_stop(self):
        dd = _compute_dd_per_contract(stop_multiplier=1.0, dd_type="eod_trailing")
        assert dd == pytest.approx(1350.0)

    def test_intraday_trailing_adjustment(self):
        dd_eod = _compute_dd_per_contract(0.75, "eod_trailing")
        dd_intra = _compute_dd_per_contract(0.75, "intraday_trailing")
        assert dd_intra > dd_eod


class TestInstrumentBans:
    def test_apex_bans_mgc(self):
        strats = [
            _make_strategy(instrument="MGC"),
            _make_strategy(instrument="MNQ", strategy_id="MNQ_TOKYO_E2"),
        ]
        banned = frozenset({"MGC", "GC", "SI"})
        filtered, excluded = _apply_instrument_bans(strats, banned)
        assert len(filtered) == 1
        assert filtered[0].instrument == "MNQ"
        assert len(excluded) == 1
        assert "banned" in excluded[0].reason.lower()

    def test_no_bans(self):
        strats = [_make_strategy()]
        filtered, excluded = _apply_instrument_bans(strats, frozenset())
        assert len(filtered) == 1
        assert len(excluded) == 0


class TestDeduplicateSessions:
    def test_keeps_best_per_session_instrument(self):
        s1 = _make_strategy(expectancy_r=0.20, strategy_id="s1")
        s2 = _make_strategy(expectancy_r=0.30, strategy_id="s2")
        deduped, excluded = _deduplicate_sessions([s1, s2])
        assert len(deduped) == 1
        assert deduped[0].strategy_id == "s2"

    def test_different_instruments_kept(self):
        s1 = _make_strategy(instrument="MGC", strategy_id="s1")
        s2 = _make_strategy(instrument="MNQ", strategy_id="s2")
        deduped, _ = _deduplicate_sessions([s1, s2])
        assert len(deduped) == 2


class TestRankStrategies:
    def test_ranks_by_expr_dd_ratio(self):
        """Higher ExpR/DD ratio ranks first (project rule: sort by ExpR, never Sharpe)."""
        s1 = _make_strategy(expectancy_r=0.10, max_drawdown_r=4.0, strategy_id="s1")
        s2 = _make_strategy(expectancy_r=0.30, max_drawdown_r=4.0, strategy_id="s2")
        ranked = _rank_strategies([s1, s2], split_factor=1.0)
        assert ranked[0].strategy.strategy_id == "s2"


class TestSelectForProfile:
    def test_dd_budget_exhaustion(self):
        profile = AccountProfile("test", "topstep", 50_000, 1, 0.75, max_slots=10)
        strats = [_make_strategy(strategy_id=f"s{i}", orb_label=f"SESSION_{i}") for i in range(5)]
        book = select_for_profile(profile, strats)
        assert book.total_slots == 2
        assert book.total_dd_used <= 2_000
        assert len(book.excluded) == 3

    def test_slot_cap(self):
        # max_slots=3 with 0.75x stop ($935/slot) and $5K budget = DD fits 5 slots
        # so the cognitive cap (3) is the binding constraint, not DD
        profile = AccountProfile("test", "self_funded", 50_000, 1, 0.75, max_slots=3)
        strats = [_make_strategy(strategy_id=f"s{i}", orb_label=f"SESSION_{i}") for i in range(5)]
        book = select_for_profile(profile, strats)
        assert book.total_slots == 3
        assert any("cognitive cap" in e.reason.lower() for e in book.excluded)

    def test_contract_cap(self):
        profile = AccountProfile("test", "tradeify", 50_000, 1, 0.75, max_slots=10)
        strats = [_make_strategy(strategy_id=f"s{i}", orb_label=f"SESSION_{i}") for i in range(5)]
        book = select_for_profile(profile, strats)
        assert book.total_contracts <= 40

    def test_empty_strategies(self):
        profile = AccountProfile("test", "topstep", 50_000)
        book = select_for_profile(profile, [])
        assert book.total_slots == 0
        assert len(book.excluded) == 0

    def test_consistency_rule(self):
        profile = AccountProfile("test", "topstep", 50_000, 1, 0.75, max_slots=10)
        dominant = _make_strategy(strategy_id="dominant", orb_label="TOKYO_OPEN", expectancy_r=0.80)
        weak = [_make_strategy(strategy_id=f"w{i}", orb_label=f"SESSION_{i}", expectancy_r=0.10) for i in range(4)]
        book = select_for_profile(profile, [dominant] + weak)
        assert book.total_slots >= 1


class TestSessionRouting:
    """Session→firm routing from playbook account grid."""

    def test_topstep_only_gets_mgc_morning(self):
        """TopStep profile: only CME_REOPEN + TOKYO_OPEN, only MGC."""
        pool = [
            _make_strategy(instrument="MGC", orb_label="CME_REOPEN", strategy_id="mgc_reopen"),
            _make_strategy(instrument="MGC", orb_label="TOKYO_OPEN", strategy_id="mgc_tokyo"),
            _make_strategy(instrument="MNQ", orb_label="CME_PRECLOSE", strategy_id="mnq_preclose"),
            _make_strategy(instrument="MGC", orb_label="NYSE_OPEN", strategy_id="mgc_nyse"),
        ]
        profile = AccountProfile(
            "test_ts",
            "topstep",
            50_000,
            1,
            0.75,
            max_slots=4,
            allowed_sessions=frozenset({"CME_REOPEN", "TOKYO_OPEN"}),
            allowed_instruments=frozenset({"MGC"}),
        )
        book = select_for_profile(profile, pool)
        # Should only have CME_REOPEN + TOKYO_OPEN, both MGC
        for e in book.entries:
            assert e.instrument == "MGC"
            assert e.orb_label in {"CME_REOPEN", "TOKYO_OPEN"}
        # MNQ_PRECLOSE excluded (wrong session + wrong instrument)
        # MGC_NYSE excluded (wrong session)
        assert any("not assigned" in ex.reason.lower() for ex in book.excluded)

    def test_tradeify_only_gets_mnq_overnight(self):
        """Tradeify profile: only overnight sessions, only MNQ."""
        pool = [
            _make_strategy(instrument="MNQ", orb_label="CME_PRECLOSE", strategy_id="mnq_pre"),
            _make_strategy(instrument="MNQ", orb_label="COMEX_SETTLE", strategy_id="mnq_comex"),
            _make_strategy(instrument="MGC", orb_label="CME_REOPEN", strategy_id="mgc_reopen"),
            _make_strategy(instrument="MNQ", orb_label="TOKYO_OPEN", strategy_id="mnq_tokyo"),
        ]
        profile = AccountProfile(
            "test_tf",
            "tradeify",
            50_000,
            1,
            0.75,
            max_slots=6,
            allowed_sessions=frozenset({"CME_PRECLOSE", "COMEX_SETTLE", "NYSE_CLOSE", "NYSE_OPEN"}),
            allowed_instruments=frozenset({"MNQ"}),
        )
        book = select_for_profile(profile, pool)
        for e in book.entries:
            assert e.instrument == "MNQ"
            assert e.orb_label in {"CME_PRECLOSE", "COMEX_SETTLE", "NYSE_CLOSE", "NYSE_OPEN"}
        # MGC_REOPEN excluded (wrong instrument)
        # MNQ_TOKYO excluded (wrong session for this profile)
        assert any("not assigned" in ex.reason.lower() for ex in book.excluded)

    def test_no_routing_allows_all(self):
        """Profile with no allowed_sessions/instruments = no filtering."""
        pool = [
            _make_strategy(instrument="MGC", orb_label="TOKYO_OPEN", strategy_id="s1"),
            _make_strategy(instrument="MNQ", orb_label="CME_PRECLOSE", strategy_id="s2"),
        ]
        profile = AccountProfile("test_open", "self_funded", 50_000, 1, 1.0, max_slots=10)
        book = select_for_profile(profile, pool)
        assert book.total_slots == 2  # Both accepted


class TestEndToEnd:
    """Integration tests with realistic multi-instrument strategy pools."""

    def test_select_from_real_strategies(self):
        pool = [
            _make_strategy(
                strategy_id=f"MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G{g}",
                instrument="MGC",
                orb_label="TOKYO_OPEN",
                expectancy_r=0.20 + g * 0.01,
                sharpe_ratio=1.0 + g * 0.1,
                max_drawdown_r=3.0 + g * 0.5,
            )
            for g in [4, 5, 6]
        ] + [
            _make_strategy(
                strategy_id="MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_VOL",
                instrument="MNQ",
                orb_label="SINGAPORE_OPEN",
                expectancy_r=0.25,
                sharpe_ratio=1.3,
                max_drawdown_r=3.5,
            ),
            _make_strategy(
                strategy_id="MES_NYSE_OPEN_E1_RR2.0_CB3_G4",
                instrument="MES",
                orb_label="NYSE_OPEN",
                expectancy_r=0.22,
                sharpe_ratio=1.1,
                max_drawdown_r=4.0,
            ),
            _make_strategy(
                strategy_id="M2K_NYSE_OPEN_E2_RR1.0_CB1_VOL",
                instrument="M2K",
                orb_label="NYSE_OPEN",
                expectancy_r=0.18,
                sharpe_ratio=0.9,
                max_drawdown_r=5.0,
            ),
        ]

        # No routing constraints (generic topstep)
        profile = AccountProfile("test", "topstep", 50_000, 1, 0.75, max_slots=6)
        book = select_for_profile(profile, pool)

        assert book.total_slots > 0
        assert book.total_dd_used <= 2_000
        # TOKYO_OPEN MGC should be deduped to 1
        tokyo_mgc = [e for e in book.entries if e.orb_label == "TOKYO_OPEN" and e.instrument == "MGC"]
        assert len(tokyo_mgc) <= 1
        assert len(book.excluded) > 0

    def test_apex_blocks_mgc(self):
        pool = [
            _make_strategy(instrument="MGC", strategy_id="mgc1"),
            _make_strategy(instrument="MNQ", strategy_id="mnq1", orb_label="SINGAPORE_OPEN"),
        ]
        profile = AccountProfile(
            "test", "apex", 50_000, 1, 0.75, max_slots=6, allowed_sessions=frozenset({"SINGAPORE_OPEN", "CME_REOPEN"})
        )
        book = select_for_profile(profile, pool)
        assert all(e.instrument != "MGC" for e in book.entries)
        assert any("banned" in ex.reason.lower() for ex in book.excluded)

    def test_self_funded_more_slots(self):
        pool = [
            _make_strategy(
                strategy_id=f"s{i}",
                orb_label=f"SESSION_{i}",
                instrument=["MGC", "MNQ", "MES", "M2K"][i % 4],
            )
            for i in range(10)
        ]
        prop_profile = AccountProfile("prop", "topstep", 50_000, 1, 0.75, max_slots=6)
        self_profile = AccountProfile("self", "self_funded", 50_000, 1, 1.0, max_slots=10)

        prop_book = select_for_profile(prop_profile, pool)
        self_book = select_for_profile(self_profile, pool)

        assert self_book.total_slots >= prop_book.total_slots


class TestCalendarGate:
    """Calendar filter skip logic (NOMON/NOTUE/NOFRI)."""

    def test_nomon_skips_monday(self):
        monday = date(2026, 3, 16)  # Monday
        assert _calendar_gate_reason("ORB_G6_NOMON", monday) is not None

    def test_nomon_allows_tuesday(self):
        tuesday = date(2026, 3, 17)  # Tuesday
        assert _calendar_gate_reason("ORB_G6_NOMON", tuesday) is None

    def test_nofri_skips_friday(self):
        friday = date(2026, 3, 20)  # Friday
        assert _calendar_gate_reason("ORB_G4_NOFRI", friday) is not None

    def test_nofri_allows_wednesday(self):
        wednesday = date(2026, 3, 18)
        assert _calendar_gate_reason("ORB_G4_NOFRI", wednesday) is None

    def test_no_filter_allows_all(self):
        assert _calendar_gate_reason("ORB_G5", date(2026, 3, 16)) is None
        assert _calendar_gate_reason(None, date(2026, 3, 16)) is None


class TestTimeSortKey:
    """Brisbane time sorting starting from 08:00."""

    def test_morning_before_evening(self):
        assert _time_sort_key("08:00") < _time_sort_key("18:00")

    def test_overnight_after_evening(self):
        assert _time_sort_key("23:30") < _time_sort_key("05:45")  # both after 08:00 in cycle

    def test_early_morning_wraps(self):
        # 05:45 AM is "tomorrow" in trading day — after 23:30
        assert _time_sort_key("05:45") > _time_sort_key("23:30")

    def test_unknown_sorts_last(self):
        assert _time_sort_key("unknown") == 9999


class TestFormatTimeAmPm:
    """HH:MM to h:MM AM/PM conversion."""

    def test_midnight(self):
        assert _format_time_ampm("00:00") == "12:00 AM"

    def test_noon(self):
        assert _format_time_ampm("12:00") == "12:00 PM"

    def test_morning(self):
        assert _format_time_ampm("08:00") == "8:00 AM"

    def test_evening(self):
        assert _format_time_ampm("23:30") == "11:30 PM"

    def test_unknown_passthrough(self):
        assert _format_time_ampm("unknown") == "unknown"
