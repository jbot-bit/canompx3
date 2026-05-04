"""Tests for scripts/tools/generate_profile_lanes.py."""

from __future__ import annotations

from scripts.tools import generate_profile_lanes
from trading_app.lane_allocator import LaneScore
from trading_app.prop_profiles import AccountProfile, DailyLaneSpec


def test_select_profile_ids_defaults_to_inactive_only() -> None:
    profile_ids = generate_profile_lanes.select_profile_ids()

    assert "topstep_50k_mnq_auto" not in profile_ids
    assert "tradeify_50k" in profile_ids


def test_select_profile_ids_include_active_adds_active_profiles() -> None:
    profile_ids = generate_profile_lanes.select_profile_ids(include_active=True)

    assert "topstep_50k_mnq_auto" in profile_ids
    assert "tradeify_50k" in profile_ids


def test_split_current_lanes_separates_valid_and_ghosts() -> None:
    valid_lane = DailyLaneSpec("VALID_A", "MNQ", "TOKYO_OPEN")
    ghost_lane = DailyLaneSpec("GHOST_B", "MNQ", "COMEX_SETTLE")

    valid, ghosts = generate_profile_lanes.split_current_lanes(
        (valid_lane, ghost_lane),
        {"VALID_A"},
    )

    assert valid == [valid_lane]
    assert ghosts == [ghost_lane]


def test_lane_cap_for_prefers_session_specific_stat_then_falls_back() -> None:
    orb_stats = {("MNQ", "TOKYO_OPEN", 5): (30.0, 45.6)}

    assert generate_profile_lanes.lane_cap_for("MNQ", "TOKYO_OPEN", 5, orb_stats) == 45.6
    assert generate_profile_lanes.lane_cap_for("MNQ", "COMEX_SETTLE", 5, orb_stats) == 120.0


def test_lane_cap_for_distinguishes_apertures() -> None:
    """O15 apertures must look up O15-keyed stats, not fall back to O5."""
    orb_stats = {
        ("MNQ", "US_DATA_1000", 5): (56.2, 94.9),
        ("MNQ", "US_DATA_1000", 15): (86.7, 146.5),
    }
    # O5 strategy gets O5 stat
    assert generate_profile_lanes.lane_cap_for("MNQ", "US_DATA_1000", 5, orb_stats) == 94.9
    # O15 strategy gets O15 stat (the bug being fixed)
    assert generate_profile_lanes.lane_cap_for("MNQ", "US_DATA_1000", 15, orb_stats) == 146.5
    # Missing aperture falls back to instrument default, not the wrong-aperture stat
    assert generate_profile_lanes.lane_cap_for("MNQ", "US_DATA_1000", 30, orb_stats) == 120.0


def test_summarize_lane_delta_reports_kept_dropped_and_added() -> None:
    current_valid = [
        DailyLaneSpec("KEEP_ME", "MNQ", "TOKYO_OPEN"),
        DailyLaneSpec("DROP_ME", "MNQ", "COMEX_SETTLE"),
    ]
    allocation = [
        LaneScore(
            strategy_id="KEEP_ME",
            instrument="MNQ",
            orb_label="TOKYO_OPEN",
            orb_minutes=5,
            rr_target=1.5,
            filter_type="COST_LT12",
            confirm_bars=1,
            stop_multiplier=0.75,
            trailing_expr=0.1,
            trailing_n=30,
            trailing_months=12,
            annual_r_estimate=20.0,
            trailing_wr=0.5,
            session_regime_expr=0.1,
            months_negative=0,
            months_positive_since_last_neg_streak=3,
            status="DEPLOY",
            status_reason="Session HOT",
        ),
        LaneScore(
            strategy_id="ADD_ME",
            instrument="MNQ",
            orb_label="US_DATA_1000",
            orb_minutes=15,
            rr_target=1.5,
            filter_type="ORB_G5_O15",
            confirm_bars=1,
            stop_multiplier=0.75,
            trailing_expr=0.2,
            trailing_n=40,
            trailing_months=12,
            annual_r_estimate=35.0,
            trailing_wr=0.55,
            session_regime_expr=0.2,
            months_negative=0,
            months_positive_since_last_neg_streak=4,
            status="DEPLOY",
            status_reason="Session HOT",
        ),
    ]

    kept, dropped, added = generate_profile_lanes.summarize_lane_delta(current_valid, allocation)

    assert kept == ["KEEP_ME"]
    assert dropped == ["DROP_ME"]
    assert added == ["ADD_ME"]


def test_print_profile_report_surfaces_dropped_valid_lane(
    monkeypatch,
    capsys,
) -> None:
    profile = AccountProfile(
        profile_id="dummy_profile",
        firm="topstep",
        account_size=50_000,
        active=False,
        allowed_sessions=frozenset({"TOKYO_OPEN", "US_DATA_1000"}),
        allowed_instruments=frozenset({"MNQ"}),
        max_slots=4,
    )
    current_valid = DailyLaneSpec("INCUMBENT", "MNQ", "TOKYO_OPEN")
    ghost_lane = DailyLaneSpec("GHOST_LANE", "MNQ", "COMEX_SETTLE")
    recommended = LaneScore(
        strategy_id="REPLACEMENT",
        instrument="MNQ",
        orb_label="US_DATA_1000",
        orb_minutes=15,
        rr_target=1.5,
        filter_type="ORB_G5_O15",
        confirm_bars=1,
        stop_multiplier=0.75,
        trailing_expr=0.2,
        trailing_n=40,
        trailing_months=12,
        annual_r_estimate=35.0,
        trailing_wr=0.55,
        session_regime_expr=0.2,
        months_negative=0,
        months_positive_since_last_neg_streak=4,
        status="DEPLOY",
        status_reason="Session HOT",
    )
    incumbent = LaneScore(
        strategy_id="INCUMBENT",
        instrument="MNQ",
        orb_label="TOKYO_OPEN",
        orb_minutes=5,
        rr_target=1.5,
        filter_type="COST_LT12",
        confirm_bars=1,
        stop_multiplier=0.75,
        trailing_expr=0.01,
        trailing_n=22,
        trailing_months=12,
        annual_r_estimate=4.0,
        trailing_wr=0.48,
        session_regime_expr=0.05,
        months_negative=1,
        months_positive_since_last_neg_streak=0,
        status="PAUSE",
        status_reason="SR alarm and weaker trailing annual R",
        sr_status="ALARM",
    )

    monkeypatch.setattr(
        generate_profile_lanes,
        "effective_daily_lanes",
        lambda _profile: (current_valid, ghost_lane),
    )
    monkeypatch.setattr(
        generate_profile_lanes,
        "compute_profile_allocation",
        lambda _profile, _scores, _orb_stats: [recommended],
    )

    generate_profile_lanes.print_profile_report(
        "dummy_profile",
        profile,
        {"INCUMBENT", "REPLACEMENT"},
        [incumbent, recommended],
        {("MNQ", "US_DATA_1000", 15): (50.0, 94.9)},
    )

    output = capsys.readouterr().out

    assert "Ghost lanes:" in output
    assert "GHOST_LANE" in output
    assert "Current valid lanes omitted by recommendation:" in output
    assert "INCUMBENT [PAUSE: SR alarm and weaker trailing annual R; annual_r=4.0; sr=ALARM]" in output
    assert "Kept current valid lanes: 0" in output
    assert "Dropped current valid lanes: 1" in output
    assert "Added new lanes: 1" in output
