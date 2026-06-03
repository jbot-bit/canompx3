from __future__ import annotations

import json

import pandas as pd
import pytest

from research.filter_utils import filter_signal
from research.mnq_single_leg_account_fit_replacement_v1 import (
    EXPECTED_PRIMARY_TRIALS,
    CandidateLane,
    IncumbentLane,
    LaneResearchState,
    build_daily_book,
    build_replacement_scenarios,
    canonical_filter_mask,
    parse_incumbent_lanes,
    score_replacement_scenario,
)


def test_declared_k_matches_locked_candidate_slot_universe() -> None:
    candidates = [
        CandidateLane("A", "MNQ", "NYSE_OPEN", 15, 2.0, "COST_LT10"),
        CandidateLane("B", "MNQ", "US_DATA_1000", 15, 1.0, "NO_FILTER"),
        CandidateLane("C", "MNQ", "US_DATA_1000", 15, 1.5, "NO_FILTER"),
        CandidateLane("D", "MNQ", "US_DATA_1000", 15, 2.0, "NO_FILTER"),
        CandidateLane("E", "MNQ", "CME_PRECLOSE", 15, 2.0, "COST_LT10"),
    ]
    incumbents = [
        IncumbentLane("L1", "MNQ", "COMEX_SETTLE", 5, 1.5, "OVNRNG_100"),
        IncumbentLane("L2", "MNQ", "US_DATA_1000", 15, 1.5, "VWAP_MID_ALIGNED"),
        IncumbentLane("L3", "MNQ", "TOKYO_OPEN", 5, 1.5, "COST_LT08"),
    ]

    scenarios = build_replacement_scenarios(candidates, incumbents)

    assert len(scenarios) == EXPECTED_PRIMARY_TRIALS
    assert scenarios[0].lanes[0].strategy_id == "A"
    assert scenarios[-1].lanes[-1].strategy_id == "E"


def test_parse_incumbent_lanes_uses_structured_fields_not_strategy_id(tmp_path) -> None:
    payload = {
        "profile_id": "topstep_50k_mnq_auto",
        "lanes": [
            {
                "strategy_id": "THIS_ID_IS_NOT_PARSEABLE",
                "instrument": "MNQ",
                "orb_label": "COMEX_SETTLE",
                "orb_minutes": 5,
                "rr_target": 1.5,
                "filter_type": "OVNRNG_100",
            },
            {
                "strategy_id": "ALSO_BAD",
                "instrument": "MNQ",
                "orb_label": "US_DATA_1000",
                "orb_minutes": 15,
                "rr_target": 1.5,
                "filter_type": "VWAP_MID_ALIGNED",
            },
            {
                "strategy_id": "BAD_TOO",
                "instrument": "MNQ",
                "orb_label": "TOKYO_OPEN",
                "orb_minutes": 5,
                "rr_target": 1.5,
                "filter_type": "COST_LT08",
            },
        ],
    }
    path = tmp_path / "allocation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    lanes = parse_incumbent_lanes(path)

    assert [(lane.orb_label, lane.orb_minutes, lane.rr_target, lane.filter_name) for lane in lanes] == [
        ("COMEX_SETTLE", 5, 1.5, "OVNRNG_100"),
        ("US_DATA_1000", 15, 1.5, "VWAP_MID_ALIGNED"),
        ("TOKYO_OPEN", 5, 1.5, "COST_LT08"),
    ]


def test_canonical_filter_mask_delegates_to_filter_signal() -> None:
    lane = CandidateLane("COST", "MNQ", "NYSE_OPEN", 15, 2.0, "COST_LT10")
    df = pd.DataFrame(
        {
            "symbol": ["MNQ", "MNQ"],
            "orb_NYSE_OPEN_size": [13.0, 14.0],
        }
    )

    mask = canonical_filter_mask(df, lane)
    expected = filter_signal(df, "COST_LT10", "NYSE_OPEN").astype(bool)

    assert mask.tolist() == expected.tolist()


def test_canonical_filter_mask_fails_closed_on_missing_required_filter_columns() -> None:
    lane = CandidateLane("COST", "MNQ", "US_DATA_1000", 15, 1.5, "COST_LT10")
    df = pd.DataFrame({"orb_US_DATA_1000_size": [10.0]})

    with pytest.raises(ValueError, match="missing required column"):
        canonical_filter_mask(df, lane)


def test_build_daily_book_keeps_zero_days_and_sums_lane_pnl() -> None:
    calendar = pd.to_datetime(["2025-01-02", "2025-01-03"])
    lane_a = pd.DataFrame(
        {
            "trading_day": [pd.Timestamp("2025-01-02").date()],
            "pnl_r": [1.0],
            "pnl_dollars": [100.0],
        }
    )
    lane_b = pd.DataFrame(
        {
            "trading_day": [pd.Timestamp("2025-01-02").date()],
            "pnl_r": [-0.5],
            "pnl_dollars": [-50.0],
        }
    )

    book = build_daily_book(calendar, [lane_a, lane_b])

    assert book["pnl_r"].tolist() == [0.5, 0.0]
    assert book["pnl_dollars"].tolist() == [50.0, 0.0]
    assert book["active_trades"].tolist() == [2, 0]


def test_score_replacement_scenario_continues_when_safe_better_but_research_hard_failed() -> None:
    days = pd.bdate_range("2025-01-02", periods=120)
    pnl = ([120.0] * 10 + [-10.0]) * 10 + [120.0] * 10
    is_book = pd.DataFrame(
        {
            "trading_day": days.date,
            "pnl_r": [value / 100.0 for value in pnl],
            "pnl_dollars": pnl,
            "active_trades": [1] * len(days),
        }
    )
    state = LaneResearchState(chordia_verdict="FAIL_CHORDIA", sr_status="ALARM_REVIEWED", oos_status="PENDING")

    row = score_replacement_scenario(
        scenario_id="S1",
        replaced_incumbent_lane="INCUMBENT",
        candidate_lane="CANDIDATE",
        book_is=is_book,
        trades_is=pd.DataFrame({"pnl_r": is_book["pnl_r"]}),
        incumbent_annual_per_max_dd=1.0,
        research_state=state,
        monte_carlo_paths=512,
        monte_carlo_seed=7,
    )

    assert row["account_safe"] is True
    assert row["annual_dollars_per_max_drawdown"] > 1.0
    assert row["verdict"] == "CONTINUE"


def test_score_replacement_scenario_kills_account_unsafe_or_worse_than_incumbent() -> None:
    is_book = pd.DataFrame(
        {
            "trading_day": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]).date,
            "pnl_r": [-2.0, -1.5, 0.5],
            "pnl_dollars": [-500.0, -300.0, 50.0],
            "active_trades": [1, 1, 1],
        }
    )
    state = LaneResearchState(chordia_verdict="PASS_CHORDIA", sr_status="CONTINUE", oos_status="PASSED")

    row = score_replacement_scenario(
        scenario_id="S2",
        replaced_incumbent_lane="INCUMBENT",
        candidate_lane="CANDIDATE",
        book_is=is_book,
        trades_is=pd.DataFrame({"pnl_r": [-2.0, -1.5, 0.5]}),
        incumbent_annual_per_max_dd=5.0,
        research_state=state,
        monte_carlo_paths=512,
        monte_carlo_seed=7,
    )

    assert row["account_safe"] is False
    assert row["verdict"] == "KILL"


def test_score_replacement_scenario_parks_underpowered_ambiguous_case() -> None:
    is_book = pd.DataFrame(
        {
            "trading_day": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]).date,
            "pnl_r": [0.3, -0.05, 0.2],
            "pnl_dollars": [40.0, -5.0, 35.0],
            "active_trades": [1, 1, 1],
        }
    )
    state = LaneResearchState(chordia_verdict="MISSING", sr_status="UNKNOWN", oos_status=None)

    row = score_replacement_scenario(
        scenario_id="S3",
        replaced_incumbent_lane="INCUMBENT",
        candidate_lane="CANDIDATE",
        book_is=is_book,
        trades_is=pd.DataFrame({"pnl_r": [0.3, -0.05, 0.2]}),
        incumbent_annual_per_max_dd=0.1,
        research_state=state,
        monte_carlo_paths=256,
        monte_carlo_seed=11,
    )

    assert row["annual_dollars_per_max_drawdown"] > 0.1
    assert row["verdict"] == "PARK"
