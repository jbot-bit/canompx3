from __future__ import annotations

from datetime import date

import pytest

from trading_app.allocation_promotion import (
    AllocationPromotionError,
    PromotionCandidate,
    apply_promotions,
)
from trading_app.prop_profiles import load_allocation_lanes


def _allocation() -> dict:
    return {
        "rebalance_date": "2026-05-03",
        "profile_id": "topstep_50k_mnq_auto",
        "lanes": [
            {
                "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
                "instrument": "MNQ",
                "orb_label": "COMEX_SETTLE",
                "orb_minutes": 5,
                "status": "DEPLOY",
                "chordia_verdict": "PASS_CHORDIA",
                "chordia_audit_age_days": 1,
                "p90_orb_pts": 48.5,
            },
            {
                "strategy_id": "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
                "instrument": "MNQ",
                "orb_label": "US_DATA_1000",
                "orb_minutes": 15,
                "status": "DEPLOY",
                "chordia_verdict": "PASS_CHORDIA",
                "chordia_audit_age_days": 1,
                "p90_orb_pts": 142.3,
            },
            {
                "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
                "instrument": "MNQ",
                "orb_label": "NYSE_OPEN",
                "orb_minutes": 5,
                "status": "PAUSE",
                "chordia_verdict": "PASS_PROTOCOL_A",
                "chordia_audit_age_days": 2,
                "p90_orb_pts": 115.6,
            },
        ],
        "paused": [
            {
                "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
                "status": "PAUSE",
                "reason": "stale allocator demotion",
                "chordia_verdict": "MISSING",
                "chordia_audit_age_days": None,
            }
        ],
        "stale": [],
    }


def _candidate(**overrides) -> PromotionCandidate:
    values = {
        "profile_id": "topstep_50k_mnq_auto",
        "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
        "decision": "PASS_REPLACE",
        "instrument": "MNQ",
        "orb_label": "NYSE_OPEN",
        "orb_minutes": 5,
        "rr_target": 1.5,
        "filter_type": "COST_LT12",
        "status": "PROVISIONAL",
        "status_reason": "PASS_REPLACE via controlled profile-construction gate",
        "chordia_verdict": "PASS_PROTOCOL_A",
        "chordia_audit_age_days": 4,
        "annual_r": 24.4,
        "trailing_expr": 0.105,
        "trailing_n": 1472,
        "trailing_wr": 0.4565,
        "months_negative": 0,
        "session_regime": "HOT",
        "avg_orb_pts": 75.1,
        "p90_orb_pts": 85.8,
        "replacement_target": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
        "replacement_target_status": "PAUSE",
        "source_path": "docs/audit/results/example.md",
        "account_risk_detail": "worst_case=$415<=2000; slots=3/7",
    }
    values.update(overrides)
    return PromotionCandidate(**values)


def test_pass_replace_promotes_candidate_and_removes_stale_blocks() -> None:
    result = apply_promotions(_allocation(), [_candidate()], rebalance_date=date(2026, 5, 11))

    active_ids = [lane["strategy_id"] for lane in result.allocation["lanes"] if lane["status"] in {"DEPLOY", "PROVISIONAL"}]
    blocked_ids = [row["strategy_id"] for row in result.allocation["paused"] + result.allocation.get("stale", [])]

    assert active_ids == [
        "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
        "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
        "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
    ]
    assert "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12" not in active_ids
    assert "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12" not in blocked_ids
    assert result.promoted == ("MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",)
    assert result.removed_lane_ids == ("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",)
    assert result.removed_block_ids == ("MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",)


def test_promoted_allocation_loads_through_runtime_lane_loader(tmp_path) -> None:
    result = apply_promotions(_allocation(), [_candidate()], rebalance_date=date(2026, 5, 11))
    path = tmp_path / "lane_allocation.json"
    path.write_text(result.to_json(), encoding="utf-8")

    lanes = load_allocation_lanes("topstep_50k_mnq_auto", allocation_path=path)

    assert [lane.strategy_id for lane in lanes] == [
        "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
        "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
        "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
    ]


def test_rejects_non_pass_decisions() -> None:
    with pytest.raises(AllocationPromotionError, match="decision"):
        apply_promotions(_allocation(), [_candidate(decision="PARK")], rebalance_date=date(2026, 5, 11))


def test_rejects_non_passing_chordia_verdict() -> None:
    with pytest.raises(AllocationPromotionError, match="Chordia"):
        apply_promotions(_allocation(), [_candidate(chordia_verdict="MISSING")], rebalance_date=date(2026, 5, 11))


def test_generic_profile_and_instrument_are_not_hardcoded() -> None:
    allocation = {
        "rebalance_date": "2026-05-03",
        "profile_id": "other_profile",
        "lanes": [],
        "paused": [],
        "stale": [],
    }
    candidate = _candidate(
        profile_id="other_profile",
        strategy_id="MGC_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12",
        decision="PASS_ADD",
        instrument="MGC",
        orb_label="TOKYO_OPEN",
        rr_target=1.0,
        replacement_target=None,
        replacement_target_status=None,
    )

    result = apply_promotions(allocation, [candidate], rebalance_date=date(2026, 5, 11))

    assert result.allocation["profile_id"] == "other_profile"
    assert result.allocation["lanes"][0]["instrument"] == "MGC"
