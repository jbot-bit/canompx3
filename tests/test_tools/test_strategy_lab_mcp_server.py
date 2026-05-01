"""Tests for scripts.tools.strategy_lab_mcp_server."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from scripts.tools import strategy_lab_mcp_server as srv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allocation_doc(*, profile_id: str = "topstep_50k_mnq_auto") -> dict[str, object]:
    return {
        "rebalance_date": "2026-05-01",
        "trailing_window_months": 12,
        "profile_id": profile_id,
        "lanes": [
            {
                "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
                "instrument": "MNQ",
                "orb_label": "NYSE_OPEN",
                "orb_minutes": 5,
                "rr_target": 1.0,
                "filter_type": "COST_LT12",
                "annual_r": 26.8,
                "trailing_expr": 0.1095,
                "trailing_n": 245,
                "trailing_wr": 0.567,
                "months_negative": 0,
                "session_regime": "HOT",
                "status": "DEPLOY",
                "status_reason": "Session HOT",
                "chordia_verdict": "PASS_PROTOCOL_A",
                "chordia_audit_age_days": 0,
            }
        ],
        "paused": [
            {
                "strategy_id": "MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K",
                "reason": "Session regime COLD (-0.0899)",
            }
        ],
        "all_scores_count": 17,
    }


def _validated_row_mnq() -> dict[str, object]:
    return {
        "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
        "instrument": "MNQ",
        "orb_label": "NYSE_OPEN",
        "orb_minutes": 5,
        "entry_model": "E2",
        "rr_target": 1.0,
        "confirm_bars": 1,
        "filter_type": "COST_LT12",
        "stop_multiplier": 1.0,
        "sample_size": 245,
        "win_rate": 0.567,
        "expectancy_r": 0.1095,
        "sharpe_ratio": 0.41,
        "max_drawdown_r": -8.2,
    }


def _fitness_payload(status: str = "FIT") -> dict[str, object]:
    return {
        "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
        "rolling_exp_r": 0.0875,
        "rolling_sample": 88,
        "rolling_window_months": 18,
        "fitness_status": status,
        "fitness_notes": "Positive rolling ExpR with stable recent Sharpe",
    }


# ---------------------------------------------------------------------------
# allocation index
# ---------------------------------------------------------------------------


def test_allocation_index_marks_active_and_paused() -> None:
    index = srv._allocation_index(_allocation_doc())
    active_id = "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12"
    paused_id = "MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K"
    assert index[active_id]["_allocation_state"] == "active"
    assert index[paused_id]["_allocation_state"] == "paused"


def test_load_allocation_doc_returns_none_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_file.json"
    assert srv._load_allocation_doc(missing) is None


def test_load_allocation_doc_returns_none_when_corrupt(tmp_path: Path) -> None:
    bad = tmp_path / "lane_allocation.json"
    bad.write_text("{not json", encoding="utf-8")
    assert srv._load_allocation_doc(bad) is None


# ---------------------------------------------------------------------------
# _readiness_verdict
# ---------------------------------------------------------------------------


def test_readiness_verdict_not_validated() -> None:
    out = srv._readiness_verdict(validated=None, fitness=None, allocation_entry=None)
    assert out["verdict"] == "NOT_VALIDATED"


def test_readiness_verdict_deployed() -> None:
    out = srv._readiness_verdict(
        validated=_validated_row_mnq(),
        fitness=_fitness_payload("FIT"),
        allocation_entry={**_allocation_doc()["lanes"][0], "_allocation_state": "active"},  # type: ignore[index]
    )
    assert out["verdict"] == "DEPLOYED"
    assert "PASS_PROTOCOL_A" in out["reason"]


def test_readiness_verdict_paused() -> None:
    out = srv._readiness_verdict(
        validated=_validated_row_mnq(),
        fitness=_fitness_payload("FIT"),
        allocation_entry={
            "_allocation_state": "paused",
            "reason": "Session regime COLD",
            "status": None,
        },
    )
    assert out["verdict"] == "PAUSED"
    assert "COLD" in out["reason"]


def test_readiness_verdict_promotable() -> None:
    out = srv._readiness_verdict(
        validated=_validated_row_mnq(),
        fitness=_fitness_payload("FIT"),
        allocation_entry=None,
    )
    assert out["verdict"] == "PROMOTABLE"


def test_readiness_verdict_validated_but_decay() -> None:
    out = srv._readiness_verdict(
        validated=_validated_row_mnq(),
        fitness=_fitness_payload("DECAY"),
        allocation_entry=None,
    )
    assert out["verdict"] == "VALIDATED_BUT_DECAY"


def test_readiness_verdict_fitness_unavailable() -> None:
    out = srv._readiness_verdict(
        validated=_validated_row_mnq(),
        fitness={"error": "no outcomes"},
        allocation_entry=None,
    )
    assert out["verdict"] == "VALIDATED_FITNESS_UNAVAILABLE"


# ---------------------------------------------------------------------------
# get_strategy_readiness orchestration
# ---------------------------------------------------------------------------


def test_get_strategy_readiness_returns_error_for_blank_id() -> None:
    assert "error" in srv._get_strategy_readiness("")


def test_get_strategy_readiness_for_deployed_strategy(tmp_path: Path) -> None:
    alloc_path = tmp_path / "lane_allocation.json"
    alloc_path.write_text(json.dumps(_allocation_doc()), encoding="utf-8")

    with (
        patch.object(srv, "_validated_row", return_value=_validated_row_mnq()),
        patch.object(srv, "_compute_fitness_payload", return_value=_fitness_payload("FIT")),
        patch.object(srv, "_allocation_path", return_value=alloc_path),
    ):
        payload = srv._get_strategy_readiness("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12")

    assert payload["verdict"] == "DEPLOYED"
    assert payload["allocation_rebalance_date"] == "2026-05-01"
    assert payload["allocation_profile_id"] == "topstep_50k_mnq_auto"


def test_get_strategy_readiness_for_unvalidated_skips_fitness(tmp_path: Path) -> None:
    alloc_path = tmp_path / "lane_allocation.json"
    alloc_path.write_text(json.dumps(_allocation_doc()), encoding="utf-8")

    fitness_calls: list[str] = []

    def _fail_fitness(strategy_id: str, rolling_months: int) -> dict[str, object]:
        fitness_calls.append(strategy_id)
        return {"error": "should not have been called"}

    with (
        patch.object(srv, "_validated_row", return_value=None),
        patch.object(srv, "_compute_fitness_payload", side_effect=_fail_fitness),
        patch.object(srv, "_allocation_path", return_value=alloc_path),
    ):
        payload = srv._get_strategy_readiness("BOGUS_ID_NOT_VALIDATED")

    assert payload["verdict"] == "NOT_VALIDATED"
    assert payload["fitness"] is None
    assert fitness_calls == []


# ---------------------------------------------------------------------------
# get_lane_allocation_summary
# ---------------------------------------------------------------------------


def test_get_lane_allocation_summary_returns_error_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "lane_allocation.json"
    with (
        patch.object(srv, "_allocation_path", return_value=missing),
        patch.object(srv, "_allocation_staleness", return_value={"status": "BLOCK", "days_old": -1}),
    ):
        payload = srv._get_lane_allocation_summary()
    assert "error" in payload


def test_get_lane_allocation_summary_profile_mismatch_errors(tmp_path: Path) -> None:
    alloc_path = tmp_path / "lane_allocation.json"
    alloc_path.write_text(json.dumps(_allocation_doc(profile_id="topstep_50k_mnq_auto")), encoding="utf-8")
    with (
        patch.object(srv, "_allocation_path", return_value=alloc_path),
        patch.object(srv, "_allocation_staleness", return_value={"status": "OK", "days_old": 1}),
    ):
        payload = srv._get_lane_allocation_summary(profile_name="some_other_profile")
    assert "error" in payload
    assert payload["actual_profile_id"] == "topstep_50k_mnq_auto"


def test_get_lane_allocation_summary_returns_active_and_paused(tmp_path: Path) -> None:
    alloc_path = tmp_path / "lane_allocation.json"
    alloc_path.write_text(json.dumps(_allocation_doc()), encoding="utf-8")
    with (
        patch.object(srv, "_allocation_path", return_value=alloc_path),
        patch.object(srv, "_allocation_staleness", return_value={"status": "OK", "days_old": 1}),
    ):
        payload = srv._get_lane_allocation_summary(profile_name="topstep_50k_mnq_auto")
    assert payload["active_count"] == 1
    assert payload["paused_count"] == 1
    assert payload["staleness"]["status"] == "OK"


# ---------------------------------------------------------------------------
# get_recent_fitness
# ---------------------------------------------------------------------------


def test_get_recent_fitness_blank_id() -> None:
    assert "error" in srv._get_recent_fitness("")


def test_get_recent_fitness_passes_through_payload() -> None:
    with patch.object(srv, "_compute_fitness_payload", return_value=_fitness_payload("WATCH")):
        payload = srv._get_recent_fitness("MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", rolling_months=12)
    assert payload["fitness"]["fitness_status"] == "WATCH"
    assert payload["rolling_months"] == 12


# ---------------------------------------------------------------------------
# list_promotable_candidates
# ---------------------------------------------------------------------------


def test_list_promotable_candidates_rejects_inactive_instrument() -> None:
    with patch.object(srv, "_validate_active_instruments", return_value={"MNQ", "MES", "MGC"}):
        out = srv._list_promotable_candidates(instrument="MCL")
    assert "error" in out
    assert "MCL" in out["error"]


def test_list_promotable_candidates_filters_allocated_and_keeps_only_fit(tmp_path: Path) -> None:
    alloc_path = tmp_path / "lane_allocation.json"
    alloc_path.write_text(json.dumps(_allocation_doc()), encoding="utf-8")

    validated_rows = [
        _validated_row_mnq(),
        {**_validated_row_mnq(), "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_OTHER"},
        {**_validated_row_mnq(), "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_DECAY_ONE"},
    ]

    fitness_by_id = {
        "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12": _fitness_payload("FIT"),
        "MNQ_NYSE_OPEN_E2_RR1.0_CB1_OTHER": _fitness_payload("FIT"),
        "MNQ_NYSE_OPEN_E2_RR1.0_CB1_DECAY_ONE": _fitness_payload("DECAY"),
    }

    def _fitness_mock(sid: str, rolling_months: int) -> dict[str, object]:
        return fitness_by_id[sid]

    with (
        patch.object(srv, "_validate_active_instruments", return_value={"MNQ"}),
        patch.object(srv, "_list_validated_rows", return_value=validated_rows),
        patch.object(srv, "_compute_fitness_payload", side_effect=_fitness_mock),
        patch.object(srv, "_allocation_path", return_value=alloc_path),
    ):
        out = srv._list_promotable_candidates(instrument="MNQ")

    candidate_ids = {c["strategy_id"] for c in out["candidates"]}
    assert "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12" not in candidate_ids  # already allocated
    assert "MNQ_NYSE_OPEN_E2_RR1.0_CB1_DECAY_ONE" not in candidate_ids  # not FIT
    assert candidate_ids == {"MNQ_NYSE_OPEN_E2_RR1.0_CB1_OTHER"}
    assert out["validated_count"] == 3
    assert out["currently_allocated_count"] == 1


def test_list_promotable_candidates_sort_handles_missing_rolling_expr(tmp_path: Path) -> None:
    alloc_path = tmp_path / "lane_allocation.json"
    alloc_path.write_text(json.dumps(_allocation_doc()), encoding="utf-8")

    validated_rows = [
        {**_validated_row_mnq(), "strategy_id": "MNQ_A"},
        {**_validated_row_mnq(), "strategy_id": "MNQ_B"},
    ]

    def _fitness_mock(sid: str, rolling_months: int) -> dict[str, object]:
        if sid == "MNQ_A":
            return {**_fitness_payload("FIT"), "rolling_exp_r": None}
        return {**_fitness_payload("FIT"), "rolling_exp_r": 0.05}

    with (
        patch.object(srv, "_validate_active_instruments", return_value={"MNQ"}),
        patch.object(srv, "_list_validated_rows", return_value=validated_rows),
        patch.object(srv, "_compute_fitness_payload", side_effect=_fitness_mock),
        patch.object(srv, "_allocation_path", return_value=alloc_path),
    ):
        out = srv._list_promotable_candidates(instrument="MNQ")

    ids_in_order = [c["strategy_id"] for c in out["candidates"]]
    assert ids_in_order == ["MNQ_B", "MNQ_A"]


# ---------------------------------------------------------------------------
# server boot (smoke)
# ---------------------------------------------------------------------------


def test_build_server_does_not_raise() -> None:
    # FastMCP build is synchronous; tool enumeration is async and varies by
    # version, so we only assert the server constructs without error here and
    # rely on the unit tests above for tool behavior.
    server = srv._build_server()
    assert server is not None
