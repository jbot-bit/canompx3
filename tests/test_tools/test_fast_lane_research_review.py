"""Tests for scripts/tools/fast_lane_research_review.py."""

from __future__ import annotations

from typing import Any

from scripts.tools import fast_lane_research_review as review


def _status(
    sid: str,
    *,
    stage: str = "ENRICHED",
    lineage: str = "FAST_LANE",
    blocker: str = "NONE",
    next_action: str = "operator_capital_review",
) -> dict[str, Any]:
    return {
        "strategy_id": sid,
        "current_stage": stage,
        "lineage_class": lineage,
        "blocker_class": blocker,
        "primary_blocker": None,
        "blocker_evidence": {},
        "next_action_token": next_action,
        "upstream_artifact_path": "docs/audit/results/example.md",
    }


def _provider(verdict: str) -> review.StrategyLabProvider:
    def _inner(strategy_id: str, rolling_months: int = 18) -> dict[str, Any]:
        return {
            "strategy_id": strategy_id,
            "verdict": verdict,
            "reason": f"strategy-lab says {verdict}",
            "rolling_months": rolling_months,
        }

    return _inner


def test_underpowered_blocker_parks_and_keeps_report_only_boundary() -> None:
    sid = "MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30"
    report = review.build_review_report(
        status_entries=[
            _status(
                sid,
                stage="REJECTED_OOS_UNPOWERED",
                blocker="UNDERPOWERED_OOS",
                next_action="operator_pick_remedy_cpcv_haircut_pool_or_park",
            )
        ],
        journal_by_strategy={
            sid: {
                "strategy_id": sid,
                "heavyweight_verdict": "DEFERRED_NOT_RUN",
                "oos_power_tier": "NA_N_BELOW_FLOOR",
            }
        },
        strategy_lab_provider=_provider("PAUSED"),
    )

    assert report["capital_boundary"] == review.CAPITAL_BOUNDARY
    assert report["entries"][0]["recommendation"] == "PARK"
    assert "underpowered_oos" in report["entries"][0]["reason_codes"]
    rendered = review.render_markdown(report)
    assert review.CAPITAL_BOUNDARY in rendered
    for phrase in review.BANNED_ACTIVE_SURFACE_PHRASES:
        assert phrase not in rendered


def test_pass_promotable_escalates_to_capital_review_not_deployment() -> None:
    sid = "MNQ_FAST_LANE_PASS"
    report = review.build_review_report(
        status_entries=[_status(sid, stage="ENRICHED")],
        journal_by_strategy={sid: {"strategy_id": sid, "heavyweight_verdict": "PASS_CHORDIA"}},
        strategy_lab_provider=_provider("PROMOTABLE"),
    )

    row = report["entries"][0]
    assert row["recommendation"] == "ESCALATE_CAPITAL_REVIEW"
    assert row["next_review_action"] == "open_separate_capital_review"
    rendered = review.render_markdown(report)
    assert "DEPLOYMENT_CANDIDATE" not in rendered
    assert "operator_deployment_decision" not in rendered


def test_direct_heavyweight_cannot_exceed_research_review() -> None:
    sid = "MNQ_DIRECT_HEAVYWEIGHT"
    report = review.build_review_report(
        status_entries=[_status(sid, lineage="DIRECT_HEAVYWEIGHT")],
        journal_by_strategy={sid: {"strategy_id": sid, "heavyweight_verdict": "PASS_CHORDIA"}},
        strategy_lab_provider=_provider("PROMOTABLE"),
        include_direct_heavyweight=True,
    )

    row = report["entries"][0]
    assert row["recommendation"] == "RECOMMEND_RESEARCH_REVIEW"
    assert "direct_heavyweight_context_only" in row["reason_codes"]


def test_default_scope_excludes_direct_heavyweight_rows() -> None:
    report = review.build_review_report(
        status_entries=[
            _status("MNQ_FAST", lineage="FAST_LANE"),
            _status("MNQ_DIRECT", lineage="DIRECT_HEAVYWEIGHT"),
        ],
        journal_by_strategy={},
        strategy_lab_provider=_provider("PROMOTABLE"),
    )

    assert [row["strategy_id"] for row in report["entries"]] == ["MNQ_FAST"]


def test_markdown_contains_boundary_banner_and_banned_phrase_absent() -> None:
    sid = "MNQ_FAST_LANE_PASS"
    report = review.build_review_report(
        status_entries=[_status(sid)],
        journal_by_strategy={sid: {"strategy_id": sid, "heavyweight_verdict": "PASS_CHORDIA"}},
        strategy_lab_provider=_provider("PROMOTABLE"),
    )

    rendered = review.render_markdown(report)

    assert "# Fast Lane Research Review" in rendered
    assert review.CAPITAL_BOUNDARY in rendered
    for phrase in review.BANNED_ACTIVE_SURFACE_PHRASES:
        assert phrase not in rendered
