"""Tests for scripts/tools/fast_lane_research_review.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.tools import fast_lane_research_review as review


def _status(
    sid: str,
    *,
    stage: str = "ENRICHED",
    lineage: str = "FAST_LANE",
    blocker: str = "NONE",
    next_action: str = "operator_capital_review",
    upstream: str = "docs/audit/results/example.md",
) -> dict[str, Any]:
    return {
        "strategy_id": sid,
        "current_stage": stage,
        "lineage_class": lineage,
        "blocker_class": blocker,
        "primary_blocker": None,
        "blocker_evidence": {},
        "next_action_token": next_action,
        "upstream_artifact_path": upstream,
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


def _write_result_md(
    tmp_path: Path,
    sid: str,
    *,
    verdict: str = "PASS_CHORDIA",
    is_n: int = 200,
    is_expr: float = 0.2,
    is_t: float = 4.2,
    oos_n: int = 40,
    oos_expr: float = 0.1,
) -> Path:
    path = tmp_path / f"{sid.lower()}-result.md"
    path.write_text(
        f"""# Chordia strict unlock audit — {sid}

## Verdict

**MEASURED verdict:** `{verdict}`
**MEASURED threshold applied:** `3.79`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 250 | {is_n} | 80.00% | 0 | 0 | {is_expr:.4f} | 0.1000 | 0.2000 | {is_t:.3f} | 0.00001 |
| OOS | 50 | {oos_n} | 80.00% | 0 | 0 | {oos_expr:.4f} | 0.0800 | 0.1000 | 1.000 | 0.20000 |
""",
        encoding="utf-8",
    )
    return path


def test_result_md_parser_extracts_truth_ledger_fields(tmp_path: Path) -> None:
    sid = "MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15"
    parsed = review.parse_result_md(_write_result_md(tmp_path, sid))

    assert parsed is not None
    assert parsed["strategy_id"] == sid
    assert parsed["heavyweight_verdict"] == "PASS_CHORDIA"
    assert parsed["is_t"] == 4.2
    assert parsed["c8_oos_status"] == "OOS_SIGN_MATCH_N_GE_30"
    assert parsed["dir_match"] is True
    assert parsed["n_unique_days"] == 200
    assert parsed["feature_family"] == "VWAP_MID"
    assert parsed["session"] == "US_DATA_1000"
    assert parsed["orb_minutes"] == 15


def test_direct_heavyweight_falls_back_to_result_md_when_journal_missing(tmp_path: Path) -> None:
    sid = "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50"
    result_md = _write_result_md(tmp_path, sid)

    report = review.build_review_report(
        status_entries=[
            _status(sid, stage="HEAVYWEIGHT_COMPLETE", lineage="DIRECT_HEAVYWEIGHT", upstream=str(result_md))
        ],
        journal_by_strategy={},
        strategy_lab_provider=_provider("PROMOTABLE"),
        include_direct_heavyweight=True,
    )

    row = report["entries"][0]
    assert row["heavyweight_verdict"] == "PASS_CHORDIA"
    assert row["evidence_source"] == "result_md"
    assert row["recommendation"] == "RECOMMEND_RESEARCH_REVIEW"
    assert "direct_heavyweight_context_only" in row["reason_codes"]


def test_truth_ledger_and_filtered_summary_are_fail_closed(tmp_path: Path) -> None:
    pass_sid = "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50"
    fail_sid = "MNQ_CME_REOPEN_E2_RR1.0_CB1_ORB_G4"
    pass_md = _write_result_md(tmp_path, pass_sid)
    fail_md = _write_result_md(tmp_path, fail_sid, verdict="FAIL_STRICT_CHORDIA", is_t=2.1, oos_expr=-0.1)

    ledger = review.build_truth_ledger(
        [
            _status(pass_sid, stage="HEAVYWEIGHT_COMPLETE", lineage="DIRECT_HEAVYWEIGHT", upstream=str(pass_md)),
            _status(fail_sid, stage="HEAVYWEIGHT_COMPLETE", lineage="DIRECT_HEAVYWEIGHT", upstream=str(fail_md)),
            _status("MNQ_ACTIVE", stage="ACTIVE_PREREG", lineage="FAST_LANE"),
        ]
    )
    summary = review.filtered_out_summary(ledger)

    assert len(ledger) == 2
    assert summary["included_for_ranking"] == 1
    assert summary["filtered_out"] == 1
    assert summary["by_reason"]["heavyweight_failed"] == 1


def test_fast_lane_heavyweight_without_journal_is_integrity_blocker(tmp_path: Path) -> None:
    sid = "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50"
    result_md = _write_result_md(tmp_path, sid)

    ledger = review.build_truth_ledger(
        [_status(sid, stage="HEAVYWEIGHT_COMPLETE", lineage="FAST_LANE", upstream=str(result_md))],
        journal_by_strategy={},
    )
    packet = review.build_capital_packet(ledger)

    assert ledger[0]["blocker"] == "journal_missing_integrity_break"
    assert review.filtered_out_summary(ledger)["included_for_ranking"] == 0
    assert packet["decisions"][sid]["bucket"] == "WATCH"


def test_capital_packet_ranks_only_pass_chordia_with_c8_pass(tmp_path: Path) -> None:
    pass_sid = "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50"
    protocol_sid = "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12"
    flip_sid = "MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15"
    ledger = [
        review.parse_result_md(_write_result_md(tmp_path, pass_sid)),
        review.parse_result_md(_write_result_md(tmp_path, protocol_sid, verdict="PASS_PROTOCOL_A")),
        review.parse_result_md(_write_result_md(tmp_path, flip_sid, verdict="PARK", oos_expr=-0.2)),
    ]

    packet = review.build_capital_packet(
        [row for row in ledger if row is not None],
        canonical_by_strategy=_canonical_context(pass_sid, protocol_sid, flip_sid),
    )

    assert [row["strategy_id"] for row in packet["ranked_candidates"]] == [pass_sid]
    assert packet["decisions"][pass_sid]["bucket"] == "WATCH"
    assert packet["decisions"][pass_sid]["reason"] == "incumbent_comparison_missing"
    assert packet["decisions"][protocol_sid]["bucket"] == "WATCH"
    assert packet["decisions"][flip_sid]["bucket"] == "REJECT"
    assert packet["rebalance_dry_run_diff"]["would_add"] == []


def test_capital_packet_correlation_displaces_redundant_ranked_candidate(tmp_path: Path) -> None:
    head_sid = "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8"
    redundant_sid = "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5"
    independent_sid = "MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50"
    ledger = [
        review.parse_result_md(_write_result_md(tmp_path, head_sid, is_t=5.0)),
        review.parse_result_md(_write_result_md(tmp_path, redundant_sid, is_t=4.8)),
        review.parse_result_md(_write_result_md(tmp_path, independent_sid, is_t=4.6)),
    ]

    def _correlation_provider(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "status": "MEASURED",
            "rho_reject_threshold": 0.70,
            "skipped": {},
            "pairs": [
                {"a": head_sid, "b": redundant_sid, "rho": 0.91, "reject": True},
                {"a": head_sid, "b": independent_sid, "rho": 0.08, "reject": False},
                {"a": independent_sid, "b": redundant_sid, "rho": 0.07, "reject": False},
            ],
        }

    packet = review.build_capital_packet(
        [row for row in ledger if row is not None],
        correlation_provider=_correlation_provider,
        canonical_by_strategy=_canonical_context(head_sid, redundant_sid, independent_sid),
    )

    assert packet["decisions"][head_sid]["bucket"] == "WATCH"
    assert packet["decisions"][head_sid]["reason"] == "incumbent_comparison_missing"
    assert packet["decisions"][redundant_sid]["bucket"] == "WATCH"
    assert packet["decisions"][redundant_sid]["correlation"]["displaced_by"] == head_sid
    assert packet["decisions"][independent_sid]["bucket"] == "WATCH"
    assert packet["decisions"][independent_sid]["reason"] == "incumbent_comparison_missing"
    assert [cluster["head"] for cluster in packet["correlation_clusters"]] == [head_sid, independent_sid]


def test_capital_packet_correlation_error_fails_closed_without_rebalance_diff(tmp_path: Path) -> None:
    sid = "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8"
    parsed = review.parse_result_md(_write_result_md(tmp_path, sid))

    def _broken_provider(rows: list[dict[str, Any]]) -> dict[str, Any]:
        raise RuntimeError("db unavailable")

    packet = review.build_capital_packet(
        [parsed] if parsed is not None else [],
        current_lanes=["CURRENT_LANE"],
        correlation_provider=_broken_provider,
        canonical_by_strategy=_canonical_context(sid),
    )

    assert packet["decisions"][sid]["bucket"] == "WATCH"
    assert packet["decisions"][sid]["reason"] == "correlation_provider_not_measured"
    assert packet["rebalance_dry_run_diff"]["would_add"] == []
    assert packet["rebalance_dry_run_diff"]["would_keep"] == ["CURRENT_LANE"]
    assert packet["rebalance_dry_run_diff"]["would_remove"] == []


def _canonical_context(*strategy_ids: str, c8: str = "PASSED", trade_day_count: int = 123) -> dict[str, Any]:
    return {
        sid: {
            "c8_oos_status": c8,
            "trade_day_count": trade_day_count,
            "sample_size": trade_day_count,
            "expectancy_r": 0.12,
            "oos_exp_r": 0.11,
            "wfe": 1.0,
        }
        for sid in strategy_ids
    }


def test_capital_packet_uses_canonical_c8_and_real_unique_days(tmp_path: Path) -> None:
    sid = "MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K"
    parsed = review.parse_result_md(_write_result_md(tmp_path, sid, is_n=1406, oos_n=80, oos_expr=0.2))

    packet = review.build_capital_packet(
        [parsed] if parsed is not None else [],
        canonical_by_strategy=_canonical_context(sid, c8="FAILED_RATIO", trade_day_count=1415),
    )

    row = packet["truth_ledger"][0]
    assert row["md_c8_oos_status"] == "OOS_SIGN_MATCH_N_GE_30"
    assert row["c8_oos_status"] == "FAILED_RATIO"
    assert row["n_unique_days"] == 1415
    assert row["md_n_fired"] == 1406
    assert row["n_unique_days_source"] == "validated_setups.trade_day_count"
    assert packet["ranked_candidates"] == []
    assert packet["decisions"][sid]["bucket"] == "WATCH"
    assert packet["decisions"][sid]["reason"] == "canonical_c8_failed_ratio"


def test_capital_packet_current_lane_absent_from_ledger_gets_decision_entry(tmp_path: Path) -> None:
    sid = "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100"
    current_sid = "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12"
    parsed = review.parse_result_md(_write_result_md(tmp_path, sid, is_t=4.5))

    packet = review.build_capital_packet(
        [parsed] if parsed is not None else [],
        current_lanes=[current_sid],
        canonical_by_strategy=_canonical_context(sid),
    )

    assert current_sid in packet["decisions"]
    assert packet["decisions"][current_sid]["bucket"] == "KEEP_CURRENT_SHADOW_ONLY"
    for removed_sid in packet["rebalance_dry_run_diff"]["would_remove"]:
        assert removed_sid in packet["decisions"]


def test_capital_packet_approve_requires_live_allocator_beats_incumbent(tmp_path: Path) -> None:
    candidate = "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100"
    incumbent = "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K"
    parsed = review.parse_result_md(_write_result_md(tmp_path, candidate, is_t=4.7))
    allocator_context = {
        "current_lanes": {incumbent: {"annual_r": 30.0, "trailing_expr": 0.16, "trailing_n": 240}},
        "metrics": {
            candidate: {"annual_r": 25.0, "trailing_expr": 0.14, "trailing_n": 250, "status": "DEPLOY"},
            incumbent: {"annual_r": 30.0, "trailing_expr": 0.16, "trailing_n": 240, "status": "DEPLOY"},
        },
    }

    packet = review.build_capital_packet(
        [parsed] if parsed is not None else [],
        current_lanes=[incumbent],
        canonical_by_strategy=_canonical_context(candidate),
        allocator_context=allocator_context,
    )

    decision = packet["decisions"][candidate]
    assert decision["bucket"] == "WATCH"
    assert decision["reason"] == "does_not_beat_incumbent_live_allocator_metrics"
    assert decision["incumbent_comparison"]["incumbent_strategy_id"] == incumbent
    assert decision["incumbent_comparison"]["beats_incumbent"] is False


def test_capital_packet_positive_approve_carries_live_allocator_proof(tmp_path: Path) -> None:
    candidate = "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100"
    incumbent = "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K"
    parsed = review.parse_result_md(_write_result_md(tmp_path, candidate, is_t=4.7))
    allocator_context = {
        "current_lanes": {incumbent: {"annual_r": 30.0, "trailing_expr": 0.16, "trailing_n": 240}},
        "metrics": {
            candidate: {"annual_r": 36.0, "trailing_expr": 0.18, "trailing_n": 260, "status": "DEPLOY"},
            incumbent: {"annual_r": 30.0, "trailing_expr": 0.16, "trailing_n": 240, "status": "DEPLOY"},
        },
    }

    packet = review.build_capital_packet(
        [parsed] if parsed is not None else [],
        current_lanes=[incumbent],
        canonical_by_strategy=_canonical_context(candidate),
        allocator_context=allocator_context,
    )

    decision = packet["decisions"][candidate]
    assert decision["bucket"] == "APPROVE_SHADOW_ONLY"
    assert decision["incumbent_comparison"]["beats_incumbent"] is True
    assert decision["incumbent_comparison"]["candidate_metrics"]["annual_r"] == 36.0
    assert packet["rebalance_dry_run_diff"]["would_add"] == []
    assert packet["rebalance_dry_run_diff"]["would_remove"] == []


def test_capital_packet_insufficient_power_or_not_rotatable_is_watch_only(tmp_path: Path) -> None:
    candidate = "MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30"
    incumbent = "MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15"
    parsed = review.parse_result_md(_write_result_md(tmp_path, candidate, is_t=4.7))
    allocator_context = {
        "current_lanes": {incumbent: {"annual_r": 20.0, "trailing_expr": 0.12, "trailing_n": 120}},
        "metrics": {
            candidate: {"annual_r": 40.0, "trailing_expr": 0.24, "trailing_n": 260, "status": "DEPLOY"},
            incumbent: {"annual_r": 20.0, "trailing_expr": 0.12, "trailing_n": 120, "status": "DEPLOY"},
        },
    }

    packet = review.build_capital_packet(
        [parsed] if parsed is not None else [],
        current_lanes=[incumbent],
        canonical_by_strategy=_canonical_context(candidate),
        allocator_context=allocator_context,
        strategy_blockers={candidate: ["insufficient_power", "not_auto_rotatable"]},
    )

    assert packet["decisions"][candidate]["bucket"] == "WATCH"
    assert packet["decisions"][candidate]["reason"] == "insufficient_power"
    assert packet["shadow_only"] is True
    assert packet["rebalance_dry_run_diff"]["would_add"] == []
    assert packet["rebalance_dry_run_diff"]["would_remove"] == []
