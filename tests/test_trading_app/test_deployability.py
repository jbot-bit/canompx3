from __future__ import annotations

import json

import duckdb

from trading_app import deployability as dep
from trading_app.deployability_state import load_latest_deployment_readiness, write_deployability_state


def _row(**overrides):
    row = {
        "strategy_id": "SID_A",
        "instrument": "MNQ",
        "orb_label": "COMEX_SETTLE",
        "orb_minutes": 15,
        "entry_model": "E2",
        "rr_target": 1.5,
        "confirm_bars": 1,
        "filter_type": "OVNRNG_100",
        "status": "active",
        "deployment_scope": "deployable",
        "sample_size": 200,
        "expectancy_r": 0.20,
        "oos_exp_r": 0.12,
        "wfe": 0.80,
        "dsr_score": 0.96,
        "years_tested": 7.0,
        "slippage_validation_status": "PASSED",
        "c8_oos_status": "PASSED",
        "robustness_status": "ROBUST",
    }
    row.update(overrides)
    return row


def _replay(**overrides):
    data = {
        "ok": True,
        "recomputed_sample_size": 200,
        "stored_sample_size": 200,
        "recomputed_expectancy_r": 0.20,
        "stored_expectancy_r": 0.20,
        "sample_size_match": True,
        "expectancy_match": True,
        "null_pnl_count": 0,
    }
    data.update(overrides)
    return data


def _fdr(**overrides):
    data = {"stored_adj_p": 0.001, "stored_k": 1000, "current_adj_p": 0.002, "current_pass": True}
    data.update(overrides)
    return data


def _c8(status="PASSED"):
    return {"status": None, "reason": None, "c8_oos_status": status, "n_oos": 40, "oos_expectancy_r": 0.12}


def _account(ok=True):
    return {"available": True, "gate_ok": ok, "operational_pass_probability": 0.91}


def _lifecycle(
    *,
    strategy_id="SID_A",
    criterion12_valid=True,
    sr_status="CONTINUE",
    blocked=False,
    review_outcome=None,
):
    return {
        "criterion12": {"valid": criterion12_valid, "reason": None, "state_age_days": 0},
        "strategy_states": {
            strategy_id: {
                "sr_status": sr_status,
                "sr_review_outcome": review_outcome,
                "sr_reviewed_at": "2026-05-10" if review_outcome else None,
                "sr_review_summary": "reviewed watch" if review_outcome else None,
                "sr_recheck_trigger": "recheck at N>=100" if review_outcome else None,
                "blocked": blocked,
                "block_source": "pause" if blocked else None,
                "block_reason": "paused" if blocked else None,
                "paused": blocked,
                "pause_reason": "paused" if blocked else None,
            }
        },
    }


def _classify(
    row=None,
    replay=None,
    fdr=None,
    c8=None,
    account=None,
    lifecycle=None,
    scope="profile",
    profile_lane_ids=None,
):
    active_row = row or _row()
    return dep._classify_strategy(
        active_row,
        replay=replay or _replay(),
        current_fdr=fdr or _fdr(),
        c8=c8 or _c8(),
        account_state=account if account is not None else _account(),
        lifecycle_state=lifecycle if lifecycle is not None else _lifecycle(strategy_id=str(active_row["strategy_id"])),
        profile_lane_ids=profile_lane_ids if profile_lane_ids is not None else {str(active_row["strategy_id"])},
        scope=scope,
    )


def test_clean_strategy_can_be_deployable_candidate():
    result = _classify()

    assert result.verdict == dep.DEPLOYABLE_CANDIDATE
    assert result.deployable is True
    assert result.institutional_language_allowed is True


def test_oos_pass_through_is_not_deployable():
    result = _classify(c8=_c8("INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH"))

    assert result.verdict == dep.BLOCKED_OOS_UNDERPOWERED
    assert any(issue.id == "c8_not_passed" for issue in result.issues)


def test_missing_slippage_blocks_deployability():
    result = _classify(row=_row(instrument="MGC", slippage_validation_status=None))

    assert result.verdict == dep.BLOCKED_SLIPPAGE
    assert any(issue.id == "slippage_missing" for issue in result.issues)


def test_mnq_e2_covered_session_infers_routine_slippage_evidence():
    result = _classify(row=_row(slippage_validation_status=None))

    assert result.verdict == dep.CONTROLLED_LIVE_PILOT_CANDIDATE
    assert result.deployable is True
    assert result.institutional_language_allowed is False
    event_tail = [issue for issue in result.issues if issue.id == "slippage_event_tail_pending"]
    assert len(event_tail) == 1
    assert event_tail[0].detail["inferred_from_routine_tbbo"] is True
    assert event_tail[0].detail["effective_status"] == "PENDING_EVENT_TAIL"


def test_mnq_e2_uncovered_session_missing_slippage_still_blocks():
    result = _classify(row=_row(orb_label="BRISBANE_1025", slippage_validation_status=None))

    assert result.verdict == dep.BLOCKED_SLIPPAGE
    assert any(issue.id == "slippage_missing" for issue in result.issues)


def test_mnq_non_e2_missing_slippage_still_blocks():
    result = _classify(row=_row(entry_model="E1", slippage_validation_status=None))

    assert result.verdict == dep.BLOCKED_SLIPPAGE
    assert any(issue.id == "slippage_missing" for issue in result.issues)


def test_mnq_event_tail_pending_is_controlled_pilot_not_institutional_language():
    result = _classify(row=_row(slippage_validation_status="PENDING_EVENT_TAIL"))

    assert result.verdict == dep.CONTROLLED_LIVE_PILOT_CANDIDATE
    assert result.deployable is True
    assert result.institutional_language_allowed is False
    assert any(issue.id == "slippage_event_tail_pending" for issue in result.issues)


def test_non_mnq_event_tail_pending_still_blocks_slippage():
    # MGC is NOT in ROUTINE_TBBO_SLIPPAGE_REGISTRY; an explicit PENDING_EVENT_TAIL
    # on an unregistered instrument falls through to slippage_not_passed (HARD).
    result = _classify(row=_row(instrument="MGC", slippage_validation_status="PENDING_EVENT_TAIL"))

    assert result.verdict == dep.BLOCKED_SLIPPAGE
    assert any(issue.id == "slippage_not_passed" for issue in result.issues)


def test_mes_event_tail_pending_is_controlled_pilot_not_institutional_language():
    # MES IS in ROUTINE_TBBO_SLIPPAGE_REGISTRY; an explicit PENDING_EVENT_TAIL on a
    # MES row must reach CONTROLLED_LIVE_PILOT_CANDIDATE, the same as MNQ.
    # Pre-audit-fix this fell through to BLOCKED_SLIPPAGE because
    # `_slippage_is_controlled_event_tail_pending` hardcoded `instrument == "MNQ"`.
    result = _classify(
        row=_row(
            instrument="MES",
            orb_label="COMEX_SETTLE",
            slippage_validation_status="PENDING_EVENT_TAIL",
        )
    )

    assert result.verdict == dep.CONTROLLED_LIVE_PILOT_CANDIDATE
    assert result.deployable is True
    assert result.institutional_language_allowed is False
    event_tail = [issue for issue in result.issues if issue.id == "slippage_event_tail_pending"]
    assert len(event_tail) == 1
    assert event_tail[0].detail["inferred_from_routine_tbbo"] is False
    assert event_tail[0].detail["effective_status"] == "PENDING_EVENT_TAIL"


def test_current_k_fdr_failure_blocks_deployability():
    result = _classify(fdr=_fdr(current_pass=False, current_adj_p=0.052))

    assert result.verdict == dep.BLOCKED_CURRENT_K_FDR
    assert any(issue.id == "current_k_fdr_fail" for issue in result.issues)


def test_current_k_fdr_uses_canonical_bh_owner(monkeypatch):
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE experimental_strategies (
            strategy_id TEXT,
            p_value DOUBLE,
            is_canonical BOOLEAN,
            orb_label TEXT,
            instrument TEXT
        )
    """)
    con.executemany(
        "INSERT INTO experimental_strategies VALUES (?, ?, TRUE, 'COMEX_SETTLE', 'MNQ')",
        [("SID_A", 0.001), ("SID_B", 0.20)],
    )
    calls = []

    def fake_bh(p_values, *, alpha=0.05, total_tests=None):
        calls.append((p_values, alpha, total_tests))
        return {
            "SID_A": {"adjusted_p": 0.01},
            "SID_B": {"adjusted_p": 0.20},
        }

    monkeypatch.setattr(dep, "benjamini_hochberg", fake_bh)

    result = dep._current_k_fdr(
        con,
        [{"strategy_id": "SID_A", "orb_label": "COMEX_SETTLE", "fdr_adjusted_p": 0.50, "discovery_k": 10}],
    )
    con.close()

    assert calls == [([("SID_A", 0.001), ("SID_B", 0.20)], 0.05, 2)]
    assert result["SID_A"]["current_adj_p"] == 0.01
    assert result["SID_A"]["current_pass"] is True


def test_e2_lookahead_filter_is_no_go_bias_or_data():
    result = _classify(row=_row(filter_type="VOL_RV12_N20"))

    assert result.verdict == dep.NO_GO_BIAS_OR_DATA
    assert any(issue.id == "e2_deployment_unsafe_filter" for issue in result.issues)


def test_e2_prior_day_direction_selector_is_no_go_bias_or_data():
    result = _classify(row=_row(filter_type="PD_GO_LONG"))

    assert result.verdict == dep.NO_GO_BIAS_OR_DATA
    assert any(issue.id == "e2_deployment_unsafe_filter" for issue in result.issues)


def test_replay_mismatch_blocks_deployability():
    result = _classify(replay=_replay(ok=False, recomputed_sample_size=198))

    assert result.verdict == dep.BLOCKED_REPLAY_MISMATCH
    assert any(issue.id == "replay_mismatch" for issue in result.issues)


def test_dsr_below_cross_check_blocks_institutional_language_only():
    result = _classify(row=_row(dsr_score=0.10))

    assert result.verdict == dep.DEPLOYABLE_CANDIDATE
    assert result.deployable is True
    assert result.institutional_language_allowed is False
    assert any(issue.id == "dsr_below_cross_check" for issue in result.issues)


def test_account_gate_failure_blocks_deployability():
    result = _classify(account=_account(ok=False))

    assert result.verdict == dep.BLOCKED_ACCOUNT_RISK
    assert any(issue.id == "account_risk_fail" for issue in result.issues)


def test_invalid_criterion12_blocks_runtime():
    result = _classify(lifecycle=_lifecycle(criterion12_valid=False))

    assert result.verdict == dep.BLOCKED_RUNTIME
    assert any(issue.id == "criterion12_invalid" for issue in result.issues)


def test_lifecycle_pause_blocks_runtime():
    result = _classify(lifecycle=_lifecycle(blocked=True, sr_status="ALARM"))

    assert result.verdict == dep.BLOCKED_RUNTIME
    assert any(issue.id == "lifecycle_blocked" for issue in result.issues)


def test_reviewed_sr_alarm_is_controlled_pilot_warning():
    result = _classify(lifecycle=_lifecycle(sr_status="ALARM", review_outcome="watch"))

    assert result.verdict == dep.CONTROLLED_LIVE_PILOT_CANDIDATE
    assert result.deployable is True
    assert result.institutional_language_allowed is False
    assert any(issue.id == "sr_alarm_watch_reviewed" for issue in result.issues)


def test_unreviewed_sr_alarm_blocks_runtime():
    result = _classify(lifecycle=_lifecycle(sr_status="ALARM"))

    assert result.verdict == dep.BLOCKED_RUNTIME
    assert any(issue.id == "sr_alarm_unreviewed" for issue in result.issues)


def test_all_active_scope_does_not_invent_account_failure_for_non_profile_row():
    result = _classify(scope="all-active", profile_lane_ids=set())

    assert result.verdict == dep.DEPLOYABLE_CANDIDATE
    assert not any(issue.id == "account_risk_missing" for issue in result.issues)
    assert any(issue.id == "profile_not_evaluated" and issue.severity == "info" for issue in result.issues)


def test_profile_scope_still_blocks_non_profile_rows_on_account_risk():
    result = _classify(profile_lane_ids=set())

    assert result.verdict == dep.BLOCKED_ACCOUNT_RISK
    assert any(issue.id == "account_risk_missing" for issue in result.issues)


def test_instrument_summary_labels_mes_and_mgc_gaps():
    # MES_A uses an orb_label NOT in the MES routine-TBBO pilot v1 registry
    # (BRISBANE_1025 is MGC's session); routine-TBBO inference must NOT apply,
    # so slippage_missing remains the surfaced hard issue.
    mes = _classify(
        row=_row(
            strategy_id="MES_A",
            instrument="MES",
            orb_label="BRISBANE_1025",
            slippage_validation_status=None,
        )
    )
    mgc = _classify(
        row=_row(
            strategy_id="MGC_A",
            instrument="MGC",
            sample_size=50,
            robustness_status="ROBUST",
            slippage_validation_status=None,
        )
    )

    summary = dep._build_instrument_summary([mes, mgc])

    assert summary["MES"]["hard_issue_counts"]["slippage_missing"] == 1
    assert summary["MGC"]["sample_size_below_100"] == 1
    assert summary["MGC"]["family_status_counts"]["ROBUST"] == 1


def test_promotion_queue_separates_evidence_gaps_from_purge_candidates():
    near = _classify(row=_row(strategy_id="NEAR", instrument="MGC", slippage_validation_status=None))
    purge = _classify(row=_row(strategy_id="PURGE", robustness_status="PURGED"))

    queue = dep._build_promotion_queue([near, purge])

    assert queue["nearest_to_deployable"]["count"] == 1
    assert queue["nearest_to_deployable"]["rows"][0]["strategy_id"] == "NEAR"
    assert queue["retire_or_purge"]["count"] == 1
    assert queue["retire_or_purge"]["rows"][0]["strategy_id"] == "PURGE"


def test_trade_context_marks_regime_conditional_filters():
    result = _classify(row=_row(filter_type="ATR70_VOL12_ORB_G5", sample_size=80))

    assert result.trade_context["archetype"] == "event_session_orb_breakout"
    assert result.trade_context["research_role"] == "standalone_lane_with_conditional_filters"
    assert result.trade_context["sample_class"] == "REGIME_CONDITIONAL_ONLY"
    assert "volatility_or_participation_regime_filter" in result.trade_context["conditional_components"]
    assert "friction_or_orb_size_regime_filter" in result.trade_context["conditional_components"]


def _state_report(strategy):
    return {
        "generated_at": "2026-05-10T00:00:00+00:00",
        "db_path": "test.db",
        "scope": "profile",
        "profile_id": "topstep_50k_mnq_auto",
        "strict": True,
        "source_truth": {"candidate_source": "validated_setups as candidate list only"},
        "resource_lit": {"multiple_testing": "docs/institutional/literature/harvey_liu_2015_backtesting.md"},
        "summary": {"total_candidates": 1, "hard_issue_counts": {}},
        "account_state": {"available": True, "gate_ok": True},
        "strategies": [strategy],
    }


def test_write_deployability_state_is_append_only_and_latest_selects_current_verdict(tmp_path):
    db_path = tmp_path / "state.db"
    first = _classify().to_dict()
    second = _classify(row=_row(instrument="MGC", slippage_validation_status=None)).to_dict()

    first_write = write_deployability_state(
        _state_report(first),
        db_path=db_path,
        rebuild_id="rid-1",
        git_sha="abc123",
    )
    second_write = write_deployability_state(
        _state_report(second),
        db_path=db_path,
        rebuild_id="rid-2",
        git_sha="def456",
    )

    assert first_write["rows_written"] == 1
    assert second_write["rows_written"] == 1

    latest = load_latest_deployment_readiness(
        db_path=db_path,
        profile_id="topstep_50k_mnq_auto",
        scope="profile",
    )
    assert len(latest) == 1
    assert latest[0]["strategy_id"] == "SID_A"
    assert latest[0]["verdict"] == dep.BLOCKED_SLIPPAGE
    assert latest[0]["rebuild_id"] == "rid-2"
    assert "rn" not in latest[0]
    assert json.loads(latest[0]["hard_issue_ids"]) == ["slippage_missing"]
    assert "source_truth" in json.loads(latest[0]["provenance_json"])

    with duckdb.connect(str(db_path), read_only=True) as con:
        row = con.execute("SELECT COUNT(*) FROM deployment_readiness_evaluations").fetchone()
    assert row[0] == 2


def test_write_deployability_state_rejects_malformed_report(tmp_path):
    bad_report = {
        "scope": "profile",
        "strategies": [{"strategy_id": "SID_A", "verdict": dep.DEPLOYABLE_CANDIDATE}],
    }

    try:
        write_deployability_state(bad_report, db_path=tmp_path / "bad.db", git_sha="abc123")
    except ValueError as exc:
        assert "missing fields" in str(exc)
    else:
        raise AssertionError("malformed deployability report should fail closed")


# --- MES routine-TBBO slippage registry coverage (Stage 1) ---

import pytest


@pytest.mark.parametrize(
    "session",
    sorted(dep.ROUTINE_TBBO_SLIPPAGE_REGISTRY["MES"].sessions),
)
def test_mes_e2_covered_session_infers_routine_slippage_evidence(session):
    result = _classify(
        row=_row(
            instrument="MES",
            orb_label=session,
            slippage_validation_status=None,
        )
    )

    assert result.verdict == dep.CONTROLLED_LIVE_PILOT_CANDIDATE
    assert result.deployable is True
    assert result.institutional_language_allowed is False
    event_tail = [issue for issue in result.issues if issue.id == "slippage_event_tail_pending"]
    assert len(event_tail) == 1
    assert event_tail[0].detail["inferred_from_routine_tbbo"] is True
    assert event_tail[0].detail["effective_status"] == "PENDING_EVENT_TAIL"
    assert event_tail[0].detail["covered_session"] is True
    assert "MES" in event_tail[0].detail["basis"]
    assert "2026-04-24-mes-e2-slippage-pilot-v1.md" in event_tail[0].detail["basis"]


def test_mes_e2_uncovered_session_missing_slippage_still_blocks():
    # US_DATA_1000 is a deployable MNQ session but is NOT in the MES pilot v1
    # session set; routine inference must NOT fire on MES for it.
    result = _classify(
        row=_row(
            instrument="MES",
            orb_label="US_DATA_1000",
            slippage_validation_status=None,
        )
    )

    assert result.verdict == dep.BLOCKED_SLIPPAGE
    assert any(issue.id == "slippage_missing" for issue in result.issues)


def test_mes_non_e2_missing_slippage_still_blocks():
    result = _classify(
        row=_row(
            instrument="MES",
            orb_label="COMEX_SETTLE",
            entry_model="E1",
            slippage_validation_status=None,
        )
    )

    assert result.verdict == dep.BLOCKED_SLIPPAGE
    assert any(issue.id == "slippage_missing" for issue in result.issues)


def test_unregistered_instrument_is_not_routine_tbbo_eligible():
    # MGC has no entry in the routine-TBBO pilot registry; matching session+model
    # alone must NOT trigger routine inference.
    result = _classify(
        row=_row(
            instrument="MGC",
            orb_label="COMEX_SETTLE",
            entry_model="E2",
            slippage_validation_status=None,
        )
    )

    assert result.verdict == dep.BLOCKED_SLIPPAGE
    assert any(issue.id == "slippage_missing" for issue in result.issues)
