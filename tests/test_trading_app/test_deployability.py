from __future__ import annotations

from trading_app import deployability as dep


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
    result = _classify(row=_row(slippage_validation_status=None))

    assert result.verdict == dep.BLOCKED_SLIPPAGE
    assert any(issue.id == "slippage_missing" for issue in result.issues)


def test_mnq_event_tail_pending_is_controlled_pilot_not_institutional_language():
    result = _classify(row=_row(slippage_validation_status="PENDING_EVENT_TAIL"))

    assert result.verdict == dep.CONTROLLED_LIVE_PILOT_CANDIDATE
    assert result.deployable is True
    assert result.institutional_language_allowed is False
    assert any(issue.id == "slippage_event_tail_pending" for issue in result.issues)


def test_non_mnq_event_tail_pending_still_blocks_slippage():
    result = _classify(row=_row(instrument="MGC", slippage_validation_status="PENDING_EVENT_TAIL"))

    assert result.verdict == dep.BLOCKED_SLIPPAGE
    assert any(issue.id == "slippage_not_passed" for issue in result.issues)


def test_current_k_fdr_failure_blocks_deployability():
    result = _classify(fdr=_fdr(current_pass=False, current_adj_p=0.052))

    assert result.verdict == dep.BLOCKED_CURRENT_K_FDR
    assert any(issue.id == "current_k_fdr_fail" for issue in result.issues)


def test_e2_lookahead_filter_is_no_go_bias_or_data():
    result = _classify(row=_row(filter_type="VOL_RV12_N20"))

    assert result.verdict == dep.NO_GO_BIAS_OR_DATA
    assert any(issue.id == "e2_lookahead_filter" for issue in result.issues)


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
    mes = _classify(row=_row(strategy_id="MES_A", instrument="MES", slippage_validation_status=None))
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
    near = _classify(row=_row(strategy_id="NEAR", slippage_validation_status=None))
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
