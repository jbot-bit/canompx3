from __future__ import annotations

from research.mnq_profile_candidate_proposal_2026_05_11 import (
    CandidateDecision,
    CandidateRecord,
    CandidateResult,
    PortfolioGate,
    _render,
    build_promotion_candidates,
    choose_dedupe_heads,
    classify_candidate,
)


def _candidate(
    strategy_id: str,
    *,
    expectancy_r: float,
    chordia_verdict: str = "MISSING",
    family_hash: str = "fam",
    session: str = "COMEX_SETTLE",
    filter_type: str = "ORB_G5",
) -> CandidateRecord:
    return CandidateRecord(
        strategy_id=strategy_id,
        instrument="MNQ",
        orb_label=session,
        orb_minutes=5,
        entry_model="E2",
        rr_target=1.5,
        confirm_bars=1,
        filter_type=filter_type,
        stop_multiplier=1.0,
        family_hash=family_hash,
        expectancy_r=expectancy_r,
        sample_size=500,
        deployability_verdict="CONTROLLED_LIVE_PILOT_CANDIDATE",
        c8_oos_status="PASSED",
        replay_ok=True,
        current_fdr_pass=True,
        family_status="WHITELISTED",
        hard_issue_ids=(),
        slippage_status="PENDING_EVENT_TAIL",
        chordia_verdict=chordia_verdict,
        chordia_audit_age_days=2,
        runtime_control_evaluated=True,
        runtime_sr_status=None,
        runtime_sr_review_outcome=None,
        profile_allowed=True,
    )


def test_choose_dedupe_heads_prefers_audited_chordia_over_higher_expr_missing() -> None:
    missing = _candidate("MNQ_COMEX_SETTLE_E2_RR2.0_CB1_COST_LT12", expectancy_r=0.25)
    audited = _candidate(
        "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12",
        expectancy_r=0.17,
        chordia_verdict="PASS_CHORDIA",
    )

    heads = choose_dedupe_heads([missing, audited])

    assert heads[audited.strategy_id] is True
    assert heads[missing.strategy_id] is False


def test_classify_active_same_session_requires_replacement_improvement() -> None:
    candidate = _candidate(
        "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100",
        expectancy_r=0.17,
        chordia_verdict="PASS_CHORDIA",
    )
    gate = PortfolioGate(
        add_delta_annual_r=10.0,
        add_delta_sharpe=0.2,
        replace_delta_annual_r=-1.0,
        replace_delta_sharpe=0.1,
        corr_gate_pass=True,
        corr_reject_reasons=(),
        replacement_target="MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
        replacement_target_status="DEPLOY",
        account_risk_ok=True,
        account_risk_detail="ok",
    )

    decision = classify_candidate(candidate, dedupe_head=True, gate=gate)

    assert decision.decision == "PARK"
    assert "replacement" in decision.primary_reason.lower()


def test_classify_current_deployed_lane_is_no_change_park() -> None:
    candidate = _candidate(
        "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
        expectancy_r=0.21,
        chordia_verdict="PASS_CHORDIA",
    )
    gate = PortfolioGate(
        add_delta_annual_r=0.0,
        add_delta_sharpe=0.0,
        replace_delta_annual_r=0.0,
        replace_delta_sharpe=0.0,
        corr_gate_pass=False,
        corr_reject_reasons=("self",),
        replacement_target=candidate.strategy_id,
        replacement_target_status="DEPLOY",
        account_risk_ok=True,
        account_risk_detail="ok",
    )

    decision = classify_candidate(candidate, dedupe_head=True, gate=gate)

    assert decision.decision == "PARK"
    assert "already selected" in decision.primary_reason.lower()


def test_classify_paused_same_session_can_pass_as_replacement_when_additive_math_clears() -> None:
    candidate = _candidate(
        "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
        expectancy_r=0.105,
        chordia_verdict="PASS_PROTOCOL_A",
        family_hash="nyse",
        session="NYSE_OPEN",
        filter_type="COST_LT12",
    )
    gate = PortfolioGate(
        add_delta_annual_r=8.0,
        add_delta_sharpe=0.1,
        replace_delta_annual_r=None,
        replace_delta_sharpe=None,
        corr_gate_pass=True,
        corr_reject_reasons=(),
        replacement_target="MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
        replacement_target_status="PAUSE",
        account_risk_ok=True,
        account_risk_detail="ok",
    )

    decision = classify_candidate(candidate, dedupe_head=True, gate=gate)

    assert decision.decision == "PASS_REPLACE"
    assert "paused" in decision.primary_reason.lower()


def test_classify_parks_when_exact_runtime_sr_gate_was_not_evaluated() -> None:
    candidate = _candidate(
        "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
        expectancy_r=0.105,
        chordia_verdict="PASS_PROTOCOL_A",
        family_hash="nyse",
        session="NYSE_OPEN",
        filter_type="COST_LT12",
    )
    candidate = CandidateRecord(
        **{
            **candidate.__dict__,
            "runtime_control_evaluated": False,
        }
    )
    gate = PortfolioGate(
        add_delta_annual_r=8.0,
        add_delta_sharpe=0.1,
        replace_delta_annual_r=None,
        replace_delta_sharpe=None,
        corr_gate_pass=True,
        corr_reject_reasons=(),
        replacement_target="MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
        replacement_target_status="PAUSE",
        account_risk_ok=True,
        account_risk_detail="ok",
    )

    decision = classify_candidate(candidate, dedupe_head=True, gate=gate)

    assert decision.decision == "PARK"
    assert "sr" in decision.primary_reason.lower()


def test_render_cites_local_literature_and_resources() -> None:
    candidate = _candidate("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100", expectancy_r=0.20)
    result = CandidateResult(
        candidate=candidate,
        decision=CandidateDecision(candidate.strategy_id, "PARK", "No Chordia audit.", ("chordia_missing",)),
        gate=None,
    )

    rendered = _render([result])

    assert "docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md" in rendered
    assert "docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md" in rendered
    assert "docs/institutional/literature/carver_2015_ch11_portfolios.md" in rendered
    assert "docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md" in rendered
    assert "resources/prop-firm-official-rules.md" in rendered


def test_build_promotion_candidates_preserves_generic_runtime_contract(monkeypatch) -> None:
    candidate = _candidate(
        "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
        expectancy_r=0.105,
        chordia_verdict="PASS_PROTOCOL_A",
        family_hash="nyse",
        session="NYSE_OPEN",
        filter_type="COST_LT12",
    )
    result = CandidateResult(
        candidate=candidate,
        decision=CandidateDecision(candidate.strategy_id, "PASS_REPLACE", "ok", ()),
        gate=PortfolioGate(
            add_delta_annual_r=8.0,
            add_delta_sharpe=0.1,
            replace_delta_annual_r=None,
            replace_delta_sharpe=None,
            corr_gate_pass=True,
            corr_reject_reasons=(),
            replacement_target="MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
            replacement_target_status="PAUSE",
            account_risk_ok=True,
            account_risk_detail="worst_case=$415<=2000; slots=3/7",
        ),
    )
    monkeypatch.setattr(
        "research.mnq_profile_candidate_proposal_2026_05_11._candidate_p90_orb",
        lambda _candidate: 85.8,
    )

    promotions = build_promotion_candidates([result])

    assert len(promotions) == 1
    promotion = promotions[0]
    assert promotion.status == "PROVISIONAL"
    assert promotion.chordia_verdict == "PASS_PROTOCOL_A"
    assert promotion.p90_orb_pts == 85.8
    assert promotion.replacement_target == "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12"


def test_build_promotion_candidates_requires_explicit_runtime_bootstrap(monkeypatch) -> None:
    candidate = _candidate(
        "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
        expectancy_r=0.105,
        chordia_verdict="PASS_PROTOCOL_A",
        family_hash="nyse",
        session="NYSE_OPEN",
        filter_type="COST_LT12",
    )
    candidate = CandidateRecord(
        **{
            **candidate.__dict__,
            "runtime_control_evaluated": False,
        }
    )
    result = CandidateResult(
        candidate=candidate,
        decision=CandidateDecision(
            candidate.strategy_id,
            "PARK",
            "Exact lane SR/runtime control was not evaluated.",
            ("runtime_sr_not_evaluated",),
        ),
        gate=PortfolioGate(
            add_delta_annual_r=8.0,
            add_delta_sharpe=0.1,
            replace_delta_annual_r=None,
            replace_delta_sharpe=None,
            corr_gate_pass=True,
            corr_reject_reasons=(),
            replacement_target="MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
            replacement_target_status="PAUSE",
            account_risk_ok=True,
            account_risk_detail="worst_case=$415<=2000; slots=3/7",
        ),
    )
    monkeypatch.setattr(
        "research.mnq_profile_candidate_proposal_2026_05_11._candidate_p90_orb",
        lambda _candidate: 85.8,
    )

    assert build_promotion_candidates([result]) == []

    promotions = build_promotion_candidates([result], allow_runtime_bootstrap=True)

    assert len(promotions) == 1
    assert promotions[0].status == "PROVISIONAL"
    assert "post-promotion SR refresh required" in promotions[0].status_reason

    blocked = build_promotion_candidates(
        [result],
        allow_runtime_bootstrap=True,
        runtime_bootstrap_checker=lambda _candidate: (False, "SR bootstrap status=ALARM"),
    )
    assert blocked == []
