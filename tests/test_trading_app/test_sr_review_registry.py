from __future__ import annotations

from trading_app.sr_review_registry import get_sr_alarm_review


def test_nyse_rr15_costlt12_sr_alarm_has_code_backed_watch_review() -> None:
    review = get_sr_alarm_review("topstep_50k_mnq_auto", "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12")

    assert review is not None
    assert review.outcome == "watch"
    assert review.reviewed_at == "2026-05-11"
    assert "WFE" in review.summary
    assert "OOS/IS" in review.summary
    # OOS/IS figure must be the canonical 112% from validated_setups
    # (oos_exp_r / expectancy_r = 0.1179 / 0.105 = 1.1229). An earlier draft
    # cited "61%" with no canonical source — see
    # memory/feedback_sr_review_registry_audit_pattern.md.
    assert "112%" in review.summary
    assert "0.1179" in review.summary
    assert "0.105" in review.summary
    # Directional risk must be carried into the recheck trigger because OOS
    # short direction is negative at N=42 (OOS power STATISTICALLY_USELESS).
    assert review.recheck_trigger is not None
    assert "N>=100" in review.recheck_trigger
    assert "short" in review.recheck_trigger.lower()


def test_comex_settle_orb_vol_2k_sr_alarm_has_code_backed_watch_review() -> None:
    review = get_sr_alarm_review(
        "topstep_50k_mnq_auto", "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K"
    )

    assert review is not None
    assert review.outcome == "watch"
    assert review.reviewed_at == "2026-05-17"
    assert "WFE 1.486" in review.summary
    assert "OOS/IS 96%" in review.summary
    # Canonical OOS/IS figures must be quoted verbatim from validated_setups
    # (oos_exp_r=0.1121, expectancy_r=0.1168). Drift detector for stale
    # registry figures — see memory/feedback_sr_review_registry_audit_pattern.md.
    assert "0.1121" in review.summary
    assert "0.1168" in review.summary
    # 2026-05-17 regime-stratified audit K=2 PARK is recorded so the
    # provenance trail survives any future audit-MD relocation.
    assert "K=2 PARK" in review.summary
    assert "fire-rate" in review.summary.lower()
    # R5 + R6 numerics carry the descriptive evidence trail required
    # by Amendment 2.7 — both must remain positive at recheck time.
    assert "+0.2494" in review.summary
    assert "+0.0291" in review.summary
    # Four-precedent citation per pre_registered_criteria.md:242 must
    # remain (drift detector against quiet removal).
    assert "L3 2026-04-12" in review.summary
    assert "L4/L6 2026-04-14" in review.summary
    assert "NYSE_OPEN_RR1.5" in review.summary
    # K=2/K=8 concur (both PARK/FAIL) -- conservative-wins rule does NOT
    # invert; this must be recorded so a future K=8 disagreement is not
    # silently treated as a softening.
    assert "concur" in review.summary.lower() or "no conservative-wins inversion" in review.summary
    assert review.recheck_trigger is not None
    assert "N>=100" in review.recheck_trigger
    assert "regime-stratified" in review.recheck_trigger
    assert "fire-rate" in review.recheck_trigger.lower()


def test_nyse_open_costlt12_rr10_sr_alarm_has_code_backed_watch_review() -> None:
    review = get_sr_alarm_review(
        "topstep_50k_mnq_auto", "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12"
    )

    assert review is not None
    assert review.outcome == "watch"
    assert review.reviewed_at == "2026-05-17"
    assert "WFE 1.9516" in review.summary
    assert "OOS/IS 114%" in review.summary
    # Canonical OOS/IS figures must be quoted verbatim
    # (oos_exp_r=0.0993, expectancy_r=0.087).
    assert "0.0993" in review.summary
    assert "0.087" in review.summary
    # K=2 CONTINUE verdict from the 2026-05-17 audit is the strongest
    # of the four lanes — must be recorded verbatim.
    assert "K=2 CONTINUE" in review.summary
    assert "p=0.234" in review.summary
    assert "p=0.552" in review.summary
    assert "+0.1282" in review.summary
    assert "+0.0817" in review.summary
    # OOS-by-direction check 2026-05-17: both long and short positive
    # on the raw cohort, so C12 directional-risk addendum does NOT
    # trigger. Must be recorded so future readers do not assume the
    # check was skipped.
    assert "OOS long ExpR=+0.0935" in review.summary
    assert "OOS short ExpR=+0.0721" in review.summary
    assert "no directional" in review.summary.lower()
    # Four-precedent citation per pre_registered_criteria.md:242.
    assert "L3 2026-04-12" in review.summary
    assert "L4/L6 2026-04-14" in review.summary
    assert "NYSE_OPEN_RR1.5" in review.summary
    assert review.recheck_trigger is not None
    assert "N>=100" in review.recheck_trigger


def test_us_data_1000_ovnrng_25_rr10_sr_alarm_has_code_backed_pause_review() -> None:
    review = get_sr_alarm_review(
        "topstep_50k_mnq_auto", "MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25"
    )

    assert review is not None
    # outcome is PAUSE per K=2/K=8 conservative-wins rule (audit MD line 12).
    # K=2 PARK + K=8 WATCH differ; K=2 PARK is conservative -> K=2 wins.
    # If a future edit silently flips this to "watch" without resolving the
    # K=2/K=8 disagreement, this test fails.
    assert review.outcome == "pause"
    assert review.reviewed_at == "2026-05-17"
    # Canonical deploy-floor figures must be quoted verbatim
    # (oos_exp_r=0.1222, expectancy_r=0.0986); these PASS even though
    # the audit verdict governs the outcome.
    assert "WFE 1.6058" in review.summary
    assert "OOS/IS 124%" in review.summary
    assert "0.1222" in review.summary
    assert "0.0986" in review.summary
    # Conservative-wins rule must be cited verbatim — it is the load-bearing
    # justification for PAUSE over WATCH on a deploy-floor-passing lane.
    assert "conservative-wins" in review.summary
    assert "K=2 PARK" in review.summary
    assert "K=8" in review.summary
    assert "p=0.00608" in review.summary
    assert "k8_p=0.04861" in review.summary
    assert "+0.0792" in review.summary
    assert "+0.0338" in review.summary
    # Four-precedent citation per pre_registered_criteria.md:242 must
    # remain in the summary as long as this is "fifth-or-later" WATCH/PAUSE
    # without an amendment block. Drift detector against quiet removal.
    assert "L3 2026-04-12" in review.summary
    assert "L4/L6 2026-04-14" in review.summary
    assert "NYSE_OPEN_RR1.5" in review.summary
    assert review.recheck_trigger is not None
    # Recheck pathway must reference resolving the K=2/K=8 disagreement
    # rather than a flat N>=100 monitored-trade re-check — pause governance
    # is audit-verdict-driven, not monitored-trade-driven.
    assert "K=2" in review.recheck_trigger
    assert "K=8" in review.recheck_trigger
