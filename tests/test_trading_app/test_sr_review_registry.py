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
