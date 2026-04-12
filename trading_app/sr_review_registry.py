"""Code-backed manual review outcomes for Criterion 12 SR alarms.

An SR ALARM means "suspend pending manual review". After review, the system
needs an explicit machine-readable outcome instead of leaving the decision in
prose only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SrAlarmReview:
    profile_id: str
    strategy_id: str
    outcome: str  # "watch" | "pause"
    reviewed_at: str
    summary: str
    recheck_trigger: str | None = None


SR_ALARM_REVIEWS: dict[tuple[str, str], SrAlarmReview] = {
    (
        "topstep_50k_mnq_auto",
        "MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5",
    ): SrAlarmReview(
        profile_id="topstep_50k_mnq_auto",
        strategy_id="MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5",
        outcome="watch",
        reviewed_at="2026-04-10",
        summary=(
            "Reviewed WATCH: edge is stable; SR alarm attributed to NYSE_OPEN volatility "
            "regime shift rather than confirmed edge decay."
        ),
        recheck_trigger="Monitor for volatility normalization and rerun SR on fresh state updates.",
    ),
    (
        "topstep_50k_mnq_auto",
        "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
    ): SrAlarmReview(
        profile_id="topstep_50k_mnq_auto",
        strategy_id="MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
        outcome="watch",
        reviewed_at="2026-04-12",
        summary=(
            "Reviewed WATCH: C6 WFE 0.52 and C8 ratio 53% still clear the literature "
            "floors; SR alarm is the only trigger."
        ),
        recheck_trigger=(
            "Re-check after N>=100 monitored trades. Retire if SR remains ALARM and "
            "(WFE < 0.50 or C8 ratio < 0.40)."
        ),
    ),
}


def get_sr_alarm_review(profile_id: str, strategy_id: str) -> SrAlarmReview | None:
    """Return the canonical manual review outcome for one SR-alarmed lane."""
    return SR_ALARM_REVIEWS.get((profile_id, strategy_id))
