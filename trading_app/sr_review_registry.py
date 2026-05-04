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
    # L3 — reviewed 2026-04-12 (literature-grounded audit post Apr 7 Phase 0).
    # Kept here unchanged: C6 WFE 0.52 and C8 ratio 53% still clear the
    # literature floors. SR alarm is the only trigger.
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
            "Re-check after N>=100 monitored trades. Retire if SR remains ALARM and (WFE < 0.50 or C8 ratio < 0.40)."
        ),
    ),
    # L4 — reviewed 2026-04-14 autonomously per L3/COMEX_SETTLE precedent.
    # Diagnostics are materially STRONGER than the L3 precedent:
    #   WFE 2.14 (>> 0.50 floor), OOS ExpR 0.104 = 116% of IS 0.089,
    #   p=0.0003, N=1521. OOS outperforms IS — no decay signal.
    # Replaces the stale RR1.5 entry: the deployed lane is RR1.0 (allocator
    # swap during Apr 13 allocator-wiring); the RR1.5 review was defending
    # an undeployed sibling.
    (
        "topstep_50k_mnq_auto",
        "MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5",
    ): SrAlarmReview(
        profile_id="topstep_50k_mnq_auto",
        strategy_id="MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5",
        outcome="watch",
        reviewed_at="2026-04-14",
        summary=(
            "Reviewed WATCH: WFE 2.14 (4x literature floor) and OOS ExpR 0.104 = "
            "116% of IS ExpR 0.089 — OOS outperforms IS, no decay signal. "
            "SR alarm is the only trigger; matches L3 standard."
        ),
        recheck_trigger=(
            "Re-check after N>=100 monitored trades. Retire if SR remains ALARM "
            "and (WFE < 0.50 or OOS/IS ratio < 0.40)."
        ),
    ),
    # L6 — reviewed 2026-04-14 autonomously per L3/COMEX_SETTLE precedent.
    # Diagnostics are STRONGER than the L3 precedent:
    #   WFE 0.90 (>> 0.50 floor), OOS ExpR 0.207 = 98% of IS 0.210 — near
    #   perfect OOS match, p<0.0001, N=701, Sharpe 0.17.
    # SR alarm on 35 monitored trades is noise on a well-validated lane.
    (
        "topstep_50k_mnq_auto",
        "MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
    ): SrAlarmReview(
        profile_id="topstep_50k_mnq_auto",
        strategy_id="MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15",
        outcome="watch",
        reviewed_at="2026-04-14",
        summary=(
            "Reviewed WATCH: WFE 0.90 and OOS ExpR 0.207 = 98% of IS 0.210 — "
            "near-perfect OOS match on N=701, p<0.0001. SR alarm on N=35 "
            "monitored trades is expected noise; matches L3 standard."
        ),
        recheck_trigger=(
            "Re-check after N>=100 monitored trades. Retire if SR remains ALARM "
            "and (WFE < 0.50 or OOS/IS ratio < 0.40)."
        ),
    ),
}


def get_sr_alarm_review(profile_id: str, strategy_id: str) -> SrAlarmReview | None:
    """Return the canonical manual review outcome for one SR-alarmed lane."""
    return SR_ALARM_REVIEWS.get((profile_id, strategy_id))
