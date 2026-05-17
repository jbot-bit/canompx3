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
    # NYSE_OPEN RR1.5 COST_LT12 — reviewed 2026-05-11 for the allocation
    # promotion pipeline. The SR alarm is path-real (alarm at monitored trade
    # 43, max SR 552.36), but the lane still clears the same review floors used
    # for the existing WATCH precedents: WFE 1.80 and OOS/IS 112%.
    # Canonical figures verified 2026-05-11 vs validated_setups:
    #   wfe=1.7986, expectancy_r=0.105, oos_exp_r=0.1179,
    #   oos_exp_r/expectancy_r = 0.1179 / 0.105 = 1.1229 (112%).
    # Earlier draft of this entry cited "61%" — that figure was not derivable
    # from any committed source. Correction trail in
    # memory/feedback_sr_review_registry_audit_pattern.md.
    (
        "topstep_50k_mnq_auto",
        "MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
    ): SrAlarmReview(
        profile_id="topstep_50k_mnq_auto",
        strategy_id="MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12",
        outcome="watch",
        reviewed_at="2026-05-11",
        summary=(
            "Reviewed WATCH: WFE 1.80 and OOS/IS 112% (validated_setups "
            "oos_exp_r/expectancy_r = 0.1179/0.105) clear the existing C12 "
            "watch floors; SR alarm is path-real (alarm trade 43, max SR 552.36) "
            "and therefore requires tight recheck rather than silent promotion. "
            "OOS short direction is negative (ExpR=-0.0451 on N=42, OOS power 0.088 "
            "STATISTICALLY_USELESS); pooled OOS sign positive only because OOS long "
            "(ExpR=+0.218 on N=35) outweighs short. Directional risk tracked in "
            "recheck trigger."
        ),
        recheck_trigger=(
            "Re-check after N>=100 monitored trades. Retire if SR remains ALARM "
            "and (WFE < 0.50 or OOS/IS ratio < 0.40), or if OOS short direction "
            "remains negative at N>=30, or if the promoted provisional lane "
            "breaches account/session risk controls."
        ),
    ),
    # COMEX_SETTLE ORB_VOL_2K — reviewed 2026-05-17. Currently SR=ALARM
    # (current_sr_stat=5.2215, alarm_trade=27, recent_10_mean_r=+0.1873 on
    # N=73 monitored trades); without a registry entry, lifecycle_state.py
    # blocks the lane Monday on the default sr_alarm fall-through.
    # Canonical figures verified 2026-05-17 vs validated_setups:
    #   wfe=1.486 (~3x literature 0.50 floor),
    #   oos_exp_r=0.1121, expectancy_r=0.1168, OOS/IS=96% (0.1121/0.1168),
    #   p=0.000135 fdr_significant=True, N=1418.
    # 2026-05-17 regime-stratified audit (docs/audit/results/
    # 2026-05-17-mnq-deployed-lanes-regime-stratified-audit-v1.md) K=2
    # verdict PARK (H1 chi-square p<0.001 fire-rate non-stationarity); raw
    # cells show fire-rate 0.89/1.00/0.996/1.00 across R2/R3/R4/R5 — the
    # H1 rejection is driven by R2 firing 445/500 sessions while later
    # regimes fire >99%. K=8 H1=FAIL k8_p=0.0000 (concurs with K=2 PARK);
    # K=8 H2=PASS k8_p=0.5055; no K=2/K=8 disagreement → conservative-wins
    # rule (audit MD line 12) does not invert the verdict; deploy-floor
    # pass governs per C12 (pre_registered_criteria.md:226-230).
    # R5 ExpR=+0.2494 on N=247 and R6 forward holdout R6_ExpR=+0.0291
    # on N=77 (descriptive only per Amendment 2.7) both remain positive.
    # Lane clears the L3-OVNRNG_100 precedent floors; treat as the same
    # WATCH pattern. Four-precedent history per
    # pre_registered_criteria.md:242: L3 2026-04-12 (COMEX_SETTLE OVNRNG_100),
    # L4 2026-04-14 (NYSE_OPEN ORB_G5), L6 2026-04-14 (US_DATA_1000 RR1.5
    # VWAP_MID_ALIGNED_O15), NYSE_OPEN RR1.5 COST_LT12 2026-05-11.
    (
        "topstep_50k_mnq_auto",
        "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K",
    ): SrAlarmReview(
        profile_id="topstep_50k_mnq_auto",
        strategy_id="MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K",
        outcome="watch",
        reviewed_at="2026-05-17",
        summary=(
            "Reviewed WATCH: WFE 1.486 (~3x literature 0.50 floor) and "
            "OOS/IS 96% (validated_setups oos_exp_r/expectancy_r = 0.1121/0.1168) "
            "clear the C12 watch floors used for L3/OVNRNG_100. "
            "Precedent: L3 2026-04-12, L4/L6 2026-04-14, NYSE_OPEN_RR1.5 "
            "2026-05-11 (pre_registered_criteria.md:242 four-precedent cite). "
            "2026-05-17 regime-stratified audit K=2 PARK is fire-rate-driven "
            "(H1 chi-square p<0.001 on R2 0.89 vs R3/R4/R5 >99%); K=8 H1=FAIL "
            "concurs with K=2 (no conservative-wins inversion). "
            "R5 ExpR is positive (+0.2494 on N=247) and R6 forward holdout "
            "ExpR remains positive (+0.0291 on N=77, descriptive only); R0 "
            "(pre-2020) excluded at runtime via HoldoutContaminationError. "
            "SR alarm is path-real (alarm_trade=27, current_sr=5.22, "
            "recent_10_mean_r=+0.1873) but does not breach deploy floors. "
            "Monitor recommendation: consider 0.5x sizing at next allocator "
            "review pending H1 root-cause."
        ),
        recheck_trigger=(
            "Re-check after N>=100 monitored trades. Retire if SR remains ALARM "
            "and (WFE < 0.50 or OOS/IS ratio < 0.40), or if recent_10_mean_r "
            "turns negative for 2 consecutive checkpoints. Re-run regime-"
            "stratified H1 audit at next monitored-trade milestone to test "
            "whether fire-rate stabilizes above 0.97 across all regimes."
        ),
    ),
    # NYSE_OPEN COST_LT12 RR1.0 — reviewed 2026-05-17. Currently SR=ALARM
    # (current_sr_stat=3.2609, alarm_trade=4, recent_10_mean_r=+0.1775 on
    # N=23 monitored trades); without a registry entry, lifecycle_state.py
    # blocks the lane Monday on the default sr_alarm fall-through.
    # Canonical figures verified 2026-05-17 vs validated_setups:
    #   wfe=1.9516 (~4x literature 0.50 floor),
    #   oos_exp_r=0.0993, expectancy_r=0.087, OOS/IS=114% (0.0993/0.087) -
    #   OOS outperforms IS, no decay signal.
    #   p=0.000447 fdr_significant=True, N=1508.
    # 2026-05-17 regime-stratified audit K=2 verdict CONTINUE (H1 p=0.234,
    # H2 p=0.552 - both regime-stability hypotheses fail to reject at
    # alpha=0.01); K=8 PASS on both H1 and H2 (k8_p=1.000). R5 ExpR=+0.1282
    # on N=257; R6 forward holdout R6_ExpR=+0.0817 on N=80 (descriptive only).
    # OOS-by-direction check 2026-05-17 (raw E2 RR1.0 CB1 NYSE_OPEN cohort,
    # pre-COST_LT12 filter): OOS long ExpR=+0.0935 N=36, OOS short
    # ExpR=+0.0721 N=44, both positive — no directional sign-flip.
    # C12 directional-risk addendum (pre_registered_criteria.md:234) does
    # NOT trigger; differs from RR1.5 sibling at line ~91 which DID trigger.
    # SR alarm at trade 4 is unusually early; treat as expected noise on
    # N=23 monitored. Sibling lane MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5 already
    # carries a WATCH entry (reviewed 2026-04-14) with the same recheck pattern.
    # Four-precedent history per pre_registered_criteria.md:242: L3 2026-04-12,
    # L4/L6 2026-04-14, NYSE_OPEN RR1.5 COST_LT12 2026-05-11.
    (
        "topstep_50k_mnq_auto",
        "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
    ): SrAlarmReview(
        profile_id="topstep_50k_mnq_auto",
        strategy_id="MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
        outcome="watch",
        reviewed_at="2026-05-17",
        summary=(
            "Reviewed WATCH: WFE 1.9516 (~4x literature 0.50 floor) and "
            "OOS/IS 114% (validated_setups oos_exp_r/expectancy_r = 0.0993/0.087) "
            "-- OOS outperforms IS, no decay signal. "
            "Precedent: L3 2026-04-12, L4/L6 2026-04-14, NYSE_OPEN_RR1.5 "
            "2026-05-11 (pre_registered_criteria.md:242 four-precedent cite). "
            "2026-05-17 regime-stratified audit K=2 CONTINUE (H1 p=0.234, "
            "H2 p=0.552); K=8 PASS on both (k8_p=1.000); R5 ExpR=+0.1282 on "
            "N=257 and R6 forward holdout ExpR=+0.0817 on N=80 (descriptive only). "
            "OOS-by-direction check 2026-05-17: OOS long ExpR=+0.0935 N=36, "
            "OOS short ExpR=+0.0721 N=44, both positive -- no directional "
            "sign-flip; C12 directional-risk addendum does NOT trigger. "
            "SR alarm at alarm_trade=4 on N=23 monitored is expected noise "
            "on a well-validated lane; matches L3 precedent."
        ),
        recheck_trigger=(
            "Re-check after N>=100 monitored trades. Retire if SR remains ALARM "
            "and (WFE < 0.50 or OOS/IS ratio < 0.40), or if recent_10_mean_r "
            "turns negative for 2 consecutive checkpoints."
        ),
    ),
    # US_DATA_1000 OVNRNG_25 RR1.0 — reviewed 2026-05-17. Currently SR=ALARM
    # (current_sr_stat=3.7252, alarm_trade=14, recent_10_mean_r=+0.1569 on
    # N=76 monitored trades). Per 2026-05-17 regime-stratified audit
    # K=2/K=8 conservative-wins rule (audit MD line 12: "the more conservative
    # verdict WINS" for any consumer treating per-lane p-values as
    # selection evidence -- registry IS such a consumer):
    #   K=2 H1 raw_p=0.00608 -> PARK at alpha=0.01.
    #   K=8 H1 k8_p=0.04861 -> tier=WATCH (passes alpha=0.05).
    # K=2 PARK is the more conservative verdict -> K=2 wins -> outcome=PAUSE.
    # This DIFFERS from COMEX_SETTLE ORB_VOL_2K (K=2 PARK + K=8 H1=FAIL both
    # concur, no inversion; deploy-floor pass governs per C12) and from
    # NYSE_OPEN COST_LT12 (K=2 CONTINUE + K=8 PASS, no PARK signal).
    # Deploy floors DO pass (WFE=1.6058, OOS/IS=124%, oos_exp_r=0.1222,
    # expectancy_r=0.0986, p=0.00005 fdr_significant=True, N=1512) and
    # recent_10_mean_r is positive -- but conservative-wins rule binds the
    # outcome regardless. Path to WATCH is regime-stratified re-run that
    # resolves the K=2/K=8 disagreement OR explicit doctrine grant.
    # Four-precedent history per pre_registered_criteria.md:242: L3
    # 2026-04-12, L4/L6 2026-04-14, NYSE_OPEN RR1.5 COST_LT12 2026-05-11.
    (
        "topstep_50k_mnq_auto",
        "MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25",
    ): SrAlarmReview(
        profile_id="topstep_50k_mnq_auto",
        strategy_id="MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25",
        outcome="pause",
        reviewed_at="2026-05-17",
        summary=(
            "Reviewed PAUSE per K=2/K=8 conservative-wins rule (audit MD "
            "line 12). 2026-05-17 regime-stratified audit K=2 PARK "
            "(H1 p=0.00608 fire-rate non-stationarity, Yates fallback), "
            "K=8 Bonferroni tier WATCH (k8_p=0.04861) -- verdicts differ "
            "and K=2 PARK is the conservative side, so K=2 governs. "
            "Precedent: L3 2026-04-12, L4/L6 2026-04-14, NYSE_OPEN_RR1.5 "
            "2026-05-11 (pre_registered_criteria.md:242 four-precedent cite). "
            "Deploy floors DO pass: WFE 1.6058 (~3x literature 0.50 floor) "
            "and OOS/IS 124% (validated_setups oos_exp_r/expectancy_r = "
            "0.1222/0.0986); R5 ExpR=+0.0792 on N=257; R6 forward holdout "
            "ExpR=+0.0338 on N=79 (descriptive only); recent_10_mean_r=+0.1569 "
            "positive. Lane is structurally sound; pause is governed by audit "
            "verdict, not deploy-floor failure."
        ),
        recheck_trigger=(
            "Re-evaluate after regime-stratified H1 re-run with current data "
            "resolves K=2/K=8 disagreement (both PASS or both FAIL). Unpause "
            "to WATCH if re-run reports K=2 PASS at alpha=0.01 AND K=8 PASS, "
            "OR if doctrine grant in pre_registered_criteria.md formalizes "
            "deploy-floor-pass override for fire-rate-only K=2 PARK with K=8 "
            "WATCH. While paused: lifecycle_state.py:241-244 blocks live "
            "routing (block_source='sr_review_pause')."
        ),
    ),
}


def get_sr_alarm_review(profile_id: str, strategy_id: str) -> SrAlarmReview | None:
    """Return the canonical manual review outcome for one SR-alarmed lane."""
    return SR_ALARM_REVIEWS.get((profile_id, strategy_id))
