---
pooled_finding: false
date: 2026-05-19
status: READ_ONLY_TRIAGE_RENDER
author: claude (session/joshd-chordia-unblock-vwap-safeguards)
scope: read-only triage of strategies still missing chordia_audit_log.yaml entries as of 2026-05-19
companion_csv: docs/audit/results/2026-05-19-chordia-audit-log-gap-triage.csv
amendments_applied: [E6]
parent_audit: docs/audit/results/2026-05-12-chordia-audit-queue-candidates.md
---

# Chordia audit-log gap — 2026-05-19 triage of canonical AUDIT_GAP_ONLY survivors

## Metadata

- **Date:** 2026-05-19
- **Universe:** `validated_setups WHERE status='active'`; queried live.
- **Anchor:** 2026-05-12 `chordia-audit-queue-candidates.csv` `queue_tier='AUDIT_GAP_ONLY'` rows (locked under v2 methodology Amendments E1-E6).
- **Live impact:** None. This document does not change `lane_allocation.json`, `chordia_audit_log.yaml`, `validated_setups`, or any live state. No pre-reg authorship.
- **Companion CSV:** `docs/audit/results/2026-05-19-chordia-audit-log-gap-triage.csv`

## Verdict token

`READ_ONLY_TRIAGE_RENDER`. Per parent audit Amendment E6 (`memory/feedback_triage_bucket_not_readiness.md`): **AUDIT_GAP_ONLY is not a readiness category, not a deployment recommendation, not a replacement candidate list.** This MD inherits that doctrine. Per-row `recommendation` values are *audit defensibility tiers*, not deployment verdicts.

## Provenance and scope reconciliation

The 2026-05-12 parent audit produced 7 AUDIT_GAP_ONLY rows in the canonical CSV. (The parent MD text says "exactly 8 rows" on line 60; the CSV truth is 7. Difference is one row — likely `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30` which was audited 2026-05-14 verdict PASS_CHORDIA t=4.45 and removed from the candidates set. CSV-vs-MD parity issue flagged here for the parent author's future correction; outside scope of this triage.)

Of those 7:
- **1 audited since 2026-05-12** — not via parent's universe, via the canonical chordia_audit_log:
  - `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30` → PASS_CHORDIA, t=4.45, 2026-05-14.
- **7 still NO_LOG today** — the universe of this triage.

Note on phrasing in the parent triage and in `memory/project_chordia_audit_unblock_real_edge_location_2026_05_19.md`: both reference "12 NO_CHORDIA_AUDIT_LOG_ENTRY candidates" / a 13-candidate CONTROLLED_LIVE_PILOT universe. That broader framing is a different cut of `validated_setups` (C8_PASSED + OOS_ExpR ≥ 0.20 + NOT_IN_LANE_ALLOC) that does not apply the 2026-05-12 v2 methodology's `DEFERRED_FILTER_EXCLUDED` filter-family exclusion. The broader cut includes many `COST_LT*` / `OVNRNG_*` / `ORB_VOL_*` rows that the canonical methodology classifies as `DEFERRED_FILTER_EXCLUDED` and explicitly NOT in the audit-gap bucket. This triage anchors on the canonical 7-row audit-gap subset, not the broader cut. The broader cut's `NO_LOG` rows are not unblock candidates under E6.

## Triage outcome

7 rows, 4 distinct audit-defensibility tiers:

| Recommendation | Count |
|---|---|
| `AUTHOR_K1_AUDIT_DEFENSIBLE` | 2 |
| `AUTHOR_K1_AUDIT_WITH_SIBLING_DISCLOSURE` | 3 |
| `NEEDS_PATHWAY_RESOLUTION_FIRST` | 1 |
| `PARK_FILTER_CLASS_DEPLOYMENT_UNSAFE` | 1 |
| **Total** | **7** |

### `AUTHOR_K1_AUDIT_DEFENSIBLE` (2)

- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P70` — N=575, IS_ExpR=0.1724, WFE=1.376, OOS_ExpR=0.1761, chordia_t_at_2026_05_12=4.582 (clears strict 3.79). Filter family INTRA_ASSET_PERCENTILE, not in DEFERRED_FILTER_EXCLUDED.
- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50` — N=833, IS_ExpR=0.1444, WFE=1.X (canonical), OOS_ExpR=0.1463, chordia_t_at_2026_05_12=4.609 (clears strict). Same filter family.

**Path:** standard K=1 Chordia audit via `research/chordia_strict_unlock_v1.py` with bounded-exact-lane strict-replay protocol matching the existing `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` 2026-05-02 audit shape. No theory grant claimed.

**Caveats:**
- Per `feedback_chordia_oos_park_vs_unverified_power_floor.md`, the runner verdict at low OOS power tier (parent CSV: STATISTICALLY_USELESS for both) needs UNVERIFIED override even on a strict-pass IS. The 2026-05-12 CSV already records `oos_power_tier=STATISTICALLY_USELESS` for both (OOS_N=47, 59); a fresh audit today should expect the same tier unless OOS data has materially expanded.
- Per `feedback_absolute_threshold_scale_audit.md`, both filters use percentile threshold; scale-stability audit not strictly required for INTRA_ASSET_PERCENTILE (relative threshold) but worth a pre-reg checkbox.

### `AUTHOR_K1_AUDIT_WITH_SIBLING_DISCLOSURE` (3)

- `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15_S075` — sibling of deployed `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` (PASS_CHORDIA, t=5.158, live since 2026-05-18).
- `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15_S075` — sibling of `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` (PASS_CHORDIA, t=4.362, displaced 2026-05-18 by RR1.5 ρ=0.852).
- `MNQ_US_DATA_1000_E1_RR1.0_CB3_VWAP_MID_ALIGNED_O15_S075` — E1/CB3 variant in same family.

**Path:** per `pre_registered_criteria.md:236` sibling-pathway disclosure rule, the audit pre-reg must (a) cite the sibling alarm or pre-existing pass as the audit's motivation, (b) limit `K_family=1` to RR/stop-agnostic mechanisms (Chan Ch 7 stop-cascade is RR-agnostic; friction filters are not), and (c) record the selection pathway in the sibling-grant section of the pre-reg YAML. The S075 stop variant changes the stop-multiplier; the mechanism cited must remain stop-agnostic.

**Caveats:**
- The deployed family head (`VWAP_MID_ALIGNED_O15` RR1.5 default-stop) is already PASS_CHORDIA. Per `feedback_chordia_unlock_deployment_gate_audit_checklist.md`, any audit on a sibling variant that materially differs (S075 stop) needs scratch_policy declared at the pre-reg root, and pooled-finding front-matter on the result MD.
- E1/CB3 row is also a different entry_model + confirm_bars combination; sibling-grant K_family=1 must extend across (entry_model, confirm_bars, stop_multiplier) axes coherently.

### `NEEDS_PATHWAY_RESOLUTION_FIRST` (1)

- `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105_S075` — `c8_oos_status=INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH`. Phase-4 grandfather marker.

**Path:** the c8 status is NOT `PASSED`. It's the Phase-4 pass-through marker (`trading_app/lane_allocator.py:301`-comment) for rows where Pathway-A had `n < 30` and the gate is permissive. Authoring a Chordia audit on top of an unresolved c8 status would record PASS_CHORDIA into a row whose deployment eligibility is not actually established — that's the antipattern in `feedback_allocator_gate_class_pattern_fail_open.md`.

**Resolution path:** Pathway-B verification or expanded OOS sample to clear c8 first. Then re-evaluate audit defensibility.

### `PARK_FILTER_CLASS_DEPLOYMENT_UNSAFE` (1)

- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG_O15` — filter type `PD_CLEAR_LONG`.

**Path:** current `lane_allocation.json` `paused[]` carries a `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG_O30` entry with `reason: "live tradeability gate: E2 deployment-unsafe: filter_type 'PD_CLEAR_LONG' selects by close"`. The filter family is currently flagged class-wide as deployment-unsafe (close-selecting → lookahead surface against E2 break-time). A Chordia audit cannot unblock this — even a PASS_CHORDIA verdict would not change the deployment-unsafe flag.

**Resolution path:** PARK or re-audit the `PD_CLEAR_LONG` filter class to determine whether the close-selecting mechanism can be re-specified to avoid the lookahead surface. Outside this triage scope.

## Class-wide pattern notes

- **`_S075` stop variant** carries audit-template-level requirements (sibling-pathway disclosure, K_family discipline) beyond what a vanilla strict-replay audit handles. 3-of-7 candidates carry this.
- **Filter-family deployment safety** is a separate question from Chordia statistical pass/fail. `PD_CLEAR_LONG` shows that a pass on Chordia is insufficient when the filter class itself is flagged.
- **All 7 STILL_NO_LOG carry `oos_power_tier_at_2026_05_12=STATISTICALLY_USELESS`.** Per `feedback_oos_does_not_accrue_holdout_is_frozen.md`, OOS is frozen — calendar-waiting does not improve power; the holdout was already sized at design time. Any fresh audit verdict on these will still be UNVERIFIED at the OOS gate. This does not prohibit authoring — Chordia is the IS-only Mode-A test — but the OOS power tier should be disclosed in every result MD body per `feedback_chordia_oos_park_vs_unverified_power_floor.md`.

## What this triage does NOT do

- It does NOT author any pre-reg.
- It does NOT recommend deployment of any candidate.
- It does NOT mutate `chordia_audit_log.yaml`, `lane_allocation.json`, `validated_setups`, or any live state.
- It does NOT claim that `AUTHOR_K1_AUDIT_DEFENSIBLE` rows would deploy if audited — even a PASS_CHORDIA + OOS-direction-match would still face the C8 gate, the SR-warmup gate, the correlation gate, and the slot-limit gate. Per E6, audit defensibility ≠ readiness.

## Recommendation to operator

If the goal is "grow Chordia-audited inventory" (per `memory/feedback_max_profit_grow_chordia_inventory_not_force_slots.md`), the 2 `AUTHOR_K1_AUDIT_DEFENSIBLE` rows are the cleanest single-author targets. Each requires:

1. A pre-reg YAML under `docs/audit/hypotheses/` declaring `scratch_policy: realized-eod`, citing the INTRA_ASSET_PERCENTILE filter family stability + the canonical session pattern as mechanism grounding (Harris 2002 Ch 4 stop-cascade is plausible for COMEX_SETTLE flow).
2. `research/chordia_strict_unlock_v1.py` run against the lane.
3. Result MD with pooled-finding front-matter declared.
4. `chordia_audit_log.yaml` entry appended with `verdict`, `t_stat`, `threshold=3.79`, `sample_size`, `note` per existing template shape (e.g., `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` 2026-05-02 entry).

That's per-row, not a batch — `feedback_chordia_missing_is_not_backlog.md` is direct: "Chordia MISSING = fail-closed doctrine, not backlog." Do not batch all 7.

## Citations

- `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.md` (parent audit, methodology source).
- `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv` (canonical 7-row AUDIT_GAP_ONLY subset).
- `docs/runtime/chordia_audit_log.yaml` (canonical audit-log truth — 33 entries at audit time).
- `docs/runtime/lane_allocation.json` (canonical deployed state — 3 lanes at audit time).
- `docs/institutional/pre_registered_criteria.md:220-242` (C12 operational extension; SR-warmup framework).
- `docs/institutional/pre_registered_criteria.md:236` (sibling-pathway disclosure rule).
- `trading_app/lane_allocator.py:770-874` (`apply_c8_gate()`).
- `memory/feedback_triage_bucket_not_readiness.md` (E6 doctrine).
- `memory/feedback_chordia_missing_is_not_backlog.md` (do-not-batch doctrine).
- `memory/feedback_chordia_oos_park_vs_unverified_power_floor.md` (OOS power override).
- `memory/feedback_max_profit_grow_chordia_inventory_not_force_slots.md` (inventory-growth doctrine).
- `memory/feedback_oos_does_not_accrue_holdout_is_frozen.md` (OOS frozen at design time).
- `memory/feedback_absolute_threshold_scale_audit.md` (scale-stability for absolute thresholds).
- `memory/feedback_allocator_gate_class_pattern_fail_open.md` (gate-class doctrine for Pathway-A grandfather rows).
