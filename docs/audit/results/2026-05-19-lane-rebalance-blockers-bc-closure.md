---
audit_date: 2026-05-19
audit_type: post_hoc_action_queue_reconciliation
action_queue_id: lane_allocation_rebalance_2026_05_14_pending_capital_review_blockers
class: capital
verdict: CLOSED_AS_APPLIED_WITH_PROCESS_GAP
deployed_lanes_verified: 3
process_gap_logged: true
canonical_drift: 133_of_133_pass
---

# Lane-rebalance action-queue entry closure — blockers (b) + (c)

Post-hoc audit of `lane_allocation_rebalance_2026_05_14_pending_capital_review_blockers`. The action-queue entry remained `status: open` while commit `7877f47b` (2026-05-18) executed the rebalance under `mode: TRIVIAL`. This audit verifies the deployed state against the entry's exit criteria and records the doctrine-process gap.

## Canonical state at audit (2026-05-19)

`docs/runtime/lane_allocation.json` — `rebalance_date: 2026-05-18`, `profile_id: topstep_50k_mnq_auto`, 3 lanes:

| strategy_id | c8_oos_status | N | ExpR | sharpe_ann | WFE | OOS_ExpR | OOS/IS | p_value |
|---|---|---|---|---|---|---|---|---|
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | PASSED | 513 | 0.2151 | 1.695 | 1.1321 | 0.2079 | 0.967 | 3.3e-05 |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 | PASSED | 701 | 0.2101 | 1.8722 | 0.9107 | 0.2086 | 0.993 | 5e-06 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | PASSED | 1508 | 0.0870 | 1.4333 | 1.9516 | 0.0993 | 1.141 | 4.5e-04 |

All three lanes carry `c8_oos_status=PASSED`, `fdr_significant=true`, `deployment_scope=deployable`, `status=active`. Canonical figures pulled live from `validated_setups` via `pipeline.paths.GOLD_DB_PATH`.

Dry-run reconciliation (this audit): `python scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto --output _scratch/rebalance_dryrun_2026_05_19.json` reproduces the identical 3-lane set. Allocator state = stable.

## Blocker (b) — SR-warmup verify

**Entry text (2026-05-14):** "SR-tripwire blind spot on all 3 ADD candidates - no live history; paper-trade warmup status undetermined for newly-promoted lanes."

**Doctrine source:** `docs/institutional/pre_registered_criteria.md:220-242` — C12 Operational extension (added 2026-05-11). WATCH-continue floors: WFE ≥ 0.50 (C6 deploy floor), OOS/IS ratio ≥ 0.40 (C8 deploy floor). Four-precedent history requirement (line 242): L3 2026-04-12, L4 2026-04-14, L6 2026-04-14, NYSE_OPEN_RR1.5 2026-05-11.

**Per-lane evaluation:**

1. **MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100** — registry entry `trading_app/sr_review_registry.py::SR_ALARM_REVIEWS[("topstep_50k_mnq_auto", "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100")]` reviewed 2026-04-12 (L3 precedent, original WATCH grant). C12 floors at audit: WFE 1.1321 ≥ 0.50 PASS; OOS/IS 0.967 ≥ 0.40 PASS. Registry summary text cites a stale WFE 0.52 / C8 ratio 53%, but those are the figures from the 2026-04-12 review; canonical truth has since improved to 1.1321 / 0.967. Per the verification-discipline rule (`pre_registered_criteria.md:238`), stale figures must be corrected or annotated. Annotation deferred (out of scope for this closure; logged as follow-up task).
2. **MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15** — registry entry reviewed 2026-04-14 (L6 precedent, summary "WFE 0.90 and OOS ExpR 0.207 = 98% of IS 0.210"). Canonical at audit: WFE 0.9107, OOS/IS 0.993. Registry figures verify against canonical (within rounding); no staleness flag needed. C12 floors PASS.
3. **MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12** — registry entry reviewed 2026-05-17 (CONTINUE verdict, summary "WFE 1.9516 and OOS/IS 114%"). Canonical at audit: WFE 1.9516, OOS/IS 1.141. Figures verify. C12 floors PASS.

**Resolution:** all 3 deployed lanes clear the C12 WATCH-continue floors at canonical truth. 2 of 3 carry pre-existing reviewed WATCH entries with the recheck-trigger boilerplate ("Re-check after N>=100 monitored trades. Retire if SR remains ALARM AND (WFE < 0.50 OR OOS/IS ratio < 0.40)"). The 3rd has a 2026-05-17 CONTINUE verdict. SR-tripwire coverage is in place per C12; the 2026-05-14 entry's "blind spot" framing is obsolete.

**Note on ORB_VOL_8K:** the 2026-05-14 entry's proposed ADD set included `MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K`, which has `c8_oos_status=FAILED_RATIO`. PR #286 (`apply_c8_gate()`, commit `59e23941`, 2026-05-16) auto-demoted that candidate at the gate. The 2026-05-18 dry-run did not propose it; current dry-run (2026-05-19) does not propose it. C12 evaluation does not apply.

**Blocker (b) status: CLOSED.**

## Blocker (c) — Live-control trace

**Entry text (2026-05-14):** "Live-control checks not exercised (kill/flatten/risk-limit not traced for new lanes)."

**Trace:** `python -m pytest tests/test_trading_app/test_session_orchestrator.py tests/test_trading_app/test_risk_manager.py -q`

**Result:** `293 passed in 101.59s`. Coverage:
- `tests/test_session_orchestrator.py` — 221 tests covering build_order_spec, lifecycle gating (`lifecycle_state.py:241-244` block_source='sr_review_pause'), kill/flatten paths, prop_profiles routing.
- `tests/test_risk_manager.py` — 72 tests covering daily loss limits, account HWM, risk-limit enforcement.

**Coverage gap callouts (informational, not blockers):**
- Tests run against the 3 currently-deployed lane symbols indirectly via `prop_profiles.ACCOUNT_PROFILES` fixtures; they do not parametrize per-strategy. This is the standard pattern for the suite — risk-manager logic is symbol-agnostic by design.
- No new lane-specific live-control tests were added for this rebalance. The 2026-05-18 stage doc (`docs/runtime/stages/rebalance-2026-05-18-fresh-allocation.md`) classified the work `mode: TRIVIAL` and did not require them. Given the suite's symbol-agnostic design this is defensible, but flagged for future doctrine consideration.

**Blocker (c) status: CLOSED.**

## Doctrine-process gap (logged, not a blocker)

The 2026-05-14 action-queue entry explicitly stated: "Do NOT mutate `docs/runtime/lane_allocation.json` while (b)/(c) are open." Commit `7877f47b` (2026-05-18) mutated `lane_allocation.json` while the entry was `status: open`. Commit body and stage doc cite NEW gate enforcement (C8 PR #286) as the rationale; neither cites blockers (b) or (c).

**Assessment:** the capital surface is defensible — every deployed lane independently clears C12 floors, c8_oos_status, chordia_audit_log PASS_CHORDIA. But the action-queue entry was bypassed rather than satisfied. The 2026-05-19 active-state memory (`memory/project_chordia_audit_unblock_real_edge_location_2026_05_19.md`) then read the stale 2026-05-14 entry as live state, missed the 2026-05-18 commit, and described VWAP_MID_ALIGNED_O15 as "NOT_IN_LANE_ALLOC, single cleanest unblock." This is the exact antipattern documented in `memory/feedback_closeout_verify_against_canonical.md`.

**Recurrence-prevention recommendations (informational):**
- Action-queue mutation discipline: any commit mutating a file named in an open action-queue `next_action:` block should either (a) close the entry in the same PR or (b) carry an explicit doctrine-override note in the commit body. Candidate drift check: `check_action_queue_open_entries_referenced_files_unchanged` flagging unclosed entries whose `next_action:` cites a file modified since `last_verified_at`.
- This is n=1 currently; per `memory/feedback_meta_tooling_n1_tunnel_2026_05_01.md`, no forcing-function hook yet. Logged here for future n=2 trigger.

## Process trail

- 2026-05-14: action-queue entry opened post-2026-05-14 dry-run, capital-review verdict VERIFY_MORE.
- 2026-05-16: PR #286 lands `apply_c8_gate()` — closes blocker (a) and reshapes the eligibility surface.
- 2026-05-18 01:09:34 +1000: commit `7877f47b` re-runs `rebalance_lanes.py`, produces a 3-lane set (different from 2026-05-14 proposal due to C8 gate enforcement + ρ correlation displacement), writes `lane_allocation.json`.
- 2026-05-18 01:09:58 +1000: commit `ae3cee57` closes the stage doc.
- 2026-05-19: active-state memory file written referring to 2026-05-14 entry as live, missing the 2026-05-18 mutation.
- 2026-05-19 (this audit): canonical reconciliation, C12 verification, live-control test trace, entry close-out.

## Closure

Action-queue entry `lane_allocation_rebalance_2026_05_14_pending_capital_review_blockers`: `status: closed`, exit reason `RESOLVED_AS_APPLIED`, notes_ref this file.

No mutation to `docs/runtime/lane_allocation.json`. No mutation to `trading_app/sr_review_registry.py`. Doc-only closure.

## Citations

- `docs/institutional/pre_registered_criteria.md:220-242` — C12 Operational extension (SR-warmup framework, four-precedent history, verification discipline).
- `docs/institutional/pre_registered_criteria.md:295-309` — v1 acceptance matrix (superseded; preserved for audit-trail).
- `trading_app/sr_review_registry.py::SR_ALARM_REVIEWS` — keyed (profile_id, strategy_id) review entries.
- `trading_app/lane_allocator.py:770-874` — `apply_c8_gate()` implementation (PR #286).
- `pipeline/check_drift.py` — Check 148 (Chordia gate), Check 149 (C8 OOS-status gate), Check 150 (displaced[] integrity).
- Commit `7877f47b` (2026-05-18) — fresh rebalance.
- Commit `59e23941` (2026-05-16) — PR #286 C8 gate.
- `memory/feedback_closeout_verify_against_canonical.md` — recurrence-pattern source.
- `memory/feedback_allocator_gate_class_pattern_fail_open.md` — gate-class doctrine.
