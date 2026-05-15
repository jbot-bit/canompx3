---
mode: IMPLEMENTATION
task: Add allocator-layer Criterion 8 OOS-status gate (BUG-class fix; mirrors PR #197 chordia gate)
queue_ref: c8_oos_status_allocator_doctrine_2026_05_14
priority: P1
close_before_new_work: true
created: 2026-05-14
verdict: BUG  # confirmed by pre-implementation gate; OVNRNG_25 deployment (95fe44c3, 2026-05-14) postdates c8 write path (6887632f, 2026-04-24) by 3 weeks
scope_lock:
  - trading_app/lane_allocator.py
  - pipeline/check_drift.py
  - tests/test_trading_app/test_lane_allocator.py
  - docs/runtime/decision-ledger.md
  - docs/runtime/action-queue.yaml

## Blast Radius

- trading_app/lane_allocator.py — add `c8_oos_status` to LaneScore; load via deployable_validated_relation SELECT; add `apply_c8_gate()` mirroring `apply_chordia_gate()`; call inline at top of build_allocation() AFTER apply_chordia_gate; surface c8 in save_allocation JSON lane + blocked_entry dicts. Reads: gold.db read-only (already open by allocator). Writes: none new — only the existing lane_allocation.json output gains a `c8_oos_status` field.
- pipeline/check_drift.py — add `check_lane_allocation_c8_gate()` mirroring `check_lane_allocation_chordia_gate()` (#136). Register in CHECKS tuple. Reads: docs/runtime/lane_allocation.json read-only via PROJECT_ROOT. Writes: none.
- tests/test_trading_app/test_lane_allocator.py — extend CREATE TABLE fixture with `c8_oos_status VARCHAR`; extend `_make_score()` defaults with `c8_oos_status="PASSED"`; add `class TestC8Gate` covering PASSED pass-through, FAILED_RATIO/NEGATIVE_OOS_EXPR/NO_OOS_DATA/INSUFFICIENT_N_PATHWAY_B_REJECT/INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH demote-to-PAUSE, None grandfather pass-through, plus a build_allocation parity test verifying both chordia and c8 gates fire.
- docs/runtime/decision-ledger.md — append BUG-verdict entry with the pre-implementation gate evidence and lane impact.
- docs/runtime/action-queue.yaml — close `c8_oos_status_allocator_doctrine_2026_05_14` with override_note; unblock `lane_allocation_rebalance_2026_05_14_pending_capital_review_blockers`.

Callers of build_allocation: only `scripts/tools/rebalance_lanes.py:118`. Calls the canonical entry point with the standard signature — the inline gate call inside build_allocation covers this path. No edits needed to rebalance_lanes.py itself.

Capital impact: OVNRNG_25 (currently DEPLOYED, FAILED_RATIO) and ORB_VOL_8K (proposed ADD, FAILED_RATIO) auto-PAUSE on the next rebalance. Both were already in the user's proposed DROP/RECONSIDER set in capital review.
---

# Stage notes

## Pre-implementation gate (confirmed BUG, not PARTIAL)

```
git log --diff-filter=A -- docs/runtime/lane_allocation.json   → 3f3dfb11 (2026-04-02 first add)
git log --all -S "c8_oos_status" -- trading_app/strategy_validator.py | tail -1
                                                                → 6887632f (2026-04-24 write path landed)
git log --all -S "OVNRNG_25" -- docs/runtime/lane_allocation.json | tail -1
                                                                → 95fe44c3 (2026-05-14 OVNRNG_25 entered the JSON)
```

OVNRNG_25 was deployed 20 days AFTER c8_oos_status was being written into validated_setups. This is not a grandfather case. The lane slipped the C8 gate because the allocator never read the column — a fail-open architectural gap, identical in shape to the pre-PR-#197 chordia situation. BUG verdict locked.

## Canonical sources reused (no re-encoding)

- Demotion shape: `lane_allocator.py:710-736` (`apply_chordia_gate` LaneScore PAUSE construction)
- Inline gate call site: `lane_allocator.py:812-815` (`apply_chordia_gate` then `apply_live_tradeability_gate` at top of `build_allocation`)
- Drift-check shape: `pipeline/check_drift.py:8842-8953` (`check_lane_allocation_chordia_gate` — uses PROJECT_ROOT, skips on missing file, fail-closed on empty `lanes[]`)
- C8 label enum source: `trading_app/strategy_validator.py:1061-1138` — labels are `NO_OOS_DATA`, `INSUFFICIENT_N_PATHWAY_B_REJECT`, `INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH`, `NEGATIVE_OOS_EXPR`, `FAILED_RATIO`, `PASSED`. SKIPPED rows get NULL.
- Phase-4 grandfather: per Criterion 8 §750 + Amendment 3.1 §950, `c8_oos_status IS NULL` on a row written before c8 became a write target (or on a `validation_status='SKIPPED'` row at line 1778) is the canonical grandfather marker — the gate must pass NULL through.

## Doctrine ordering

Apply chordia FIRST, then c8, then live-tradeability. Chordia is the broader "is this strategy still validated under strict replay" gate (Criterion 4); c8 is the narrower OOS Sharpe ratio gate (Criterion 8). Live-tradeability handles E2 deployment-safety filters and runs last. Once one gate demotes a score to PAUSE, downstream gates leave it alone (guard at top of each gate function).

## Done criteria

1. `python pipeline/check_drift.py` returns 0; the new `check_lane_allocation_c8_gate` reports clean (or reports OVNRNG_25 as a violation pre-rebalance, then clears after a fresh rebalance demotes it).
2. `python -m pytest tests/test_trading_app/test_lane_allocator.py -k c8 -v` passes (full suite); plus the parity-test passes.
3. `python scripts/tools/rebalance_lanes.py --output /tmp/rebalance_dryrun.json` confirms OVNRNG_25 and ORB_VOL_8K demoted to PAUSE; the proposal collapses to whatever survives.
4. Decision-ledger + action-queue updated.
