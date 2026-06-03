# Lane Bench State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only lane bench state surface that classifies every profile-scoped deployable-shelf row into lifecycle state, primary blocker, and next action.

**Architecture:** Add a small pure classifier and CLI under `scripts/tools/lane_bench_state.py`, reusing the existing allocator score fields and `chordia_unlock_batch` inventory hints. The tool must not mutate live allocation, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`; it only writes stamped artifacts under `artifacts/research/lane_bench_state_<date>/`.

**Tech Stack:** Python stdlib dataclasses/csv/json/argparse, existing `trading_app.lane_allocator` read path, existing `trading_app.prop_profiles` profile/allocation readers, pytest and ruff.

---

### Task 1: Pure Lifecycle Classifier

**Files:**
- Create: `scripts/tools/lane_bench_state.py`
- Test: `tests/test_tools/test_lane_bench_state.py`

- [x] **Step 1: Write failing tests**

```python
from scripts.tools import chordia_unlock_batch as cub
from scripts.tools import lane_bench_state as lbs


def _candidate(strategy_id: str, *, status="DEPLOY", reason="ok", verdict="MISSING", c8="PASSED", annual=10.0, expr=0.05):
    return cub.UnlockCandidate(
        strategy_id=strategy_id,
        instrument="MNQ",
        session="NYSE_OPEN",
        orb_minutes=15,
        entry_model="E2",
        rr_target=1.5,
        filter_type="NO_FILTER",
        confirm_bars=1,
        trailing_expr=expr,
        trailing_n=100,
        annual_r_estimate=annual,
        status=status,
        status_reason=reason,
        chordia_verdict=verdict,
        chordia_audit_age_days=None,
        c8_oos_status=c8,
    )


def test_missing_chordia_positive_lane_is_exact_replay_work():
    rows = lbs.build_bench_rows(
        [_candidate("A")],
        allowed_instruments=frozenset({"MNQ"}),
        allowed_sessions=frozenset({"NYSE_OPEN"}),
        active_strategy_ids=frozenset(),
        family_hints={},
    )
    assert rows[0].state == "EXACT_LANE_READY_FOR_REPLAY"
    assert rows[0].primary_blocker == "MISSING_CHORDIA"
    assert rows[0].next_action == "RUN_STRICT_UNLOCK"


def test_chordia_pass_not_active_is_allocator_eligible_bench():
    rows = lbs.build_bench_rows(
        [_candidate("A", verdict="PASS_CHORDIA")],
        allowed_instruments=frozenset({"MNQ"}),
        allowed_sessions=frozenset({"NYSE_OPEN"}),
        active_strategy_ids=frozenset(),
        family_hints={},
    )
    assert rows[0].state == "ALLOCATOR_ELIGIBLE_BENCH"
    assert rows[0].next_action == "EVALUATE_ALLOCATION_SLOT"


def test_blockers_precede_missing_chordia():
    candidates = [
        _candidate("C8", c8="NEGATIVE_OOS_EXPR"),
        _candidate("UNSAFE", status="PAUSE", reason="live tradeability gate: close-selected"),
        _candidate("NEG", annual=-1.0, expr=-0.01),
    ]
    rows = lbs.build_bench_rows(
        candidates,
        allowed_instruments=frozenset({"MNQ"}),
        allowed_sessions=frozenset({"NYSE_OPEN"}),
        active_strategy_ids=frozenset(),
        family_hints={},
    )
    assert [(r.strategy_id, r.state, r.primary_blocker) for r in rows] == [
        ("C8", "PARKED", "C8_OOS"),
        ("UNSAFE", "PARKED", "LIVE_TRADEABILITY"),
        ("NEG", "PARKED", "NEGATIVE_CURRENT_EDGE"),
    ]
```

- [x] **Step 2: Run tests to verify red**

Run: `python -m pytest tests/test_tools/test_lane_bench_state.py -q`

Expected: import error because `scripts.tools.lane_bench_state` does not exist.

- [x] **Step 3: Implement classifier**

Create dataclass `BenchRow` with fields: rank, strategy_id, state, primary_blocker, next_action, instrument, session, orb_minutes, entry_model, rr_target, filter_type, trailing_expr, trailing_n, annual_r_estimate, chordia_verdict, c8_oos_status, family_priority, active_in_profile.

Implement `build_bench_rows(candidates, allowed_instruments, allowed_sessions, active_strategy_ids, family_hints)` with fail-closed precedence:

1. Active but blocker present -> `LIVE_ACTIVE_REVIEW`.
2. Active and no blocker -> `LIVE_ACTIVE`.
3. Profile block -> `PARKED`.
4. Live-tradeability, C8, stale, cold regime, non-positive current edge -> `PARKED`.
5. `MISSING` Chordia -> `EXACT_LANE_READY_FOR_REPLAY`.
6. `PARK`, `FAIL_CHORDIA`, `FAIL_BOTH` -> `PARKED` or `KILLED`.
7. `PASS_CHORDIA` -> `ALLOCATOR_ELIGIBLE_BENCH`.
8. `PASS_PROTOCOL_A` -> `ALLOCATOR_ELIGIBLE_BENCH` with 1-contract action.

- [x] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_tools/test_lane_bench_state.py -q`

Expected: all tests pass.

### Task 2: CLI and Artifacts

**Files:**
- Modify: `scripts/tools/lane_bench_state.py`
- Test: `tests/test_tools/test_lane_bench_state.py`

- [x] **Step 1: Add report/artifact tests**

Test that `render_report()` includes `read-only`, `state counts`, and does not claim live mutation.

- [x] **Step 2: Implement CLI**

CLI flags:

```bash
python scripts/tools/lane_bench_state.py --profile topstep_50k_mnq_auto --rebalance-date 2026-05-30 --inventory-cells <cells.csv> --output-dir artifacts/research/lane_bench_state_2026_05_30
```

The CLI reads `compute_lane_scores()`, applies live-tradeability and C8 gates, classifies rows, writes `manifest.json`, `bench.csv`, and `report.md`.

- [x] **Step 3: Verify CLI**

Run:

```bash
python -m pytest tests/test_tools/test_lane_bench_state.py -q
ruff check scripts/tools/lane_bench_state.py tests/test_tools/test_lane_bench_state.py
python -m py_compile scripts/tools/lane_bench_state.py tests/test_tools/test_lane_bench_state.py
python scripts/tools/lane_bench_state.py --profile topstep_50k_mnq_auto --rebalance-date 2026-05-30 --inventory-cells C:\Users\joshd\.codex\worktrees\f5ee\canompx3\artifacts\research\orb_edge_inventory_2026_05_30\cells.csv --output-dir artifacts\research\lane_bench_state_2026_05_30
```

Expected: focused tests, ruff, py_compile pass; artifact manifest says no live mutation and reports state counts.

### Task 3: Handoff

**Files:**
- Modify: `HANDOFF.md`

- [x] **Step 1: Add compact baton line**

Record the new `lane_bench_state` tool and artifact path. State explicitly that it is a read-only classifier, not deployment authority.

- [x] **Step 2: Final verification snapshot**

### Task 4: Chordia Evidence Factory Follow-Up

**Files:**
- Create: `scripts/tools/chordia_evidence_factory.py`
- Test: `tests/test_tools/test_chordia_evidence_factory.py`
- Artifact: `artifacts/research/chordia_evidence_factory_2026_05_31/`

- [x] **Step 1: Add failing tests**

Covered default-stop prereg draft readiness, non-default-stop blocking, no-theory strict draft shape, and result-MD parsing into audit-log proposals.

- [x] **Step 2: Implement factory**

Added a proposal-only CLI that consumes `bench.csv`, filters `EXACT_LANE_READY_FOR_REPLAY` + `MISSING_CHORDIA`, writes `prereg_drafts/`, `run_manifest.csv`, `manifest.json`, `report.md`, and `audit_log_proposal.yaml`.

- [x] **Step 3: Preserve strict runner boundary**

`*_S075` and other non-default stop lanes are `BLOCKED_NON_DEFAULT_STOP` by default because `research/chordia_strict_unlock_v1.py` audits default-stop `orb_outcomes` only. The next action is an outcome-builder rebuild plus stop-specific runner, not a fake Chordia replay.

- [x] **Step 4: Generate artifact and verify**

Generated `artifacts/research/chordia_evidence_factory_2026_05_31/` with 12 priority-0 work items: 7 `PREREG_DRAFT_READY`, 5 `BLOCKED_NON_DEFAULT_STOP`, and 0 audit-log proposals because no measured result MDs were supplied.

### Task 5: Full Queue Batch Planner

**Files:**
- Modify: `scripts/tools/chordia_evidence_factory.py`
- Modify: `tests/test_tools/test_chordia_evidence_factory.py`
- Artifact: `artifacts/research/chordia_evidence_factory_full_2026_05_31/`

- [x] **Step 1: Add failing tests**

Covered `limit=0` full-queue semantics, deterministic batch shard partitioning, blocked-row visibility inside shards, command shard emission, and loader-clean generated drafts.

- [x] **Step 2: Implement full-queue planning**

`limit=0` now means uncapped. `plan_batch_shards()` partitions all work items into deterministic `batch_###` shards and `write_factory_artifacts()` emits `batch_summary.csv`, per-batch CSV manifests, and proposal-only PowerShell command shards.

- [x] **Step 3: Fix generated prereg schema**

Generated drafts now include the conditional-role `role` block and hypothesis-level scope/filter fields required by `trading_app.hypothesis_loader`. Verification loaded all 479 generated full-queue draft preregs with zero failures.

- [x] **Step 4: Generate full artifact**

Generated `artifacts/research/chordia_evidence_factory_full_2026_05_31/` from the current bench CSV with all 708 exact replay rows: 479 `PREREG_DRAFT_READY`, 229 `BLOCKED_NON_DEFAULT_STOP`, 29 batches at batch size 25, and no audit-log proposals because no measured result MDs were supplied.

Run `git status --short` and report touched files plus verification commands.

### Task 6: Replay Bridge and First Measured Unlock

**Files:**
- Create: `scripts/tools/chordia_replay_batch_bridge.py`
- Test: `tests/test_tools/test_chordia_replay_batch_bridge.py`
- Modify: `research/chordia_strict_unlock_v1.py`
- Modify: `docs/runtime/chordia_audit_log.yaml`
- Artifacts:
  - `artifacts/research/chordia_replay_batch_bridge_2026_05_31_batch_001/`
  - `artifacts/research/chordia_replay_batch_bridge_2026_05_31_batch_001_run5/`
  - `artifacts/research/lane_bench_state_2026_05_31_after_chordia_apply/`
  - `artifacts/research/chordia_evidence_factory_after_apply_2026_05_31_p5/`

- [x] **Step 1: Add bridge tests**

Covered batch acceptance planning, skip behavior for factory-blocked rows, active prereg activation with `execution_gate.allowed_now=true`, no audit/live mutation in bridge artifacts, fail-closed replay execution when the active prereg file is missing, strict-runner loadability of factory drafts, and direct script-entrypoint importability.

- [x] **Step 2: Implement replay bridge**

Added `scripts/tools/chordia_replay_batch_bridge.py`. It consumes a factory `batch_###.csv`, writes an `activation_plan.csv`, can copy reviewed draft preregs into `docs/audit/hypotheses/`, can run the existing strict replay runner, and converts measured result markdown into proposal-only `audit_log_proposal.yaml`. The bridge does not append `docs/runtime/chordia_audit_log.yaml`, mutate live allocation, or write `validated_setups`.

- [x] **Step 3: Fix runner integration blockers**

`research/chordia_strict_unlock_v1.py` now inserts the repo root on `sys.path` so the documented direct invocation works:

```bash
python research/chordia_strict_unlock_v1.py --hypothesis-file <path>
```

The runner also explicitly accepts `metadata.template_version: chordia_strict_v1` as the heavyweight exact-lane template emitted by the evidence factory. Unknown template versions remain refused.

- [x] **Step 4: Execute one bounded replay**

Activated batch-001 ready drafts and ran one strict replay for:

```text
MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15
```

Measured result:

```text
PASS_CHORDIA
IS N=1545
IS ExpR=+0.0934
IS t=3.936 vs strict no-theory threshold 3.79
OOS N=86
OOS ExpR=+0.0895
```

Result artifacts:

```text
docs/audit/hypotheses/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-no-filter-o15-chordia-unlock-v1.yaml
docs/audit/results/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-no-filter-o15-chordia-unlock-v1.md
docs/audit/results/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-no-filter-o15-chordia-unlock-v1.csv
docs/audit/results/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-no-filter-o15-chordia-unlock-v1.summary.csv
```

- [x] **Step 5: Apply reviewed audit-log unlock**

Appended the measured `PASS_CHORDIA` row to `docs/runtime/chordia_audit_log.yaml` for the exact lane above. This removes the `MISSING_CHORDIA` blocker only; it does not allocate the lane live.

- [x] **Step 6: Refresh bench/factory truth**

Refreshed read-only bench state after the audit-log append:

```text
artifacts/research/lane_bench_state_2026_05_31_after_chordia_apply/
ALLOCATOR_ELIGIBLE_BENCH: 15
EXACT_LANE_READY_FOR_REPLAY: 707
KILLED: 12
LIVE_ACTIVE: 3
PARKED: 111
```

The measured lane moved to:

```text
ALLOCATOR_ELIGIBLE_BENCH / NONE / EVALUATE_ALLOCATION_SLOT
```

Regenerated the full queue from the refreshed bench with broad priority fallback:

```text
artifacts/research/chordia_evidence_factory_after_apply_2026_05_31_p5/
707 work items
478 PREREG_DRAFT_READY
229 BLOCKED_NON_DEFAULT_STOP
29 batches
```

### Task 7: Priority-0 Batch Carry-Forward and Apply

**Files:**
- Modify: `scripts/tools/lane_bench_state.py`
- Modify: `tests/test_tools/test_lane_bench_state.py`
- Modify: `scripts/tools/chordia_evidence_factory.py`
- Modify: `tests/test_tools/test_chordia_evidence_factory.py`
- Modify: `docs/runtime/chordia_audit_log.yaml`
- Modify: `docs/runtime/fast_lane_status.yaml`
- Artifacts:
  - `artifacts/research/chordia_replay_batch_bridge_2026_05_31_p0_batch_001_run_all/`
  - `artifacts/research/lane_bench_state_2026_05_31_after_p0_batch_apply/`
  - `artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p0/`
  - `artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/`

- [x] **Step 1: Preserve family priority through refreshed benches**

Added `--family-hints-csv` to `lane_bench_state.py`. This lets the state machine carry strict-inventory `family_priority` hints forward from a prior `bench.csv` when the original ORB inventory cells artifact is not present in the current worktree. This prevents p0 queues from silently collapsing into broad priority-5 work after the first audit-log apply.

- [x] **Step 2: Keep runner/audit-log verdict vocabulary canonical**

Added verdict mapping in `chordia_evidence_factory.py` so strict runner output `FAIL_STRICT_CHORDIA` becomes audit-log `FAIL_CHORDIA`. Runner detail remains in the result markdown; runtime state uses the canonical verdict token.

- [x] **Step 3: Run and apply the remaining default-stop priority-0 batch**

Ran 8 strict replays through the bridge artifact:

```text
artifacts/research/chordia_replay_batch_bridge_2026_05_31_p0_batch_001_run_all/
```

Applied reviewed audit-log rows:

```text
PASS_CHORDIA:
- MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O30
- MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER

FAIL_CHORDIA:
- MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O15
- MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER
- MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER
- MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER_O15
- MNQ_NYSE_OPEN_E2_RR1.5_CB1_NO_FILTER
- MNQ_NYSE_OPEN_E2_RR2.0_CB1_NO_FILTER
```

- [x] **Step 4: Refresh current bench and queue**

Latest refreshed state:

```text
artifacts/research/lane_bench_state_2026_05_31_after_p0_batch_apply/
ALLOCATOR_ELIGIBLE_BENCH: 17
EXACT_LANE_READY_FOR_REPLAY: 699
KILLED: 18
LIVE_ACTIVE: 3
PARKED: 111
```

Priority-0 replay queue is now only non-default-stop blocked:

```text
artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p0/
5 work items
5 BLOCKED_NON_DEFAULT_STOP
```

Broad queue remains available for further batch replay:

```text
artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/
699 work items
470 PREREG_DRAFT_READY
229 BLOCKED_NON_DEFAULT_STOP
28 batches
```

- [x] **Step 5: Final verification**

Ran focused tests, lint, py_compile, and drift after the status refresh:

```text
114 passed
ruff: All checks passed
py_compile: passed
pipeline/check_drift.py: NO DRIFT DETECTED: 170 checks passed [OK], 0 skipped, 21 advisory
```

### Task 8: Reviewed Audit-Log Applicator

**Files:**
- Create: `scripts/tools/chordia_audit_log_apply.py`
- Test: `tests/test_tools/test_chordia_audit_log_apply.py`

- [x] **Step 1: Add the missing reviewed-apply stage**

The bridge stays proposal-only by design. Added a separate applicator for the final reviewed step so future batches do not require hand-editing `docs/runtime/chordia_audit_log.yaml`.

Safety behavior:

```text
--write requires --reviewed
proposal_only must be true
verdicts must use canonical audit-log tokens
existing strategy_ids are skipped idempotently
live_mutation=false
validated_setups_mutation=false
```

- [x] **Step 2: Verify against the already-applied p0 proposal**

Dry-run:

```text
scripts/tools/chordia_audit_log_apply.py --proposal artifacts/research/chordia_replay_batch_bridge_2026_05_31_p0_batch_001_run_all/audit_log_proposal.yaml --format json
applied_count: 0
skipped_existing_count: 8
```

This confirms the manual p0 apply is represented in `docs/runtime/chordia_audit_log.yaml` and the applicator is idempotent.

### Task 9: Broad Batch-Cycle Automation and Taxonomy Repair

**Files:**
- Create: `scripts/tools/chordia_batch_cycle.py`
- Test: `tests/test_tools/test_chordia_batch_cycle.py`
- Modify: `scripts/tools/chordia_evidence_factory.py`
- Modify: `tests/test_tools/test_chordia_evidence_factory.py`
- Modify: `docs/runtime/chordia_audit_log.yaml`
- Modify: `docs/runtime/fast_lane_status.yaml`
- Artifacts:
  - `artifacts/research/chordia_batch_cycle_2026_05_31_broad_batch_001/`
  - `artifacts/research/chordia_batch_cycle_2026_05_31_broad_batch_001b/`
  - `artifacts/research/lane_bench_state_2026_05_31_after_broad_batch_001b_apply/`
  - `artifacts/research/chordia_evidence_factory_after_broad_batch_001b_apply_2026_05_31_p5/`

- [x] **Step 1: Add a repeatable batch-cycle wrapper**

Added `scripts/tools/chordia_batch_cycle.py` as the one-command operator loop
over the smaller Chordia tools:

```text
factory batch -> activate drafts -> strict replay -> proposal artifacts
-> reviewed audit-log apply -> refresh status/bench/factory
```

The wrapper reports:

```text
live_mutation=false
validated_setups_mutation=false
```

It is still an evidence factory and readiness state-machine tool. It is not a
live allocator and does not promote strategies.

- [x] **Step 2: Run two broad default-stop batches**

After the p0 queue was exhausted except non-default-stop lanes, ran two p5
broad batches:

```text
artifacts/research/chordia_batch_cycle_2026_05_31_broad_batch_001/
15 strict replays
15 returncode 0
15 strict failures

artifacts/research/chordia_batch_cycle_2026_05_31_broad_batch_001b/
11 strict replays
11 returncode 0
11 strict failures
```

All 26 measured rows were reviewed/applied. No new bench lanes were unlocked.
The queue reduction came from turning unknown rows into measured failures.

- [x] **Step 3: Repair audit verdict taxonomy**

Found a parser/mapping defect while broad replays were being applied:
`FAIL_STRICT_CHORDIA` rows with measured `t_stat < 3.0` were being collapsed
into `FAIL_CHORDIA`. That was too coarse.

Correct taxonomy:

```text
runner FAIL_STRICT_CHORDIA + t_stat < 3.0       -> audit FAIL_BOTH
runner FAIL_STRICT_CHORDIA + 3.0 <= t_stat < 3.79 -> audit FAIL_CHORDIA
```

Patched the factory parser to read explicit measured-result fields such as:

```text
**IS t-stat:** `2.965`
**IS sample size:** `1188`
**IS ExpR:** `0.0500`
```

Repaired:

```text
docs/runtime/chordia_audit_log.yaml
artifacts/research/chordia_batch_cycle_2026_05_31_broad_batch_001/audit_log_proposal.yaml
artifacts/research/chordia_batch_cycle_2026_05_31_broad_batch_001b/audit_log_proposal.yaml
```

Post-repair proposal verdict mix:

```text
broad_batch_001:  11 FAIL_BOTH, 4 FAIL_CHORDIA
broad_batch_001b:  7 FAIL_BOTH, 4 FAIL_CHORDIA
```

- [x] **Step 4: Refresh current state**

Latest bench:

```text
artifacts/research/lane_bench_state_2026_05_31_after_broad_batch_001b_apply/
ALLOCATOR_ELIGIBLE_BENCH: 17
EXACT_LANE_READY_FOR_REPLAY: 673
KILLED: 44
LIVE_ACTIVE: 3
PARKED: 111
```

Latest broad factory:

```text
artifacts/research/chordia_evidence_factory_after_broad_batch_001b_apply_2026_05_31_p5/
673 work items
444 PREREG_DRAFT_READY
229 BLOCKED_NON_DEFAULT_STOP
27 batches
```

Current honest blocker:

```text
MISSING_CHORDIA remains the main throughput blocker: 673 rows.
Non-default-stop support remains the main automation coverage blocker: 229 rows.
The live allocator gate itself was not loosened.
```

- [x] **Step 5: Verification**

Reran focused verification after the taxonomy repair:

```text
118 passed
ruff: All checks passed
py_compile: passed
pipeline/check_drift.py: NO DRIFT DETECTED: 170 checks passed [OK], 0 skipped, 21 advisory
```

The first drift run hit the 3-minute command timeout; rerun with a longer
timeout completed green.

### Task 10: Supported Non-Default Stop Replay

**Files:**
- Modify: `research/chordia_strict_unlock_v1.py`
- Modify: `scripts/tools/chordia_evidence_factory.py`
- Modify: `tests/test_research/test_chordia_strict_unlock_v1_emissions.py`
- Modify: `tests/test_tools/test_chordia_evidence_factory.py`
- Modify: `tests/test_tools/test_chordia_replay_batch_bridge.py`
- Modify: `docs/runtime/chordia_audit_log.yaml`
- Modify: `docs/runtime/fast_lane_status.yaml`
- Artifacts:
  - `artifacts/research/chordia_evidence_factory_after_stop_support_2026_05_31_p5/`
  - `artifacts/research/chordia_batch_cycle_2026_05_31_stop_support_batch_001/`
  - `artifacts/research/chordia_batch_cycle_2026_05_31_stop_support_batch_002/`
  - `artifacts/research/chordia_batch_cycle_2026_05_31_stop_support_batch_003/`
  - `artifacts/research/lane_bench_state_2026_05_31_after_stop_support_batch_003_apply/`
  - `artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/`

- [x] **Step 1: Replace the stop-support assumption with code-backed proof**

Previous state treated every non-default-stop row as requiring an outcome-builder
rebuild and a separate runner. That was too broad: this repo already has one
canonical tight-stop transform, `trading_app.config.apply_tight_stop()`, used by
discovery, validation, allocation, and account-survival paths.

The strict runner now supports only stop multipliers already declared in
`trading_app.config.STOP_MULTIPLIERS`. Supported non-default stops are replayed
by applying the canonical MAE/friction transform to the default-stop
`orb_outcomes` rows before statistics are computed. Unsupported stop levels
still fail closed.

- [x] **Step 2: Preserve stop metadata through prereg drafts**

The evidence factory now emits:

```text
scope.stop_multiplier
hypotheses[0].scope.stop_multipliers
```

for each exact lane. `_S075` lanes now produce `PREREG_DRAFT_READY` rows by
default instead of `BLOCKED_NON_DEFAULT_STOP`.

Regenerated factory:

```text
artifacts/research/chordia_evidence_factory_after_stop_support_2026_05_31_p5/
673 work items
673 PREREG_DRAFT_READY
0 blocked stop rows
27 batches
```

- [x] **Step 3: Run three reviewed batches after stop support**

Batch 001:

```text
artifacts/research/chordia_batch_cycle_2026_05_31_stop_support_batch_001/
25 activated, 25 run, 25 applied
4 PASS_CHORDIA
12 FAIL_BOTH
7 FAIL_CHORDIA
2 PARK
```

Batch 002:

```text
artifacts/research/chordia_batch_cycle_2026_05_31_stop_support_batch_002/
25 activated, 25 run, 25 applied
13 FAIL_CHORDIA
12 FAIL_BOTH
```

Batch 003:

```text
artifacts/research/chordia_batch_cycle_2026_05_31_stop_support_batch_003/
25 activated, 25 run, 25 applied
3 PASS_CHORDIA
14 FAIL_BOTH
8 FAIL_CHORDIA
```

All runs returned exit code 0. No live allocation or `validated_setups` mutation.

- [x] **Step 4: Refresh latest state**

Latest bench:

```text
artifacts/research/lane_bench_state_2026_05_31_after_stop_support_batch_003_apply/
ALLOCATOR_ELIGIBLE_BENCH: 24
EXACT_LANE_READY_FOR_REPLAY: 598
KILLED: 110
LIVE_ACTIVE: 3
PARKED: 113
```

Latest factory:

```text
artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/
598 work items
598 PREREG_DRAFT_READY
24 batches
```

Current honest blocker:

```text
The stop-support blocker is removed for supported S075 lanes.
The remaining blocker is throughput: 598 exact lanes still need measured strict replay.
Allocator-eligible bench is 24, not 100+.
```

- [x] **Step 5: Verification**

```text
144 passed
ruff: All checks passed
py_compile: passed
audit-log taxonomy check: 0 FAIL_CHORDIA rows with t_stat < 3.0
pipeline/check_drift.py: NO DRIFT DETECTED: 170 checks passed [OK], 0 skipped, 21 advisory
```

### Task 11: Queue Drain and Review Hardening

**Files:**
- Add: `scripts/tools/chordia_batch_queue_runner.py`
- Modify: `scripts/tools/chordia_audit_log_apply.py`
- Modify: `scripts/tools/chordia_batch_cycle.py`
- Modify: `scripts/tools/chordia_evidence_factory.py`
- Modify: `pipeline/check_drift.py`
- Modify: `tests/test_tools/test_chordia_audit_log_apply.py`
- Modify: `tests/test_tools/test_chordia_batch_cycle.py`
- Modify: `tests/test_tools/test_chordia_batch_queue_runner.py`
- Modify: `tests/test_research/test_cherry_pick_journal.py`
- Runtime surfaces: `docs/runtime/chordia_audit_log.yaml`, `docs/runtime/fast_lane_status.yaml`,
  `docs/runtime/promote_queue.yaml`

- [x] **Step 1: Automate reviewed throughput without live deployment**

Added a queue runner that repeatedly runs the reviewed batch-cycle flow against
the refreshed factory `batch_001`. The pipeline remains bounded:

```text
factory queue -> active prereg copy -> strict replay -> proposal-only bridge
-> reviewed audit-log apply -> derived bench/factory refresh
```

It reports `live_mutation=false` and `validated_setups_mutation=false`; it does
not touch allocation.

- [x] **Step 2: Code-review fixes**

Fixed two review findings:

```text
audit applicator: reject duplicate strategy_id rows inside a proposal file
batch cycle: refuse reviewed apply if any strict replay in the batch exits non-zero
```

Also corrected a derived-state drift check: the documented doctrine says
`REVOKED`, `PARKED`, and plain queued promote rows are exempt from cherry-pick
journal matching. The implementation now only requires journal entries for
`ESCALATED` promote-queue rows with heavyweight preregs.

- [x] **Step 3: Drain the strict replay queue**

Final state:

```text
artifacts/research/lane_bench_state_2026_05_31_stop_support_queue_drain_no_status_b7_001_apply/
ALLOCATOR_ELIGIBLE_BENCH: 165
LIVE_ACTIVE: 3
KILLED: 529
PARKED: 151
MISSING_CHORDIA: 0

artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_no_status_b7_001_apply_p5/
work_item_count: 0
batch_count: 0
```

Current audit-log verdict counts:

```text
PASS_CHORDIA: 165
PASS_PROTOCOL_A: 3
PARK: 46
FAIL_CHORDIA: 257
FAIL_BOTH: 274
```

The honest result is 165 allocator-eligible bench lanes plus 3 currently live
active lanes. This is not an automatic promotion to 168 live-active lanes.

- [x] **Step 4: Verification**

```text
151 focused Chordia/lane tests passed
47 cherry-pick/fast-lane-status tests passed
ruff: All checks passed
py_compile: passed
audit_behavioral.py: passed
audit_integrity.py: passed
pipeline/check_drift.py: NO DRIFT DETECTED: 170 checks passed [OK], 0 skipped, 21 advisory
```
