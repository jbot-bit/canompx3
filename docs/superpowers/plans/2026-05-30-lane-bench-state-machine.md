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

- [ ] **Step 1: Write failing tests**

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

- [ ] **Step 2: Run tests to verify red**

Run: `python -m pytest tests/test_tools/test_lane_bench_state.py -q`

Expected: import error because `scripts.tools.lane_bench_state` does not exist.

- [ ] **Step 3: Implement classifier**

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

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_tools/test_lane_bench_state.py -q`

Expected: all tests pass.

### Task 2: CLI and Artifacts

**Files:**
- Modify: `scripts/tools/lane_bench_state.py`
- Test: `tests/test_tools/test_lane_bench_state.py`

- [ ] **Step 1: Add report/artifact tests**

Test that `render_report()` includes `read-only`, `state counts`, and does not claim live mutation.

- [ ] **Step 2: Implement CLI**

CLI flags:

```bash
python scripts/tools/lane_bench_state.py --profile topstep_50k_mnq_auto --rebalance-date 2026-05-30 --inventory-cells <cells.csv> --output-dir artifacts/research/lane_bench_state_2026_05_30
```

The CLI reads `compute_lane_scores()`, applies live-tradeability and C8 gates, classifies rows, writes `manifest.json`, `bench.csv`, and `report.md`.

- [ ] **Step 3: Verify CLI**

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

- [ ] **Step 1: Add compact baton line**

Record the new `lane_bench_state` tool and artifact path. State explicitly that it is a read-only classifier, not deployment authority.

- [ ] **Step 2: Final verification snapshot**

Run `git status --short` and report touched files plus verification commands.
