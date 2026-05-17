---
task: Add `displaced[]` provenance bucket to lane_allocation.json so correlation/DD/hysteresis rejections are auditable from JSON alone
mode: IMPLEMENTATION
scope_lock:
  - trading_app/lane_allocator.py
  - scripts/tools/rebalance_lanes.py
  - tests/test_trading_app/test_lane_allocator.py
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift.py
---

## Blast Radius

- `trading_app/lane_allocator.py` — modifies write path. `build_allocation` gets one new optional kwarg `displaced_out: list[dict] | None = None` (default None → silent, preserving current behavior). `save_allocation` gets one new optional positional/kwarg `displaced: list[dict] | None = None` that writes to the new top-level `displaced` key. 9 callers exist (1 production: `rebalance_lanes.py`; 1 generator: `generate_profile_lanes.py`; 1 audit tool: `allocator_gate_audit.py`; 1 backtest: `backtest_allocator.py`; 5 research scripts). Only `rebalance_lanes.py` is modified to use the new mechanism. The other 8 callers pass no new kwarg and observe **zero behavioral change**.
- `scripts/tools/rebalance_lanes.py` — modifies write path. Declares `displaced: list[dict] = []` before `build_allocation`, passes it in, hands it to `save_allocation`.
- `tests/test_trading_app/test_lane_allocator.py` — additive only. 4 new tests covering correlation/dd_budget/hysteresis/missing_cost_spec rejection gates. **Zero edits to existing ~20 tests** — they assert on `result[0].strategy_id`-style which is unchanged.
- `pipeline/check_drift.py` — adds one new check `check_lane_allocation_displaced_bucket`. Reads `docs/runtime/lane_allocation.json` (read-only). ADVISORY when `displaced` key absent (grandfather window for old JSON); FAIL when key present and entries malformed.
- `tests/test_pipeline/test_check_drift.py` — adds 4 injection tests (one per rejection_gate value per `feedback_regex_alternation_sibling_coverage.md`).
- Reads canonical: `docs/runtime/lane_allocation.json` (drift check; read-only).
- Writes: `docs/runtime/lane_allocation.json` (only when rebalance is actually run; this stage does NOT trigger a rebalance commit).

## Why now (motivation)

Today's audit on `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30` needed to re-run the rebalancer in a scratch path just to discover the lane was correlation-displaced. The JSON alone cannot answer "why is this lane absent from `lanes[]`?" — only `paused[]` (status=PAUSE from chordia/c8/live-tradeability hard gates) and `stale[]` (status=STALE) are persisted. Candidates that clear all hard gates and lose at the correlation / DD-budget / hysteresis soft gates are silently `continue`'d in `build_allocation` at lines 1016, 1031, 1048 — they leave no trace in the JSON.

This is the F6 finding from `docs/handoffs/2026-05-17-capital-review-handoff-vwap-o30-audit.md`. Class is **schema-non-self-describing**, severity MEDIUM. Per `institutional-rigor.md` § 6 (no silent failures): each soft-gate `continue` is a silent decision being discarded.

## Design (Option A — mutable out-list, 4 captured gates)

### Signature changes

```python
# trading_app/lane_allocator.py:922
def build_allocation(
    scores: list[LaneScore],
    *,
    max_slots: int = 5,
    max_dd: float = 3000.0,
    allowed_instruments: frozenset[str] | None = None,
    allowed_sessions: frozenset[str] | None = None,
    stop_multiplier: float = 0.75,
    prior_allocation: list[str] | None = None,
    orb_size_stats: dict | None = None,
    correlation_matrix: dict | None = None,
    displaced_out: list[dict] | None = None,  # NEW — append-only diagnostic
) -> list[LaneScore]:
```

```python
# trading_app/lane_allocator.py:1199
def save_allocation(
    scores: list[LaneScore],
    allocation: list[LaneScore],
    rebalance_date: date,
    profile_id: str,
    output_path: str | Path | None = None,
    orb_size_stats: dict | None = None,
    displaced: list[dict] | None = None,  # NEW — written to top-level "displaced" key
) -> Path:
```

### Captured rejection_gate values (4)

| value | trigger site | meaning | dict fields |
|---|---|---|---|
| `correlation` | `build_allocation` line 1015-1016 | rho > RHO_REJECT_THRESHOLD vs already-selected lane | `strategy_id`, `rejection_gate`, `displaced_by` (sid that won the slot), `rho`, `status_at_rejection` |
| `dd_budget` | line 1031-1032 | adding this lane's worst-case DD would exceed profile.max_dd | `strategy_id`, `rejection_gate`, `displaced_by` (None — no specific winner), `lane_dd`, `dd_used_at_rejection`, `max_dd`, `status_at_rejection` |
| `hysteresis` | line 1044-1048 | improvement vs prior in-session lane < HYSTERESIS_PCT | `strategy_id`, `rejection_gate`, `displaced_by` (the prior lane's sid that was rescued), `improvement_pct`, `status_at_rejection` |
| `missing_cost_spec` | line 1020-1021 | `COST_SPECS.get(lane.instrument) is None` — config drift trip-wire | `strategy_id`, `rejection_gate`, `instrument`, `status_at_rejection` |

### NOT captured (and why)

- **`slot_full`** — fires for every ranked candidate past `max_slots`. ~745+ rows of noise per rebalance. Skip.
- **`same_session_legacy`** — only fires when `correlation_matrix=None`, which is unit-tests-only in production. Skip.
- **`apply_*_gate` rejections** (chordia / c8 / live-tradeability) — already captured in `paused[]` via status=PAUSE. Don't double-write.
- **`allowed_instruments` / `allowed_sessions` profile filters** (line 970-972) — operate on candidate set before greedy loop; rejection here means "profile doesn't want this lane at all", not "lost to a better lane". Out of scope.

### New drift check

```python
# pipeline/check_drift.py — new check
def check_lane_allocation_displaced_bucket() -> list[str]:
    """`displaced` key in lane_allocation.json must be present with valid entries.

    Grandfather: if `displaced` key absent, return advisory-only (allocator
    may not have been re-run since the field was added). If present, every
    entry must have a `strategy_id` (str) and `rejection_gate` in the locked
    enum. A `missing_cost_spec` entry is a LOUD FAIL (config drift).
    """
    ALLOWED = {"correlation", "dd_budget", "hysteresis", "missing_cost_spec"}
    ...
```

Drift check description: `"lane_allocation.json displaced[] entries must have valid rejection_gate"`.

### Test plan (4 new lane_allocator tests + 4 new drift-check injection tests)

**`tests/test_trading_app/test_lane_allocator.py`** — append at end:

1. `test_displaced_out_captures_correlation_rejection` — 2 lanes high rho, max_slots=2, assert `displaced_out` has 1 entry with `rejection_gate=correlation, displaced_by=<winner sid>, rho>=0.70`.
2. `test_displaced_out_captures_dd_budget_rejection` — 2 expensive lanes, max_dd low, assert `displaced_out` has 1 entry with `rejection_gate=dd_budget, lane_dd>max_dd-dd_used`.
3. `test_displaced_out_captures_hysteresis_rejection` — `prior_allocation` set, new candidate <20% better, assert `displaced_out` has 1 entry with `rejection_gate=hysteresis, displaced_by=<prior sid>`.
4. `test_displaced_out_captures_missing_cost_spec` — monkeypatch `COST_SPECS` to drop one instrument, assert `displaced_out` has 1 entry with `rejection_gate=missing_cost_spec`.

**`tests/test_pipeline/test_check_drift.py`** — append 4 injection tests:

5. `test_check_lane_allocation_displaced_bucket_advisory_when_absent` — JSON without `displaced` key → check returns empty list (advisory only).
6. `test_check_lane_allocation_displaced_bucket_passes_empty_list` — JSON with `"displaced": []` → check returns empty list.
7. `test_check_lane_allocation_displaced_bucket_rejects_unknown_gate` — JSON with entry `rejection_gate="bogus_value"` → check returns 1 violation.
8. `test_check_lane_allocation_displaced_bucket_rejects_missing_cost_spec` — JSON with entry `rejection_gate="missing_cost_spec"` → check returns 1 violation (LOUD FAIL on trip-wire).

## Acceptance criteria

All four required per `institutional-rigor.md` § 8 ("Done = tests pass + dead code swept + drift check passes + self-review passed"):

1. **Existing tests untouched and green:** `pytest tests/test_trading_app/test_lane_allocator.py -v` shows all pre-existing tests pass with **zero edits to their assertions**. (Required by Option A's design contract.)
2. **New tests green:** 4 new lane_allocator tests pass; 4 new drift-check injection tests pass.
3. **Drift check green:** `python pipeline/check_drift.py` → 133 checks passed, 0 fails (was 132).
4. **Dead-code sweep:** `grep -rn "displaced_out" trading_app/ scripts/ tests/ research/ pipeline/` shows only the intended sites (rebalance_lanes.py + lane_allocator.py + tests). No orphan reads, no other callers picking up the kwarg.
5. **Falsifiable live verification:** re-run the rebalancer to a scratch path:
   ```bash
   python scripts/tools/rebalance_lanes.py --date 2026-05-17 --output "$TEMP/verify.json"
   python -c "import json, os; d=json.load(open(os.path.expandvars(r'%TEMP%\verify.json'))); [print(x) for x in d.get('displaced', []) if 'VWAP_MID_ALIGNED_O30' in x['strategy_id']]"
   ```
   Must print at least one entry for `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30` with `rejection_gate=correlation` and `displaced_by` pointing at `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`.
6. **Self-review pass:** behavioral audit per `institutional-rigor.md` § 1 (run mentally against the 12 sub-rules).

## What this stage does NOT do

- Does NOT mutate `docs/runtime/lane_allocation.json` on disk. A real rebalance commit (the 3-drop-2-add capital-review decision) is a SEPARATE thread per `docs/handoffs/2026-05-17-capital-review-handoff-vwap-o30-audit.md`.
- Does NOT ratchet the drift check from ADVISORY to LOUD-FAIL. That ratchet happens on the next-commit after the first post-fix rebalance has landed `displaced[]` in the canonical JSON (per `feedback_doctrine_supersession_banner_pattern.md` grandfather-then-enforce cadence).
- Does NOT modify the 8 non-rebalance callers of `build_allocation`. They keep silent-skip behavior (zero `displaced_out` passed → no capture).
- Does NOT add `slot_full` or `same_session_legacy` capture (excluded as noise; rationale in design § "NOT captured").

## Risk register

| Risk | Mitigation |
|---|---|
| `displaced_out=[]` mutable default trap | Use `displaced_out: list[dict] \| None = None`, never `= []` |
| Caller passes same list twice → double-append on idempotent re-call | Acceptable; rebalance is single-shot. Document in docstring. |
| Drift check fires on grandfathered old JSON | Advisory-only when key absent; only LOUD-FAIL when malformed entry present |
| `missing_cost_spec` trip-wire never tested in prod | Test #4 monkeypatches `COST_SPECS` to force it |
| Memory blowup from large displaced lists | 4-gate capture excludes `slot_full` (~745 noise rows); production runs typically produce <20 displaced entries |
| Test parity between new tests and old tests | Old tests check `result[...]`; new tests check `displaced_out[...]` separately. No overlap. |
