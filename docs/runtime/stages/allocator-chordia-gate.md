---
task: allocator-chordia-gate
mode: IMPLEMENTATION
phase: 1/1
spec: action-queue.yaml entry `allocator_chordia_gate` (P1, opened 2026-05-01)
created: 2026-05-01
scope_lock:
  - trading_app/lane_allocator.py
  - trading_app/chordia.py
  - docs/runtime/chordia_audit_log.yaml
  - pipeline/check_drift.py
  - tests/test_trading_app/test_lane_allocator.py
  - tests/test_pipeline/test_check_drift.py
  - docs/runtime/stages/allocator-chordia-gate.md
---

# Stage: Allocator Chordia gate — prevent silent rebalance-bypass class

## Why

Chordia revalidation 2026-05-01 found 6 of 7 candidate lanes had never been
audited against Criterion 4 (Chordia 2018 t-stat). `rebalance_lanes.py` +
`lane_allocator.build_allocation()` emit a DEPLOY-ranked JSON with **no**
Chordia awareness today; the only reason live config doesn't already include
FAIL_BOTH lanes is the post-hoc revert-to-HEAD. Hand-editing JSON is a
band-aid; the gate belongs in the allocator. Without this stage, the next
monthly rebalance silently re-promotes unaudited lanes.

## Architectural decision (resolved during blast-radius)

**No new gold.db table.** `validated_setups` already carries `sharpe_ratio`,
`sample_size`, `promoted_at` — everything needed to recompute the Chordia
t-stat inline in `compute_lane_scores`. The only persisted artifact is a
small `docs/runtime/chordia_audit_log.yaml` listing strategy_ids that have
**theoretical support** (Harvey-Liu T=3.00 hurdle); everything else defaults
to `has_theory=False` (Chordia 2018 T=3.79 hurdle — strictest prior).

This collapses stage scope item (1) "audit log table" → a thin doctrine
YAML, and item (2)/(3) populate `LaneScore` from existing data + the gate
function in `trading_app/chordia.py`.

## Blast Radius

(From subagent `a9984a0fd2109eb78`, 2026-05-01.)

- `trading_app/lane_allocator.py`
  - `LaneScore` dataclass: add `chordia_verdict: str | None = None` and
    `chordia_audit_age_days: int | None = None` (None defaults — the
    `_make_score()` factory in tests has 20+ call sites; required fields
    would break every one).
  - `compute_lane_scores()`: query `sharpe_ratio`, `sample_size`,
    `promoted_at` from `validated_setups`; call `chordia_gate(...)` per
    strategy with `has_theory` resolved from doctrine YAML; populate the
    two new fields on every LaneScore.
  - `build_allocation()`: insert refusal logic at top of candidate loop —
    if `chordia_verdict` is `FAIL_BOTH` (or `None` / missing) or
    `chordia_audit_age_days > 90`, mutate to PAUSED with
    `status_reason="chordia gate: <reason>"` and skip selection.
  - `save_allocation()` lane dict: emit `chordia_verdict` and
    `chordia_audit_age_days` so the drift check can read them.
- `trading_app/chordia.py`: add `load_chordia_audit_log(path) -> dict[str, bool]`
  reader (strategy_id → has_theory). Pure function, no DB access.
- `docs/runtime/chordia_audit_log.yaml`: NEW — initial population from the
  2026-05-01 revalidation result doc. Default `has_theory=False`; entries
  with theory cite `docs/institutional/literature/<file>`.
- `pipeline/check_drift.py`: add new check
  `check_lane_allocation_chordia_gate` — appended to `CHECKS` list. Reads
  `docs/runtime/lane_allocation.json`; fails if any lane has
  `chordia_verdict in (None, "FAIL_BOTH")` or
  `chordia_audit_age_days > 90`. Non-advisory, no DB.
- Tests:
  - `tests/test_trading_app/test_lane_allocator.py`: add `_make_score()`
    kwargs for the new fields (default None safe); add 4 tests on the gate
    (FAIL_BOTH → PAUSED, missing → PAUSED, stale 91d → PAUSED, PASS_CHORDIA
    → unchanged).
  - `tests/test_pipeline/test_check_drift.py`: add negative test injecting
    a `lane_allocation.json` with FAIL_BOTH lane → drift check fails.

## Out of scope (do NOT touch this stage)

- `scripts/tools/rebalance_lanes.py` — read-only caller; no edits needed,
  the gate sits inside `build_allocation`.
- `trading_app/prop_profiles.py` — no LaneSpec or daily_lanes change.
- `trading_app/live/session_orchestrator.py` — already reads
  `lane_allocation.json`; if the gate prevents FAIL_BOTH lanes from
  appearing, orchestrator inherits the fix automatically.
- All `research/*.py` callers of `compute_lane_scores` / `build_allocation`
  — the signatures are unchanged; default-None fields keep them working.
- Any new gold.db table or schema migration.

## Acceptance

- `python pipeline/check_drift.py` passes (including new check).
- `python -m pytest tests/test_trading_app/test_lane_allocator.py -v` passes,
  with 4 new gate-tests showing.
- `python -m pytest tests/test_pipeline/test_check_drift.py -v` passes,
  with 1 new negative-injection test for the chordia drift check.
- Dry-run:
  `python scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto
  --dry-run` produces a JSON whose `lanes[]` contains only strategies with
  `chordia_verdict in ("PASS_CHORDIA","PASS_PROTOCOL_A")` and
  `chordia_audit_age_days <= 90`. (Per chordia revalidation 2026-05-01,
  this should be the H3 NYSE_OPEN lane only until further audits land.)
- `LaneScore` field count grows by exactly 2; no existing tests break from
  signature change.
- Self-review walked through: (a) strategy with sharpe_ratio NULL →
  graceful PAUSE, (b) audit_age exactly 90 → PASS, 91 → FAIL, (c)
  has_theory=True boundary at T=3.00 vs 2.99, (d) doctrine YAML missing
  file → fail-closed (treat as has_theory=False), (e) doctrine YAML lists
  unknown strategy_id → ignored, no warning spam.

## Decision refs

- `docs/audit/results/2026-05-01-chordia-revalidation-deployed-lanes.md`
- `docs/runtime/chordia-revalidation-decision-audit-2026-05-01.md`
- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`
- `docs/institutional/pre_registered_criteria.md` (Criterion 4)
