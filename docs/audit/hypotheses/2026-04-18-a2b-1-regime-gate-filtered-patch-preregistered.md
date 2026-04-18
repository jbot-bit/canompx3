# A2b-1 — regime-gate filtered patch — scope (Stage-1)

- phase: A2b-1 / Phase 2 of `docs/plans/2026-04-18-multi-phase-audit-roadmap.md`
- status: **SCOPE — awaiting user approval before Stage-2 implementation**
- author: audit/a2b-1-regime-gate-phase2
- created: 2026-04-18
- parent scope: `docs/audit/hypotheses/2026-04-18-a2b-portfolio-optimization-audit-scope.md`
- upstream empirical: `docs/audit/results/2026-04-18-regime-gate-empirical-verification.md` (Phase 2a)
- upstream adversarial: `docs/audit/results/2026-04-18-portfolio-audit-adversarial-reopen.md`

## 1. Problem statement (one paragraph)

The adaptive lane allocator (`trading_app/lane_allocator.py`) classifies each lane as `DEPLOY`/`PAUSE` using a session-level regime gate (`_compute_session_regime`). The gate currently pools all E2/RR1.0/CB1/O5 trades on `(instrument, orb_label)` over a 6-month trailing window **with no lane-filter applied**. Each lane, however, only trades the subset of sessions where its filter fires. The adversarial portfolio re-audit flagged this as a bug: the gate judges the lane's deployment regime against a sample the lane does not actually trade. Phase 2a quantified the impact empirically.

## 2. Empirical grounding (Phase 2a result)

From `docs/audit/results/2026-04-18-regime-gate-empirical-verification.md`, 30 profile-eligible lanes on the 2026-04-18 `topstep_50k_mnq_auto` rebalance:

| code | count | meaning |
|---|---:|---|
| `AGREE_SIGN` | 23 | UNFILT and FILT_POOLED agree on sign |
| `SIGN_FLIP` | 0 | deployment verdict would flip under the patch |
| `FILT_EMPTY` | 7 | lane's filter fires on 0 trades in the 6mo baseline pool |
| `UNFILT_EMPTY` | 0 | |

Verdict: **BUG_LATENT**. No current deployment sign flip — the patch is not immediately corrective — but 7 lanes (all `COST_LT12` on MNQ) have undefined patch behavior until a fallback policy is specified. Also, `FILT_POOLED` vs `FILT_LANE` magnitudes diverge materially (e.g., MNQ TOKYO_OPEN E2/2.0 ORB_G5: `+0.0897` pool vs `+0.2701` lane), which is informational for the design choice below.

## 3. Patch specification (minimal, bug-fix-first)

### 3.1 Core change

Extend `trading_app/lane_allocator.py::_compute_session_regime` with optional filter gating:

```python
def _compute_session_regime(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    orb_label: str,
    rebalance_date: date,
    *,
    filter_key: str | None = None,
    orb_minutes: int | None = None,   # reserved; kept = 5 below
) -> float | None:
```

Behavior:
- `filter_key=None` → existing behavior byte-for-byte (baseline pool, no filter).
- `filter_key is not None` and in `ALL_FILTERS` → load the pool as now, then gate via canonical `research.filter_utils.filter_signal(df, filter_key, orb_label)` and return `AVG(pnl_r)` over the fired subset.
- `filter_key not in ALL_FILTERS` → treat as unknown; **log a warning** and return the unfiltered value (fail-open to prior behavior, never silent).
- Fired-subset size < `REGIME_MIN_FIRE_N` (new constant, default `10`) → return unfiltered value and log `REGIME_FIRED_THIN` (fail-open to prior). This is the `FILT_EMPTY` fallback.

Reasoning:
- `filter_key=None` default keeps the function signature backward-compatible with every existing call site.
- Fail-open fallback chosen over fail-closed PAUSE because Phase 2a showed the 7 `FILT_EMPTY` lanes have positive UNFILT regime and are currently DEPLOY'd — fail-closed would flip 7 lanes to PAUSE on a patch that Phase 2a explicitly labelled BUG_LATENT. Fail-open preserves current behavior where the patch is undefined; any future decision to tighten the fallback is a separate scope.

### 3.2 Call-site change in `compute_lane_scores`

The one call site currently passes no filter:

```python
session_regime = _compute_session_regime(con, inst, orb, rebalance_date)
```

Change to:

```python
session_regime = _compute_session_regime(
    con, inst, orb, rebalance_date, filter_key=ft
)
```

where `ft` is the lane's `filter_type` already in scope in that loop.

### 3.3 Variant NOT in scope

`FILT_LANE` semantics (use lane's own `entry_model`/`rr_target`/`confirm_bars`/`orb_minutes` + filter) are deferred. Reason: that is a larger reimplementation (the regime gate stops being session-level, becomes lane-level), and the FILT_POOLED pass is the literal bug fix the adversarial audit flagged. Phase 2a showed FILT_LANE is dramatically different and likely more informative — it is tracked as a follow-up (potential A2b-1b or Phase 2c).

## 4. Files in scope (scope_lock)

- `trading_app/lane_allocator.py` — the edit above, plus a `REGIME_MIN_FIRE_N = 10` constant near the other window constants.
- `tests/test_trading_app/test_lane_allocator.py` (or new file under same path) — 5 new tests listed in §6.

Nothing else. No changes to `prop_profiles.py`, `validated_shelf.py`, `lane_correlation.py`, `config.py`, or any research script.

## 5. Literature grounding

- **Chan (2008) ch 7** — regime-switching framework. The regime classifier must be computed on the sample the strategy trades, not a pooled superset.
- **Pepelyshev-Polunchenko (2015)** — Shiryaev-Roberts monitoring of filtered-lane drift (already wired via `sr_status`). The regime gate is the offline companion and should use the same sample the online monitor uses.
- **Carver (2015) ch 11-12** — forecast-weighting over trailing windows. The trailing-ExpR input must be the filtered ExpR, not the baseline.
- **`.claude/rules/institutional-rigor.md` Rule 4** — delegate to canonical sources. Filter application goes through `research.filter_utils.filter_signal` → `ALL_FILTERS[key].matches_df`. No re-encoding.
- **`.claude/rules/integrity-guardian.md` §6 "Never silent failures"** — unknown filter and thin-fire fallbacks are logged, not swallowed.

## 6. Tests required (all pre-registered)

T1. `test_regime_gate_backward_compatible`
- Calling `_compute_session_regime(con, inst, orb, date)` with no `filter_key` kwarg returns exactly the existing float/None for a known case. Byte-equal to pre-patch reference value captured in the test fixture.

T2. `test_regime_gate_filter_none_equals_default`
- Explicit `filter_key=None` returns the same value as the default-arg call (guards against kwarg-parsing regressions).

T3. `test_regime_gate_filter_applied_changes_result`
- For a lane where Phase 2a showed FILT_POOLED ≠ UNFILT (e.g., MNQ EUROPE_FLOW OVNRNG_100: UNFILT `+0.1534` vs FILT_POOLED `+0.2157`), the patched call with `filter_key="OVNRNG_100"` returns the FILT_POOLED value to 4 decimals.

T4. `test_regime_gate_filter_empty_falls_back_to_unfiltered`
- For a `FILT_EMPTY` lane (e.g., COST_LT12 on MNQ COMEX_SETTLE), the patched call returns the UNFILT value and the test asserts the caller can observe the `REGIME_FIRED_THIN` fallback via captured log.

T5. `test_regime_gate_unknown_filter_key_falls_back_warns`
- Passing `filter_key="DOES_NOT_EXIST"` returns the UNFILT value and emits a warning. Fails if the implementation raises.

## 7. Kill criteria (pre-registered — if ANY fires, Stage-2 HALTS)

K1. **Existing test regression.** `pytest tests/test_trading_app/test_lane_allocator.py` must remain green after the patch. Any pre-existing test failing on the patched tree → HALT and revert.

K2. **Drift regression.** `python pipeline/check_drift.py` must not add new failures attributable to this commit. Baseline captured pre-patch; post-patch diff ≤ 0.

K3. **Reproduction break.** `python research/audit_allocator_rho_excluded.py` reproduction (from Phase 1) must still match `lane_allocation.json` exactly (`filter_key=None` path preserves backward compatibility). If reproduction breaks, patch defaults are wrong.

K4. **Reproduction with filter.** A new companion reproduction script (or a unit test) must confirm that passing `filter_key=ft` to `_compute_session_regime` reproduces the Phase 2a FILT_POOLED column for every audited lane to 4 decimals. Deviation → HALT.

K5. **New-test regression limit.** All 5 tests in §6 must pass first attempt. Any test needing rework for reasons other than obvious typo → HALT.

K6. **Live-lane verdict flip without user approval.** If the patched `compute_lane_scores` output for the 2026-04-18 rebalance flips any lane's status vs the shipped `lane_allocation.json`, Stage-2 must NOT push. Require explicit user approval with the flipped-lane table in-hand before any merge. (Phase 2a says this should not happen — 0 SIGN_FLIP — but K6 guards against implementation bugs.)

## 8. Rollback plan

- Single-file change. If any of K1-K6 fires post-merge: `git revert <commit>` is sufficient. No data migration, no schema change, no downstream rebuild.
- State file (`data/state/sr_state.json`) untouched.
- `lane_allocation.json` untouched unless a scheduled rebalance runs; recompute trivially via `python scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto`.

## 9. Success criteria (for Stage-2 acceptance)

- All 5 new tests pass
- All pre-existing `test_lane_allocator.py` tests pass
- Drift check passes
- K4 reproduction matches Phase 2a FILT_POOLED column to 4 decimals for all 30 lanes
- K6 passes (no lane flips)
- Pushed branch with one commit; PR body references Phase 2a result MD

## 10. Out of scope (explicit — do not expand)

- Changing the ranking objective (A2b-2 DSR — separate phase)
- Sizing changes (A2b-3 Half-Kelly — separate phase)
- FILT_LANE semantics (possible A2b-1b — separate phase)
- Any change to `prop_profiles.py`, `validated_shelf.py`, `config.py`, or any research script outside the tests
- Any change to the 6-month window or baseline pool dimensions — patch only adds optional filter gating to the existing window/pool

## 11. User approval gate

**Stage-2 (implementation) will not start until:**

1. User reviews this scope doc
2. User confirms the minimal FILT_POOLED + fail-open fallback is the right shape (vs FILT_LANE, vs fail-closed PAUSE fallback)
3. User explicitly says `proceed` or equivalent

If the user wants a different shape (e.g., fail-closed, or FILT_LANE semantics upfront), Stage-1 will be revised and re-iterated before any code is written.

## 12. Provenance

- Adversarial portfolio re-audit verdict on regime gate → `docs/audit/results/2026-04-18-portfolio-audit-adversarial-reopen.md`
- Empirical Phase 2a quantification → `docs/audit/results/2026-04-18-regime-gate-empirical-verification.md` (this audit's direct input)
- Multi-phase sequencing → `docs/plans/2026-04-18-multi-phase-audit-roadmap.md`
- Scope parent → `docs/audit/hypotheses/2026-04-18-a2b-portfolio-optimization-audit-scope.md`
- Related 2026-04-19 Mode-A revalidation audit → `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md` (parallel finding; does not alter A2b-1 scope but strengthens the case that regime-related statistics across the allocator stack are drift-laden relative to Mode A).
