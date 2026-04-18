# A2b-1 — regime-gate filtered patch — scope (Stage-1)

- phase: A2b-1 / Phase 2 of `docs/plans/2026-04-18-multi-phase-audit-roadmap.md`
- status: **SCOPE — awaiting user approval before Stage-2 implementation**
- author: audit/a2b-1-regime-gate-phase2
- created: 2026-04-18
- revised: 2026-04-18 (post adversarial re-audit; see §10a)
- parent scope: `docs/audit/hypotheses/2026-04-18-a2b-portfolio-optimization-audit-scope.md`
- upstream empirical: `docs/audit/results/2026-04-18-regime-gate-empirical-verification.md` (Phase 2a)
- upstream adversarial: `docs/audit/results/2026-04-18-portfolio-audit-adversarial-reopen.md`

## 1. Problem statement (one paragraph)

The adaptive lane allocator (`trading_app/lane_allocator.py`) classifies each lane as `DEPLOY`/`PAUSE` using a session-level regime gate (`_compute_session_regime`). The gate currently pools all E2/RR1.0/CB1/O5 trades on `(instrument, orb_label)` over a 6-month trailing window **with no lane-filter applied**. Each lane, however, only trades the subset of sessions where its filter fires. The adversarial portfolio re-audit flagged this as a bug: the gate judges the lane's deployment regime against a sample the lane does not actually trade. Phase 2a quantified the impact empirically.

## 2. Empirical grounding (Phase 2a result — CORRECTED)

**Adversarial re-audit note (2026-04-18 later).** The first Phase 2a run
(commit `c9744076`) shipped with a harness bug: the `_load_pool` SELECT
used `EXCLUDE (trading_day, symbol, orb_minutes)` which stripped `symbol`
from the DataFrame, causing `CostRatioFilter.matches_df` to fail-closed
silently (it requires `symbol` to look up per-instrument `COST_SPECS`).
All 7 `FILT_EMPTY` verdicts in that run were artifacts. Fixed in commit
`99d59aa3`; re-run verdict below is the honest one.

From the corrected `docs/audit/results/2026-04-18-regime-gate-empirical-verification.md`, 30 profile-eligible lanes on the 2026-04-18 `topstep_50k_mnq_auto` rebalance:

| code | count | meaning |
|---|---:|---|
| `AGREE_SIGN` | 30 | UNFILT and FILT_POOLED agree on sign |
| `SIGN_FLIP` | 0 | deployment verdict would flip under the patch |
| `FILT_EMPTY` | 0 | lane's filter fires on 0 trades in the 6mo baseline pool |
| `UNFILT_EMPTY` | 0 | |

Corrected verdict: **BUG_COSMETIC on the 2026-04-18 window.** No current
deployment sign flip and no undefined fallback. The patch is strictly
defensive / forward-looking — it adds filter awareness that may matter
on future rebalances if the UNFILT pool and the filter-fired subset
diverge in sign.

`FILT_POOLED` vs `FILT_LANE` magnitudes still diverge materially on 15
of 30 lanes (|diff| ≥ 0.05 R; max `+0.2252` R on MNQ US_DATA_1000 E2/1.5
VWAP_MID_ALIGNED). That divergence is the stronger argument for a
future `FILT_LANE` variant (possible A2b-1b), not for this minimal
`FILT_POOLED` bug-fix patch.

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
) -> float | None:
```

Behavior:
- `filter_key=None` → existing behavior byte-for-byte (baseline pool, no filter).
- `filter_key is not None` and in `ALL_FILTERS` → load the pool as now, JOIN `daily_features` (keeping `symbol` — CostRatioFilter dep per Phase 2a fix commit `99d59aa3`), gate via canonical `research.filter_utils.filter_signal(df, filter_key, orb_label)` and return `AVG(pnl_r)` over the fired subset.
- `filter_key not in ALL_FILTERS` → **log a warning** and return the unfiltered value (fail-open to prior behavior, never silent). Never raise — unknown filters must not crash a rebalance.
- Fired-subset size < `MIN_TRAILING_N` (reuse the existing canonical constant) → return unfiltered value and log `REGIME_FIRED_THIN`.

Reasoning:
- `filter_key=None` default keeps the function signature backward-compatible with every existing call site.
- Corrected Phase 2a shows 0 `FILT_EMPTY` lanes on the 2026-04-18 rebalance, so the thin-fire fallback is defensive scaffolding — it does not change any current deployment verdict. Fail-open (to UNFILT + log) is chosen over fail-closed PAUSE because if the filter ever fires on 0 trades in a future 6mo window the UNFILT pool is the best regime signal available, and silent PAUSE of a lane in DEPLOY status without user review would violate `.claude/rules/integrity-guardian.md` §3 fail-closed-with-evidence.
- Orphan `orb_minutes` kwarg is **NOT** added (`.claude/rules/institutional-rigor.md` Rule 5 — no dead parameters). Pool stays at `orb_minutes = 5` matching the existing function's hardcoded value; if ever generalised that is a separate scope.
- `REGIME_MIN_FIRE_N` reuses the existing canonical `MIN_TRAILING_N = 20` constant (no new knob invented).

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

### 3.3 Variant NOT in scope — AND THE HONEST TRADEOFF

`FILT_LANE` semantics (use lane's own `entry_model` / `rr_target` / `confirm_bars` / `orb_minutes` + filter) are deferred.

**Honest disclosure.** The adversarial audit's actual complaint was that
the regime gate reads the wrong sample for the lane. `FILT_POOLED` only
partially addresses this: 20 of the 30 profile-eligible lanes have at
least one of RR, CB, or `orb_minutes` differing from the baseline pool
(`E2/1.0/1/5`). For those 20 lanes, `FILT_POOLED` is still the wrong
sample, just less wrong. `FILT_LANE` is the sample-accurate fix.

Reason to ship `FILT_POOLED` first anyway:
- 0 SIGN_FLIP on 2026-04-18 under either variant (verified against CSV).
- `FILT_POOLED` is a one-file backward-compatible additive change; `FILT_LANE` changes the regime from "session-level shared across all lanes on that (instrument, session)" to "lane-specific", which has downstream implications for how the gate is explained, cached, and monitored (the SR monitor is lane-specific; the regime gate currently is not).
- Shipping `FILT_POOLED` leaves the regime as a session-level object and simply adds filter awareness within it. `FILT_LANE` is better audited after A2b-2 (DSR ranking) since those two together change what the allocator optimises.

If this Stage-1 scope returns user feedback "go straight to FILT_LANE", the scope doc is revised and Stage-2 work targets FILT_LANE instead.

## 4. Files in scope (scope_lock)

- `trading_app/lane_allocator.py` — the edit above, plus a `REGIME_MIN_FIRE_N = 10` constant near the other window constants.
- `tests/test_trading_app/test_lane_allocator.py` (or new file under same path) — 5 new tests listed in §6.

Nothing else. No changes to `prop_profiles.py`, `validated_shelf.py`, `lane_correlation.py`, `config.py`, or any research script.

## 5. Literature grounding

Honest scope: the literature supports the *framework* (regimes matter, filters matter), not the *specific claim* ("the regime classifier must read the filter-fired subset"). That specific claim is a project-local inference from the adversarial audit + Phase 2a, labelled as such below.

- **Chan (2008) `Algorithmic_Trading_Chan.pdf` ch 7** — regime-switching framework (supports the existence and practical importance of regime gates; does NOT specifically claim the classifier must match the deployed sample).
- **Pepelyshev-Polunchenko (2015)** — Shiryaev-Roberts online drift monitor, already wired per `sr_status` via `trading_app.sr_monitor`. Offline regime gate is a project-local design; pairing it with an SR monitor that reads the filter-fired subset is the analogue, not a directly cited claim.
- **Project-local rationale (NOT a literature claim).** The adversarial portfolio re-audit (`docs/audit/results/2026-04-18-portfolio-audit-adversarial-reopen.md`) flagged the UNFILT regime as a sample-mismatch bug; Phase 2a (this scope's empirical input) quantified materiality. The patch is motivated by internal audit, not external citation.
- **`.claude/rules/institutional-rigor.md` Rule 4** — delegate to canonical sources. Filter application goes through `research.filter_utils.filter_signal` → `ALL_FILTERS[key].matches_df`. No re-encoding.
- **`.claude/rules/integrity-guardian.md` §3 fail-closed with evidence** — unknown-filter and thin-fire paths log and fall back; never crash, never silent.

## 6. Tests required (all pre-registered)

**Test fixture discipline.** Tests that rely on live `gold.db` numeric values are fragile as the DB grows. The patched tests must compute their reference value from a frozen canonical call in the SAME test (compare patched-with-filter vs unpatched-without-filter ROUND-TRIP) rather than hard-coding `+0.1620` etc. This keeps tests green as the DB moves forward and still exercises the patched code path.

T1. `test_regime_gate_backward_compatible`
- Calling `_compute_session_regime(con, inst, orb, date)` with no `filter_key` kwarg returns the same float/None as `filter_key=None`. No hardcoded numeric expectation — the test asserts round-trip equivalence.

T2. `test_regime_gate_filter_applied_changes_or_matches_unfiltered`
- For a fixture lane where the filter fires on a strict subset of the pool (injected fixture data, not live DB), patched `filter_key="TEST_FILTER"` returns the AVG over the filtered subset; `filter_key=None` returns AVG over the full pool; the two values must differ when the filter is a non-trivial subset.

T3. `test_regime_gate_filter_thin_fire_falls_back`
- Fixture with a filter that fires on N < `MIN_TRAILING_N` rows → patched call returns the unfiltered AVG and emits `REGIME_FIRED_THIN` log line. Assert log line present via `caplog`.

T4. `test_regime_gate_unknown_filter_key_falls_back_warns`
- Passing `filter_key="DOES_NOT_EXIST"` returns the UNFILT value and emits a warning. Asserts the function does NOT raise.

T5. `test_compute_lane_scores_passes_filter_type_to_regime`
- Integration: monkeypatch `_compute_session_regime` to record its kwargs, run `compute_lane_scores(fixture_date)`, assert every call receives `filter_key=<lane.filter_type>`. Guards the call-site wiring.

All 5 tests use synthetic fixtures (no live `gold.db` dependency) except T5 which uses a small read-only seeded DB. No OOS consumption.

## 7. Kill criteria (pre-registered — if ANY fires, Stage-2 HALTS)

K1. **Existing test regression.** `pytest tests/test_trading_app/test_lane_allocator.py` must remain green after the patch. Any pre-existing test failing on the patched tree → HALT and revert.

K2. **Drift regression.** `python pipeline/check_drift.py` must not add new failures attributable to this commit. Baseline captured pre-patch; post-patch diff ≤ 0.

K3. **Reproduction break.** `python research/audit_allocator_rho_excluded.py` reproduction (from Phase 1) must still match `lane_allocation.json` exactly (`filter_key=None` path preserves backward compatibility). If reproduction breaks, patch defaults are wrong.

K4. **Reproduction with filter.** A new companion reproduction script (or a unit test) must confirm that passing `filter_key=ft` to the patched `_compute_session_regime` reproduces the Phase 2a FILT_POOLED column for every audited lane to 4 decimals **against a frozen snapshot of the Phase 2a CSV**, not against a live re-query (which drifts as the DB grows). Snapshot lives at `tests/fixtures/regime_gate/2026-04-18-filt-pooled-snapshot.csv`. Deviation → HALT.

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

## 10a. Adversarial re-audit of this scope doc (2026-04-18 self-review)

Findings from re-auditing the first draft of this document. All addressed above; listed here as an honesty trail.

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | CRITICAL | Phase 2a `_load_pool` used `EXCLUDE (trading_day, symbol, orb_minutes)` which silently made `CostRatioFilter` return all-False. 7 `FILT_EMPTY` verdicts were harness artifacts. | Fixed commit `99d59aa3`; this doc's §2 rewritten with corrected counts. |
| 2 | HIGH | First draft argued fail-open fallback was required to avoid flipping 7 COST_LT12 lanes to PAUSE. With the fix those 7 lanes have valid FILT data; the fallback argument collapses. | Rewritten in §3.1 reasoning; fallback kept as defensive scaffolding only. |
| 3 | HIGH | `FILT_POOLED` only fixes 10 of 30 lanes sample-accurately (20 lanes have `RR/CB/orb_minutes` differing from `E2/1.0/1/5`). First draft did not disclose this. | §3.3 rewritten with explicit disclosure. `FILT_LANE` is the sample-accurate variant; deferred to A2b-1b. |
| 4 | MEDIUM | First draft added an orphan `orb_minutes` kwarg as "reserved". That is a Rule 5 dead-parameter violation. | Removed from §3.1 signature. |
| 5 | MEDIUM | First draft invented `REGIME_MIN_FIRE_N = 10` with no justification. | Replaced with the existing canonical `MIN_TRAILING_N = 20`. |
| 6 | MEDIUM | First draft's T3/T4 locked live numeric values from Phase 2a CSV — brittle as DB grows. | §6 rewritten to use synthetic fixtures + round-trip equivalence; K4 uses a frozen snapshot CSV stored under `tests/fixtures/regime_gate/`. |
| 7 | MEDIUM | Literature citations (Chan ch7, Carver ch11-12) supported the framework but were over-applied to specific claim. | §5 rewritten to label project-local inference explicitly. |
| 8 | MEDIUM | Referenced `integrity-guardian.md §6` — that file has no §6. Actual relevant section is §3 (fail-closed with evidence). | Corrected. |

## 10b. Is A2b-1 the right next step at all?

Corrected Phase 2a verdict is BUG_COSMETIC on the 2026-04-18 window. Given the bug-fix-first roadmap ordering was based on the pre-fix assumption that 7 FILT_EMPTY lanes had undefined patch behavior (not true), the case for A2b-1 weakens:

| argument for | weight |
|---|---|
| Defensive / forward-looking — once a filter-fired subset and pool diverge in sign on a future rebalance, the unpatched gate will silently mis-deploy | medium |
| One-file additive change with low implementation cost and clean rollback | medium |
| Aligns the offline regime gate with the online SR monitor's sample | low-medium |

| argument against |  |
|---|---|
| 0 SIGN_FLIP on current rebalance — no immediate EV | |
| `FILT_POOLED` is a half-fix for 20 of 30 lanes; shipping it now commits to a variant that may be superseded by `FILT_LANE` | |
| A2b-2 (DSR ranking) and A2b-3 (Half-Kelly sizing) plausibly have larger EV | |

Honest recommendation: user has two coherent options.
- **Proceed with A2b-1 `FILT_POOLED`** as a defensive patch, accepting it is not a current-EV move.
- **Defer A2b-1** to after A2b-2 / A2b-3 and revisit as `FILT_LANE` at that point.

Either is defensible. Stage-1 will hold until the user selects.

## 11. User approval gate

**Stage-2 (implementation) will not start until the user picks one:**

A. **Proceed with `FILT_POOLED` as scoped above** — defensive patch, BUG_COSMETIC verdict acknowledged.
B. **Defer A2b-1 entirely** — skip ahead to A2b-2 (DSR ranking) per the multi-phase roadmap; revisit later as `FILT_LANE`.
C. **Promote A2b-1 to `FILT_LANE` upfront** — sample-accurate variant; larger blast radius (regime becomes lane-specific not session-specific). Stage-1 would be rewritten before any code.

Stage-1 holds at this scope doc. No code touches `lane_allocator.py` until the user selects A, B, or C (or proposes a different fourth shape).

## 12. Provenance

- Adversarial portfolio re-audit verdict on regime gate → `docs/audit/results/2026-04-18-portfolio-audit-adversarial-reopen.md`
- Empirical Phase 2a quantification → `docs/audit/results/2026-04-18-regime-gate-empirical-verification.md` (this audit's direct input)
- Multi-phase sequencing → `docs/plans/2026-04-18-multi-phase-audit-roadmap.md`
- Scope parent → `docs/audit/hypotheses/2026-04-18-a2b-portfolio-optimization-audit-scope.md`
- Related 2026-04-19 Mode-A revalidation audit → `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md` (parallel finding; does not alter A2b-1 scope but strengthens the case that regime-related statistics across the allocator stack are drift-laden relative to Mode A).
