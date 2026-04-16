# Postmortem: E2 Canonical Window Fix

**Date:** 2026-04-07
**Branch:** `e2-canonical-window-fix`
**Severity:** HIGH (lookahead-bias backtest divergence from live execution)
**Detection:** internal audit during multi-instrument fakeout-honesty review
**Resolution:** 9-stage refactor consolidating ORB window calculation into single canonical source `pipeline.dst.orb_utc_window`
**Bug class:** parallel-models drift (Chan Ch 1 p4 violation)

---

## TL;DR

Three independent implementations of "compute ORB window end UTC" had drifted apart. The backtest path (`trading_app/outcome_builder.py`) silently fell back to `break_ts` when canonical inputs were missing, scanning E2 (stop-market) entries from the close-confirmed break bar instead of from the canonical ORB window close. The live execution path (`trading_app/execution_engine.py`) scanned from the canonical ORB window close. Result: backtest and live engine produced different E2 entries on fakeout days, violating the institutional invariant from Chan, *Algorithmic Trading* (Wiley 2013) Ch 1 p4.

The fix promotes one canonical function `pipeline.dst.orb_utc_window(trading_day, orb_label, orb_minutes)` as the single source of truth, makes the backtest path fail-closed (raise `ValueError` on missing canonical inputs), deletes 536 lines of dead code (`trading_app/nested/builder.py`), and adds 5 structural drift checks + a backtest-live convergence test corpus to prevent regression.

**No production data was rebuilt** — empirical pre-check using `SESSION_CATALOG` ground truth confirmed all 5 deployed lanes (MGC/MNQ/MES) had 0 affected rows. The 4,045 NULL `break_delay_min` rows surfaced by an earlier audit are SIL-only (a dead instrument per CLAUDE.md). Stage 5 is structural hardening + future-proofing, not corrective.

---

## 1. Discovery

### How the bug surfaced

During a fakeout-honesty review of E2 entries across active instruments, an analyst noticed that 35.5% of E2 entries in `orb_outcomes` had `entry_ts` strictly earlier than `break_ts`. That should be impossible: E2 is a stop-market entry that fires on the first bar to TOUCH the ORB high/low after the ORB window closes; by construction `entry_ts` must be at or after `orb_window_end_utc`, and `break_ts` (the first close-confirmed break bar) must be at or after `entry_ts`.

`entry_ts < break_ts` is the *expected* state on fakeout days — when a bar pierces the ORB high intra-bar but closes back inside the ORB, E2 fires on the touch bar but no close-confirmed break occurs until a later bar. The 35.5% rate was therefore consistent with a healthy ratio of fakeout days, not a bug per se.

The red flag came when the same query was run against `validated_setups` and showed that the deployed lane configurations resolved to a *different* set of strategies than the analyst expected. Tracing the discrepancy backwards through the strategy validator and outcome builder revealed two separate issues:

1. **Live execution engine** (`trading_app/execution_engine.py:443-487`, Phase 1.5) correctly scanned for E2 touches from the canonical ORB window close (`orb_utc_window_end_utc`).
2. **Backtest engine** (`trading_app/outcome_builder.py:411` `compute_single_outcome`) had two silent fallback paths:
   - L455: `e2_scan_start = orb_end_utc if orb_end_utc is not None else break_ts`
   - L782: `orb_end_utc = break_ts - timedelta(minutes=break_delay) if break_delay is not None else break_ts`

When either fallback fired, the backtest scanned from the LATER close-confirmed break bar instead of the canonical ORB window close. On fakeout days this excluded the true E2 entry. The backtest and live engine would therefore produce different entries on the same trading day — exactly the look-ahead bias Chan Ch 1 p4 forbids.

### Why the bug had not been caught earlier

- **Synthetic test fixtures** in `tests/test_trading_app/test_outcome_builder.py` passed an explicit `orb_end_utc` value, never triggering the silent fallback.
- **Production builds** (`scripts/tools/build_outcomes_fast.py`) similarly passed an explicit `orb_end_utc` value computed from `daily_features.orb_*_break_ts - daily_features.orb_*_break_delay_min`, which is *itself* derived from a third independent implementation of the same calendar logic in `pipeline/build_daily_features.py:_orb_utc_window`.
- The three implementations agreed on most days but diverged when `daily_features.break_delay_min` was NULL (e.g. days where the ORB window closed but no close-confirmed break ever occurred). On those days the backtest fell back to `break_ts` and produced the wrong scan start.
- No drift check enforced "single source of truth for ORB window end UTC". No convergence test compared backtest entries against live engine entries.

---

## 2. Root cause

**Parallel-models drift.** Three independent implementations of "compute UTC start/end for an ORB window on a given trading day, given the session label and aperture" existed in three different files:

1. `pipeline/build_daily_features.py:_orb_utc_window` — local helper, used to populate `daily_features.orb_*_break_ts` and `break_delay_min`.
2. `trading_app/execution_engine.py:395-411` — inline resolver in `on_trading_day_start`, used to construct `LiveORB` window boundaries for the live engine.
3. `trading_app/outcome_builder.py:782` — derived from `break_delay_min` in `build_outcomes`, used as fallback when `orb_end_utc` was not explicitly passed.

Each implementation was correct in isolation. The drift was at the *interface*: when an `orb_end_utc` value flowed from path #1 → path #3 via the `daily_features.break_delay_min` field, the value was correct on most days but undefined on NULL-delay days. Path #3 silently fell back to `break_ts`, masking the missing data and producing fakeout-blind backtests on those days.

The deeper root cause is the **silent fallback pattern** itself. `if orb_end_utc is not None else break_ts` is functionally equivalent to "guess if the canonical input is missing" — exactly the kind of band-aid forbidden by `.claude/rules/institutional-rigor.md` rule 6 ("No silent failures"). The right behavior is to fail-closed: raise `ValueError` if canonical inputs are missing, forcing the caller to provide them.

Three parallel implementations × two silent fallbacks = inevitable divergence.

---

## 3. The fix

### Design

Pass 1 of the design proposed eight patches across the three parallel implementations. Pass 2 self-critique identified that:

1. The three independent implementations were the root cause, not the symptoms.
2. Patching each silent fallback would leave the duplication intact, allowing the next dev to add a fourth implementation with the same bug.
3. Empirical SESSION_CATALOG ground-truth check showed all 5 deployed lanes had 0 affected rows for active instruments — the 4,045 NULL-delay rows were SIL only (a dead instrument per CLAUDE.md). The fix is structural hardening + future-proofing, not corrective.

The Pass 2 plan consolidated the three implementations into one canonical function in `pipeline/dst.py` and made every consumer call it.

### Implementation (9 stages)

| Stage | Description | Files | Commit |
|---|---|---|---|
| 1 | Promote `orb_utc_window()` and `compute_trading_day_utc_range()` to canonical `pipeline.dst` functions; add 200 deterministic test cases including DST transition coverage | `pipeline/dst.py`, `tests/test_pipeline/test_orb_utc_window.py` | `e8b1537` |
| 2 | Migrate `build_daily_features` to import from `pipeline.dst` (re-export shim for backwards compatibility) | `pipeline/build_daily_features.py` | `e3d5208` |
| 3 | Migrate `execution_engine` to use canonical `orb_utc_window`; replace unverified Pardo Ch.4 citation with verified Chan Ch 1 p4 verbatim quote from `resources/Algorithmic_Trading_Chan.pdf` | `trading_app/execution_engine.py` | `92187bd` |
| 4 | Extract `resample_to_5m` and `_verify_e3_sub_bar_fill` helpers from `trading_app/nested/builder.py` to `trading_app/entry_rules.py` (rescue real helpers before deletion) | `trading_app/entry_rules.py`, `trading_app/nested/audit_outcomes.py` | `d412160` |
| 5 | **THE FIX:** refactor `compute_single_outcome` to fail-closed for E2 entries — accept either explicit `orb_end_utc` OR `(trading_day, orb_label, orb_minutes)` triple; raise `ValueError` if neither is supplied; delete the L455 silent fallback to `break_ts` and the L782 derivation from `break_delay_min` | `trading_app/outcome_builder.py`, `scripts/tools/build_outcomes_fast.py`, `scripts/tools/build_mes_outcomes_fast.py`, `trading_app/nested/audit_outcomes.py` | `8cd3c69` |
| 6 | Regression tests pinning the fail-closed contract: synthetic fakeout day produces entry on the touch bar (not the close-confirmed break bar); E2 without canonical args raises `ValueError`; explicit `orb_end_utc` and canonical-triple paths are equivalent | `tests/test_trading_app/test_outcome_builder.py` | `fd0e871` |
| 7 | Delete `trading_app/nested/builder.py` (536 lines of dead code targeting a `nested_outcomes` table that never existed in `init_db.py`; also embedded a buggy duplicate E2 path) | `trading_app/nested/builder.py` (deleted), `tests/test_trading_app/test_nested/` (import migration) | `a92c299` |
| 8.1 | Add 5 structural drift checks to `pipeline/check_drift.py`: canonical-source uniqueness, no silent break_ts fallback, `compute_single_outcome` canonical kwargs (signature inspection), `nested/builder.py` absent, resample helpers' canonical home | `pipeline/check_drift.py` | `14531a5` |
| 8.2 | 13 paired negative tests for the 5 new drift checks: each injects a controlled violation via `tmp_path` + `monkeypatch` and asserts detection | `tests/test_pipeline/test_check_drift.py` | `14531a5` |
| 8.3 | Backtest-live convergence test on a 30-day fixture corpus (CME_REOPEN + NYSE_OPEN, fakeout/clean_break/no_break scenarios). Asserts backtest E2 `entry_ts` and `entry_price` match live `ExecutionEngine` output exactly. 21 parametrized tests + 2 sanity tests = 23/23 passing | `tests/test_integration/test_backtest_live_convergence.py` (NEW) | `2d8619d` |
| 9 | Documentation hygiene: this postmortem; canonical sources table updates in `.claude/rules/integrity-guardian.md` and `CLAUDE.md`; memory reconciliation | `.claude/rules/integrity-guardian.md`, `CLAUDE.md`, `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`, `~/.claude/projects/.../memory/MEMORY.md`, `~/.claude/projects/.../memory/e2_fakeout_bias.md` | (this commit) |

### Why no production data rebuild

- Empirical pre-check using `SESSION_CATALOG` ground truth (the actual session start times computed from `pipeline.dst.DYNAMIC_ORB_RESOLVERS`, not the derived `daily_features.break_delay_min` field) showed 0 affected rows for the 5 deployed lanes (MGC/MNQ/MES).
- The 4,045 NULL `break_delay_min` rows surfaced by an earlier audit are SIL only — a dead instrument per CLAUDE.md `DEAD_ORB_INSTRUMENTS`.
- Deployed lane fitness (`ExpR`, `sample_size`, `win_rate`, `sharpe_ann`) is byte-identical pre/post the refactor for active instruments because no `orb_outcomes` row was rebuilt.
- A SIL data cleanup is cosmetic and out of scope.

---

## 4. Verification

### Test evidence (post-Stage 7, baseline `a92c299`)

```
Targeted affected-suite tests:
  PYTHONPATH=. pytest \
    tests/test_trading_app/test_outcome_builder.py \
    tests/test_trading_app/test_stress_hardcore.py \
    tests/test_trader_logic.py \
    tests/test_trading_app/test_outcome_builder_utc.py \
    tests/test_trading_app/test_execution_engine.py \
    tests/test_trading_app/test_nested/ \
    tests/test_trading_app/test_early_exits.py \
    tests/test_pipeline/test_orb_utc_window.py \
    tests/test_pipeline/test_build_daily_features.py \
    tests/test_pipeline/test_dst.py
  -> 718 passed, 12 skipped in ~9s
```

```
Wide regression sweep (3,200+ tests):
  3195 passed, 8 failed in 490s
```
(The 8 failures are pre-existing test infrastructure issues — empty worktree DB and missing MES DBN dir — not regressions. Verified by running the same tests on the populated main worktree where they pass.)

### Stage 8 evidence

```
5 new drift checks all pass on current codebase:
  Check 85: Canonical orb_utc_window source ........... PASSED [OK]
  Check 86: No silent break_ts fallback ............... PASSED [OK]
  Check 87: compute_single_outcome canonical kwargs ... PASSED [OK]
  Check 88: nested/builder.py absent .................. PASSED [OK]
  Check 89: resample helpers in entry_rules ........... PASSED [OK]

Total drift state: 1 violation across 81 checks passed [OK]
  (the 1 violation is pre-existing Check 57 for a partial daily_features
   build on 2026-04-06 — a data freshness issue unrelated to this refactor;
   tracked separately, see HANDOFF.md)

13 paired negative tests pass (each injects a violation, asserts detection):
  TestCanonicalOrbUtcWindowSource ............ 3/3 PASSED
  TestNoSilentBreakTsFallback ................ 4/4 PASSED
  TestComputeSingleOutcomeCanonicalKwargs .... 2/2 PASSED
  TestNestedBuilderAbsent .................... 2/2 PASSED
  TestResampleHelpersInEntryRules ............ 2/2 PASSED

Backtest-live convergence test (the load-bearing test):
  21 parametrized days (CME_REOPEN + NYSE_OPEN, fakeout/clean_break/no_break)
  + 2 sanity tests (fakeout actually triggers, no_break actually doesn't)
  = 23/23 PASSED in 0.14s
```

The convergence test is the direct test of Chan Ch 1 p4 for E2: it builds the same bars, runs both `compute_single_outcome(entry_model='E2', ...)` and `ExecutionEngine.on_bar(...)`, and asserts the resulting `entry_ts` and `entry_price` are byte-identical. If this test ever fails, the backtest has drifted from live execution and the bug has returned.

---

## 5. Literature grounding

| Source | Citation | Verification status |
|---|---|---|
| **Chan, *Algorithmic Trading* (Wiley 2013) Ch 1 p4** | *"If your backtesting and live trading programs are one and the same, and the only difference between backtesting versus live trading is what kind of data you are feeding into the program (historical data in the former, and live market data in the latter), then there can be no look-ahead bias in the program."* | **VERIFIED** verbatim from `resources/Algorithmic_Trading_Chan.pdf` PDF p22 |
| Pardo, *Evaluation and Optimization of Trading Strategies* Ch 4 | (was previously cited at `execution_engine.py:446`) | **NOT VERIFIED** — local PDF is 30 pages of front matter only. **Replaced with Chan citation in Stage 3 (commit `92187bd`).** |
| Fitschen, *Building Reliable Trading Systems* (2013) Ch 2 title | *"Developing a Strategy So It Trades Like It Back-Tests"* | Verified as chapter title; chapter body is about curve-fitting, not lookahead bias per se — supporting reference only. |

The Chan citation is the load-bearing literature reference for the entire refactor. It is now embedded as a comment block at `trading_app/execution_engine.py:446-457` and referenced from the Stage 5 fail-closed `ValueError` message in `trading_app/outcome_builder.py:493`.

---

## 6. Pass 1 → Pass 2 corrections

| Pass 1 claim | Pass 2 correction | Source |
|---|---|---|
| "4,045 production rows are poisoned" | All 4,045 are SIL (dead instrument per CLAUDE.md). MGC/MNQ/MES have 0 affected rows. | Per-symbol slice of NULL `break_delay_min` |
| "Data rebuild for active instruments mandatory" | Optional cleanup only — SIL is dead anyway. | Empirical: active instruments are clean against canonical SESSION_CATALOG |
| "Drop `orb_end_utc` parameter from `compute_single_outcome`" | Keep parameter; raise on `None` for E2; accept either explicit value or `(trading_day, orb_label, orb_minutes)` triple. | 77 test call sites use synthetic timestamps not tied to real sessions; dropping the parameter would cascade-break them. |
| "Delete `nested/builder.py` outright" | Extract `resample_to_5m` and `_verify_e3_sub_bar_fill` first (used by `audit_outcomes.py`), then delete. | Helpers are in production use. |
| "Pardo Ch 4 grounds the fix" | NOT VERIFIED — local Pardo PDF is front matter only. Chan Ch 1 p4 is the verified citation. | Local PDF extraction; see `resources/Algorithmic_Trading_Chan.pdf` PDF p22 |

---

## 7. Lessons

### What worked

- **Empirical pre-check before claiming scope.** The Pass 1 claim "4,045 poisoned production rows" turned out to be SIL-only when sliced by symbol. Always slice the affected rows by your highest-cost dimension (symbol, in this case) before estimating remediation cost.
- **Pass 2 self-critique.** Catching the parallel-models root cause required stepping back from the patch list and asking "what are these all symptoms of?" The institutional rigor rule "refactor when you see a pattern of bugs" applies even when you only see two of the same bug — three parallel implementations of one calculation IS the pattern.
- **Convergence test as the load-bearing assertion.** Stage 1's snapshot equivalence proved backtest == backtest. Stage 6's fakeout-honesty test pinned the fail-closed contract. Neither proved backtest == live. The Stage 8.3 convergence test directly proves the Chan Ch 1 p4 invariant for E2 and is the single most important test in the entire refactor.
- **Negative tests forced API hardening.** The Stage 8.2 negative tests for the new drift checks initially failed with `ValueError` from `relative_to(PROJECT_ROOT)` — because monkey-patched scan dirs live outside `PROJECT_ROOT`. The fix (try/except fallback) makes the production code more robust to off-tree scans, not just test-friendly. Tests aren't just verifying behavior — they're stress-testing the API contract.
- **Helper rescue before deletion.** Stage 4 extracted `resample_to_5m` and `_verify_e3_sub_bar_fill` to `entry_rules.py` BEFORE Stage 7 deleted `nested/builder.py`, preventing a hidden import break in `audit_outcomes.py`. Always grep callers before deleting.

### What didn't work

- **Silent fallbacks.** `if X is not None else Y` is "guess if the canonical input is missing." Forbidden by `.claude/rules/institutional-rigor.md` rule 6, but easy to accidentally write because Python idiom encourages it. The fix is fail-closed `raise ValueError` with a long, specific error message that names the expected canonical inputs. Stage 5's error message is intentionally verbose because the next reader needs to understand which inputs they're missing and why a fallback would be wrong.
- **Citing PDFs from training memory.** The Pardo Ch.4 citation at `execution_engine.py:446` was unverified — the local PDF is 30 pages of front matter only. CLAUDE.md's "Local Academic / Project-Source Grounding Rule" forbids this. The fix is to label any citation that hasn't been extract-verified as "from training memory — not verified against local PDF" OR replace it with a verified citation. This refactor replaced the Pardo citation with a verified verbatim Chan quote from the local PDF.
- **Trusting metadata.** The `daily_features.break_delay_min` field looks like it should be the source of truth for "how long after the ORB window did the close-confirmed break occur" — but it's *derived* from another implementation of `_orb_utc_window` in `build_daily_features.py`. Trusting this derived value as a canonical input is exactly the kind of "trust metadata, not the canonical computation" that `.claude/rules/integrity-guardian.md` rule 7 forbids. The fix is to compute from the canonical source (`pipeline.dst.orb_utc_window`) every time, even if it means recomputing.

### Actionable rules added

1. **`pipeline.dst.orb_utc_window` is the single canonical source** for "compute ORB window end UTC". Added to `.claude/rules/integrity-guardian.md` § 2 canonical sources table and `CLAUDE.md` Volatile Data Rule. Drift Check 85 enforces uniqueness.
2. **No silent fallback to `break_ts`** in any new outcome-builder code. Drift Check 86 enforces.
3. **`compute_single_outcome` E2 path requires canonical inputs** — `(trading_day, orb_label, orb_minutes)` triple OR explicit `orb_end_utc`. Drift Check 87 enforces signature.
4. **`trading_app/nested/builder.py` is permanently deleted** — Drift Check 88 enforces.
5. **`resample_to_5m` and `_verify_e3_sub_bar_fill` live in `trading_app/entry_rules.py`** — Drift Check 89 enforces via `__module__` inspection.
6. **Backtest-live convergence is non-negotiable for entry-model code.** Any new entry model must add a parametrized convergence test in `tests/test_integration/test_backtest_live_convergence.py` covering at least one multi-day fixture per session-with-different-DST-behavior.

---

## 8. Related work and follow-ups

### In-scope (this refactor)

- All 9 stages above. Branch `e2-canonical-window-fix` merges as one logical unit.

### Out of scope (separate work, tracked elsewhere)

- **Phase B: live filter staleness in `session_orchestrator._build_daily_features_row`.** Yesterday's `break_delay_min` is used as a proxy for live filters. Different bug class. Track for follow-up after this refactor merges.
- **SIL data rebuild.** SIL is dead per CLAUDE.md `DEAD_ORB_INSTRUMENTS`. The 4,045 NULL `break_delay_min` rows are cosmetic only.
- **Full Option C refactor: backtest drives `execution_engine` bar-by-bar.** Would unify backtest and live engine into a single code path, eliminating the need for the convergence test entirely. Larger scope; the canonical-source approach in this refactor achieves the same correctness guarantee with smaller blast radius.
- **Pre-existing Check 57 (MGC 2026-04-06 partial `daily_features` build).** Data freshness issue: bar ingestion stopped at 2026-04-06 09:59 Brisbane, leaving the trading day partially built. Not a code drift issue. Resolution: wait for ingestion to catch up, OR add a "skip incomplete days" feature to `build_daily_features.py` (separate scope).

---

## 9. Decision log

- **Why one combined commit per stage instead of 17 mechanical commits per the plan:** Each combined commit covers one logical concern (Stage N) and is tightly coupled within itself. Splitting would be cosmetic and risk introducing edit errors. Each commit message documents what's in it and which plan items it addresses.
- **Why extract helpers before deleting `nested/builder.py`:** `audit_outcomes.py` imported them. Deleting first would have broken that import without necessarily failing CI (the file is rarely run directly). Stage 4 + Stage 7 split prevents this class of bug.
- **Why fail-closed `ValueError` instead of a default value or warning:** The previous silent fallback was a default value (fall back to `break_ts`). That's exactly what produced the bug. Warnings are noise and would be filtered. The only remediation that actually prevents the bug is making the missing-input case impossible to reach without explicit caller awareness, and the only mechanism for that in Python is `raise`.
- **Why a deterministic synthetic 30-day corpus instead of replaying real production days:** Real production days are non-reproducible (the bar data is in `gold.db` which the worktree doesn't have populated, and the bars themselves are TIMESTAMPTZ-bound to the current DB state). Synthetic corpus with fixed seeds is reproducible across machines and CI. The fixtures are designed to cover the failure modes (fakeout, clean_break, no_break) rather than approximate real distributions, which is appropriate for a contract test.
- **Why parametrize across both CME_REOPEN and NYSE_OPEN:** These two sessions have materially different DST behavior — CME_REOPEN is dynamic per `SESSION_CATALOG`, NYSE_OPEN bumps the calendar date by one day (the Brisbane 00:30 case handled by `pipeline.dst.orb_utc_window`'s `hour < TRADING_DAY_START_HOUR_LOCAL` branch). Convergence on both gives confidence that the canonical function handles both branches correctly under both DST regimes.
