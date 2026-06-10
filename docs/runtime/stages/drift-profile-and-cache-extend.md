---
task: Reduce the ~3-min drift gate. PIVOTED after profiling (2026-06-09) — the profile showed file-deps cache can only harvest ~15-25s of 202s; real cost is DB-bound (73s) + promote-queue (50s). Operator chose direction = attack DB cost (design+GO). Smaller committed item = port a cache-ELIGIBILITY column into the EXISTING scripts/tools/profile_check_drift.py (NOT a new --profile flag; that was reverted to avoid a parallel surface, institutional-rigor §4).
mode: DONE
scope_lock:
  - scripts/tools/profile_check_drift.py
  - tests/test_tools/test_profile_check_drift.py
---

## Status: Stage 1 DONE (2026-06-09)
Profiler ported (cache-eligibility column), 4/4 tests, profile captured. Stage 2
(wire ineligible-but-static checks) + DB-cost attack are NEXT, design+GO required —
see memory baton project_drift_gate_reduction_profiled_NEXT_2026_06_09.md. This
stage file is closed; downstream work opens a fresh stage.

## Scope Lock
- scripts/tools/profile_check_drift.py
- tests/test_tools/test_profile_check_drift.py

## Blast Radius
- pipeline/check_drift.py — Stage 1: ADD a `--profile` store_true flag + per-check
  `perf_counter()` timing around the existing `check_fn()` dispatch, and a sorted
  table emitted after the loop. OFF by default → the commit/push/CI paths are
  byte-for-byte unchanged (no flag passed). Exit-code semantics untouched.
  Stage 2 (later pass): registry-only additions to CHECK_DEPS / CHECK_TREE_DEPS —
  NO check-logic edits.
- tests/test_tools/test_check_drift_profile.py — NEW. Asserts `--profile` prints a
  table and does not change exit code on a clean tree.
- .githooks/pre-push — Stage 1.5: one-line `--skip-crg-advisory` addition (mirror
  pre-commit:550), advisory-only, zero stale-PASS risk. Conditional on profile.
- tests/test_drift_cache.py — Stage 2: per-check completeness/injection tests for
  any newly-wired cache-eligible check.
- Reads: git (HEAD sha, common-dir), gold.db (read-only, for requires_db checks).
  Writes: none beyond the existing .git/.drift-cache PASS files.

## Eligibility bar (Stage 2 — the load-bearing safety constraint)
A check is cache-eligible ONLY if its inputs are a known, enumerable, STATIC file
set read from disk. NOT eligible: requires_db checks, checks that query gold.db
internally even when requires_db=False (e.g. FAST_LANE promote-queue #171 — already
deliberately excluded, see CHECK_TREE_DEPS comment), unbounded-tree walks, or
runtime-derived inputs. An incomplete CHECK_DEPS entry is a stale-PASS on a capital
gate (false-PASS is the unsafe direction). Each new entry needs an injection test:
mutate each declared dep → MISS; mutate a non-declared file the check reads → must
not silently HIT. Cold-recheck (Check 207) stays on as the backstop, never primary.

## Forbidden
- Routing --fast into pre-commit/pre-push (skips 38 blocking capital checks).
- Shipping a CHECK_DEPS entry without a completeness/injection test.
- Caching any requires_db / live-state / unbounded-input check.
- Editing check LOGIC while wiring deps (registry-only in Stage 2).
