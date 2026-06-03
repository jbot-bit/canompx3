# Stage: Speed up check_drift.py — tree-hash cache + threadpool

task: Make pipeline/check_drift.py fast. Root cause (re-profiled 2026-06-03, this
  session): 50 checks >200ms summing 264.9s total; the worst single check (FAST_LANE
  PROMOTE queue, BLOCKING + non-DB) is now 43.7s (was 80.9s before the 2026-06-03
  adversarial-review fixes e6910886/5df768f3 — number refreshed). Top non-DB
  cache-eligible targets re-measured: FAST_LANE PROMOTE 43.7s, DSR ref-lock 14.5s,
  theory_grant parity 13.5s, doc-hygiene 11.3s, Phase4 SHA manifest 6.1s (~89s total).
  The #2 overall (Validated_setups C4, 16.3s) is DB → NOT cache-eligible (DB content
  unhashed; cold-recheck's requires_db guard correctly forbids it). Two-tier fix: (1) extend
  _drift_cache with a cheap stat/content tree-digest dep so the big tree-scanners
  hit cache on a clean tree (measured: 535ms content-hash vs 80,900ms scan = 150x);
  (2) ThreadPoolExecutor over the non-DB checks to compress the I/O-bound long tail.

mode: IMPLEMENTATION

## Scope Lock
- pipeline/_drift_cache.py — add tree-digest dep support (new dep-spec form)
- pipeline/check_drift.py — declare tree-deps for slow scanners; threadpool the non-DB loop
- tests/test_pipeline/test_drift_cache.py — tests for tree-digest cache (fail-closed, invalidation)
  (NOTE: corrected path — the existing cache tests live at tests/test_pipeline/, NOT pipeline/tests/.)
- tests/test_pipeline/test_check_drift_parallel.py — new: threadpool equivalence + ordering

## Blast Radius
- _drift_cache.py — fail-closed cache module; consumed ONLY by check_drift.py main loop.
  New API is additive (existing `cache_key(label, list[str])` path untouched). Reads:
  filesystem stat/bytes; Writes: .git/.drift-cache/ (gitignored). No DB, no capital path.
- check_drift.py — THE canonical guardrail. Changes: (a) CHECK_DEPS gains tree-spec
  entries for ~5 slow checks; (b) the non-DB branch of the main `for` loop runs under a
  ThreadPoolExecutor. DB checks STAY serial on the shared read-only connection (duckdb
  conn is not thread-safe for concurrent execute). Output ordering preserved by collecting
  results then printing in registry order. Exit-code semantics unchanged.
- Risk: a wrong tree-dep = a blocking check that caches PASS while real drift exists =
  silent failure on a commit gate. Mitigation: content-hash (not mtime) tree digest;
  known-violation injection test per cached check; fail-closed on any hash error.
- Adversarial-audit gate REQUIRED (pipeline/ + blocking-gate semantics) before close.

## Plan (staged, upstream-first)

Stage 1 — _drift_cache tree-digest API (no check_drift wiring yet)  [DONE 2026-06-03]
  - Add `tree_cache_key(label, file_deps, tree_deps)` where tree_deps is a list of
    (glob_root, pattern) pairs. Digest = content-hash of sorted(matched files). Fail-closed
    identical to existing path. Existing cache_key untouched.
  - LANDED: `pipeline/_drift_cache.py` `tree_cache_key()` (path+content digest so add/remove/
    rename invalidate, not just edit; "tree" namespace prevents collision with cache_key;
    ValueError on out-of-root glob match → None/MISS). Unwired: grep confirms zero production
    callers — additive only.
  - ADVERSARIAL-AUDIT (evidence-auditor, 2026-06-03): verdict CONDITIONAL → all findings closed:
    * CRITICAL (fixed): a missing/typo'd `glob_root` globs EMPTY with NO error → phantom-empty
      tree → would serve a stale PASS over real drift once wired into a blocking check. Fix:
      `if not root.is_dir(): return None` guard (fail-closed). Mutation-proven by 2 new tests
      (`*_nonexistent_glob_root_fails_closed`, `*_root_is_a_file_fails_closed` — both FAIL when
      the guard is removed).
    * Test-adequacy (fixed): the path component of the digest was not proven load-bearing. The
      airtight probe is an order-preserving rename among IDENTICAL-content files — a content-only
      digest collides there; the path-bound digest does not. Test
      `*_path_is_load_bearing_rename_among_identical_content` FAILS when `{rel}:` is dropped.
    * Docstring overstatement (fixed): "fail-closed... any glob raises" now also names the
      non-existent-root case.
  - 12 tree tests + 10 pre-existing = 22 in test_drift_cache.py; +8 canary = 30 pass, ruff clean.
  - STAGE 1 COMPLETE & AUDIT-CLOSED. Stage 2 (wire the 5 slow scanners) may proceed; note the
    `is_dir()` guard means every Stage-2 tree-dep root must be a real, correctly-spelled dir or
    the check silently runs uncached (slow but honest) — verify each declared root resolves.
  - Tests: hit/miss, invalidation on add/remove/edit a tree file, fail-closed on unreadable.

Stage 2 — wire the 5 slow tree-scanners into the cache
  - check_fast_lane_promote_orphans (80.9s) → tree-deps: docs/audit/results/*FAST_LANE*,
    docs/audit/hypotheses/*.yaml, the ledger + graveyard + action-queue files, promote_queue.yaml.
  - theory_grant parity (27s), fast_lane_status rollup (8.3s), DSR lock (17s), doc-hygiene (14s)
    — declare each one's actual input tree after tracing its reads.
  - Per check: known-violation injection test proving a real drift still fails through the cache.

Stage 3 — ThreadPoolExecutor over non-DB checks
  - Split CHECKS into db / non-db. Run non-db concurrently (workers = min(8, cpu-2)); keep
    db serial. Collect (idx, label, violations) → print in registry order. --fast/--quiet/
    --skip-advisory/--skip-crg-advisory semantics preserved. Exit code unchanged.
  - Test: parallel run produces identical violation set + identical exit code as serial.

## Verify (per stage + final)
- python -m scripts.tools.profile_check_drift (before/after wall-time)
- targeted: pytest pipeline/tests/test_drift_cache.py pipeline/tests/test_check_drift_parallel.py
- full: python pipeline/check_drift.py (clean exit 0, count unchanged)
- cold-recheck parity: check_drift_cache_meta_recheck must still pass
- adversarial-audit gate (evidence-auditor) before deleting this stage file
