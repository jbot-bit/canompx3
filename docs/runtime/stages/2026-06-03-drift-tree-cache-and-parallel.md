# Stage: Speed up check_drift.py — tree-hash cache + threadpool

task: Make pipeline/check_drift.py fast. Root cause (profiled 2026-06-03): 6 of 197
  checks consume 174s of 264s total; the worst single check (FAST_LANE PROMOTE
  queue, BLOCKING + non-DB) is 80.9s re-scanning 1,037 result MDs + 749 hypothesis
  YAMLs from scratch on every run, including every commit. Two-tier fix: (1) extend
  _drift_cache with a cheap stat/content tree-digest dep so the big tree-scanners
  hit cache on a clean tree (measured: 535ms content-hash vs 80,900ms scan = 150x);
  (2) ThreadPoolExecutor over the non-DB checks to compress the I/O-bound long tail.

mode: IMPLEMENTATION

## Scope Lock
- pipeline/_drift_cache.py — add tree-digest dep support (new dep-spec form)
- pipeline/check_drift.py — declare tree-deps for slow scanners; threadpool the non-DB loop
- pipeline/tests/test_drift_cache.py — tests for tree-digest cache (fail-closed, invalidation)
- pipeline/tests/test_check_drift_parallel.py — new: threadpool equivalence + ordering

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

Stage 1 — _drift_cache tree-digest API (no check_drift wiring yet)
  - Add `tree_cache_key(label, file_deps, tree_deps)` where tree_deps is a list of
    (glob_root, pattern) pairs. Digest = content-hash of sorted(matched files). Fail-closed
    identical to existing path. Existing cache_key untouched.
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
