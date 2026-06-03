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

Stage 2 — wire the slow tree-scanners into the cache  [IN PROGRESS 2026-06-03]
  - DONE: check_fast_lane_promote_orphans (Check #171, 43.7s — the dominant cost). Wired via NEW
    `CHECK_TREE_DEPS` dict (label → {file_deps, tree_deps}) + dispatch-loop branch calling
    `_drift_cache.tree_cache_key`. Existing flat-`CHECK_DEPS` path byte-for-byte unchanged
    (2 proven checks unaffected). Dep set traced from scripts.research.fast_lane_promote_queue.scan():
      file_deps = scanner source (fast_lane_promote_queue.py, fast_lane_trial_ledger.py,
        research/oos_power.py, trading_app/config.py) + 5 runtime YAMLs (action-queue,
        promote_queue, trial_ledger, trial_corrections, graveyard_digest);
      tree_deps = docs/audit/results/*fast-lane*.md (incl. .revocation.md sidecars), docs/audit/
        hypotheses/*.yaml (933 files — the re-parse cost; also K_global=len(ledger)).
    Source files ARE in the dep set (institutional-rigor §4 — verdict tracks the code).
  - NEW import-time guard `_assert_check_dep_dicts_valid()` (sibling of `_assert_slow_labels_valid`):
    mutual-exclusivity of the two dep dicts + labels-real + non-DB. Fail-loud at import.
  - Tests (6 new, all pass; full module now 28): real-dispatch hit→cold-recheck parity on the LIVE
    check; structural dep-completeness proof (every file the scanner reads is covered); invalidation
    on source-edit + hypothesis-add; the 3 config invariants.
  - DIAGNOSIS (2026-06-03, this session): the prior "cache fires but ZERO speedup (221s≈224s)"
    was NOT a broken cache. Proven: cache HIT 0.5s vs real check 62s; key deterministic+cheap;
    read_pass True. Root cause = the honesty backstop `check_drift_cache_meta_recheck` RE-RUNS
    every cached-hit check cold. With ONE cached check it is a full recheck (documented), so warm
    = 0.5s(hit) + 55s(meta recheck) = net zero. The speedup only exists once MANY checks are cached
    AND the meta-recheck SAMPLES rather than full-rechecks.
  - LOAD-BEARING FIX (done): meta-recheck split into two layers — (1) STRUCTURAL validation
    (in-CHECKS, non-db) runs for EVERY cached-hit label every run (cheap); (2) the expensive cold
    fn() re-run is SAMPLED to exactly ONE label/run via stateless `_drift_cache.meta_recheck_sample_index`
    (index = sha256(sorted labels + git HEAD) % n → stable per-commit, rotates across commits, NO
    persisted counter to race between concurrent worktree drift runs; fail-closed to index 0 if HEAD
    unreadable). Sampling is defense-in-depth: complete dep sets make stale PASS impossible by key
    construction; the static `*_covers_every_file_*` completeness tests are the primary
    under-declaration guard, the sampled cold re-run is the runtime backstop with bounded
    (per-few-commits) detection latency.
  - WIRED (done): +2 cleanly-hashable non-DB checks (real per-check profiler costs, NOT stale est):
      * AM3.3 theory_grant parity (#162, 25.4s) — file_deps: chordia_audit_log.yaml + trading_app/
        chordia.py (T-threshold constants = verdict logic); tree_deps: hypotheses/*.yaml.
      * DSR reference-universe lock (Crit 5, 24.6s) — tree_deps: hypotheses/*.yaml only.
    Both verified: cold ~25s → warm HIT, key stable+cheap (2.8s/1.5s).
  - DELIBERATELY NOT WIRED (rigor — these have non-file-content verdict inputs = stale-PASS hazard):
      * doc-hygiene contracts (19.7s): reads ARBITRARY prereg `execution.entrypoint` path EXISTENCE
        (check_drift.py:2336-2337) — dep set is data-dependent, not statically enumerable. Deleting
        an entrypoint script flips PASS→FAIL with NO hashed-file change → stale PASS. REJECTED.
      * Phase4 SHA migration manifest (8.8s): verdict depends on GIT HISTORY state (`git cat-file -e`,
        blob SHA at a historical commit; lines 7028-7030+). Not file-content-hashable — same class as
        a requires_db check. REJECTED.
    Net cacheable set: 3 checks (was 1). Cacheable cost ≈ 80+25+25 = 130s of the 263s total.
  - Adversarial-audit gate (evidence-auditor) REQUIRED before stage close (pipeline/ + blocking-gate).

  - ADVERSARIAL-AUDIT (evidence-auditor, 2026-06-03 post-clear): verdict **FAIL (CRITICAL)** —
    independently confirmed by the main thread. The FAST_LANE PROMOTE check (#171) was WRONGLY
    cached: although its CHECKS tuple is requires_db=False, scan() internally opens gold.db via
    `_resolve_oos_window_days()` (`SELECT MAX(trading_day) FROM orb_outcomes`,
    fast_lane_promote_queue.py:337) and that value flips entries REJECTED_OOS_UNPOWERED↔QUEUED
    (build_entry OOS-power pre-flight, fast_lane_promote_queue.py:800-827). DB content is NOT
    file-hashable → a warmed cache serves a STALE PASS on a BLOCKING capital gate as new bars land.
    Same non-cacheable class as the already-rejected Phase4-SHA (git-history) + doc-hygiene
    (entrypoint-existence) checks — the `requires_db=False` flag was the trap (it gates cache
    eligibility but the check reads the DB internally without the flag set).
  - FIX (done, operator-approved Option A "un-wire #171"): removed FAST_LANE from CHECK_TREE_DEPS
    (replaced with a DELIBERATELY-NOT-CACHED rejection note at the dict head documenting the DB
    verdict-input). Added anti-regression test `test_fast_lane_check_is_deliberately_not_cached`
    (asserts FAST_LANE ∉ CHECK_TREE_DEPS ∪ CHECK_DEPS). Removed the 3 FAST_LANE-specific cache
    tests (coverage/invalidation/slow-parity) since the check is no longer cached. The 2 genuinely
    file-hashable checks (theory_grant #162, DSR) are UNAFFECTED and remain cached.
  - NET cacheable set CORRECTED: **2 checks** (theory_grant + DSR ≈ 50s), not 3. The auditor's
    secondary finding (`fast_lane_structural_hash.py` undeclared) is MOOT — un-wiring #171 removes
    the dep-completeness concern entirely.
  - FOLLOW-UP (not built, separate audit-gated stage): a gold.db-content-aware key (fold
    MAX(trading_day) into the digest) could recover #171's ~62s under its own correctness review.
  - POST-FIX VERIFICATION: cold→warm timing re-measured (FAST_LANE no longer cached); 34/34
    non-slow cache tests pass + 2 slow parity tests pass; ruff clean; import guard passes.

Stage 3 — ThreadPoolExecutor over non-DB checks  [IN PROGRESS 2026-06-03 post-clear]
  - Split CHECKS into db / non-db. Run non-db concurrently (workers = min(8, cpu-2)); keep
    db serial. Collect (idx, label, violations) → print in registry order. --fast/--quiet/
    --skip-advisory/--skip-crg-advisory semantics preserved. Exit code unchanged.
  - Test: parallel run produces identical violation set + identical exit code as serial.

  REFINED DESIGN (this session — concurrency for compute, serial for EFFECTS):
  The main loop currently mutates shared state per-iteration (prints, counters,
  all_violations, _CACHE_HITS_THIS_RUN). Naive threading races all of it. Correct shape:
    * Phase A (SERIAL, registry order): classify each check → skip / requires_db / cacheable.
      For cacheable non-DB checks compute cache_key + read_pass HERE (cheap hash read) and
      record cache hits into _CACHE_HITS_THIS_RUN. This guarantees ALL hits are recorded
      before the meta-recheck runs. Build a work-list of (idx,label,fn) needing a real fn().
    * Phase B (PARALLEL): run the cache-MISS / non-cacheable NON-DB fn()s concurrently into
      a results dict keyed by idx. Each fn() is pure (returns list[str]); no shared mutation.
      Each parallel fn() is wrapped so stdout is captured per-thread (advisory checks print
      inline) — prevents interleaved output. DB checks NOT in this pool.
    * Phase C (SERIAL, registry order): for each check replay print + counter increment +
      all_violations.extend, using the Phase-A/B results. Run requires_db fn()s here on the
      shared con (serial). Run the meta-recheck fn() LAST (it reads the now-complete
      _CACHE_HITS_THIS_RUN; its sampler already does sorted(set(...)) so it is order-immune).
  CONSTRAINTS proven from code read:
    - duckdb _shared_con is NOT thread-safe → all requires_db checks serial (Phase C).
    - check_drift_cache_meta_recheck MUST run last (import-asserted at CHECKS[-1]) and reads
      _CACHE_HITS_THIS_RUN → cannot be in the parallel pool; runs last in Phase C.
    - _drift_cache.write_pass on a cacheable MISS must happen after its fn() returns — do it
      in Phase C (serial) keyed off the Phase-B result, so cache writes are not raced.
    - --quiet _QuietSink + per-check stdout: capture per parallel task, replay in Phase C.
  - Scope additions: scripts/tools/profile_check_drift.py is read-only (verify only).

  IMPLEMENTED (this session, CHECKPOINT — NOT yet stage-closed):
    * pipeline/check_drift.py: import concurrent.futures; new helpers _run_one_check_capturing
      (per-thread stdout capture, fail-closed exception→violation) + _drift_worker_count
      (min(8,cpu-2), DRIFT_WORKERS env override). Main loop rewritten into the A/B/C phases
      above. requires_db checks + meta cold-recheck stay SERIAL (Phase C). Cache read in
      Phase A (records _CACHE_HITS_THIS_RUN before meta runs); cache WRITE in Phase C (no race).
    * tests/test_pipeline/test_check_drift_parallel.py: NEW. 9 tests — serial-vs-parallel
      verdict+exit-code equivalence (subprocess, DRIFT_WORKERS=1 vs 8); meta-recheck-present;
      worker-count override + fallback bounds.
  DELIBERATE BEHAVIOR CHANGE (documented, defensible — NOT pure equivalence):
    The OLD serial path did NOT wrap non-DB check_fn() in try/except — a raising non-DB check
    CRASHED the entire drift run (unhandled). The NEW worker catches it → single-line
    `EXCEPTION:` violation → exit 1, other checks still report. This is STRICTLY fail-closed
    (exception → violation, never a false PASS) and matches the DB path's pre-existing
    try/except. Net: one broken check no longer aborts all reporting. Surfaced to operator.
  VERIFIED:
    * Serial (DRIFT_WORKERS=1) vs parallel (=8): `diff` of sorted verdict lines = IDENTICAL
      (198 == 198 lines, both exit 1). This is the load-bearing equivalence proof.
    * Module imports clean; _assert_check_dep_dicts_valid passes; CHECKS[-1] meta assert holds.
    * 34/34 non-slow test_drift_cache.py pass (Stage 1/2 cache unaffected); 7 worker-count
      tests pass; ruff clean.
    * NOTE: this worktree's full drift is RED for PRE-EXISTING ENVIRONMENTAL reasons (33
      "All imports resolve" failures from missing local research/output artifacts + 4 unrelated
      checks) — NOT caused by Stage 3 (serial==parallel both show 170 passed / 37 violations).
      So "drift exits 0" is NOT a usable closeout gate in this worktree; equivalence is.
  TEST-HARNESS GOTCHA FIXED: pytest-timeout's `thread` watchdog races pytest's capture
    manager on subprocess-spawning tests → one full-drift run intermittently returned ZERO
    captured verdict lines (false FAIL). Fix = run with `-p no:timeout` (UNLOAD the plugin);
    `mark.timeout(0)` does NOT work (still loads plugin — the n=4 watchdog-race close). Test
    now asserts-nonempty-with-stderr so a real crash can't hide as an empty mis-compare.
  OWED BEFORE STAGE CLOSE (could NOT run — PRE-CLEAR blocked subagent spawns):
    (1) ADVERSARIAL-AUDIT GATE (evidence-auditor) on the Stage-3 commit — MANDATORY
        (pipeline/ blocking-gate, [judgment]). Focus: thread-safety of the shared duckdb con
        (must stay serial — verify no DB check leaked into the pool), _CACHE_HITS_THIS_RUN
        ordering vs meta-recheck, cache-write race in Phase C, output-ordering equivalence,
        the deliberate exception-handling change.
    (2) pytest-wrapper confirmation of the 2 slow equivalence tests with -p no:timeout
        (manual diff already proves the underlying equivalence; this is harness confirmation).
  NEXT SESSION: /clear, then run the audit on the Stage-3 commit; if PASS, delete this file.

## STATUS 2026-06-03 (post-clear task #4): AUDIT-FAIL FIXED, commit ca0a2b7e.
##   Stage 2 cache: 2 checks cached (theory_grant, DSR); FAST_LANE un-wired (DB leak).
##   DO NOT DELETE THIS FILE YET. Remaining before close:
##   (1) confirmatory adversarial audit of the FIX commit ca0a2b7e (un-wire #171) —
##       a [judgment] pipeline/ blocking-gate change; was BLOCKED by PRE-CLEAR context
##       (no subagent spawns). The fix is a REMOVAL (reduces cache surface, closes the
##       hazard, cannot add a stale-PASS path) + fully tested, so risk is low, but the
##       gate is still owed. Run evidence-auditor on ca0a2b7e next session.
##   (2) Stage 3 (ThreadPoolExecutor over non-DB checks) — NOT STARTED. Optional; the
##       real long-tail compression. The cache speedup alone is now small (~25s net,
##       2 checks) so Stage 3 is where the remaining wall-time win lives.
##   Then delete this file.

## Verify (per stage + final)
- python -m scripts.tools.profile_check_drift (before/after wall-time)
- targeted: pytest pipeline/tests/test_drift_cache.py pipeline/tests/test_check_drift_parallel.py
- full: python pipeline/check_drift.py (clean exit 0, count unchanged)
- cold-recheck parity: check_drift_cache_meta_recheck must still pass
- adversarial-audit gate (evidence-auditor) before deleting this stage file
