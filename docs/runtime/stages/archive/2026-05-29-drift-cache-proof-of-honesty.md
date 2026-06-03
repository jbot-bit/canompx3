---
task: Stage 3 (narrow) — per-check content-hash cache, ONE check, proof-of-honesty
mode: CLOSED_PENDING_AUDIT
scope_lock:
  - pipeline/_drift_cache.py
  - pipeline/check_drift.py
  - tests/test_pipeline/test_drift_cache.py
blast_radius:
  - pipeline/_drift_cache.py — NEW; content-hash cache read/write under .git/.drift-cache/. Fail-closed: any error → MISS (run real check). Only PASS results cached; FAIL never cached. Zero importers besides check_drift.py.
  - pipeline/check_drift.py — ADD parallel CHECK_DEPS dict (label→dep paths) + cache wrap in runner for labels in CHECK_DEPS only + meta cold-recheck drift check. CHECKS tuple shape UNCHANGED (4-tuple) — verified strict 4-tuple unpack at :5231 (check #64) and :15038 (runner); a 5th element = ValueError at import. Deps in parallel dict, zero unpack sites touched.
  - tests/test_pipeline/test_drift_cache.py — NEW; honesty proofs (miss-on-dep-edit, corrupt→real-run, changed-dep→real-run, FAIL-never-cached).
  - Reads: trading_app/config.py, trading_app/ai/sql_adapter.py (the cached check's deps). Writes: .git/.drift-cache/ only (inside .git/, never tracked). No repo writes.
updated: 2026-05-29T13:45:00+10:00
agent: claude
---

## Purpose
Prove the drift-cache honesty mechanism on exactly ONE check before any speed-harvest
expansion. Cache key = sha256(label + content-hash of declared deps). Input change → key
miss → real run. Honesty preserved by construction; fail-closed on every error path.

## Cached check (the ONE)
- Label: `ENTRY_MODELS sync`  (check_drift.py:893, check_entry_models_sync)
- is_advisory=False (blocking), requires_db=False (file-only).
- Deps (enumerated, provably complete): trading_app/config.py, trading_app/ai/sql_adapter.py
- Speed win ~0 (its 0.68s is amortized import cost) — this stage proves the MECHANISM, not a speedup.

## Acceptance
- Cache module fail-closed: read/parse/hash error → MISS. FAIL verdict never written.
- ENTRY_MODELS sync cached; absent-from-CHECK_DEPS labels never cached (opt-in default-uncacheable).
- Meta cold-recheck: every cached-HIT label re-run cold once, cached verdict must == cold verdict.
- Tests prove: (a) miss on dep edit, (b) corrupt cache → real run, (c) changed dep → real run, (d) FAIL never cached.
- Full `python pipeline/check_drift.py` passes (no regression to the 4-tuple gate).

## Adversarial-audit gate
check_drift.py is a truth-layer verification path → evidence-auditor independent pass owed
AFTER this stage, BEFORE any expansion to more checks. Meta cold-recheck lands in THIS stage.

### Audit verdict (2026-05-29, independent evidence-auditor context)
CONDITIONAL — zero critical issues. All 5 core promises execution-verified with citations:
fail-closed on every error path; FAIL never cached; stale-dep detection works AND the
`ENTRY_MODELS sync` dep set is complete (ENTRY_MODELS is a literal in config.py; sql_adapter.py
derives `VALID_ENTRY_MODELS = set(ENTRY_MODELS)`); meta cold-recheck runs last (import-time
assert), bypasses cache, fails closed on requires_db hits; CHECKS 4-tuple shape unchanged.

Single gap closed in-stage: the unit tests patched `_CACHE_HITS_THIS_RUN` directly, leaving the
real runner-dispatch→hit→meta-recheck path uncovered. Added
`test_runner_dispatch_real_hit_then_meta_recheck_passes` — earns a genuine cache hit through the
real `read_pass` path (only the cache dir isolated to tmp) and asserts the real meta-recheck
reproduces the PASS cold. 9/9 cache tests pass; full drift 169/0 with Check 190 (meta
cold-recheck) PASSED live.

Latent note for future expansion (NOT a bug today): if a refactor decouples
`VALID_ENTRY_MODELS` from `config.ENTRY_MODELS` into a third file, CHECK_DEPS would be
under-declared until the meta cold-recheck catches it on the next run.
