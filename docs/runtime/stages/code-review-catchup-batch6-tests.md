---
task: code-review-catchup-batch6-tests
mode: IMPLEMENTATION
scope_lock:
  - tests/test_pipeline/test_check_drift_context.py
  - tests/test_trading_app/test_bot_dashboard.py
  - tests/test_trading_app/test_notifications.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_stress_hardcore.py
  - tests/test_trading_app/test_lane_allocator.py
blast_radius: |
  Batch 6 (chunk 1 of N) in the Opus-only code-review catch-up sweep. Reviews
  the 8 oldest test-touching commits since 2026-04-24 not already covered by
  Batches 1-5. Tests are not capital-class on their own — but a tautological
  test (assertion that the implementation behaves like itself) creates false
  confidence that propagates to every future review. The bias surface here is
  "looks like it protects, but doesn't". Read-only review with in-session
  implementation per CLAUDE.md mandate. Companion sub-batches will follow for
  the remaining ~150 test-touching commits.
---

# Code Review Catch-Up — Batch 6 (Tests, chunk 1)

**Stage:** Batches 1-5 DONE, Batch 6 (chunk 1) PENDING.
**Model required:** Opus 4.7 — Sonnet PROHIBITED per plan (HoldToKill +
BrokerDispatcher were Sonnet misses; tautological tests are the same
"looks-fine-on-skim" class).
**Last commit on main:** `c9f27a72` — `[code-review batch 5] pipeline catch-up`.
**Date opened:** 2026-05-15.

## Why tests are the right Batch 6 surface

Tests are not capital-class on their own. But:

1. A tautological test (asserts the implementation against itself) gives green
   CI forever while the underlying behavior silently drifts. The
   `iso_utc_silent_none` class-pattern (`feedback_iso_utc_silent_none_class_pattern.md`)
   is the canonical example — silent-None helper had passing tests for months
   because tests asserted "returns None on unknown type" instead of "warns".
2. Drift checks without injection tests have shipped in this repo twice
   (Batch 4 NQ-mini empty-dict edge case; Batch 5 Chordia threshold check).
   Both surfaced only because we audited the test. CI passed both.
3. Mock-and-assert-the-mock patterns survive review because the test "passes"
   — but it asserts the mock's wiring, not the SUT's behavior.

Tests are the immune system's immune system. This batch audits whether it
itself is sound.

## Batches 1-5 outcomes (DONE — for context)

- **Batch 1** (pre-existing): HoldToKill canonical-field fix `f75157fe`, CSRF middleware `db8df761`, A.6 ORB caps per-aperture `82510553`.
- **Batch 2** (`17d8b5cd`): 2 operator-clarity findings on session_orchestrator.
- **Batch 3** (`337c2ed4`): CSRF `PYTEST_CURRENT_TEST` env-var-only bypass closed.
- **Batch 4** (`eafb7e0f`): 4 findings — NQ-mini drift check, `_normalize_writable_path` consolidation, magic-number scope extension, `_INACTIVITY_BLOCK_DAYS` canonicalization.
- **Batch 5** (`c9f27a72`): 1 MEDIUM — Chordia threshold check shipped without injection tests; 6 injection tests added; float-repr LOW surfaced and fixed in-session.

## Batch 6 chunk 1 — confirmed commits in scope

Chronological, oldest first (run `git log --since="2026-04-24" --format="%H|%s" --no-merges --reverse -- tests/ | head -8` to refresh):

| # | Hash | Subject | Real diff (whitespace-ignored) |
|---|------|---------|-------------------------------|
| 1 | `a9f92a89` | audit(codex): 3-day Codex commit sweep + hardening | +51 to `test_check_drift_context.py` (new) |
| 2 | `43a123ad` | A+ follow-up: test teeth + symmetric pid-alive rename (#100) | +27/-36 in `test_bot_dashboard.py` |
| 3 | `5be52bdc` | fix(live): notify() returns bool so preflight self-test can detect broken Telegram | +32 in `test_notifications.py` (new section) |
| 4 | `ca363e1a` | fix(live): seed F-1 XFA EOD balance to $0 in signal-only (closes B6) | +92 in `test_session_orchestrator.py` |
| 5 | `9b16c4eb` | [mechanical] fix: Ralph Loop iter 170 — dead parameter break_ts | -2 in `test_stress_hardcore.py` (1445/-1447 is CRLF churn) |
| 6 | `e02c529d` | fix(live): close 5 silent-failure paths before unattended demo run | +132 in `test_session_orchestrator.py` |
| 7 | `a6494e76` | fix(test): align test_preflight mock with post-B2 notify() bool contract | +1/-1 in `test_bot_dashboard.py` |
| 8 | `9809f1b8` | [mechanical] fix: Ralph Loop iter 171 — import RHO_REJECT_THRESHOLD | +2/-2 in `test_lane_allocator.py` (907/-907 is CRLF churn) |

Verify each on resume — squashes/reverts/rebases may have changed history.

## Batch 6 focus areas

For each test commit, ask:

1. **Tautology check** — does the test assert the SUT's behavior, or does it
   assert the mock's wiring? Mock-and-then-assert-the-mock-was-called is
   tautological. Real assertion: SUT output / side effect / observable state.
2. **Drift-check injection rule** — if the parent commit added a `check_*`
   function in `pipeline/check_drift.py`, does the test inject a known
   violation and confirm the check FIRES? "It passes 132/132" is necessary
   but not sufficient. Batch 5 caught the missing injection on
   `check_chordia_result_threshold_matches_prereg`.
3. **Silent-None pattern** — does the test assert on an explicit warning /
   log / raise, or does it accept silent None as success? Per
   `feedback_iso_utc_silent_none_class_pattern.md`, operator-visible helpers
   must WARN on unknown types, and tests must catch that warning.
4. **Test fixture time bombs** — any hardcoded `last_verified_at`,
   `datetime.utcnow()`, or SLA threshold without clock injection? Per
   `feedback_test_clock_injection.md`.
5. **CRLF/whitespace-only diffs** — the diff-stat lies on Windows/macOS
   line-ending changes. Use `git diff -w --shortstat` to see real change.
   Two of the eight commits above are whitespace-noise; the third (a6494e76)
   is a 1-line fix.
6. **Contract alignment** — when a fix changes a function's return type
   (e.g. `notify()` now returns `bool`), every call-site test must update,
   not just one. Audit for cousin tests that still mock the old contract.

## Disconfirming checks (run BEFORE accepting a commit as clean)

```bash
# 1. Does the test file under review actually have assertions, or just calls?
grep -c "^\s*assert\|self\.assert" tests/test_pipeline/test_check_drift_context.py
grep -c "^\s*assert\|self\.assert" tests/test_trading_app/test_notifications.py

# 2. Are mocks asserted-on the mock, or on the SUT?
grep -n "mock_.*\.assert_called\|MagicMock\|create_autospec" <file>

# 3. Drift-check injection test pattern (every new check_<x> in check_drift.py
#    should have at least one injection test in tests/test_pipeline/test_check_drift*.py)
grep -n "def check_" pipeline/check_drift.py | wc -l   # check count
grep -rn "def test.*inject\|inject_violation\|tmp_path.*\.write_text" tests/test_pipeline/ | wc -l

# 4. CRLF noise vs real change
git diff -w --shortstat <hash>^..<hash> -- tests/

# 5. Hardcoded time bombs
grep -rn "datetime.utcnow()\|datetime\.now()\|2026-\|2025-\|last_verified_at\s*=\s*['\"]" tests/test_trading_app/ tests/test_pipeline/ | grep -v "# clock-injected\|monkeypatch"
```

## Per-session protocol (battle-tested from Batches 1-5)

1. **Stay on Opus 4.7.** Sonnet caught BrokerDispatcher dead-class miss; Opus is required for tautology-class catches.
2. Refresh commit list with the scope command — never trust the cached table above blindly.
3. For each commit: PREMISE → TRACE → EVIDENCE → VERDICT.
4. Implement findings same-session per CLAUDE.md mandate.
5. Run `python pipeline/check_drift.py` (target 132/132, Batch 5 baseline).
6. Run `python -m pytest tests/test_pipeline/ tests/test_trading_app/ -x -q`.
7. Write commit message to `tmp/batch6-commit-msg.txt` (NEVER PowerShell here-string).
8. `git add` only scope-locked files; `git commit -F tmp/batch6-commit-msg.txt`.
9. Update this stage file post-commit with outcome table + handoff for chunk 2.

## Deferred findings (carried forward from Batch 5 — do NOT lose)

Same list as Batch 5 — Batch 6 does NOT pull these in (scope-creep prevention):

1. **BrokerDispatcher dead-class** — wire or delete.
2. **Pyright cluster** — `session_orchestrator.py`, `bot_dashboard.py`, `check_drift.py` (Batch 4 surfaced) plus test fixture types.
3. **Tradovate `verify_bracket_legs` ID-heuristic** — audit before back-to-back paper-trade activation.
4. **SSE lazy-stop on last-subscriber-disconnect** — reference-counted stop in `unsubscribe`.
5. **NQ-mini Stage 2 wiring** — separate branch.

## Bias guards (do NOT skip)

1. **Tests are themselves code** — they have bugs. The Batch 5 Chordia injection
   test surfaced a float-repr bug *in the check it was testing*. Tests can
   reveal upstream bugs only if they actually exercise the check's output.
2. **CI green is not a verdict** — every commit below passed CI. The audit
   asks a different question (tautology, injection, contract-cousin sweep).
3. **CRLF churn hides 2-line bugs** — `+1445/-1447` looks like a rewrite; it
   isn't. Run `git diff -w` first, every time.
4. **One commit at a time** — Batch 5 protocol is PREMISE → TRACE → EVIDENCE →
   VERDICT per commit, not per-batch. Resist the urge to skim.

## Audit-gate exemption decision tree

Per `.claude/rules/adversarial-audit-gate.md`:

- **TEST-ONLY** (no production change; adds/edits a `tests/` file only) →
  exempt by class with explicit one-liner in commit body.
- **TEST + DRIFT-CHECK INJECTION** (adds a test that exercises a check_drift
  function) → still exempt by class; the check itself was already audit-gated
  in its parent batch.
- **TEST + PRODUCTION FIX** (rare in Batch 6 — only if a test exposes a real
  behavior bug that we fix in-session) → adversarial audit MANDATORY before
  next phase commits.

## Resume instructions (post-/clear)

1. Re-read this file.
2. Confirm Opus 4.7.
3. Confirm `main` is at `c9f27a72` (Batch 5) or later.
4. Refresh commit list (table above may be stale).
5. Begin commit-by-commit review starting with `a9f92a89`.

## Remaining plan after this chunk

- **Batch 6 chunks 2-N** — work the remaining ~150 test-touching commits in
  oldest-first chunks of ~8.
- **Deferred cleanup batch** — pyright + BrokerDispatcher + verify_bracket_legs + SSE + NQ-mini Stage 2.

## State at handoff (2026-05-15 PM)

- `main` HEAD: `c9f27a72` (Batch 5 commit).
- Drift baseline: 132/132 (0 skipped, 20 advisory).
- pytest baseline: 1424 PASS.
- Working tree on Batch 6 scope: clean. Pre-existing unmodified files
  (`.claude/agents/*.md`, `.claude/rules/subagent-budget.md`,
  `docs/runtime/session-checkpoint-2026-05-14-go-live.md`,
  `docs/runtime/stages/code-review-catchup-batch3-resume.md`,
  `resources/Harris_*`, `tmp/`) are NOT in scope and will remain untouched.

## Batch 6 chunk 1 outcome (2026-05-15)

All 8 target commits reviewed under PREMISE → TRACE → EVIDENCE → VERDICT.

| # | Commit | Subject (truncated) | Verdict |
|---|--------|---------|---------|
| 1 | `a9f92a89` | Codex 3-day sweep — drift-check context tests | **FINDING (LOW) — fixed in-session** |
| 2 | `43a123ad` | A+ follow-up: test teeth + pid-alive rename | CLEAN |
| 3 | `5be52bdc` | notify() bool contract tests | CLEAN |
| 4 | `ca363e1a` | F-1 XFA $0 seed tests | CLEAN |
| 5 | `9b16c4eb` | dead `break_ts` param removal | CLEAN (CRLF-noise diff) |
| 6 | `e02c529d` | overnight resilience hardening tests | **FINDING (MED) — deferred** |
| 7 | `a6494e76` | align test_preflight mock bool | CLEAN |
| 8 | `9809f1b8` | RHO_REJECT_THRESHOLD import rename | CLEAN (CRLF-noise diff) |

### LOW finding fixed in-session (commit 1 / `a9f92a89`)

`check_doc_hygiene_contracts` regex catches four placeholder tokens:
`PENDING | TO_FILL_(?!AFTER_COMMIT)[A-Z_]+ | UNSTAMPED | TO_BE_STAMPED`. The
4 injection tests shipped in `a9f92a89` covered only `PENDING` and
`TO_FILL_*`, plus the exempt token and a real-SHA happy path. A future
regex-refactor accidentally dropping `UNSTAMPED` or `TO_BE_STAMPED` from
the alternation would ship silently.

Fix:
- Added `test_catches_commit_sha_unstamped` and
  `test_catches_commit_sha_to_be_stamped` to
  `tests/test_pipeline/test_check_drift_context.py`.
- Mutation-probed: simulated a regex with `UNSTAMPED` removed; the new
  test would fail (confirmed via inline regex patch in Python).
- 15/15 tests pass in `test_check_drift_context.py` (13 prior + 2 new).
- Drift 132/132 PASS (baseline maintained).

### MED finding deferred (commit 6 / `e02c529d`)

`TestOvernightResilienceHardening` shipped 6 tests for 5 fixes (F8, F6, F2, R2 ×2, F5):
- **F8 / F6 / F2** — `open(so.__file__).read()` + `assert "..." in src`
  source-string-grep tests. These survive any refactor preserving the
  literal strings and could break on benign comment rewording. The author
  explicitly chose this pattern because constructing a SessionOrchestrator
  with a `cancel_bracket_orders`-raising broker is heavy — but
  `build_orchestrator()` + `FakeBrokerComponents` *is* serviceable in
  the same file (F5/R2 prove it). Behavioral coverage achievable.
- **R2 (×2) / F5** — CLEAN behavioral tests. F5 patches `query_equity`
  side-effect and asserts `update_equity(None)` is called — proper
  mutation-probe shape.

**Why deferred, not fixed in-session:**
1. Refactoring F8/F6/F2 to behavioral tests requires building broker test
   doubles that simulate `cancel_bracket_orders` raising, journal-unhealthy,
   and startup-with-None-equity-on-XFA. Each is a stand-alone test-double
   piece of work.
2. Scope-creep prevention: institutional-rigor § 1 + workflow-preferences
   "trivial-change tier" — this fix is non-trivial and would expand Batch
   6 scope dramatically.
3. Tests *do* provide accidental-deletion regression protection, which
   was their stated purpose.

Added to deferred bucket as item 6.

### Verification

- Drift: 132/132 PASS, 0 skipped, 20 advisory (baseline maintained).
- pytest `tests/test_pipeline/test_check_drift_context.py`: 15 PASS.
- pytest `tests/test_pipeline/` full: 1426 PASS (Batch 5 baseline 1424 + 2
  new injection tests = exact count).
- No scope expansion mid-flight; only
  `tests/test_pipeline/test_check_drift_context.py` modified.
- All 8 commits reviewed PREMISE → TRACE → EVIDENCE → VERDICT.

## Deferred findings (updated post-chunk-1)

1. **BrokerDispatcher dead-class** — unchanged from Batch 5.
2. **Pyright cluster** — unchanged (note: `test_check_drift_context.py:69`
   surfaced 3 unused-binding pyright warnings during Batch 6 chunk 1 edit;
   PRE-EXISTING fixture pattern in lines well above the edit point. Joins
   the existing cluster).
3. **Tradovate `verify_bracket_legs` ID-heuristic** — unchanged.
4. **SSE lazy-stop on last-subscriber-disconnect** — unchanged.
5. **NQ-mini Stage 2 wiring** — unchanged.
6. **NEW — F8/F6/F2 behavioral coverage refactor** — replace
   source-string-grep tests in `TestOvernightResilienceHardening`
   (`tests/test_trading_app/test_session_orchestrator.py:1720+`) with
   broker-double-driven behavioral tests. Needs test-double for
   cancel_bracket_orders-raises, journal-unhealthy, startup-None-equity.

## Next chunk

`git log --since="2026-04-24" --format="%H|%s" --no-merges --reverse -- tests/`
returns 158 entries. Chunk 1 covered the oldest 8. Chunk 2 picks up at #9
(after `9809f1b8`). Same protocol.
