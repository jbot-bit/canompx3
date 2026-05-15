---
task: code-review-catchup-batch5
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - pipeline/build_bars_5m.py
  - pipeline/build_daily_features.py
  - pipeline/cost_model.py
  - pipeline/dst.py
  - pipeline/asset_configs.py
  - pipeline/paths.py
  - tests/test_pipeline/test_check_drift.py
blast_radius: |
  Batch 5 of 6 in the Opus-only code-review catch-up sweep. Reviews 8 pipeline
  commits since 2026-04-24 — drift checks, daily-features build, cost model,
  DST/session catalog, asset configs, canonical paths. Capital-class:
  pipeline is upstream of every backtest, every validator, and every live
  decision. A silent bug here contaminates every downstream layer. Read-only
  review with same-session implementation per CLAUDE.md mandate.
---

# Code Review Catch-Up — Batch 5 Resume Point (Pipeline)

**Stage:** Batches 1-4 DONE, Batch 5 PENDING
**Model required:** Opus 4.7 — Sonnet PROHIBITED per plan (HoldToKill + BrokerDispatcher were Sonnet misses; pipeline silent failures are the same class)
**Last commit on main:** `eafb7e0f` — `[code-review batch 4] allocator + HWM constants + NQ-mini drift defense`
**Date opened:** 2026-05-15

## Why this matters

Pipeline is the truth layer. Every research script, every validator, every fitness
check, every live lane allocation reads its inputs from `daily_features`,
`orb_outcomes`, `cost_model.COST_SPECS`, `dst.SESSION_CATALOG`,
`asset_configs.ACTIVE_ORB_INSTRUMENTS`, `paths.GOLD_DB_PATH`. A silent
contamination here propagates to every downstream layer without warning.

Pipeline bugs found in prior batches (E2 break-bar look-ahead, GARCH silent NULL,
cryptography/authlib MCP drift) all surfaced this way: one upstream silent
failure → weeks of polluted research. The drift checks Josh added since Apr-24
are the immune system; this batch audits whether the immune system itself is
sound.

## Batches 1-4 Outcomes (DONE — for context)

- **Batch 1** (pre-existing): HoldToKill canonical-field fix `f75157fe`, CSRF middleware `db8df761`, A.6 ORB caps per-aperture `82510553`
- **Batch 2** (`17d8b5cd`): 2 operator-clarity findings on session_orchestrator (signal_record type rename, bracket-cancel warning split). No CRIT/HIGH on capital-class commits.
- **Batch 3** (`337c2ed4`): 1 defense-in-depth finding — CSRF `PYTEST_CURRENT_TEST` env-var-only bypass closed (now requires `pytest` in sys.modules). SSE lifecycle, HoldToKill, poller retirement, localhost binding, session-isolation auto-recover all PASS.
- **Batch 4** (`eafb7e0f`): 4 findings closed —
  - HIGH: new drift check `check_nq_mini_substitution_wired_or_unused` guards Stage 1 NQ-mini contract from silent activation before Stage 2 wiring lands.
  - MEDIUM: `_normalize_writable_path` consolidated to canonical `lane_allocator.normalize_writable_path` (one documented inline exception in `prop_profiles.py` due to pre-existing module-load circular import — `validate_dd_budget()` at L1254 → `load_allocation_lanes` → `lane_allocator` during its partial-load. Function-level import does NOT break the cycle because the function is invoked at module-load. Documented with NOTE block).
  - MEDIUM: `check_magic_number_rationale` scope extended to `account_hwm_tracker.py` + `pre_session_check.py`.
  - LOW: `_INACTIVITY_BLOCK_DAYS` derives from canonical `STATE_STALENESS_FAIL_DAYS`.
  - Verification: drift 132/132, pytest 480p/1s, imports OK.

## Deferred Findings (carried forward — do NOT lose)

Stack for a separate cleanup batch after Batch 6 (or sooner if Josh prioritizes):

1. **BrokerDispatcher dead-class** — 41 lines of API-parity infrastructure with zero production callsites on current `main`. Per `memory/feedback_code_review_dead_class_detection.md`. Either wire it or delete it.
2. **Pre-existing pyright errors cluster** — Josh's explicit ask: "we need to fix them after". Pattern:
   - `trading_app/live/session_orchestrator.py:2942/2959` — `Optional[None]` member access (cancel, query_order_status without None guard)
   - `tests/test_trading_app/test_session_orchestrator.py` — test fixture types bypass BrokerAuth / BrokerRouter / BrokerPositions / Bar protocols
   - `trading_app/live/bot_dashboard.py:369/412/430/989-990` — same `object | None` / `Popen` / `ConvertibleToFloat` cluster
   - `pipeline/check_drift.py` lines 1915/1949/1950/1953/2481/3025/3887/3895/3903 — `Object of type "None" is not subscriptable` cluster surfaced during Batch 4 finalization. PRE-EXISTING (not introduced by Batch 4 — verified at line 7607-7610 edit). Same fix class: proper Optional-narrowing or assertion before subscript.
   - Class-bug pattern — proper test fixture base classes + None-guards, not patching. Per `institutional-rigor.md` §3.
3. **Tradovate verify_bracket_legs ID-heuristic mis-attribution risk** — if two entries land on same contract back-to-back, higher-ID-than-entry heuristic could mis-attribute legs. Mitigated by short verification window. Audit before any back-to-back paper-trade activation.
4. **SSE lazy-stop on last-subscriber-disconnect** — `_sse_start_watchers` is lazy-start but watcher tasks never decrement-and-stop when subscriber count returns to 0. Not a memory leak (bounded set), but wasted CPU polling. Add reference-counted stop in `unsubscribe` if/when CPU becomes a concern.
5. **NQ-mini Stage 2 wiring (Batch 4 spawn)** — separate branch. Action-queue id: `nq_mini_stage2_wiring_2026_05_15`. Touches `trading_app/live/session_orchestrator.py:~2317` (build_order_spec call site), `trading_app/live/webhook_server.py`, ≥1 populated `ACCOUNT_PROFILES` row, integration test. Exit criterion: `check_nq_mini_substitution_wired_or_unused` continues PASS with populated profile (proves callsite landed). Driver memo: `memory/mini_vs_micro_commission_fix.md` (~77% commission reduction).

## Batch 5 Scope (RESUME HERE)

**Scope command (run first to refresh diff — DO NOT trust the commit list below blindly):**
```bash
git log --since="2026-04-24" --format="%H %s" --no-merges -- pipeline/ | head -30
```

**Confirmed commits in scope (from plan — verify on resume):**

- `b98df2ec` — pipeline drift / research-guard work (verify subject + diff)
- `c68ecc3c` — pipeline drift / research-guard work
- `115cccf8` — pipeline drift / research-guard work
- `4a219b0b` — pipeline drift / research-guard work
- `9633fee6` — pipeline drift / research-guard work
- `b1068bac` — verify on main first (may have been reverted/superseded)
- `b700d4ad` — pipeline drift / research-guard work
- `d2b3ba5b` — pipeline drift / research-guard work

If any of these are no longer on `main` (squashed, reverted, rebased away),
note in the commit body and skip — do not chase ghosts.

## Batch 5 Focus Areas

Pipeline-specific bias surfaces. For each commit, ask:

1. **Drift-check correctness** — does the new check (a) inject a known violation
   and confirm it fires, (b) measure what it claims to measure, (c) fail-LOUD
   not fail-quiet on its own internal errors? Per integrity-guardian § 7.
2. **Canonical-source delegation** — does any new pipeline code re-encode logic
   that lives in a canonical source (COST_SPECS, SESSION_CATALOG,
   ASSET_CONFIGS, GOLD_DB_PATH, orb_utc_window)? Per institutional-rigor § 4.
3. **Silent failure surfaces** — `except Exception: return []` patterns;
   `if x is None: continue` without explicit reason; NaN/NaT/pd.NA treated as
   `is None`. Per institutional-rigor § 6.
4. **Look-ahead bias guards** — any new feature column or transform must NOT
   leak future bars into a same-day predictor. E2 break-bar look-ahead registry
   (29 entries, 24 TAINTED) is the canonical example.
5. **Schema/JOIN integrity** — `daily_features` has 3 rows per (day, symbol)
   for orb_minutes ∈ {5,15,30}. Triple-join trap is mandatory. CTE/subquery
   guard for non-ORB-specific columns. Per `.claude/rules/daily-features-joins.md`.
6. **Hardcoded check counts** — never `"all 17 checks"`; always compute
   dynamically. Drift count is volatile.
7. **Magic numbers** — every literal threshold needs `# rationale: <source>`
   or `@research-source` annotation. The Batch 4 extension covers
   `account_hwm_tracker.py` + `pre_session_check.py`; pipeline files have
   their own audit history.

## Disconfirming Checks (run BEFORE accepting any commit as clean)

```bash
# 1. Verify drift-check count is dynamic (not hardcoded)
grep -n "131\|132\|all .* checks\|len(CHECKS)\|total_checks" pipeline/check_drift.py | head -20

# 2. Verify no inlined canonical values
grep -rn "COST_SPECS\|SESSION_CATALOG\|ACTIVE_ORB_INSTRUMENTS\|GOLD_DB_PATH" pipeline/ | head -30
grep -rn "/tmp/gold\.db\|c:\\\\db\\\\gold\.db" pipeline/ scripts/  # MUST return zero hits

# 3. Verify daily_features JOIN guard
grep -n "orb_minutes" pipeline/build_daily_features.py | head -20

# 4. Verify orb_utc_window is the single source for ORB window timing
grep -rn "orb_utc_window\|break_ts.*fallback\|ORB.*end.*compute" pipeline/ trading_app/ | head -20

# 5. Verify each new drift check has an injection test
ls tests/test_pipeline/test_check_drift.py
grep -n "def test_.*injection\|def test_.*fires\|inject_violation" tests/test_pipeline/test_check_drift.py | head

# 6. Verify GARCH dependency check landed (canonical example — should be PASS)
grep -n "check_garch_dependency_importable\|arch" pipeline/check_drift.py | head
```

## Per-Session Protocol (from Batch 4 — battle-tested)

1. **Stay on Opus 4.7.** If you see this resume file under Sonnet, STOP and tell Josh to flip the model.
2. Run scope command — get actual diff, not cached summary.
3. For each pipeline commit: PREMISE → TRACE → EVIDENCE → VERDICT.
4. Implement findings same-session per CLAUDE.md mandate.
5. Run `python pipeline/check_drift.py` (target 132/132 — Batch 4 baseline + any new checks added in Batch 5).
6. Run `python -m pytest tests/test_pipeline/ -x -q` first; expand to `tests/test_trading_app/` if any cross-module change.
7. Write commit message to `tmp/batch5-commit-msg.txt` (NEVER PowerShell here-string — `@` leaks).
8. `git add` only scope-locked files; `git commit -F tmp/batch5-commit-msg.txt`.
9. Update this resume file post-commit with hash + outcome summary.

## Bias Guards (do NOT skip on resume)

1. **Drift checks are themselves code** — they have bugs. The Batch 4 NQ-mini
   check missed the empty-dict edge case until finalization caught it. Every
   new drift check needs an injection test that proves it fires on a known
   violation. "It passes 132/132" is necessary but not sufficient.

2. **Pipeline silent failures are the worst class** — they contaminate every
   downstream layer without warning. The GARCH ImportError swallow + the
   E2 break-bar look-ahead are the canonical examples. Treat every
   `except Exception` block as guilty until proven loud.

3. **"It already passed CI"** is not a verdict. CI ran on the PR diff; this
   audit is asking different questions (canonical delegation, magic-number
   rationale, fail-loud invariants). The audit can FAIL clean CI.

4. **One-way pipeline → trading_app dependency** — verify no new pipeline
   code imports from `trading_app/`. That direction is forbidden per
   CLAUDE.md § Architecture.

5. **Volatile-data rule** — never cite cost/session/instrument counts from
   memory or docs. Query live: `pipeline.cost_model.COST_SPECS`,
   `pipeline.dst.SESSION_CATALOG`, `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`.

## Commit Message File Pattern (REQUIRED — battle-tested)

```
1. Write tmp/batch5-commit-msg.txt
2. git add <scoped files>
3. git commit -F tmp/batch5-commit-msg.txt
```

PowerShell here-strings leak `@` — verified twice in Batch 2 attempts, once in
Batch 4. File pattern is the only reliable path.

## Audit-gate exemption decision tree

Per `.claude/rules/adversarial-audit-gate.md`:

- **DEFENSE-ONLY** (drift check added; no truth-layer behavior change; no new
  failure mode introduced) → exempt with explicit one-liner in commit body.
  Example: Batch 4 NQ-mini drift check.
- **TRUTH-LAYER FIX** (changes how a value is computed, persisted, or read on
  the live path) → adversarial audit MANDATORY before next phase commits.
  Dispatch `evidence-auditor` subagent post-commit.
- **MECHANICAL** (rename, comment, doc-only, test-only) → exempt by class.

If in doubt, dispatch the audit. The C1 kill-switch race (iter 174 F4 fix) is
the canonical proof that single-agent self-review is insufficient on
exposure-creating paths.

## Resume Instructions (post-/clear)

After `/clear`:

1. Re-read this file: `docs/runtime/stages/code-review-catchup-batch5-resume.md`
2. Confirm on **Opus 4.7** — `/model opus` if not.
3. Confirm current `main` is at `eafb7e0f` (Batch 4 commit) or later: `git log -1 --format=%h`.
4. Run the scope command above to get actual commit list (not cached).
5. Begin commit-by-commit review starting with the oldest in scope.
6. Same protocol: PREMISE → TRACE → EVIDENCE → VERDICT, fix in-session, commit `-F`.

## Remaining Plan Batches After This One

- **Batch 6** — Tests (77 test-touching commits since 2026-04-24; focus on additions not modifications; verify each new test exercises the new code path and is not a tautology).
- **Deferred cleanup batch** — pyright cluster (now includes `pipeline/check_drift.py` cluster surfaced during Batch 4) + BrokerDispatcher dead-class + verify_bracket_legs ID-heuristic + SSE lazy-stop + NQ-mini Stage 2 wiring.

## State at handoff (2026-05-15 PM)

- `main` HEAD: `eafb7e0f` (Batch 4 commit).
- Drift baseline: 132/132 (0 skipped, 20 advisory).
- pytest scope_lock tests: 480p/1s.
- Working tree at handoff: clean on Batch 5 scope; pre-existing unmodified
  files (`.claude/agents/*.md`, `.claude/rules/subagent-budget.md`,
  `docs/runtime/session-checkpoint-2026-05-14-go-live.md`,
  `resources/Harris_*`, `tmp/`) are NOT in Batch 5 scope and were untouched.
- `/clear` next; resume from this file on fresh Opus 4.7 session.

## Batch 5 Outcome (2026-05-15)

All 8 target commits reviewed under PREMISE → TRACE → EVIDENCE → VERDICT.

| Commit | Subject (truncated) | Verdict |
|---|---|---|
| `b98df2ec` | pyright-clean build_bars_5m via `_scalar` helper | CLEAN |
| `c68ecc3c` | ASCII-clean CHECKS labels + binding regression-guard | CLEAN |
| `115cccf8` | verdict-vocab runner-crash + intersection-order + layout-lock | CLEAN |
| `4a219b0b` | verdict-token vocabulary doctrine + binding parity check | CLEAN |
| `9633fee6` | Chordia threshold drift check + prereg authoring guidance | **FINDING (MED) — fixed in-session** |
| `b1068bac` | MinBTL K-budget gate + `/nogo` command | CLEAN |
| `b700d4ad` | registry-driven routine-TBBO slippage + coverage drift gate | CLEAN |
| `d2b3ba5b` | SR-monitor current_sr_stat fields + stale-pause advisory | CLEAN |

**One MEDIUM finding closed in-session:**

- `check_chordia_result_threshold_matches_prereg` shipped (commit `9633fee6`)
  with ZERO injection tests — violates the immune-system-immune-system rule
  ("check the checker"). Fix:
  - Added `TestChordiaResultThresholdMatchesPrereg` (6 cases) covering
    live-clean tripwire, binding-date mismatch, pre-sentinel advisory routing,
    matching-threshold no-false-positive, prereg-without-result-MD silent skip,
    non-chordia-prereg silent skip.
  - Surfaced LOW operator-clarity bug: violation message used Python float
    repr `3.0` instead of MD-literal `3.00`. Fixed by formatting thresholds
    as `:.2f` so the message mirrors the prereg / result file literal text.

**Verification:**
- Drift: 132/132 PASS, 0 skipped, 20 advisory (baseline maintained).
- pytest: 1424 PASS (1418 baseline + 6 new injection tests = exact count).
- Self-review: float-repr bug surfaced by injection test itself — proof
  the test class actually probes the check rather than being tautological.
- No scope expansion mid-flight; only `tests/test_pipeline/test_check_drift.py`
  and `pipeline/check_drift.py` modified.
