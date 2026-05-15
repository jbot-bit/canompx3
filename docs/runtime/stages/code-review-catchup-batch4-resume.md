---
task: code-review-catchup-batch4
mode: IMPLEMENTATION
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/live/account_hwm_tracker.py
  - trading_app/live/risk_manager.py
blast_radius: |
  Batch 4 reviews 4 commits touching HWM tracking, allocator pathing, NQ-mini
  symbol substitution, and magic-number rationale audit. Capital impact is
  HIGH (HWM is the drawdown-tier gating, symbol-substitution can mis-route
  orders to wrong instrument). Read-only review with same-session
  implementation per CLAUDE.md mandate.
---

# Code Review Catch-Up — Batch 4 Resume Point

**Stage:** Batches 1-3 DONE, Batch 4 PENDING
**Model required:** Opus 4.7 — Sonnet PROHIBITED per plan (HoldToKill + BrokerDispatcher were Sonnet misses)
**Last commit:** `337c2ed4` — `[code-review batch 3] CSRF: close PYTEST_CURRENT_TEST env-var-only bypass`
**Date:** 2026-05-15

## Batches 1-3 Outcomes (DONE)

- **Batch 1** (pre-existing): HoldToKill canonical-field fix `f75157fe`, CSRF middleware `db8df761`, A.6 ORB caps per-aperture `82510553`
- **Batch 2** (`17d8b5cd`): 2 operator-clarity findings on session_orchestrator (signal_record type rename, bracket-cancel warning split). No CRIT/HIGH on capital-class commits.
- **Batch 3** (`337c2ed4`): 1 defense-in-depth finding — CSRF `PYTEST_CURRENT_TEST` env-var-only bypass closed (now requires `pytest` in sys.modules). SSE lifecycle, HoldToKill, poller retirement, localhost binding, session-isolation auto-recover all PASS.

## Batch 4 Outcomes (DONE)

**Date:** 2026-05-15
**Commit:** `b544f4e8` (initial hash `3387a046` pre-amend) — `[code-review batch 4] allocator + HWM constants + NQ-mini drift defense`
**Verification:** drift 132/132, pytest 480p/1s on scope_lock tests, imports OK

Findings closed (4):

1. **HIGH (NQ-mini fail-open trap)** — new drift check `check_nq_mini_substitution_wired_or_unused` in `pipeline/check_drift.py`. Guards against any future `ACCOUNT_PROFILES` row populating `execution_symbol_map=` before Stage 2 wires `resolve_execution_symbol()` callsites in `trading_app/live/`. Empty-dict case treated as identity (matches loader semantics).
2. **MEDIUM (allocator path-helper class-bug)** — `_normalize_writable_path` consolidated to `trading_app.lane_allocator.normalize_writable_path` (public). `pre_session_check` imports it. `prop_profiles` keeps an inline copy with documented NOTE explaining the module-load reentrancy (validate_dd_budget at L1254 → load_allocation_lanes → lane_allocator while lane_allocator partially loaded).
3. **MEDIUM (drift coverage gap)** — `check_magic_number_rationale` scope extended to `account_hwm_tracker.py` + `pre_session_check.py`.
4. **LOW (duplicated 30-day constant)** — `pre_session_check._INACTIVITY_BLOCK_DAYS` now derives from `account_hwm_tracker.STATE_STALENESS_FAIL_DAYS` (new public alias).

Self-review trade-offs (called out in commit body):
- Empty-dict `execution_symbol_map={}` is identity (matches `AccountProfile.__post_init__`).
- AST matcher is Name-form only; Attribute-form `prop_profiles.AccountProfile(...)` not targeted.

## Deferred Findings (do NOT lose)

Stack for a separate cleanup batch after Batch 6 (or sooner if Josh prioritizes — he flagged the pyright cluster as "stop sweeping under the rug"):

1. **BrokerDispatcher dead-class** — 41 lines of API-parity infrastructure with zero production callsites confirmed on current main. Per `memory/feedback_code_review_dead_class_detection.md`. Either wire it or delete it.
2. **Pre-existing pyright errors cluster** — Josh's explicit ask: "we need to fix them after". Pattern:
   - `trading_app/live/session_orchestrator.py:2942/2959` — `Optional[None]` member access (cancel, query_order_status without None guard)
   - `tests/test_trading_app/test_session_orchestrator.py` — test fixture types bypass BrokerAuth / BrokerRouter / BrokerPositions / Bar protocols
   - `trading_app/live/bot_dashboard.py:369/412/430/989-990` — same `object | None` / `Popen` / `ConvertibleToFloat` cluster
   - Class-bug pattern — proper test fixture base classes + None-guards, not patching. Per `institutional-rigor.md` §3.
3. **Tradovate verify_bracket_legs ID-heuristic mis-attribution risk** — if two entries land on same contract back-to-back, higher-ID-than-entry heuristic could mis-attribute legs. Mitigated by short verification window. Audit before any back-to-back paper-trade activation.
4. **SSE lazy-stop on last-subscriber-disconnect** — `_sse_start_watchers` is lazy-start but watcher tasks never decrement-and-stop when subscriber count returns to 0. Not a memory leak (bounded set), but wasted CPU polling. Add reference-counted stop in `unsubscribe` if/when CPU becomes a concern.
5. **NQ-mini Stage 2 wiring (NEW from Batch 4)** — separate branch. Touches `trading_app/live/session_orchestrator.py:2317` (the `build_order_spec` call site), `trading_app/live/webhook_server.py`, at least one populated `ACCOUNT_PROFILES` row, and an integration test exercising the substitution + qty-divisor math. Action-queue id: `nq_mini_stage2_wiring_2026_05_15`. Driver memo: `memory/mini_vs_micro_commission_fix.md` (~77% commission reduction = $26K→$52K/yr per contract). Exit criterion: drift check `check_nq_mini_substitution_wired_or_unused` continues PASS after a profile populates `execution_symbol_map=` (proves Stage 2 callsite landed).

## Batch 4 Scope (RESUME HERE)

**Scope command (run first to refresh diff — DO NOT trust the commit list below blindly):**
```
git diff $(git log --since="2026-04-24" --format="%H" origin/main -- trading_app/prop_profiles.py trading_app/live/account_hwm_tracker.py | tail -1)^..HEAD -- trading_app/prop_profiles.py trading_app/live/account_hwm_tracker.py trading_app/live/risk_manager.py
```

**Confirmed commits in scope (2026-05-15 audit):**

- `8a40a142` HWM: Stages 1-4 — HWM warning tier + tracker integrity + orchestrator + inactivity window (PR #129) — **HIGH** capital
- `db8a7666` fix(allocator): harden lane allocation pathing — **HIGH** capital
- `8bef5eb1` feat(profiles): NQ-mini execution-layer Stage 1 — AccountProfile symbol-substitution (PR #158) — **HIGH** capital (mis-route risk)
- `170b6085` feat(drift): Pass Three magic-number rationale audit + retag constants (PR #147) — judgment

Note: `risk_manager.py` has only one pre-range commit (`2a632285`) — no changes since 2026-04-24 needing review.

## Batch 4 Focus Areas

1. **HWM inactivity window threshold** — is the threshold a magic number or a named constant? Does it have a drift check? Capital impact: HWM tier transitions gate withdrawals and drawdown limits.
2. **HWM tracker integrity** — what happens if HWM file is corrupt / missing / partially written? Fail-closed or fail-open?
3. **Symbol-substitution contract (NQ-mini)** — can a misconfigured AccountProfile silently route MES orders to NQ (or vice versa)? Verify fail-closed validation at substitution time, not at order-submit time.
4. **Allocator lane allocation pathing harden** — `db8a7666` changed pathing across live consumers. Verify all downstream consumers (session_orchestrator, pre_session_check, lane_allocator) read from canonical path; no stale-path fallback.
5. **Magic-number rationale audit** — did it actually retag constants with rationale, or just add comments? Verify drift check enforces.

## Disconfirming Checks

Run BEFORE accepting any commit as clean:

```bash
# HWM inactivity window
grep -n "inactivity\|INACTIVITY\|inactive_hours\|_HOURS\|_THRESHOLD" trading_app/live/account_hwm_tracker.py
grep -rn "HWM" pipeline/check_drift.py

# Symbol substitution
grep -n "symbol_substitution\|substitute_symbol\|symbol_map\|nq_mini\|NQ_MINI" trading_app/prop_profiles.py
grep -n "symbol_substitution\|substitute" trading_app/live/

# Allocator pathing — all consumers read canonical
grep -rn "lane_allocation\.json\|LANE_ALLOCATION" trading_app/ pipeline/

# Magic-number retag
grep -rn "@drift-rationale\|@const-source\|# rationale:" trading_app/prop_profiles.py | head
```

## Per-Session Protocol (from plan)

1. **Stay on Opus 4.7.** If you see this resume file under Sonnet, STOP and tell Josh to flip the model.
2. Run scope command — get actual diff, not cached summary
3. For each capital-class commit: PREMISE → TRACE → EVIDENCE → VERDICT
4. Implement findings same-session per CLAUDE.md mandate
5. Run `python pipeline/check_drift.py` (target 131/131)
6. Run `python -m pytest tests/test_trading_app/ -x -q` (3118 baseline)
7. Commit with `[code-review batch 4]` prefix using `-F tmp/batch4-commit-msg.txt` (NOT `-m heredoc` — PowerShell leaks `@` into commit subjects)
8. Update plan, mark Batch 4 DONE with date

## Bias Guards (do NOT skip on resume)

1. **HWM is capital-class** — drawdown tier transitions gate real money. Treat every threshold as a potential silent-failure surface.
2. **Symbol-substitution mis-route is HIGH capital** — wrong-instrument fills can blow account limits. Verify the validation path is BEFORE order submission, not after fill.
3. **"Magic-number rationale audit" sounds like doc-only** — verify it actually changed behavior (drift check) or admit it's pure-doc and rate it judgment-only.
4. **`db8a7666` allocator pathing** — class-bug class. Per `feedback_allocator_gate_class_pattern_fail_open.md`, allocator-class code must have paired validator + apply gate + drift check. Verify all three landed.

## Commit Message File Pattern (REQUIRED)

```
1. Write tmp/batch4-commit-msg.txt
2. git add <scoped files>
3. git commit -F tmp/batch4-commit-msg.txt
```

PowerShell here-strings leak `@` — verified twice in Batch 2 attempts. File pattern is the only reliable path.

## Resume Instructions (post-/clear)

After `/clear`:
1. Re-read this file: `docs/runtime/stages/code-review-catchup-batch4-resume.md`
2. Confirm on **Opus 4.7** — `/model opus` if not
3. Confirm current `main` is at `337c2ed4` (Batch 3 commit) or later: `git log -1 --format=%h`
4. Run the scope command above
5. Begin commit-by-commit review starting with `8a40a142` (HWM)
6. Same protocol: PREMISE → TRACE → EVIDENCE → VERDICT, fix in-session, commit `-F`

## Remaining Plan Batches After This One

- **Batch 5** — Pipeline (drift checks + research guards): `b98df2ec`, `c68ecc3c`, `115cccf8`, `4a219b0b`, `9633fee6`, `b1068bac` (verify on main first), `b700d4ad`, `d2b3ba5b`
- **Batch 6** — Tests (77 test-touching commits since 2026-04-24; focus on additions not modifications)
- **Deferred cleanup batch** — pyright cluster + BrokerDispatcher dead-class + verify_bracket_legs ID-heuristic + SSE lazy-stop
