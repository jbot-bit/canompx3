---
task: deferred-bucket-audit
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/broker_dispatcher.py
  - trading_app/live/tradovate/order_router.py
  - trading_app/live/rithmic/order_router.py
  - trading_app/live/session_orchestrator.py
  - trading_app/live/bot_dashboard.py
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift_context.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_tradovate.py
  - docs/runtime/stages/deferred-bucket-audit.md
---

# Deferred-Bucket Audit (Batches 1-5 latent findings)

**Stage:** PENDING -> PASS 1.
**Model:** Opus 4.7 (per CLAUDE.md and Batch 6 chunk-1 outcome).
**Last commit on main:** verify on resume via `git rev-parse HEAD`.
**Date opened:** 2026-05-15.

## Blast Radius

- `trading_app/live/broker_dispatcher.py` — Pass 1 (delete candidate, zero prod callsites confirmed via grep).
- `tests/test_trading_app/test_tradovate.py` (22 BrokerDispatcher refs) — Pass 1 collateral; delete those tests with the class.
- `trading_app/live/tradovate/order_router.py` — Pass 5 READ-ONLY audit of `verify_bracket_legs`; capital-class surface.
- `trading_app/live/session_orchestrator.py`, `trading_app/live/bot_dashboard.py`, `pipeline/check_drift.py` — Pass 4 Pyright cluster; type-only changes, no behavior.
- `tests/test_pipeline/test_check_drift_context.py` — Pass 4 unused fixture bindings.
- `tests/test_trading_app/test_session_orchestrator.py:1720+` — Pass 6 F8/F6/F2 source-string-grep -> behavioral refactor.
- SSE path (TBD on Pass 3 — locate first; deferred from Batch 2/3).
- Reads: gold.db (read-only), git log, source files above. Writes: scoped files only. No DB writes.

## Passes (cheapest -> most expensive)

1. **BrokerDispatcher** — DELETE (zero prod callsites; ~250 dead test lines).
2. **NQ-mini Stage 2** — DECIDE (schedule on named branch or DROP). No code change.
3. **SSE lazy-stop** — IMPLEMENT ref-counted stop on last unsubscribe.
4. **Pyright cluster** — RESOLVE per-file, treat each finding individually.
5. **verify_bracket_legs** — READ-ONLY AUDIT (capital-class; defer fix to own stage).
6. **F8/F6/F2 test refactor** — REFACTOR three source-string-grep tests to behavioral via existing `build_orchestrator()` + `FakeBrokerComponents`.

## Per-pass protocol

1. Opus 4.7.
2. PREMISE -> TRACE -> EVIDENCE -> VERDICT.
3. Implement same-session per CLAUDE.md.
4. `python pipeline/check_drift.py` after each pass; target 132/132.
5. `python -m pytest tests/test_pipeline/ tests/test_trading_app/ -x -q` after each pass; target 1426+ PASS minus deleted tests.
6. One commit per pass.

## Bias guards

- Don't extend scope mid-pass. Note new items, continue.
- `verify_bracket_legs` is capital-class — audit before touching. Own stage if a bug surfaces.
- Pyright "cluster" is a roll-up — treat each finding individually.
- NQ-mini Stage 2 is a DECISION, not implementation.

## Stop conditions

- Capital-class finding outside scope -> STOP, open named stage.
- Drift check breaks -> STOP, fix-or-revert before next pass.
- Context past Tier 3 (~80K) -> STOP after current pass, commit, `/clear`.

## Verification (end of stage)

1. Drift: `python pipeline/check_drift.py` -> 132/132 PASS.
2. Tests: full pytest green (baseline minus deleted BrokerDispatcher tests).
3. Dead-code: `grep -rn "BrokerDispatcher" trading_app/ tests/` -> zero.
4. Pyright on 4 scope files -> zero errors.
5. Deferred bucket = 0 items, or each remaining item has its own named follow-up stage file.

---

## Outcomes

### Pass 1 — BrokerDispatcher DELETE (DONE, commit dccb108e)

- Deleted `trading_app/live/broker_dispatcher.py` (215 lines).
- Deleted `TestBrokerDispatcher` from `test_tradovate.py` (-261 lines, 22 tests).
- Updated two stale docstrings in tradovate/rithmic order_routers.
- Drift 132/132. pytest 4522 pass / 10 skip. No regressions.

### Pass 2 — "NQ-mini Stage 2" DROP (decision-only, no code change)

**Disambiguation surfaced during audit:** the "NQ-mini" name conflates
two different things. Recording both so they don't recollide later.

- **Code-side "Stage 2"** = wiring `resolve_execution_symbol(profile,
  strategy_symbol)` into `trading_app/live/session_orchestrator.py` and
  `webhook_server.py` so a profile's `execution_symbol_map={"MNQ": "NQ"}`
  actually substitutes the symbol before the broker order is built.
  Stage 1 (the dataclass field) landed in PR #158 (commit `8bef5eb1`,
  2026-04-27). Stage 2 (the callsites) never landed.
- **Plan-side "Phase 4"** = `docs/audit/2026-04-15-topstep-scaling-reality-audit.md:242,254,268`:
  self-funded $50K AMP/EdgeClear account, **1 NQ mini instead of 10 MNQ
  micros = ~77% commission cut** (`memory/mini_vs_micro_commission_fix.md`).
  Reaffirmed `chatgpt_bundle/CANONICAL_VALUES.md:121,130`. This is the
  bigger-contracts plan the user asked about. Phase 4 is 12+ months out
  from Phase 1 (current live state, 4 MNQ lanes on TopStep XFA as of
  2026-05-14).

**Need it eventually?** YES — "Self-funded with 1 NQ mini is the single
biggest revenue-per-unit-effort lever post Phase 4" (audit doc:268). The
plan to go bigger contracts is alive, just gated on Phase 1-3 first.

**Ready for it now?** NO. Current state: Phase 1 only. Blocking:
(a) self-funded broker integration (AMP/EdgeClear via Rithmic — not built),
(b) Phase 1-3 must prove edge live first, (c) the Stage 2 wiring itself.

**Why DROP from this audit (rather than wire Stage 2 now):**

1. The fail-closed drift check `check_nq_mini_substitution_wired_or_unused`
   (`pipeline/check_drift.py:7537`) already prevents the capital-class
   silent-mis-route. It fails the build the moment any profile populates
   `execution_symbol_map=` without a callsite under `trading_app/live/`.
2. No ACCOUNT_PROFILES row currently populates the map — Stage 1 is
   dormant. There is no live trap.
3. Wiring Stage 2 now = building for hypothetical 12-month-out work,
   exactly what CLAUDE.md § "Don't design for hypothetical future
   requirements" forbids.
4. When Phase 4 arrives, populating `execution_symbol_map` is the natural
   trigger — and the drift check refuses the commit until callsites exist.
   The forcing function is already in place.

**Side note kept for posterity:** the bigger-contracts plan (1 NQ mini
self-funded, Phase 4) is NOT dropped — it's still the canonical Phase 4
target. What's dropped is *pre-wiring code for it now*. The drift check
is the bridge.

### Pass 3 — SSE lazy-stop IMPLEMENT (DONE)

**Gap closed:** `_sse_cancel_watchers` was only called at FastAPI lifespan
shutdown (`bot_dashboard.py:102`). When the last browser tab closed, the
five file-polling watchers (heartbeat 1s, state 0.5s, signals JSONL tail,
alerts tail, bars 2s) kept polling until the dashboard process itself
exited — the "zombie SSE streams" surface.

**Implementation:**
- New async helper `_sse_lazy_stop_if_idle()` (`bot_dashboard.py:2731`)
  cancels watchers only when `subscriber_count == 0` AND `_sse_tasks` is
  non-empty. Mirrors the existing lazy-start pattern.
- SSE endpoint finally-block (`bot_dashboard.py:2974`) now awaits
  `_sse_lazy_stop_if_idle()` after `_sse_broker.unsubscribe(queue)`.
- Two new tests in `test_bot_dashboard_sse.py`:
  - `test_sse_lazy_stop_cancels_watchers_when_last_subscriber_unsubscribes`
    — exercises the ref-count: first unsubscribe leaves watchers running
    (one subscriber survives), second (last) unsubscribe cancels all 5.
  - `test_sse_lazy_stop_noop_when_no_watchers` — TOCTOU-rejection-path
    safety; lazy-stop called before lazy-start must not raise.

**Verification:** drift 132/132, dashboard suites 40/40 pass (was 38/38;
+2 new tests).

**Side note:** lazy-stop is sync-on-subscriber-count, not async-task-based.
Two concurrent unsubscribes from the same loop are serialized by Python's
event-loop scheduling — no race within a single FastAPI worker. The
single-uvicorn-worker assertion in `run_dashboard()` upholds that
invariant; multi-worker would need a different model.
