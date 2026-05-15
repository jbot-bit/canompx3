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
