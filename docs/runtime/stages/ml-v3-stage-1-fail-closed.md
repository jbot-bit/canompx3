---
mode: IMPLEMENTATION
slug: ml-v3-stage-1-fail-closed
task: Replace silent fail-open ML path with env-gated three-state system (disabled / enabled+models / enabled+missing=RuntimeError)
agent: Claude Code (institutional ML V3 sprint)
created: 2026-04-11
parent_plan: docs/audit/ml_v3/2026-04-11-stage-0-verification.md
---

# Stage: ML V3 — Stage 1: Fail-closed ML gate

## Scope Lock

- trading_app/ml/config.py
- trading_app/ml/predict_live.py
- trading_app/live/session_orchestrator.py
- trading_app/paper_trader.py
- tests/test_trading_app/test_ml/test_predict_live.py
- tests/test_trading_app/test_session_orchestrator.py
- tests/test_trading_app/test_paper_trader.py

## Blast Radius

Touches ML init paths only. Downstream: `trading_app/execution_engine.py` consumes `ml_predictor` parameter unchanged — None path already handled correctly (verified in Stage 0 audit of predict_live.py:257-261). No ExecutionEngine changes. No API change at the caller level beyond the new `require_models` kwarg (defaults to False, backwards-compatible). Current production behavior unchanged with `ML_ENABLED` unset — bot currently has no models, current fail-open path and new `ml_predictor=None` path both result in "every trade passes through ExecutionEngine.ml gate." New RuntimeError path fires ONLY under explicit `ML_ENABLED=1` opt-in. Tests: existing `test_predict_live.py` (7 fail-open tests, unchanged — backwards compat verified), add 4 new fail-closed tests to same file; existing `test_session_orchestrator.py` (gets 2 new tests); existing `test_paper_trader.py` (gets 1 new test). Drift checks: `pipeline/check_drift.py` ML bundle validation at lines 2925-2994 is unaffected — still runs on committed models, independent of runtime gate. Memory / HANDOFF: will update post-merge. No schema changes, no DB migrations, no config file format changes.

## Purpose

Verified in Stage 0 (`docs/audit/ml_v3/2026-04-11-stage-0-verification.md` § Q5): current `LiveMLPredictor.predict()` returns `(0.5, True, 0.5)` silently when models are missing. Session orchestrator wraps init in try/except and sets `self._ml_predictor = None` on failure, fail-open. This is ambiguous — no distinction between "ML intentionally off" and "ML failed to load, flying blind." Also: `models/ml/` directory does not exist on disk at all; `MODEL_DIR` is referenced but never asserted.

Fix: `ML_ENABLED` env var in `trading_app/ml/config.py`. Three deterministic states:

| ML_ENABLED | models/ml/ state | behavior |
|---|---|---|
| unset / 0 | any | `self._ml_predictor = None`, log INFO, ExecutionEngine takes all trades |
| 1 | missing OR empty OR no model for this instrument | RuntimeError at orchestrator startup — refuse to boot |
| 1 | model present | Current path: load, gate trades |

Policy lives in callers (session_orchestrator, paper_trader), enforced via `LiveMLPredictor(..., require_models=True)` opt-in parameter. Existing tests that construct `LiveMLPredictor(db_path="dummy.db")` with synthetic bundles keep working because default `require_models=False`.

## Acceptance Criteria

1. Tests pass — new tests for three states + existing `tests/test_trading_app/test_ml/` suite stays green. Command output shown.
2. Dead code swept — `grep -r "fail-open"` in touched files shows updated or removed comments; no stale "fail-open" references in the init paths.
3. `python pipeline/check_drift.py` passes at current baseline (87 passing, 0 failing, 7 advisory).
4. Self-review scan — no silent exception handlers, no re-encoded logic (delegates to LiveMLPredictor existing behavior), explicit log lines on all paths, no `_ = x` lint silencers.

## Rollback Plan

If any acceptance criterion fails and cannot be fixed within this stage: `git reset --hard HEAD~N`, re-open this stage file with the failure mode documented, do NOT leave partial state.
