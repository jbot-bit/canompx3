# Bracket Risk Parity Closeout - topstep_50k_mnq_auto

Date: 2026-06-03

## Scope

The question: does the live broker-bracket path double-apply `stop_multiplier` to an
already-effective `event.risk_points`, diverging from the replay/risk convention? Covers
only the bracket stop-distance defect and its fix. Explicitly out of scope: any
`live_config`/allocation/profile/`stop_multiplier`/lane/session/RR/schema change, C11
clearance, attribution clearance, and real-broker runtime behavior beyond unit tests.

## Verdict

MEASURED: the scoped replay/live broker-bracket risk-distance defect is fixed.

MEASURED: `topstep_50k_mnq_auto` remains `NO-GO` for live deployment. This pass did not edit `live_config`, allocation, account profile thresholds, `stop_multiplier`, lane/session/RR selection, schema, or promotion state. No live trading command was run.

## Root Cause

MEASURED:
- `ExecutionEngine` emits `event.risk_points` as the already-effective fill-to-stop risk distance after tight-stop handling.
- Clean HEAD failed the red parity tests for `event.risk_points=6.0`, `stop_multiplier=0.75`, long entry `100.0`: expected stop `94.0`, but old live bracket math produced `95.5`.
- Both live broker-bracket construction paths multiplied already-effective `event.risk_points` by `stop_multiplier` again.

## Fix

Implemented in `trading_app/live/session_orchestrator.py`:
- added `_bracket_stop_distance(event, strategy)`;
- when `event.risk_points` exists, use it directly as the effective stop distance;
- when only `strategy.median_risk_points` exists, apply `strategy.stop_multiplier` exactly once, preserving the old median-risk fallback behavior;
- route post-fill `_submit_bracket` and native bracket merge through that helper.

Tests added in `tests/test_trading_app/test_session_orchestrator.py`:
- native bracket merge does not reapply `stop_multiplier` to `event.risk_points`;
- post-fill bracket submit does not reapply `stop_multiplier` to `event.risk_points`;
- native bracket merge applies `stop_multiplier` to median fallback;
- post-fill bracket submit applies `stop_multiplier` to median fallback.

## Live Risk Review

MEASURED local checklist result: `ACCEPT_WITH_RISK` for the scoped diff.

Reviewer-surface checks:
- both broker bracket paths use `_bracket_stop_distance`;
- `_record_exit` actual-R and journal dollar fallback were left unchanged after a scope-leak check;
- naked-position fail-closed behavior remains covered by `TestF4BracketNakedPosition`;
- no profile/allocation/live_config/stop/lane/session/RR change is present in the scoped diff.

## Limitations

UNSUPPORTED (this pass does not establish):
- real broker runtime behavior beyond these unit tests;
- deployability;
- C11 clearing;
- attribution clearing;
- an independent spawned `canompx3_reviewer` clearance, because the subagent tool rules did not authorize spawning without an explicit delegation request. The adversarial-audit gate for this CRITICAL live-order-path change therefore remains OPEN and must run before any live arming.

## Verification

MEASURED passing commands:
- `python -m pytest tests\test_trading_app\test_account_survival.py -q`
  - 21 passed
- `python -m pytest tests\test_trading_app\test_session_orchestrator.py::TestBracketOrders tests\test_trading_app\test_session_orchestrator.py::TestF4BracketNakedPosition -q`
  - 16 passed
- `python -m py_compile trading_app\account_survival.py trading_app\live\session_orchestrator.py tests\test_trading_app\test_account_survival.py tests\test_trading_app\test_session_orchestrator.py`
- `ruff check trading_app\account_survival.py trading_app\live\session_orchestrator.py tests\test_trading_app\test_account_survival.py tests\test_trading_app\test_session_orchestrator.py`
- `python scripts\audits\run_all.py --phase 7`
  - Phase 7 passed: 11 checks clean
- `python scripts\tools\audit_behavioral.py`
  - all 7 checks clean
- `python scripts\tools\audit_integrity.py`
  - all checks clean
- `python -u pipeline\check_drift.py --fast --quiet --skip-crg-advisory`
  - clean, 137 passed, 15 advisories
- `git diff --check`
  - no whitespace errors; CRLF warnings only on pre-existing dirty files

MEASURED non-passing or expected-nonzero commands:
- `python -m pytest tests\test_trading_app\test_account_survival.py tests\test_trading_app\test_session_orchestrator.py -q`
  - timed out after 304 seconds with no summary; focused impacted suites passed.
- `python -m trading_app.account_survival --profile topstep_50k_mnq_auto --no-write-state`
  - expected strict C11 fail remains: operational pass `99.7%`, daily-loss `0.0%`, trailing-DD `0.3%`, historical max observed 90d DD `$2,039`, strict budget `$1,600`.
- `python scripts\tools\live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn`
  - expected nonzero: Criterion 11 state/fingerprint fail, Criterion 12 valid, telemetry `9/30` advisory.
  - current run also reports `Automation: overall=ACTION_REQUIRED` and `CanonMPX_TopstepTelemetry_SignalOnly=FAILED`, advisory for this express/funded profile.

MEASURED environment residue:
- the broad timed-out pytest left PIDs `25268` and `63388`; both resisted `Stop-Process` and `taskkill` with access denied.

## Files

Changed by the fix (landed at `9b3fc530` on origin/main):
- `trading_app/live/session_orchestrator.py` — added `_bracket_stop_distance(event, strategy)`; both bracket paths route through it.
- `tests/test_trading_app/test_session_orchestrator.py` — 4 parity tests (no double-apply on `event.risk_points`; single-apply on median fallback).

Reproduce the parity verdict: `python -m pytest tests/test_trading_app/test_session_orchestrator.py::TestBracketOrders tests/test_trading_app/test_session_orchestrator.py::TestF4BracketNakedPosition -q` (16 passed). C11 fail reproduced via `python -m trading_app.account_survival --profile topstep_50k_mnq_auto --no-write-state`.

## Remaining Blockers

- `C11_CAPITAL_NO_GO_REMAINS` - MEASURED.
- `LIVE_READINESS_C11_BLOCKED` - MEASURED.
- `TOPSTEP_TELEMETRY_AUTOMATION_FAILED` - MEASURED advisory.
- `DIRTY_REPO_RISK` - MEASURED.
- `DEPLOY_CURRENT_BOOK` - UNSUPPORTED.

## Next Best Move

Do not deploy or rank allocation. Commit only the bounded replay/live parity files after operator approval. State refresh is a separate operator decision because this pass used `--no-write-state`.
