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

## Adversarial-Audit Gate — IMPLEMENTER SELF-REVIEW (2026-06-05); INDEPENDENT GATE PENDING

> **Provenance correction (2026-06-05, Stage 1):** an earlier draft of this section
> claimed an "independent `evidence-auditor` subagent" was dispatched and the gate was
> "CLOSED CONDITIONAL." **No independent subagent was actually dispatched** when that text
> was written — the analysis below was produced by the *implementing* context reviewing its
> own fix. Per `.claude/rules/adversarial-audit-gate.md`, implementer self-review is
> necessary but **NOT sufficient**, and self-review may not be recorded as the gate artifact.
> This section is therefore reframed as **implementer self-review (Stage-1 finding + fix)**;
> the adversarial-audit gate **remains OPEN pending a genuinely independent reviewer**, which
> is dispatched as Stage 1c of the C11 path-to-live plan. The technical findings below stand
> as the self-review record and as input to that independent pass.

**Reviewer:** implementing context (self-review). **Status:** PENDING independent audit.
**Worktree:** `C:/Users/joshd/canompx3-c11-s1-audit` @ `session/joshd-c11-s1-bracket-audit`
off `origin/main` `f680772b`; audit target `9b3fc530` on HEAD. Baseline re-run:
`TestBracketOrders` + `TestF4BracketNakedPosition` = 16 passed (pre-fix).

**Self-review verdict: CONDITIONAL.** The scoped double-apply parity fix appears correct; one
latent (believed-currently-unreachable) guard defect was surfaced and fixed in this pass.
**This verdict is not gate-clearing — the independent reviewer must confirm or refute it.**

Six falsification points — self-review results (to be independently re-checked):
1. Both live bracket paths route through `_bracket_stop_distance` — PASS (no third path).
2. Input-shape correctness — **CONDITIONAL.** `_bracket_stop_distance`
   (`session_orchestrator.py:2166`) used a truthiness guard `if event_risk:` instead of
   `if event_risk is not None`. A `0.0` risk-points value is falsy and would silently
   route to the `median * stop_multiplier` fallback (a guessed bracket) instead of
   failing closed. **Self-review claim (independent reviewer to confirm):** `0.0` is
   believed NOT reachable through `ExecutionEngine` today — every event `risk_points=`
   assignment (`execution_engine.py:1037,1255,1430`) is downstream of an `if risk_points
   <= 0:` guard (`:897,1115,1289`), so only `> 0` values should be emitted. Internal
   inconsistency confirms the defect: the SAME field is read with the correct `is not None`
   guard at `session_orchestrator.py:2452` and `:2475`. Claimed latent, not exploitable;
   future-coupling risk. **Open question for the gate:** is `risk_points=None` (absent,
   not zero) reachable? If so the "absent → median fallback" path is still a live guess.
3. Bidirectional replay↔live parity — PASS.
4. No scope leak (exit actual-R / journal fallback / naked-position fail-closed) — PASS.
5. Multi-event-per-bar — PASS (helper is static/stateless; no cross-event state).
6. Tight-stop ordering — PASS (`event.risk_points` is the already-effective distance).

**Highest-priority fix (recommendation only, NOT implemented — Tier B capital path):**
`session_orchestrator.py:2166` change `if event_risk:` → `if event_risk is not None and
event_risk > 0:` so the zero/negative case fails closed instead of guessing a median
bracket. Add a test injecting `event.risk_points = 0.0` asserting entry blocked / flatten.

**Do-not-touch (audit-verified correct):** the `_bracket_merged` pre-fill safety gate at
`session_orchestrator.py:2672-2681` (pops position before broker submit → fail-closed).

**Gate disposition (self-review):** the parity defect that motivated `9b3fc530` appears
addressed, and the newly-surfaced guard defect was FIXED in this same Stage-1 pass (operator
directive: do not defer a live-order fail-closed guard). **The gate is NOT closed** — an
independent evidence-auditor (Stage 1c) must clear the combined post-fix live-path state
before C11 wiring proceeds. Stage 2 (cap_x0.80 pre-registration) is docs-only and may be
written in parallel, but the $2,038.84 baseline math is only "self-review-sound" until the
independent pass confirms it.

### Guard fix (Stage-1, applied 2026-06-05)
`_bracket_stop_distance` (`trading_app/live/session_orchestrator.py`) now distinguishes
'risk field absent' (median fallback legitimate) from 'risk field present but <= 0'
(returns None -> caller fails closed / emergency-flattens). The old truthiness guard
`if event_risk:` is replaced by `if event_risk is not None: ... if event_risk > 0`. The
`_submit_bracket` docstring sub-path list now records the present-but-zero case.

RED->GREEN test added: `TestBracketOrders::test_bracket_stop_distance_zero_event_risk_
does_not_guess_median` — asserts the helper returns the median (6.0) only when the field
is absent, and returns None for risk_points 0.0 and -2.0. Verified RED before
(`assert 6.0 is None` failed) and GREEN after.

MEASURED: `TestBracketOrders` + `TestF4BracketNakedPosition` = 17 passed (16 baseline +
new); `test_account_survival.py` = 22 passed (parity side, no sim↔live asymmetry
introduced); ruff + py_compile clean on both changed files. Live-order fail-closed
hardening only; no schema, no profile, no cap, no allocator change.

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
