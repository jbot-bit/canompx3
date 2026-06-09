---
task: |
  IMPLEMENTATION — App Overhaul STAGE 1 of 5: Single source of preflight truth (cohesion seam).
  The "disjointed" complaint has a precise root cause — FOUR divergent preflight implementations
  across three process boundaries (START_BOT.bat, dashboard subprocess, orchestrator boot,
  orchestrator self-tests). Cohesion = collapse them into ONE shared verdict every surface reads.

  Extract the existing 15-gate engine (PreflightContext@196, CheckResult@218, CheckFn@252,
  PREFLIGHT_CHECKS@965, run-loop@1031-1057 in scripts/run_live_session.py) into one importable
  module `trading_app/live/preflight.py` returning a structured PreflightReport
  (per-gate {id, title, status, message, remediation} + overall + strict_block).

  scripts/run_live_session.py thins to a CLI/printing wrapper importing the module.
  bot_dashboard.py replaces its `_run_preflight_subprocess` parse path with an in-process import
  call; strict_zero_warn becomes a report FIELD (kills START_BOT-vs-dashboard flag divergence).

  BEHAVIOR-PRESERVING. No capital-semantics change. Pinned by existing
  tests/test_scripts/test_run_live_session_preflight.py + new module unit test.
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/preflight.py
  - scripts/run_live_session.py
  - trading_app/live/bot_dashboard.py
  - pipeline/check_drift.py
  - tests/test_scripts/test_run_live_session_preflight.py
  - tests/test_trading_app/test_preflight_module.py
depends: none (upstream seam — Stages 2-5 consume this module)
blocker: none — anchors verified live 2026-06-09; design audited (CONDITIONAL→absorbed)
done_when: |
  1. `python pipeline/check_drift.py` passes (count self-reported).
  2. `pytest tests/test_scripts/test_run_live_session_preflight.py
     tests/test_trading_app/test_preflight_module.py -q` GREEN.
  3. grep proves BOTH run_live_session.py AND bot_dashboard.py import the gate
     logic from trading_app/live/preflight.py — no duplicate gate definitions remain.
  4. Standalone `python scripts/run_live_session.py --preflight` produces identical
     gate count + verdicts as before (behavior-preserving check).
  5. Self-review passed; dead code swept (grep for old _run_preflight_subprocess refs).
not_this_stage: |
  - Gate [12] chicken-and-egg semantics change → Stage 2 (CAPITAL, operator sign-off).
  - Orchestrator boot consuming the module → Stage 3.
  - Any frontend / chart liveness → Stage 4.
  - Launch/shutdown unification → Stage 5.
implementation_status: STAGE_1_DONE_COMMITTED_9b7c2a8f — STAGE_1b_DASHBOARD_ADAPTER_OPERATOR_DEFERRED
resume_pointer: |
  STATUS (verified 2026-06-09 via git + grep, /next session):
  Stage 1 (3 of 4 Turn-3 steps) DONE + COMMITTED as `9b7c2a8f` (local-only; operator front-end owns push):
    - trading_app/live/preflight.py created (engine moved + structured run_preflight()/PreflightReport). VERIFIED exists.
    - scripts/run_live_session.py thinned, re-imports moved names from preflight.py. VERIFIED imports (line 69).
    - pipeline/check_drift.py _extract_preflight_emitted_tokens re-pointed to preflight.py. VERIFIED (line 1046).
  REMAINING done_when items (#2 module test, #3 dashboard in-process re-point, #5 dead-code sweep of
  _run_preflight_subprocess) are STAGE 1b = dashboard in-process adapter, which the OPERATOR DEFERRED
  to its own scope (see memory/project_app_overhaul_stage1_preflight_seam_IMPL_DONE_dashboard_NEXT_2026_06_09.md).
  bot_dashboard.py STILL uses _run_preflight_subprocess (line 753/852) BY DESIGN until Stage 1b.
  NEXT (operator-gated, NOT autonomous /next): push/merge 9b7c2a8f, then schedule Stage 1b.
  Capital-adjacent but behavior-preserving — no operator sign-off gate on Stage 1 itself
  itself (Stages 2/3/5 are the capital-sign-off stages).
---

# Stage 1 — Single Source of Preflight Truth

## Purpose
Kill the four-way preflight divergence that makes the app "feel disjointed". One importable
module, one verdict shape, every surface reads it.

## Approach (behavior-preserving extraction)
1. Create `trading_app/live/preflight.py` housing the moved gate engine + a `PreflightReport`
   dataclass exposing per-gate `{id, title, status, message, remediation}`, `overall`, `strict_block`.
2. `run_live_session.py` imports the engine; keeps ONLY CLI parsing + printing.
3. `bot_dashboard.py` calls the module in-process instead of parsing subprocess stdout.
4. Verify byte-for-byte gate parity via the existing preflight test + a new module unit test.

## Blast Radius
- `trading_app/live/preflight.py` — NEW module; receives the moved 15-gate engine, probes, dataclasses, `PREFLIGHT_CHECKS`, `_passing_check_is_advisory`, and a new structured `run_preflight() -> PreflightReport`. Zero callers until the launcher + dashboard re-point to it.
- `scripts/run_live_session.py` — re-imports every moved name back into its namespace (keeps `monkeypatch.setattr(rls, …)` + `rls.CheckResult`/`rls.PreflightContext`/`rls._check_*` test swaps resolving); `_run_preflight` delegates to `run_preflight()` then renders; `main()` unchanged. Source-grep tests pin `checks_total = len(PREFLIGHT_CHECKS)` + `_probe_brackets`/`_probe_fill_poller` textually present here.
- `trading_app/live/bot_dashboard.py` — `_run_preflight_subprocess` replaced by in-process `run_preflight()` call shaped through a fail-closed adapter into the EXACT existing cache-entry dict (`status`/`checks`/`passed`/`total`/`overall`/`has_warnings`/`has_failures`/`output`); arm-guard (`action_start`) + SSE UI consume those keys unchanged.
- `pipeline/check_drift.py` — `_extract_preflight_emitted_tokens` re-pointed from `run_live_session.py` to `preflight.py` (the new honest residence); `check_preflight_status_token_parity` must still fail-closed on a known-bad token. Capital-path guard.
- `tests/test_scripts/test_run_live_session_preflight.py` — 71 tests, MUST stay green untouched (parity proof).
- `tests/test_trading_app/test_preflight_module.py` — NEW; structured-report-vs-printed parity, fail-closed adapter, anti-rubber-stamp guard on `preflight.py`.
- Reads: `gold.db` read-only (daily_features freshness probe), broker API read-only (account/contract probes). Writes: none (preflight is exit-after-report; no order routing, no DB writes).

## Verification profile
drift + preflight test + module test + grep-no-duplicate-gates + standalone --preflight parity.
