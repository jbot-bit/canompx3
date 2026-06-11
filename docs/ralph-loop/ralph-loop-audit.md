# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 227

## RALPH AUDIT — Iteration 227 (COMPLETED)
## Date: 2026-06-11
## Infrastructure Gates: fast drift PASS (150/0, --fast); ruff PASS
## Scope: trading_app/live/session_orchestrator.py — circuit_breaker integration seams :2684/:2981/:2411/:2430

---

## Full-File Audit Results

### trading_app/live/session_orchestrator.py — SCANNED (stale re-audit, last iter 214)

**Hash check:** stored=6e96afbb1c0fd16a, current=6e96afbb1c0fd16a — SCAN_NEEDED (findings=5 from ledger).

**Note on prior ledger `findings=5`:** iter 214 findings appear to have been addressed. No open findings in deferred ledger for this file. Fresh audit conducted.

**Seven Sins Scan — iteration 227:**
- S1 (Silent failure): Many `except Exception as e` clauses — all reviewed. All either log+notify+re-raise or have documented best-effort `# noqa: BLE001` justifications (dashboard snapshot, bracket cancel). CLEAN for capital-path code. `except Exception: pass` at :1436-1437 has `# noqa: BLE001 — Dashboard state is best-effort — never kill the trading loop` — acceptable.
- S2 (Fail-open): CB check at :2684 blocks entries when open (fail-closed). Exit path at :2981 logs CRITICAL and proceeds — intentional (cannot leave positions open). `_confirmed_flat_at_broker()` at :3385-3411 returns False on generic exception (fail-closed). CLEAN.
- S3 (Canonical violation): No hardcoded instrument literals, session times, cost specs, DB paths, holdout dates. CLEAN.
- S4 (Impact awareness): Tests exist (`test_orchestrator_circuit_wiring.py` 7/7 PASS, `test_session_orchestrator.py` 48/49 — 1 pre-existing failure). CLEAN for new code.
- S5 (Evidence over assertion): All findings traced via PREMISE→TRACE→EVIDENCE. CLEAN.
- S6 (Spec compliance): No spec violations found. CLEAN.
- S7 (Metadata trust): Comments on CB design accurate vs code. CLEAN.

**Domain-specific checks:**
- ORB window timing: No hardcoded times. `orb_utc_window()` used correctly via `LiveORBBuilder`. CLEAN.
- Session hardcoding: No session literals. CLEAN.
- E0 fill-on-touch: No `close_outside`/`closed_outside`. CLEAN.
- Holdout date: No `date(2026...` literals. CLEAN.
- DST contamination: No fixed clock times. CLEAN.
- DB path: `GOLD_DB_PATH` used (no direct hardcoded paths). CLEAN.
- Cost inline: `cost_spec.point_value` from injected spec. CLEAN.

**Ralph-specific extensions scan:**
- Async safety: `time.sleep(0.5)` at :2392 is in a retry backoff inside `_retry_fill_poller` — not in the main event loop. `asyncio.sleep` used in async context. CLEAN.
- State persistence gap: `_kill_switch_fired`, `_close_time_forced`, `_blocked_strategies` all persisted via `_safety_state.save()` on mutation. `_circuit_breaker` is intentionally in-memory (correct for per-session HTTP probe). CLEAN.
- Contract drift: CB integration via `auth.failure_hook = self._circuit_breaker` wired at init (:343); all callers confirmed matching. CLEAN.
- Look-ahead bias: Not applicable (live execution). CLEAN.

**Candidate findings investigated:**

**SO-227-01: `_circuit_breaker` not reset at daily rollover (LOW, DEFERRED)**
- PREMISE: `_check_trading_day_rollover` resets `_consecutive_engine_errors = 0` (line 1889) but has no `_circuit_breaker.reset()` or `record_success()` call.
- TRACE: `:1889` resets engine CB; `:342` creates HTTP CB; `grep "_circuit_breaker"` shows no rollover reset
- EVIDENCE: If HTTP CB opens (5+ broker failures near EOD), it persists into new trading day. Entries blocked until 30s probe window opens at `:2684`.
- MITIGATING: `should_allow_request()` returns True after 30s (probe path); `record_success()` on any successful HTTP call resets CB; `failure_hook` on auth propagates HTTP successes.
- VERDICT: DEFER — LOW severity; self-healing mechanism prevents permanent block; correct behavior for a per-session transient CB.

**SO-227-02: Re-notify uses `_consecutive_engine_errors` not `_circuit_breaker.is_open` (ACCEPTABLE)**
- PREMISE: Heartbeat re-notify at :3615 checks engine-level CB, not HTTP-level CB.
- TRACE: `:3615` checks `_consecutive_engine_errors >= 5`; `:342` has separate `_circuit_breaker`
- VERDICT: ACCEPTABLE — two independent CBs for different failure classes. Engine CB persists until rollover (intended for 30+ min outage re-notify); HTTP CB self-heals in 30s (no re-notify needed).

**SO-227-03: Pre-existing test failure (DOCUMENTED, NOT INTRODUCED)**
- `test_run_starts_watchdog_task` fails with `CancelledError` on notify async task teardown.
- Pre-existing — not introduced this iteration. 48/49 pass.
- VERDICT: PRE-EXISTING, not owned by this iteration.

**SO-227-04: `_emergency_flatten`/`_flatten_broker_positions` no explicit CB update (ACCEPTABLE)**
- PREMISE: Direct `order_router.submit` calls without `record_success()`/`record_failure()`.
- EVIDENCE: `auth.failure_hook = self._circuit_breaker` at :343 — HTTP-level failures routed through `BrokerHTTPClient` automatically. No gap.
- VERDICT: ACCEPTABLE — failure_hook provides implicit CB updates; emergency paths bypass CB gate by design.

**CB seams :2684/:2981/:2411/:2430 — CLEAN:**
- :2684 ENTRY CB check: correct gate ordering (after signal_only, before on_entry_sent). Fail-closed.
- :2981 EXIT CB check: logs CRITICAL, proceeds (intentional — can't leave positions open). Correct.
- :2411 exit success: `record_success()` properly resets CB after successful exit submit.
- :2430 exit failure (last retry): `record_failure()` properly opens CB after exhausted retries.

**Cluster summary:**
- FIXED: 0
- DEFERRED: 1 (SO-227-01, LOW)
- ACCEPTABLE: 2 (SO-227-02, SO-227-04)
- PRE-EXISTING: 1 (SO-227-03)
- CLEAN: all CB seam checks

## Files Fully Scanned

- trading_app/live/circuit_breaker.py (iter 226)
- trading_app/live/session_orchestrator.py (iter 227)
- trading_app/db_manager.py (iter 225)
- trading_app/live/tradovate/order_router.py (iter 224)
- trading_app/live/tradovate/auth.py (iter 223)
- pipeline/build_daily_features.py (iter 222)
- pipeline/outcome_builder.py (iter 221)
- pipeline/orb_calculator.py (iter 220)
- trading_app/live/tradovate/http.py (iter 219)
- trading_app/live/projectx/order_router.py (iter 218)
- trading_app/live/fill_poller.py (iter 217)
- trading_app/strategy_fitness.py (iter 215)
- trading_app/live/position_tracker.py (iter 214)
- trading_app/risk_manager.py (iter 213)
- trading_app/live/execution_engine.py (iter 212)
- trading_app/chordia.py (iter 211)
- pipeline/strategy_validator.py (iter 210)
- trading_app/derived_state.py (iter 209)
- pipeline/strategy_discovery.py (iter 208)
- trading_app/live/hwm_tracker.py (iter 207)
- trading_app/live/session_safety_state.py (iter 206)
- pipeline/build_features.py (iter 205)
- pipeline/check_drift.py (iter 204)
- trading_app/live/alert_engine.py (iter 203)
- trading_app/live/signal_log_rotator.py (iter 202)
- trading_app/live/copy_order_router.py (iter 201)
- pipeline/asset_configs.py (iter 196)
- trading_app/prop_profiles.py (iter 195)

## Next Iteration Targets (P0-P4)

- P0: Pre-existing `test_run_starts_watchdog_task` failure investigation (CancelledError on notify teardown)
- P1: Unscanned high-centrality files — check import_centrality.json for next target
- P2: Stale re-audits — files modified since last scan
