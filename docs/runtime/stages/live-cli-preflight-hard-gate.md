---
task: Close the fail-open --live launch path — run canonical preflight unconditionally for every order-routing launch (CLI + dashboard), block on failure, no skip flag.
mode: IMPLEMENTATION
scope_lock:
  - scripts/run_live_session.py
  - tests/test_scripts/test_run_live_session_preflight.py
blast_radius: "scripts/run_live_session.py (single inline gate added in main() after --live CONFIRM, before planned-launch write / lock / orchestrator construction). Delegates to the EXISTING _run_preflight (no re-encoding). Closes the hole on BOTH entry paths: direct CLI --live and dashboard launch (bot_dashboard.py:2825 launches --live --auto-confirm with NO --preflight; its _run_preflight_subprocess at :834 is an advisory UI pre-check, not a hard gate on the launch process). Signal-only and --demo semantics preserved — the _check_* functions self-SKIP on signal_only and where appropriate on demo. Reads gold.db + lifecycle_state read-only; NO schema change, NO capital-config change, does NOT touch account_survival.py or prop_profiles.py (Stage 2, deferred until peer C11 work settles). Decision: HARD GATE, no --skip-preflight escape (operator confirmed, fail-closed doctrine)."
---

## Context

Codex adversarial review (2026-06-05) + ground-truth verification confirmed:
`_run_preflight` (the only caller of `_check_survival_report` [C11], `_check_sr_state`
[C12], `_check_live_readiness_report`, `_check_project_pulse_for_live`,
`_check_repo_drift_for_live`, `_check_telemetry_maturity`) runs ONLY inside
`if args.preflight:` (run_live_session.py:1166), which always `sys.exit`s. The
`--live` branch flows to `SessionOrchestrator(...)` → `asyncio.run(session.run())`
with no survival/C11/readiness gate. The orchestrator's only startup lifecycle read
(`_load_paused_lane_blocks`, session_orchestrator.py:1103-1143) loads per-strategy
blocks and is fail-open — it never refuses launch on `not gate_ok`.

## Blast Radius
- scripts/run_live_session.py — add ONE inline gate in `main()` after the `--live`
  CONFIRM (~line 1234) and BEFORE planned-launch write (1241), lock acquisition,
  dashboard subprocess, and `SessionOrchestrator(...)`. Single-instrument AND
  multi-instrument-profile paths both gated. Delegates to the canonical
  `_run_preflight(...)` already used by the `--preflight` block.
- tests/test_scripts/test_run_live_session_preflight.py — regression: a failing
  C11/readiness check makes `--live --auto-confirm` exit non-zero and NEVER
  construct the orchestrator.
- Reads: gold.db, lifecycle_state (read-only). Writes: none.
- Does NOT touch: account_survival.py, prop_profiles.py, session_orchestrator.py.

## Decisions (operator-confirmed)
- HARD GATE — no `--skip-preflight` flag. Override = fix the failing gate.
- Stage 1 ONLY this session. Stage 2 (strict-DD-budget-in-sim) deferred until
  peer C11 cap/prereg work lands (avoids account_survival collision + C11 verdict
  thrash).

## Acceptance
- Tests pass (show output).
- `python pipeline/check_drift.py` passes.
- Dead code swept.
- Self-review + adversarial-audit gate (CRITICAL on trading_app/live-adjacent
  capital path) before landing.
