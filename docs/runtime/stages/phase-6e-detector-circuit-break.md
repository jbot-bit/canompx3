---
slug: phase-6e-detector-circuit-break
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Phase 6e sub-step 2.c — Alert 2 Circuit Break detector (TDD)
worktree: deploy/live-trading-buildout-v1 (C:/Users/joshd/canompx3-deploy-live)
---

# Stage: Phase 6e sub-step 2.c — Alert 2 Circuit Break detector

## Task

Build detector 2/7: Alert 2 Daily Circuit Break (`daily_pnl_halt_r = -5.0`).
Pure function; fires when `daily_r <= thresholds.daily_pnl_halt_r` (at-or-
below, halt-flavored — contrast with Alert 1's strict-less-than warn).

Canonical classifier extension: `("daily_circuit_break", "critical",
("DAILY CIRCUIT BREAK",))`.

## Scope Lock

- trading_app/live/detectors/circuit_break.py
- trading_app/live/alert_engine.py
- tests/test_trading_app/test_detectors_circuit_break.py

## Blast Radius

- New detector module; pure function, no I/O, no state.
- `alert_engine.py` additive-only.
- Zero consumers at commit time.
- No canonical-config touches.

## Parallel-concept note

`trading_app/risk_manager.py` RiskLimits.max_daily_loss_r defaults to -5.0
and is used for trade-BLOCKING enforcement. This detector is MONITORING
only. Documented explicitly in `circuit_break.py` docstring.

## Acceptance criteria

1. All detector + alert_engine tests green.
2. `pipeline/check_drift.py` exit 0.
3. Pre-commit hook passes.
4. Self-review: at-or-below semantics, canonical marker, additive rule.
5. Dead-code sweep clean.

## Non-goals

- No Alert 3+.
- No monitor_runner, dashboard, orchestrator hook.
- No changes to risk_manager.
