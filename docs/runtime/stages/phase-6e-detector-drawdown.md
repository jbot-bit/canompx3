---
slug: phase-6e-detector-drawdown
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Phase 6e sub-step 2.b — Alert 1 Drawdown detector (TDD)
worktree: deploy/live-trading-buildout-v1 (C:/Users/joshd/canompx3-deploy-live)
---

# Stage: Phase 6e sub-step 2.b — Alert 1 Drawdown detector

## Task

Build the first of 7 Phase 6e detectors: Alert 1 Drawdown
(`daily_pnl_warn_r = -3.0`). Detector is a pure function that takes daily
PnL (in R multiples) and a `MonitorThresholds` instance and returns a list
of canonical-marker message strings. monitor_runner (sub-step later) calls
the detector once per cadence and dispatches messages through
`alert_engine.record_operator_alert()`, which classifies via the canonical
`_ALERT_RULES` table.

To route the detector's output cleanly, `_ALERT_RULES` is extended with one
new marker: `("drawdown_warn", "warning", ("DRAWDOWN WARN",))`. This is an
intentional canonical-source extension, not a fork — `classify_operator_alert`
remains the single place where messages → (level, category) mapping happens.

## Scope Lock

- trading_app/live/detectors/__init__.py
- trading_app/live/detectors/drawdown.py
- trading_app/live/alert_engine.py
- tests/test_trading_app/test_detectors_drawdown.py

## Blast Radius

- `trading_app/live/detectors/` is a new sub-package. No existing callers.
- `drawdown.py` — pure function, no I/O, no state, no side effects.
- `alert_engine.py` — ADDS one row to `_ALERT_RULES`. No signature changes,
  no existing-rule edits, no reordering. Existing tests continue to pass.
- Zero consumers at commit time. Sub-step 2.i (monitor_runner) is first
  consumer.
- No canonical-config touches (config.py, cost_model, prop_profiles,
  asset_configs, sessions, holdout policy).
- No gold.db writes. No schema changes. No broker. No network.

## Parallel-concept note

`trading_app/risk_manager.py` RiskLimits has `drawdown_warning_r: float = -3.0`
with `<=` semantics used for trade BLOCKING. This detector serves the
MONITORING (alert-only) role and uses `<` semantics. Both values default to
-3.0 because both derive from the 2026-02-08 Phase 6 design. Documented
explicitly in `drawdown.py` docstring.

## Acceptance criteria

1. `pytest tests/test_trading_app/test_detectors_drawdown.py
    tests/test_trading_app/test_alert_engine.py` green.
2. `python pipeline/check_drift.py` exit 0.
3. Pre-commit hook passes.
4. Self-review: strict-less-than semantics intentional. Marker string is
   canonical and unique. Alert_engine change is additive-only.
5. Dead-code sweep shows only expected call sites.

## Non-goals (explicit)

- No Alert 2 Circuit Break this sub-step (sub-step 2.c).
- No monitor_runner orchestrator (sub-step 2.i).
- No dashboard panels.
- No session_orchestrator hook.
- No changes to risk_manager's parametric drawdown_warning_r.
