---
slug: phase-6e-detectors-nan-guards
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Phase 6e correction — add NaN input guards to all 4 detectors per institutional rigor #6
worktree: deploy/live-trading-buildout-v1 (C:/Users/joshd/canompx3-deploy-live)
---

# Stage: Phase 6e correction — detector NaN guards

## What went wrong

M2.5 on 2.e flagged NaN vulnerability in expr_ratio.py; verified systemic
across all 4 detectors shipped so far. Institutional-rigor Rule #6
violation.

  - drawdown / circuit_break: NaN `<` threshold returns False -> silent
    no-fire.
  - wr_drift / expr_ratio: NaN-contaminated computation -> garbage
    "nan" in message.

## Scope Lock

- trading_app/live/detectors/drawdown.py
- trading_app/live/detectors/circuit_break.py
- trading_app/live/detectors/wr_drift.py
- trading_app/live/detectors/expr_ratio.py
- tests/test_trading_app/test_detectors_drawdown.py
- tests/test_trading_app/test_detectors_circuit_break.py
- tests/test_trading_app/test_detectors_wr_drift.py
- tests/test_trading_app/test_detectors_expr_ratio.py

## Changes

Per detector: add `import math`; add NaN guard as the FIRST check;
return `[]` on any NaN input. One NaN-input test per detector.

Silent-return-[] rather than raise: detectors are observers, not
enforcers; upstream data corruption is a separate alert class.

## Acceptance criteria

1. All detector + alert_engine tests green.
2. `pipeline/check_drift.py` exit 0.
3. Pre-commit hook passes.
4. NaN guard is first check in every detector; no semantic changes.
5. Dead-code sweep clean.

## Blast Radius

- 4 detector modules: `import math` + 3-line NaN guard as FIRST check.
- 4 test modules: new NaN-input test per detector.
- `alert_engine.py` UNCHANGED.
- Zero consumers at commit time (monitor_runner is sub-step 2.i).
- No canonical-config touches. No gold.db. No broker. No network.

## Non-goals

- M2.5 LOW findings (`__all__`, positional args).
- Upstream PerformanceMonitor NaN detection.
- New detectors.
