---
slug: phase-6e-detectors-input-validation-refactor
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Phase 6e refactor — extract shared detector input validator (None + NaN) per institutional rigor #3
worktree: deploy/live-trading-buildout-v1 (C:/Users/joshd/canompx3-deploy-live)
---

# Stage: Phase 6e refactor — shared detector input validator

## Why refactor, not patch

Third input-validation patch cycle (strict-<, NaN, now None). Rule #3
triggered: stop patching, extract shared validator for the 4 existing
+ 3 future detectors.

## Pre-reg

`has_missing_input(*values: float | None) -> bool` returns True iff any
value is None or NaN. Inf passes through (extreme-value alerts MUST
fire, not hide).

Every detector uses this as the FIRST guard. `math.isnan` imports
removed where no longer used.

## Scope Lock

- trading_app/live/detectors/_validation.py
- trading_app/live/detectors/drawdown.py
- trading_app/live/detectors/circuit_break.py
- trading_app/live/detectors/wr_drift.py
- trading_app/live/detectors/expr_ratio.py
- tests/test_trading_app/test_detectors_validation.py
- tests/test_trading_app/test_detectors_drawdown.py
- tests/test_trading_app/test_detectors_circuit_break.py
- tests/test_trading_app/test_detectors_wr_drift.py
- tests/test_trading_app/test_detectors_expr_ratio.py

## Blast Radius

- New _validation.py: single pure function, ~10 LOC.
- 4 existing detectors: guard line swap, import cleanup.
- New test module + extensions to 4 existing test modules.
- alert_engine.py UNCHANGED. No canonical-config. No gold.db. No broker.

## Acceptance criteria

1. All tests green.
2. `pipeline/check_drift.py` exit 0.
3. Pre-commit hook passes.
4. `has_missing_input` is FIRST guard in every detector.
5. Dead `math` imports removed.

## Non-goals

- No existing-semantic changes.
- No new detectors.
- No `math.isfinite` (inf intentional pass-through).
- No PerformanceMonitor work.
