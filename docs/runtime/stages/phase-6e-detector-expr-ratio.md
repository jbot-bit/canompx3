---
slug: phase-6e-detector-expr-ratio
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Phase 6e sub-step 2.e — Alert 4a ExpR Ratio detector (TDD) with pre-reg
worktree: deploy/live-trading-buildout-v1 (C:/Users/joshd/canompx3-deploy-live)
---

# Stage: Phase 6e sub-step 2.e — Alert 4a ExpR Ratio detector

## Pre-registration (locked BEFORE any code)

Alert 4a of a composite Alert 4 (2.f handles the SR alarm).

### Canonical authority

- `docs/plans/2026-02-08-phase6-live-trading-design.md` line 425:
  "| ExpR Drift | Rolling 50-trade ExpR < 50% of backtest | CRITICAL |"
- `docs/plans/2026-04-21-phase-6e-monitoring-design.md` § 4 rows 5-6:
  `expr_window_trades=50`, `expr_ratio_threshold=0.50`.

### Hypothesis

H0: rolling_expr >= 0.5 * baseline_expr (within band).
H1: rolling_expr < 0.5 * baseline_expr (>50% decay -> critical).

### Gate (locked, no post-hoc relaxation)

Fire iff ALL of:
  1. n_trades >= thresholds.expr_window_trades (50)
  2. baseline_expr > 0 (defensive; upstream validation's job)
  3. rolling_expr < thresholds.expr_ratio_threshold * baseline_expr
     (STRICT `<` per 2026-02-08 line 425)

No tolerance: RHS is a single multiplication, not a float subtraction,
so IEEE-754 boundary noise is not material at operator-relevant scale.

### Severity

CRITICAL per 2026-02-08 line 425. Classifier rule:
`("expr_drift", "critical", ("EXPR DRIFT",))`.

### Signature

```
check_expr_ratio(
    *,
    strategy_id: str,
    rolling_expr: float,
    baseline_expr: float,
    n_trades: int,
    thresholds: MonitorThresholds,
) -> list[str]
```

### Edge cases (pre-declared)

- baseline_expr <= 0 -> [] (documented; upstream rejects non-positive-
  expectancy strategies).
- rolling_expr == 0.5 * baseline_expr -> [] (strict `<`).
- n_trades < window -> [] (UNVERIFIED).
- Negative rolling_expr: fires if gate holds (valid critical signal).

## Scope Lock

- trading_app/live/detectors/expr_ratio.py
- trading_app/live/alert_engine.py
- tests/test_trading_app/test_detectors_expr_ratio.py

## Blast Radius

- Pure function; no I/O, no state.
- `alert_engine.py` additive only.
- Zero consumers at commit time.

## Acceptance criteria

1. New + prior detector + alert_engine tests green.
2. `pipeline/check_drift.py` exit 0.
3. Pre-commit hook passes.
4. Self-review: strict `<` matches line 425; baseline<=0 documented;
   severity=critical; threshold injection test present.
5. Dead-code sweep clean.

## Non-goals

- SR alarm (2.f).
- PerformanceMonitor accessors (2.i).
- monitor_runner / dashboard / orchestrator.
