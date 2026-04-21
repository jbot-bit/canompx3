---
slug: phase-6e-detector-wr-drift
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Phase 6e sub-step 2.d — Alert 3 WR Drift detector (TDD)
worktree: deploy/live-trading-buildout-v1 (C:/Users/joshd/canompx3-deploy-live)
---

# Stage: Phase 6e sub-step 2.d — Alert 3 WR Drift detector

## Task

Detector 3/7: Alert 3 Win-rate Drift. Fires when:
  (baseline_wr - rolling_wr) * 100 >= thresholds.wr_delta_pp
  AND n_trades >= thresholds.wr_window_trades

Pure function; caller supplies scalars. Rolling-WR accessor on
PerformanceMonitor + paper_trades data sourcing deferred to sub-step 2.i
(monitor_runner).

Canonical classifier extension: `("wr_drift", "warning", ("WR DRIFT",))`.

## Scope Lock

- trading_app/live/detectors/wr_drift.py
- trading_app/live/alert_engine.py
- tests/test_trading_app/test_detectors_wr_drift.py

## Blast Radius

- New pure-function detector; no I/O, no state.
- `alert_engine.py` additive rule only.
- Zero consumers at commit time.

## Semantics (locked)

- Window gate: n_trades < window_trades -> return [] (UNVERIFIED).
- Drift gate: (baseline_wr - rolling_wr) * 100 >= wr_delta_pp (inclusive).

## Acceptance criteria

1. New + existing detector + alert_engine tests green.
2. `pipeline/check_drift.py` exit 0.
3. Pre-commit hook passes.
4. Self-review: under-N returns []; inclusive drift gate; additive rule.
5. Dead-code sweep clean.

## Non-goals

- No monitor_runner / dashboard / orchestrator work.
- No PerformanceMonitor changes.
- No paper_trades queries.
