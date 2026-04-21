---
slug: phase-6e-monitor-thresholds
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Phase 6e sub-step 2.a — locked threshold dataclass (TDD) per design doc § 4
---

# Stage: Phase 6e sub-step 2.a — locked threshold dataclass

## Task

Create a frozen dataclass carrying the 11 numeric contracts from
`docs/plans/2026-04-21-phase-6e-monitoring-design.md § 4`, and a companion test
module that locks every value against § 4 verbatim. Pure contract module. No
logic. TDD order: tests first (red), implementation second (green).

Source authority for all 11 numeric values:
- 10 values from `docs/plans/2026-02-08-phase6-live-trading-design.md § 6e`
  (the 2026-02-08 Phase 6 design; authoritative per the 2026-04-21 6e design
  doc header).
- 1 value (`sr_alarm_arl0 = 1000`) from shadow pre-reg G9
  kill_criteria.shiryaev_roberts_alarm + Pepelyshev-Polunchenko 2015 Eq. 11.

Unblocks: subsequent 6e sub-steps (detectors → monitor_runner → dashboard →
orchestrator hook) consume this dataclass.

## Scope Lock

- `trading_app/live/monitor_thresholds.py` (new)
- `tests/test_trading_app/test_monitor_thresholds.py` (new)

## Blast Radius

- New module is a pure frozen dataclass — no side effects, no I/O, no state.
- Zero callers at time of commit (sub-step 2.b will be first consumer).
- Read-only consumption of the 2026-04-21 6e design doc (§ 4) for value source.
- No canonical-config touches (config.py, cost_model, prop_profiles,
  asset_configs, session catalog).
- No gold.db writes. No schema changes. No network. No broker.
- No changes to existing live/* modules (sr_monitor, cusum_monitor,
  alert_engine, performance_monitor, bot_dashboard, session_orchestrator).
- Pre-commit hook runs lint + drift + fast tests; expected to pass.

## Approach

1. Write `tests/test_trading_app/test_monitor_thresholds.py` first.
   - One test per § 4 numeric (11 tests).
   - `FrozenInstanceError` on mutation (frozen dataclass invariant).
   - Source-text assertions for `@revalidated-for` + design-doc citation
     (Research Provenance Rule per CLAUDE.md).
2. Run tests → red (import fails; module missing).
3. Write `trading_app/live/monitor_thresholds.py` — minimal frozen dataclass
   with 11 default fields matching § 4 exactly, plus module docstring carrying
   provenance annotations.
4. Run tests → green.
5. Run `python pipeline/check_drift.py` → green.
6. Run `grep -r "monitor_thresholds" trading_app/ tests/` — expect only the
   two files in scope_lock (sub-step 2.b is first downstream consumer; that's
   next stage).

## Acceptance criteria

1. `pytest tests/test_trading_app/test_monitor_thresholds.py` green.
2. `python pipeline/check_drift.py` exit 0.
3. Pre-commit hook passes (lint, drift, behavioural audit, M2.5 scan).
4. Self-review: confirm each of 11 values matches § 4 of
   `docs/plans/2026-04-21-phase-6e-monitoring-design.md` verbatim. No
   post-hoc tuning. No additional fields. No logic.
5. Dead-code sweep: `grep -rn "monitor_thresholds" trading_app/ tests/` shows
   only the two scope-locked files.

## Non-goals (explicit)

- No detector modules this stage.
- No monitor_runner orchestrator this stage.
- No dashboard panels this stage.
- No session_orchestrator hook this stage.
- No consumption call-sites in existing live/* modules this stage.
- No reading from gold.db, paper_trades, or any live data.

## Commit plan

- Single commit. Subject: `feat(6e): locked threshold dataclass per § 4 + tests`.
- Body cites § 4 and the shadow pre-reg G9 (sr_alarm_arl0 provenance).
