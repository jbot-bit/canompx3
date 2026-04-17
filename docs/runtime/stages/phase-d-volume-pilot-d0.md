---
task: Phase D Volume Pilot D-0 — discrete size-scaling backtest on MNQ COMEX_SETTLE
mode: IMPLEMENTATION
stage: 1_complete
total_stages: 1
slug: phase-d-volume-pilot-d0
created: 2026-04-17
updated: 2026-04-17
status: COMPLETE — D-0 verdict PASS (Sharpe uplift 54.12%, gate 15%)
result: docs/audit/results/2026-04-17-phase-d-d0-backtest.md
---

# Phase D Volume Pilot — Stage D-0 Backtest

## Why
Phase D spec (`docs/audit/hypotheses/2026-04-15-phase-d-volume-pilot-spec.md`)
H1: discrete rel_vol size-scaling on MNQ COMEX_SETTLE O5 RR1.5 produces Sharpe
uplift >= 15% over binary 1x deployment. D-0 is the IS-only backtest that
gates the 15-week full Phase D build. If D-0 fails the 15% uplift gate, the
rest of the Phase D roadmap is killed before infrastructure spend.

rel_vol_HIGH_Q3 is BH-global validated at K=14,261 on 5 independent cells
(comprehensive scan Apr 15). Per-trade Sharpe on binary deployment is
modest (0.05-0.17). Hypothesis: continuous-ish size scaling (discrete 3-tier)
extracts signal magnitude rather than just signal presence.

## Scope Lock
- research/phase_d_volume_sizing_pilot_d0.py (NEW — executor)
- docs/audit/results/2026-04-17-phase-d-d0-backtest.md (NEW — result report)

## Out of scope (deferred to D-1 onwards)
- trading_app/forecast_combiner.py (D-1 live infrastructure)
- trading_app/risk_manager.py / execution_engine.py updates (D-1/D-2)
- pipeline/backtest.py extension (D-1+)
- Any trading_app/ modification
- 2026 OOS evaluation (held sacred per Mode A; D-0 is IS-only)
- Other lanes (TOKYO_OPEN, SINGAPORE_OPEN) — those are D-3 scope

## Data contract
- Read-only: daily_features.rel_vol_COMEX_SETTLE, orb_outcomes (pre-2026)
- Trading window: trading_day < 2026-01-01 (Mode A sacred boundary)
- Instrument: MNQ only
- Session: COMEX_SETTLE
- Aperture: O5
- RR: 1.5
- Entry model: E2
- Cost: canonical pipeline.cost_model.COST_SPECS (no override)

## Sizing rule (per spec section 3.2 discrete bucketing)
- Compute P33, P67 of rel_vol_COMEX_SETTLE on IS-only, MNQ-only
- size = 0.5x if rel_vol < P33
- size = 1.0x if P33 <= rel_vol <= P67
- size = 1.5x if rel_vol > P67
- Baseline: size = 1.0x on every signal-day (unscaled)

## Acceptance
- [ ] Script py_compile clean + ruff clean
- [ ] Script uses canonical pipeline.paths.GOLD_DB_PATH (no hardcoded path)
- [ ] Script uses canonical pipeline.cost_model cost application
- [ ] Script uses canonical trading_app.holdout_policy.HOLDOUT_SACRED_FROM
- [ ] P33 / P67 calibrated on IS-only, MNQ-only, COMEX_SETTLE-only rel_vol rows
- [ ] Never touches 2026 rows (assert fail-closed)
- [ ] Baseline 1x backtest matches existing MNQ_COMEX_SETTLE_E2_RR1.5 PnL
      within numerical tolerance (sanity check against canonical)
- [ ] Scaled backtest reports: N_trades, WR, ExpR, Sharpe_ann, MaxDD, per-year breakdown
- [ ] Primary gate: Sharpe_scaled / Sharpe_baseline >= 1.15 -> PASS
- [ ] Secondary gates (all must hold for PASS):
      - MaxDD_scaled <= 1.5 * MaxDD_baseline
      - No single year Sharpe_scaled < 0.8 * Sharpe_baseline
      - corr(size_multiplier, pnl_r) > 0.05 (signal actually predicts)
- [ ] T0 tautology check: |corr(size_multiplier, existing OVNRNG_100 filter fire)| < 0.70
- [ ] Result MD written with PASS/FAIL verdict + all gate evaluations + per-year table
- [ ] Drift check passes (via uv run python -m pipeline.check_drift)

## Kill conditions (halt D-0, do NOT proceed to D-1)
- Primary gate fails (Sharpe uplift < 15% or any secondary gate red)
- Baseline sanity check fails (current code does not match canonical PnL)
- rel_vol signal correlation with pnl is negative or below 0.05
- T0 tautology fires (size_multiplier subsumed by existing OVNRNG_100)

## Verification commands
- uv run python research/phase_d_volume_sizing_pilot_d0.py
- uv run python -m pytest tests/ -x -q  (smoke — nothing in pipeline/ or trading_app/ touched)
- uv run python -m pipeline.check_drift
- ruff check research/phase_d_volume_sizing_pilot_d0.py
- python -m py_compile research/phase_d_volume_sizing_pilot_d0.py

## Preflight assertions (before running)
- Confirm daily_features.rel_vol_COMEX_SETTLE column exists and populated
  for MNQ pre-2026 (verified at stage-file write: 5853 rows, values spanning
  0.34-2.58 observed on last 5 IS days)
- Confirm orb_outcomes has MNQ COMEX_SETTLE O5 RR1.5 E2 rows pre-2026
- Confirm pipeline.cost_model.COST_SPECS['MNQ'] returns expected friction

## Gate for moving to D-1
- ALL acceptance criteria green
- D-0 verdict = PASS committed to docs/audit/results/
- Result reviewed against `docs/institutional/pre_registered_criteria.md`
  (DSR at multiple K framings if applicable, temporal stability, per-year)
- User approval of 4-week D-1 signal-only shadow timeline
