# Phase D — Carver Forecast Combiner & Continuous Size Scaling (PRE-REG STUB)

**Status:** NOT EXECUTED. Pre-registration stub for future session.
**Date authored:** 2026-04-15
**Companion:** `docs/audit/hypotheses/2026-04-15-comprehensive-deployed-lane-gap-audit.md` §5.6

## Hypothesis

H1: Combining multiple validated signals into a continuous forecast (-10 to +10
per Carver Ch 8) and scaling position size proportional to forecast strength
produces higher risk-adjusted returns than the current binary-lane approach.

H2 (testable first): On MNQ TOKYO_OPEN, combining the 4 validated overlay
signals from 2026-04-15 comprehensive scan into a confluence score (+1 per TAKE
signal firing, -1 per SKIP signal firing) predicts ExpR linearly, and size
scaling by that score delivers Sharpe uplift ≥ 20% over binary deployment.

Mechanism: Each validated signal carries independent information (low
inter-correlation). Summed forecasts are closer to true Signal-to-Noise,
per Carver Ch 10 pp 161-175.

Literature grounding: `docs/institutional/literature/carver_2015_systematic_trading_ch9_10.md`
(volatility targeting + Kelly-linked sizing + forecast combination).

## Scope (when executed)

- Phase 1 — forecast combiner for MNQ TOKYO_OPEN only (proof of concept)
  - Signals to combine: rel_vol_HIGH_Q3 (TAKE, +1), rel_vol_LOW_Q1 (SKIP, -1),
    bb_volume_ratio_HIGH (TAKE, +1), bb_volume_ratio_LOW (SKIP, -1),
    atr_vel_LOW (SKIP, -1), plus F5_BELOW_PDL long (TAKE, +1 — from P3 finding)
  - Continuous combined score in [-3, +3]
  - Position size = max(0, min(2, 1 + 0.5 × combined_score)) — clamp 0 to 2x
  - Compare Sharpe vs existing binary TOKYO_OPEN lane

- Phase 2 — extend to all 6 deployed lanes using lane-specific validated signals
- Phase 3 — vol targeting: target 12% annualized per-lane vol via inverse-realized-vol sizing

## Comparison baseline

Current `topstep_50k_mnq_auto` deployment at 1x size per lane.

## Pre-registered criteria

1. Inter-signal correlation < 0.3 for any 2 signals combined (else collinearity weakens combination)
2. Each component signal has its own T0-T8 validation before inclusion
3. Phase 1 TOKYO_OPEN Sharpe uplift ≥ 20% on IS + dir_match on OOS
4. No new data mining — only signals pre-committed from prior validated research
5. Max position size 2x (vs 1x baseline) — aggressive but bounded

## Infrastructure required

1. New module `trading_app/forecast_combiner.py` — signal weighting, combination, clamping
2. Config schema extension in `prop_profiles.py` to carry forecast-dependent sizing per lane
3. Risk manager update to size orders based on combined forecast
4. Backtest framework update to re-simulate with continuous sizing
5. Shadow mode for 2 weeks before live

## Estimated timeline

- 1 week: forecast_combiner module + tests
- 1 week: risk manager + config integration
- 1 week: backtest framework + validation
- 2 weeks: live shadow
- Total: 5 weeks to live

## Kill criteria

- Phase 1 TOKYO_OPEN Sharpe uplift < 10% → reassess signal set
- Any deployed lane shows ExpR degradation under scaling → abandon
- Inter-signal correlation > 0.5 → need orthogonalization first

## Linkage to Phase C (E_RETEST)

Phase C and Phase D are independent but composable. E_RETEST changes how we
enter; forecast combiner changes size. Both can apply to the same trade.
Sequence: execute Phase C first (simpler, cell-specific), then D (broader,
portfolio-wide).
