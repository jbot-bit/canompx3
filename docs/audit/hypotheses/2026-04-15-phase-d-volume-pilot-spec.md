# Phase D — Volume Size-Scaling Pilot (SPEC)

**Status:** Pre-registration SPEC for when this session's research matures. Build deferred.
**Date authored:** 2026-04-15
**Replaces stub:** `docs/audit/hypotheses/phase-d-carver-forecast-combiner.md` — this is the concrete instantiation for the volume pilot, keeping the general stub as the parent framework.

---

## 1. Source findings this spec builds on

Institutional-grade confirmed:
- **`rel_vol_HIGH_Q3` passes BH-global at K=14,261** across 5 independent (instrument, session) combos. Volume confirmation is UNIVERSAL. (`docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`)
- **T0-T8 audit** of 4 volume cells: all CONDITIONAL, each fails only T3 (WFE> 0.95 leakage-suspect) due to thin 3-month OOS. Otherwise clean on T0/T1/T2/T6/T7/T8. (`docs/audit/results/2026-04-15-t0-t8-audit-volume-cells.md`)
- **Confluence scan**: composite 2-factor AND gates add only 20% marginal edge over single-factor `rel_vol`. Phase D prioritises single-factor continuous forecast over confluence gate.

---

## 2. Hypothesis

H1: **Continuous size scaling based on `rel_vol` percentile produces Sharpe uplift ≥ 15% over binary 1x deployment on any of the 6 deployed MNQ lanes.**

Mechanism (Carver Ch 9-10 grounded):
- High rel_vol = institutional participation confirmation → higher SNR
- Low rel_vol = no follow-through, high noise → lower SNR
- Linear-interpolated sizing between 0.5x (low rel_vol) and 1.5x (high rel_vol) delivers Carver-style vol-adjusted allocation

H2 (secondary): `rel_vol_LOW_Q1` as hard-skip (size=0) produces Sharpe uplift by trading fewer low-quality days.

---

## 3. Implementation scope — MINIMAL FIRST

### 3.1 Pilot lane

**MNQ COMEX_SETTLE O5 RR1.5** — deployed lane with:
- 3 BH-family confluence survivors in scan (including the V4 T0-T8 CONDITIONAL cell)
- Existing filter OVNRNG_100 but 1.5% fire rate (effectively unfiltered at deployment)
- Short-direction volume confluence concentration

### 3.2 Forecast calculation

```
forecast_raw = rel_vol_percentile  # 0-1 percentile vs IS distribution
forecast_clamped = clip(forecast_raw, 0.1, 0.9)  # avoid extremes
size_multiplier = 0.5 + (forecast_clamped - 0.33) * 2.0  # linear: P33→0.5x, P67→1.5x
size_multiplier = clip(size_multiplier, 0.0, 1.5)
```

Alternative discrete bucketing (simpler, more testable):
```
if rel_vol < P33: size = 0.5x (or 0 if hard skip)
if P33 ≤ rel_vol ≤ P67: size = 1.0x (baseline)
if rel_vol > P67: size = 1.5x
```

Recommend **discrete bucketing** for Stage 1 pilot — cleaner A/B testing vs continuous.

### 3.3 Deployment stages

| Stage | What | Duration | Gate to next |
|-------|------|----------|--------------|
| D-0 | Backtest rebuild on MNQ COMEX_SETTLE with discrete size-scaling | 1 week | Sharpe uplift > 15% IS |
| D-1 | Signal-only shadow of size-scaled lane alongside 1x lane | 4 weeks | Live Sharpe tracks IS within ±30% |
| D-2 | Flip to live with size-scaled, monitor 30 days | 30 days | No account drawdown > 1.5x baseline |
| D-3 | Extend to TOKYO_OPEN + SINGAPORE_OPEN (lanes with volume confluence) | 2 weeks | Repeat Stage 0-2 per lane |
| D-4 | Extend to all 6 deployed lanes with per-lane-calibrated P33/P67 thresholds | 4 weeks | — |

Full pilot: ~12 weeks signal-only + live shadow before all 6 lanes size-scaled.

---

## 4. Pre-registered criteria (must pass before any size-scaling promotion)

### 4.1 Backtest stage (D-0 gate)

- Sharpe_scaled / Sharpe_baseline ≥ 1.15 on IS
- Max drawdown scaled ≤ 1.5x baseline drawdown
- No single year shows scaled Sharpe < 0.8x baseline
- OOS ExpR maintains dir_match on single-lane forecast
- Correlation of size_multiplier with pnl_r > 0.05 (signal actually predicts)

### 4.2 Signal-only shadow stage (D-1 gate)

- Live signal-only forecast correlates with intended size within 5% (measurement error only)
- No dirft in live `rel_vol` percentile distribution vs historical (Shiryaev-Roberts monitor — `pre_registered_criteria.md` criterion 12)
- Zero unfilled orders or execution errors
- Would-have-been P&L tracks expected within 30%

### 4.3 Live deployment stage (D-2 gate)

- Actual account-level realized Sharpe uplift > 10% (haircut from IS 15% for slippage / timing)
- No breach of account rules (Topstep max DD, daily loss limits)
- Forecast not drifting outside calibration range

---

## 5. Kill criteria (stop and revert)

- D-0 Sharpe uplift < 10% → rethink size scaling approach or abandon
- D-1 live-forecast correlation < 0.70 with intended → execution bug, fix before D-2
- D-2 drawdown > 1.5x baseline → size multiplier too aggressive, reduce cap to 1.25x
- Any stage: regime change detected (Shiryaev-Roberts alarm) → halt and re-audit

---

## 6. Infrastructure required

1. `trading_app/forecast_combiner.py` — new module
   - Function `compute_rel_vol_forecast(trading_day, session, instrument) → size_multiplier`
   - Persists P33/P67 per (session, instrument) from IS calibration
   - Exports to `ACCOUNT_PROFILES` / allocator
2. `pipeline/backtest.py` extension — simulate size-scaled lanes
3. `trading_app/risk_manager.py` update — accept forecast-dependent size per lane
4. `trading_app/execution_engine.py` update — compute rel_vol at session start, look up forecast, size order
5. Dashboard / monitoring — live forecast tracking + Shiryaev-Roberts alarm

### 6.1 Canonical source delegation (per `institutional-rigor.md` rule 4)

- `rel_vol_{session}` columns come from `pipeline/build_daily_features.py`. Forecast uses exactly these columns, never re-computes.
- P33/P67 calibration uses ONLY IS data (`trading_day < HOLDOUT_SACRED_FROM`).
- Sacred 2026 holdout remains untouched.

---

## 7. Timeline estimate

- Week 1: D-0 backtest on MNQ COMEX_SETTLE only
- Week 2-5: D-1 signal-only shadow
- Week 6-9: D-2 live deployment COMEX_SETTLE
- Week 10-11: D-3 TOKYO_OPEN + SINGAPORE_OPEN rollout
- Week 12-15: D-4 all 6 lanes rollout

Total: ~15 weeks for full Phase D deployment if nothing blocks.

---

## 8. Expected outcomes

Honest range:

- Optimistic: +20% per-lane Sharpe uplift from size-scaling, ~25-35% portfolio Sharpe uplift (correlation-adjusted). New annual R ~220-260 vs ~176 currently.
- Pessimistic: +8% per-lane, marginal portfolio benefit (~5% uplift), Stage 0-1 only before abandoning.
- Realistic middle: +12-15% per-lane Sharpe. $10-12K/yr/ct → $13-14K/yr/ct on MNQ.

---

## 9. Why this pilot vs alternatives

- vs Phase C (E_RETEST): Phase C requires new entry model in outcome_builder (heavy schema change). Phase D reuses existing E2, only changes size. Faster ship.
- vs Phase E (non-ORB SC2): blocked by literature acquisition (Dalton/Murphy). Phase D needs no new literature beyond already-extracted Carver.
- vs binary overlay filters: confluence scan showed K=14K global only 13 survivors, confluence adds only 20% marginal. Continuous size scaling captures the full signal without K-blowup.

---

## 10. Decision rule — execute when

- MNQ COMEX_SETTLE lane has > 6 months trailing stable performance (to establish strong baseline)
- User approves 15-week dedicated build timeline
- No competing priority (e.g., tick-delta ingestion which could supersede rel_vol as primary feature — see next-phase research plan)

**Current recommendation:** queue Phase D as NEXT major build after volume findings mature through signal-only shadow in current portfolio.
