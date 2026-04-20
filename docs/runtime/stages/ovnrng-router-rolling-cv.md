---
slug: ovnrng-router-rolling-cv
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Rolling CV + ablation for OVNRNG allocator router (PR #62 follow-up). Determines deployment-readiness.
---

# Stage: OVNRNG router rolling CV + ablation

## Plan

### Question (trader/quant framing)

PR #62 established: ovn/atr is an allocator signal (walk-forward
ΔSR_ann=+0.87 on single fold). Before writing a pre-reg / building
the infra (~1 week), verify:

1. **Does the signal hold across multiple train/test folds?** One
   fold could be luck; 3+ folds with consistent results is a
   deployment-worthy signal.
2. **Is ovn/atr SPECIAL, or is any high-vol-vs-low-vol binning
   variable equivalent?** If `atr_20_pct` or `garch_forecast_vol_pct`
   produces similar router gains, the signal is general
   "vol-regime-conditional session selection" (Chan 2008 Ch 7), not
   ovn/atr-specific. That's still deployable but changes how the
   pre-reg is written.
3. **Does the best-session-per-bin map stabilize?** If bin→session
   choices drift 2+ sessions across folds, the router is picking up
   regime shifts faster than the train window can learn them —
   signal exists but deployment is brittle.

### Does this make money in reality?

Router top-1 trades ~165/yr with ExpR +0.12R → +19.5R/yr/contract
OOS. Current 6-lane portfolio trades ~1400/yr at weighted ExpR ~+0.15
= +210R/yr. **Router is NOT a replacement** for the portfolio — it's
a 7th strategy with ~9x less volume but potentially higher SR. The
realistic deployment path is overlay (additional capital allocation),
not replacement.

### Theory grounding

- **Chan 2008 Ch 7** (canonical extract: `docs/institutional/literature/
  chan_2008_ch7_regime_switching.md`): vol-regime-conditional strategy
  activation. Directly grounds "bin by volatility proxy, route to
  best-matched session."
- **Chordia et al 2018** (`chordia_et_al_2018_two_million_strategies.md`):
  factor-segmented discovery with strict t-threshold (≥3.79).
  Router in WF passes t=6.00 (IS) and t=3.03 on test (Step 4 of
  PR #62); exceeds the strict threshold.
- **Carver 2015 Ch 10** (`carver_2015_ch11_portfolios.md`, adjacent
  section): forecast combination — bin-conditional switching is a
  discrete forecast combination rule.

### Task decomposition

1. Build rolling CV with 4 annual folds:
   - Fold 1: train 2019+2020+2021 → test 2022
   - Fold 2: train 2020+2021+2022 → test 2023
   - Fold 3: train 2021+2022+2023 → test 2024
   - Fold 4: train 2022+2023+2024 → test 2025

2. For each fold, compute:
   - Train-derived best session per bin
   - Test-period ROUTER SR_ann (apply train map)
   - Test-period CONTROL SR_ann (train bin-agnostic top-1 session)
   - Router − Control ΔSR_ann
   - Router n, ExpR on test

3. **Map stability measure**: across 4 folds × 5 bins = 20 choices,
   count unique session assignments per bin. Fully stable = 1
   unique session per bin across folds. Fully unstable = 4 different
   sessions chosen. Score per bin and aggregate.

4. **Ablation on binning variable**:
   - Run router framework on same data but bin by `atr_20_pct`
     (known pre-session, 252-day rolling percentile of ATR-20)
   - Run same framework binning by `garch_forecast_vol_pct` (where
     non-NULL)
   - Compare per-fold SR_ann of router vs control across binning
     variables

5. Final verdict:
   - **ROUTER_DEPLOY_READY**: router > control in ≥3 of 4 folds
     AND map stable (majority bin has same session in ≥3 of 4 folds)
     AND ovn/atr shows advantage over ablation variables (or similar
     if the signal is general regime routing)
   - **ROUTER_MARGINAL**: router > control in 2 of 4 folds
   - **ROUTER_BRITTLE**: router > control in ≤1 of 4 folds OR map
     drifts every fold (signal regime-shifts faster than train window)

### Scope Lock

- `research/audit_ovnrng_router_rolling_cv.py` (new)
- `docs/audit/results/2026-04-21-ovnrng-router-rolling-cv.md` (new)

### Blast Radius

- Read-only research. Zero production-code touch.
- Canonical data: `orb_outcomes`, `daily_features`.
- No new filters. No pre-reg. No deployment change.
- **2026 OOS sacred — NOT touched.**

### Acceptance criteria

1. Script runs on current `gold.db` without exceptions.
2. MD contains per-fold train map + test ROUTER vs CONTROL SR_ann table.
3. MD contains map-stability summary (unique sessions per bin across
   folds).
4. MD contains ablation: ovn/atr vs atr_20_pct vs garch_forecast_vol_pct
   per-fold SR_ann.
5. MD states clear verdict (DEPLOY_READY / MARGINAL / BRITTLE).
6. MD discusses: wrong question? wrong test? implementation vs signal?
7. Theory citations (Chan 2008 Ch 7, Chordia 2018, Carver Ch 10)
   reference canonical local extracts.
8. `python pipeline/check_drift.py` passes.
9. No production code touched.

## Non-goals

- Not writing router pre-reg (next turn if DEPLOY_READY).
- Not touching 2026 OOS.
- Not proposing deployment change.
- Not building allocator infra (~1 week separate build).
- Not cross-instrument (MES/MGC) — MNQ-only this turn.
