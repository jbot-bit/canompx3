---
slug: l2-atr-p50-stability-audit
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: L2 SINGAPORE_OPEN ATR_P50 filter stability audit — is the +0.073R IS edge stable over time?
---

# Stage: L2 ATR_P50 stability audit

## Task

PR #52 found L2 MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 is FULLY
filter-dependent — unfiltered IS baseline is net-negative (ExpR −0.010, t=−0.38),
filter restores the lane (filt ExpR +0.063, t=+1.73, Welch fire-vs-non-fire
p=0.002). If the ATR_P50 edge decays in recent years, the lane has no
remaining edge and should be paused.

PR #47 described the filter's 2026 behaviour as "rolling-percentile
instability on sparse session data" with fire rate 29%(2019)→80%(2025)
erratic. On closer reading:

- `atr_20_pct` is computed instrument-wide (rolling 252d of `atr_20`),
  not session-specific. Sparse-session data cannot destabilize the
  percentile computation itself.
- The fire-rate drift is therefore either: (a) SINGAPORE_OPEN-eligible
  days sample a biased slice of `atr_20_pct` that has trended upward,
  OR (b) `atr_20` volatility regime has shifted such that a larger
  fraction of days rank above the 252-day median.

Either way, the operational question is **does the +0.073R IS lift
from the Welch test hold in recent years, or is it pre-2023-heavy?**

## Scope Lock

- `research/audit_l2_atr_p50_stability.py` (new)
- `docs/audit/results/2026-04-21-l2-atr-p50-stability.md` (new)

## Blast Radius

- Read-only research. Zero production-code touch.
- Canonical data only: `orb_outcomes`, `daily_features`.
- No new filters. No config changes. No downstream consumers.

## Approach

1. Load MNQ SINGAPORE_OPEN E2 RR1.5 CB1 canonical universe (pre-filter).
2. For each year 2019-2026:
   - Distribution of `atr_20_pct` (mean, median, p25, p75) on the
     SINGAPORE_OPEN-eligible days for that year. Establishes whether
     the fire-rate drift is distribution-shift or sampling-bias.
   - ATR_P50 fire rate (should match PR #47).
   - ExpR on fire days, ExpR on non-fire days, delta, Welch p.
   - N on each side.
3. Cumulative lift test: compute rolling 3-year Welch p of
   fire-vs-non-fire delta. If it drifts from p<0.01 (early) to p>0.10
   (recent), filter is decaying.
4. Instrument-wide comparison: distribution of `atr_20_pct` on ALL MNQ
   trading days vs SINGAPORE_OPEN-only days per year. A divergence
   means SINGAPORE_OPEN samples a biased slice.
5. Era split: Welch test on fire-vs-non-fire IS half-vs-half. If late
   half Welch p > 0.05 while early half p < 0.01, filter is decaying.
6. Classify the filter as:
   - STABLE: filter Welch p < 0.05 in both IS halves, delta consistent
   - DECAYING: filter Welch p drifts from <0.05 to >0.10 between halves
   - UNSTABLE_DISTRIBUTION: `atr_20_pct` distribution non-uniform on
     SINGAPORE_OPEN days, suggesting the filter threshold no longer
     targets what it was designed to target
   - HOLDING: recent years retain the Welch advantage

## Acceptance criteria

1. Script runs to completion on current `gold.db` without exceptions.
2. MD contains per-year table (fire rate, delta, Welch p, atr_20_pct
   distribution stats).
3. MD contains rolling 3-year Welch p time series.
4. MD contains early-vs-late IS Welch comparison.
5. Final verdict: STABLE / DECAYING / UNSTABLE_DISTRIBUTION / HOLDING
   with reasoning.
6. `python pipeline/check_drift.py` passes (no production code touched).

## Non-goals (explicit)

- Not proposing filter replacement. If DECAYING, a replacement pre-reg
  is a separate work item.
- Not changing deployment state. If DECAYING, recommendation to user
  is a separate decision turn.
- Not re-running PR #47's fire-rate-by-year (already on main).
