# L2 MNQ_SINGAPORE_OPEN ATR_P50 Filter Stability Audit

**Date:** 2026-04-21
**Branch:** `research/l2-atr-p50-stability`
**Script:** `research/audit_l2_atr_p50_stability.py`
**Parent:** `docs/audit/results/2026-04-20-6lane-unfiltered-baseline-stress.md` (PR #52)

---

## Background

PR #52 found L2 `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` is FULLY
filter-dependent: unfiltered IS baseline is net-negative (ExpR −0.010,
t=−0.38), filter restores lane viability (Welch fire-vs-non-fire
p=0.002, Δ=+0.073R). If the filter's edge has decayed, the lane has no
remaining signal.

PR #47 described the fire-rate behaviour (29%→80% erratic) as
"rolling-percentile instability on sparse session data." This audit
rechecks that diagnosis and tests whether the +0.073R IS lift holds
across the sample.

---

## Result summary

**Verdict: HOLDING (marginal).** The filter still discriminates across
both IS halves, but the edge is noisy year-over-year and depends
disproportionately on two strong years (2022, 2024).

Mechanism re-diagnosis: **NOT "sparse-session instability."** The KS
test shows SINGAPORE_OPEN-eligible days sample the `atr_20_pct`
distribution identically to all-MNQ (D=0.007, p=1.0). Fire-rate drift
mirrors MNQ's instrument-wide vol-regime cycling, not session-sampling
bias.

---

## Step 1 — Per-year fire rate, delta, Welch fire-vs-non-fire

| Year | N | fire% | atr_pct μ | atr_pct p25 | atr_pct p75 | ExpR_fire | ExpR_nonf | Δ | Welch t | p |
|------|---|-------|-----------|-------------|-------------|-----------|-----------|---|---------|---|
| 2019 | 171 | 29.2% | 31.6 | 5.5 | 55.0 | +0.064 | −0.163 | +0.228 | +1.33 | 0.187 |
| 2020 | 259 | 79.9% | 67.7 | 59.7 | 82.9 | +0.078 | +0.049 | +0.029 | +0.18 | 0.859 |
| 2021 | 259 | 42.1% | 42.1 | 9.1 | 72.2 | −0.058 | −0.035 | −0.023 | −0.17 | 0.863 |
| **2022** | 258 | 59.3% | 57.8 | 32.7 | 82.9 | +0.189 | −0.169 | **+0.359** | **+2.64** | **0.009** |
| 2023 | 258 | 18.2% | 28.8 | 9.9 | 43.5 | −0.053 | −0.059 | +0.006 | +0.04 | 0.970 |
| **2024** | 259 | 78.8% | 69.5 | 54.0 | 90.9 | −0.015 | −0.401 | **+0.386** | **+2.60** | **0.011** |
| 2025 | 258 | 56.6% | 56.0 | 21.0 | 88.1 | +0.147 | +0.000 | +0.147 | +1.04 | 0.300 |
| 2026 | 72 | 75.0% | 73.5 | 52.0 | 87.3 | +0.216 | +0.120 | +0.095 | +0.30 | 0.765 |

Only 2 of 8 years individually reach Welch p<0.05 (2022, 2024). Most
years show no discrimination at annual power (n≈260). In 2021 the filter
slightly HURTS (−0.023R). The aggregate IS Welch p=0.002 from PR #52 is
driven primarily by the two outlier years.

## Step 2 — Distribution shift check (SINGAPORE_OPEN vs all-MNQ)

| Year | SGO mean | SGO med | ALL mean | ALL med | Δmean | Δmed |
|------|---------|--------|---------|---------|------|------|
| 2019 | 31.6 | 23.0 | 32.6 | 23.6 | −0.99 | −0.63 |
| 2020 | 67.7 | 73.0 | 68.6 | 73.0 | −0.97 | +0.00 |
| 2021 | 42.1 | 38.9 | 41.0 | 36.5 | +1.15 | +2.38 |
| 2022 | 57.8 | 59.9 | 57.5 | 59.5 | +0.33 | +0.40 |
| 2023 | 28.8 | 25.8 | 28.8 | 25.4 | +0.06 | +0.40 |
| 2024 | 69.5 | 77.4 | 69.7 | 77.4 | −0.16 | +0.00 |
| 2025 | 56.0 | 66.5 | 54.8 | 62.7 | +1.25 | +3.77 |
| 2026 | 73.5 | 84.5 | 73.8 | 84.5 | −0.25 | +0.00 |

**KS test D=0.007, p=1.0.** The two samples are indistinguishable.

This refutes PR #47's "sparse-session" explanation for the fire-rate
drift. SINGAPORE_OPEN's `atr_20_pct` mean/median track all-MNQ's year-by-year
to within ±2pp. The drift from 29%→80%→erratic is an MNQ-wide
vol-regime cycling artifact — the `atr_20` distribution itself has shifted
year-over-year (low-vol 2019, high-vol 2020, medium 2021, etc.), and the
252-day rolling window captures those shifts. The filter fires often in
high-vol years and rarely in low-vol years *on every session*, not
specially on SINGAPORE_OPEN.

## Step 3 — Early-vs-late IS Welch test

IS midpoint: 2022-08-31.

| Half | n | fire_n | nonf_n | ExpR_fire | ExpR_nonf | Δ | Welch t | p |
|------|---|--------|--------|-----------|-----------|---|---------|---|
| Early | 861 | 493 | 368 | +0.081 | −0.062 | +0.143 | +1.97 | 0.049 |
| Late | 861 | 423 | 438 | +0.043 | −0.118 | +0.161 | +2.22 | 0.027 |

Both halves discriminate at p<0.05. Late half is marginally stronger.
No decay signature.

## Step 4 — Trailing 3-year Welch fire-vs-non-fire

| End year | N | ExpR_fire | ExpR_nonf | Δ | Welch t | p |
|---------|---|-----------|-----------|---|---------|---|
| 2021 | 689 | +0.036 | −0.070 | +0.105 | +1.32 | 0.189 |
| 2022 | 776 | +0.083 | −0.067 | +0.150 | +1.91 | 0.056 |
| 2023 | 775 | +0.065 | −0.076 | +0.141 | +1.81 | 0.071 |
| 2024 | 775 | +0.058 | −0.141 | +0.199 | +2.63 | 0.009 |
| 2025 | 775 | +0.040 | −0.092 | +0.131 | +1.71 | 0.088 |
| 2026 | 589 | +0.074 | −0.108 | +0.182 | +1.88 | 0.061 |

Rolling-3y p hovers around 0.06–0.09 recently. Always positive sign on
delta, but only 2024 hits p<0.05 cleanly.

---

## Classification

**HOLDING, but marginal.** The strict early-vs-late test says filter
still discriminates in both halves (p<0.05). But:

- Only 2 of 8 years individually significant (2022, 2024)
- Recent 3y rolling p is borderline (0.06–0.09)
- 2021 shows mild reverse sign (filter hurts by −0.023R)
- 2026 OOS too small (n=72) to inform stability view

This is not a DECAYING filter. It is a NOISY filter — 15-year edge
survives at p<0.05 on 1700 trades, but the per-year behaviour is
inconsistent. For a FULLY filter-dependent lane (baseline −0.010R),
depending on a filter this noisy is a structural concern even when
the aggregate test holds.

## What the mechanism actually is

ATR_P50 selects days where MNQ's recent vol (ATR-20) ranks above its
own 252-day median. On SINGAPORE_OPEN (which trades in Asian afternoon
relative to Brisbane), the filter effectively says: "take the ORB
breakout when MNQ has been running hot lately."

This is a **vol-regime gate**, not a session-specific filter. It does
not interact with SINGAPORE_OPEN session structure at all — the same
filter would select the same days regardless of which session we
backtested. The 2022/2024 outliers correspond to the largest vol-regime
transitions in MNQ's post-2019 history, where the filter captures
breakouts in the late-regime half and excludes the early-regime half.

## Operational implications

1. **Don't pause L2.** The filter still discriminates in both IS halves
   and in full-sample. But:

2. **Don't treat L2 as "safe with edge."** The edge is entirely
   filter-borne, the filter is noisy, and 2026 OOS is too small to
   re-confirm. Aggregate edge survives on two strong years out of
   seven.

3. **Priority candidate for replacement research.** A filter that
   fires on more stable selection criteria — e.g., GARCH forecast
   percentile (already populated in `daily_features`), ATR velocity
   regime, or an ATR threshold normalized to the 60-day rolling
   median (short-window, less regime-cycling) — could plausibly
   provide the same vol-regime gate with tighter per-year
   discrimination. Would be a separate Pathway-B pre-reg.

4. **PR #47 diagnosis needs amending.** PR #47 attributed the
   fire-rate drift to "rolling-percentile instability on sparse
   session data." The actual mechanism is MNQ-wide vol-regime
   cycling. This is a minor documentation fix — the operational
   conclusion (filter at 29%→80% drift) holds, but the explanation
   changes. Not worth its own PR; note it here and reference on next
   touch of that file.

5. **Lane sizing posture UNCHANGED.** L2 is deployed at 137 trade
   trailing N and +0.2407R trailing ExpR in the allocator — a
   disproportionate contributor. The audit does not suggest
   immediate sizing change.

---

## Next-best tests (not pursued here)

1. **Alternative vol-regime filter for L2.** Candidates:
   `garch_forecast_vol_pct >= 50`, `atr_vel_regime == 'expanding'`,
   `atr_20 > rolling_median(atr_20, 60)`. Run a Pathway-B K=1
   confirmatory (or small family scan) on L2 with the new filter
   and compare Welch discrimination per-year stability. Would need
   its own pre-reg.

2. **Cross-session ATR_P50 replication.** Does ATR_P50 show the same
   per-year noisiness on other sessions where it's registered? If
   yes, the noise is ATR-20 regime-dependence, not session-specific.
   Would inform whether replacement is a per-lane or per-filter
   project.

3. **L4 COST_LT12 audit (followup).** L4 is FILTER_VESTIGIAL in PR #52
   (Welch p=0.59). COST_LT12 on NYSE_OPEN fires 95–100% in IS; audit
   whether it ever added edge, or has always been a pass-through cost
   gate. Read-only, ~30 min.

---

## Provenance

- Canonical data: `orb_outcomes`, `daily_features` (`gold.db`).
- Filter: `ATR_P50` via `research.filter_utils.filter_signal`
  (`trading_app.config.OwnATRPercentileFilter`, min_pct=50).
- Feature: `daily_features.atr_20_pct` — rolling 252d percentile
  of `atr_20` (pre-computed, prior-only window, `pipeline.build_daily_features`
  line 1381–1393).
- Holdout: 2026-01-01 (Mode A sacred).
- Read-only; no production code touched.
