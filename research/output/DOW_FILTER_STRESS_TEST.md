# DOW Filter Stress Test — Results

**Date:** 2026-03-13
**Design doc:** `docs/plans/2026-03-13-dow-filter-stress-test-design.md`
**Scope:** All 3 DOW skip filters (NOFRI, NOMON, NOTUE) across all instruments

## Executive Summary

| Filter | Session | Verdict | Action |
|--------|---------|---------|--------|
| **NOMON** (skip Monday) | LONDON_METALS | **PLAUSIBLE BUT UNPROVEN** | Keep in live_config with kill criteria |
| **NOFRI** (skip Friday) | CME_REOPEN | **LIKELY NOISE** | Remove from grid |
| **NOTUE** (skip Tuesday) | TOKYO_OPEN | **LIKELY NOISE** | Remove from grid |

---

## Phase 1: Triage Results

### T1 — FDR Inclusion Verification: PASS (all 3 filters)

All DOW composite strategies have non-null p-values in `experimental_strategies` and are included in the BH FDR correction family. No orphans. FDR treats them as separate hypotheses alongside their base filters.

| Filter | Instruments with canonical strategies | FDR-significant |
|--------|--------------------------------------|-----------------|
| NOFRI | M2K(840) MES(820) MGC(810) MNQ(732) | MNQ: 3 |
| NOMON | M2K(840) MES(840) MGC(840) MNQ(588) | MNQ: 21 |
| NOTUE | M2K(840) MES(840) MGC(828) MNQ(672) | MGC: 3, MNQ: 16 |

The red-team concern about "post-hoc layering outside FDR" is **unfounded** — DOW filters are inside the correction (`strategy_validator.py:1105-1143`).

### T2 — Paired WITH/WITHOUT Comparison: NO JK-SIGNIFICANT IMPROVEMENT (any filter)

For FDR-significant DOW strategies that have a validated base equivalent, Jobson-Korkie tests show no statistically significant Sharpe improvement from adding the DOW filter.

**NOMON (live filter — MNQ LONDON_METALS):**
- 21 FDR-significant strategies; several bases not validated (DOW version lifted them over threshold)
- Best JK p-value: 0.107 (MNQ E2 RR1.5 CB1 G8 O15_S075, +37.9% ExpR improvement)
- 2 strategies show NEGATIVE ExpR difference (filter is harmful)
- Pattern: O15_S075 aperture shows largest improvements; S075-only and O30 variants are flat

**NOFRI (MNQ CME_REOPEN):**
- 2 FDR-significant with base comparison; JK p=0.42
- +21.6% ExpR improvement but not significant

**NOTUE:** No FDR-significant strategies had validated base equivalents for comparison.

### T3 — Sample Size Destruction: CONDITIONAL PASS

| Filter | Validated | All CORE | Class Changes | Avg Sample Loss |
|--------|-----------|----------|---------------|-----------------|
| NOFRI | 12 | Yes | 0 | 15.8% |
| NOMON | 84 | Yes | 0 | 19.7% |
| NOTUE | 115 | 97 CORE, 18 REGIME | **7 CORE→REGIME** | 19.1% |

**NOTUE fails:** 7 MGC TOKYO_OPEN strategies classified as CORE but the DOW-filtered N drops below 100 (range 84-94). These are misclassified — they should be REGIME tier.

### T4 — Brisbane/Exchange DOW Alignment Spot-Check: PASS

50/50 spot checks match (30 random across 3 sessions + 20 DST transition dates). Zero mismatches. The `day_of_week` column in `daily_features` is correct for all DOW-filtered sessions.

### T5 — Year-by-Year Stability

**NOMON (MNQ, Monday vs Others on LONDON_METALS E2 CB1 RR1.0 G6):**
| Year | Monday AvgR | Others AvgR | Monday Worse? |
|------|-------------|-------------|---------------|
| 2021 | -0.0406 | +0.1483 | YES |
| 2022 | -0.1039 | +0.0267 | YES |
| 2023 | +0.0009 | +0.0663 | YES |
| 2024 | -0.1404 | +0.0696 | YES |
| 2025 | +0.1415 | +0.0489 | no |
| 2026 | +0.1412 | +0.0677 | no |

Monday worse in 4/6 years (67%). Max single-year contribution 49% of aggregate. **VERDICT: STABLE** (barely — 2025-2026 shows Monday recovering).

**NOFRI (MNQ, Friday vs Others on CME_REOPEN):**
Friday worse in 4/6 years (67%) BUT max single-year contribution **173%** — 2021 Friday was strongly positive, dominating the aggregate. **VERDICT: UNSTABLE**.

**NOTUE (MGC, Tuesday vs Others on TOKYO_OPEN):**
Tuesday worse in 2/6 years (33%), max contribution 71%. **VERDICT: UNSTABLE**.

**NOTUE (MNQ):**
Tuesday worse in 2/6 years (33%), max contribution 402%. **VERDICT: UNSTABLE**.

---

## Phase 2: Permutation Test (NOMON only — survivors from Phase 1)

**Configuration:** MNQ LONDON_METALS E2 CB1 RR1.0 G6, N=3893 trades (764 Monday, 3129 other).

**Test statistic:** ExpR(skip Monday) - ExpR(all days) = improvement from applying the filter.

**5000 random shuffles** of DOW labels (preserving per-DOW counts).

| Metric | Real Improvement | Null 95th pct | p-value | Percentile |
|--------|-----------------|---------------|---------|------------|
| ExpR | +0.0181 | +0.0123 | **0.0064** | 99.4th |
| Sharpe | +0.0196 | +0.0133 | **0.0064** | 99.4th |

**Direct t-test (Monday vs Others):**
- Monday mean: -0.022R (N=764)
- Others mean: +0.070R (N=3129)
- t = -2.44, **p = 0.015** (two-sided)

**VERDICT: Monday is genuinely different from random DOW assignment at p=0.006.**

---

## Microstructure Plausibility (V1)

### NOMON — Monday LONDON_METALS
**Hypothesized mechanisms:**
- Post-weekend institutional ramp-up (metals desks re-establishing positions)
- Weekend gap risk (gold gaps from Sunday night futures open)
- Monday morning liquidity thinness in Brisbane-time London metals

**Literature support:** NONE. Searched all 15 reference PDFs:
- Fitschen: "No day has such a large impact that it should receive special consideration." Skipping Monday *hurts* stock performance.
- Chan: Equity calendar effects "unprofitable in recent years." Commodity seasonality only annual (demand-driven).
- No book discusses metals-specific DOW patterns.

**Assessment:** The effect is statistically real (p=0.006 permutation, p=0.015 t-test) but has **no institutional grounding**. This is the definition of PLAUSIBLE BUT UNPROVEN — the data says yes, the theory says nothing.

### NOFRI — Friday CME_REOPEN
**Killed in Phase 1.** Year-by-year unstable (2021 outlier dominates).

### NOTUE — Tuesday TOKYO_OPEN
**Killed in Phase 1.** No mechanism, unstable, classification integrity issues.

---

## Kill Criteria (V2) — NOMON Only

Since NOMON survives as PLAUSIBLE BUT UNPROVEN, define forward-looking kill conditions:

1. **Rolling monitor:** If trailing 12-month ExpR of Monday trades exceeds the mean of other days' ExpR for 2 consecutive rolling windows, remove the filter
2. **Year-end review:** At each annual rebuild, re-run T2 (paired JK comparison) and T5 (year-by-year stability). If Monday is no longer worse in >= 60% of available years, remove.
3. **Regime break:** 2025-2026 data shows Monday recovering (+0.14R vs +0.05-0.07R for others). If this persists through end of 2026, the Monday effect may have decayed — trigger full re-evaluation.
4. **Sample threshold:** If Monday N falls below 150 per year (currently ~150/yr), the effect can't be reliably measured.

**Note:** The NOMON effect is strongest in 2021-2024 and appears to be fading in 2025-2026. This is consistent with a decaying calendar artifact rather than permanent microstructure.

---

## Live Portfolio Action (V3)

| Filter | Current State | Action |
|--------|--------------|--------|
| NOMON | `MNQ_LONDON_METALS_E2_ORB_G6_NOMON` in live_config.py:198 | **KEEP** with kill criteria monitoring. Effect is real (p=0.006) but unproven mechanism + possible decay. |
| NOFRI | In grid via `config.py:733-735` | **REMOVE from grid.** Unstable, no evidence of real effect. |
| NOTUE | In grid via `config.py:739-742` | **REMOVE from grid.** Unstable, classification errors, no mechanism. |

---

## Recommendations

1. **Remove NOFRI and NOTUE** from `get_session_filters()` in `config.py`. They add degrees of freedom to the grid without evidence of real effects. Removing them shrinks the hypothesis space and gives remaining strategies slightly more FDR headroom.

2. **Keep NOMON but add decay monitoring.** The permutation test is strong (p=0.006), but the year-by-year data shows the effect fading in 2025-2026. If Monday remains non-toxic through 2026, the filter should be removed.

3. **Do not add new DOW filters** without first establishing a plausible mechanism AND pre-registering the hypothesis before testing. The literature is clear: DOW effects at this granularity are almost always noise.

4. **Fix NOTUE classification errors.** 7 MGC TOKYO_OPEN strategies are CORE-classified but their DOW-filtered N is below 100 (REGIME territory). Even though the filter is being removed, the classification logic should account for the sample size reduction from any filter overlay.

---

## Methodology Notes

- Permutation test used 5000 iterations (p-value resolution 0.0002)
- DOW labels shuffled preserving per-DOW counts (permutation, not bootstrap)
- Test statistic: ExpR(filtered) - ExpR(all) for permutation; Welch's t-test for direct comparison
- Year-by-year analysis used weighted average for "others" bucket
- JK test used rho=0.85 (shared-trades assumption for DOW filter overlaps)
- All queries used `orb_outcomes` joined with `daily_features` for proper filter application
