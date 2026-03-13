# DOW Filter Stress Test — Audit Design

**Date:** 2026-03-13
**Status:** Approved
**Goal:** Determine whether DOW skip filters capture real market microstructure or noise

## Context

Three DOW skip filters exist in the discovery grid:

| Filter | Session | Skip Day | Mechanism Hypothesis |
|--------|---------|----------|---------------------|
| NOFRI | CME_REOPEN | Friday (4) | Weekend positioning, reduced liquidity |
| NOMON | LONDON_METALS | Monday (0) | Weekend gap recovery, institutional ramp-up |
| NOTUE | TOKYO_OPEN | Tuesday (1) | Unknown — requires investigation |

**Live exposure:** Only `MNQ_LONDON_METALS_E2_ORB_G6_NOMON` is in `live_config.py:198-206`.
Priority target is NOMON — the only filter generating real trade signals.

## Architecture Understanding

### How DOW Filters Enter the Grid

1. `config.py:563-565` — Three `DayOfWeekSkipFilter` instances defined (NOFRI, NOMON, NOTUE)
2. `config.py:582-596` — `_make_dow_composites()` combines each with size filters (G4/G5/G6/G8)
3. `config.py:733-742` — `get_session_filters()` wires DOW composites to specific sessions only
4. `strategy_discovery.py:1079-1083` — Each DOW composite counted as a separate filter variant in `total_combos`
5. `strategy_discovery.py:1086-1089` — `get_filters_for_grid()` enumerates all composites per session

### How BH FDR Treats DOW Strategies

`strategy_validator.py:1105-1143` applies BH FDR across ALL strategies for an instrument — DOW composites are corrected alongside their base filters. No special grouping, no exemption. ORB_G6_NOMON competes in the same FDR family as ORB_G6.

**This means the red-team concern about "post-hoc layering outside FDR" is unfounded.** DOW filters are inside the correction.

### DOW Alignment Guard

`dst.py:114-146` maintains `DOW_ALIGNED_SESSIONS` (11 sessions) and `DOW_MISALIGNED_SESSIONS` (NYSE_OPEN only, offset -1). `validate_dow_filter_alignment()` is fail-closed: raises `ValueError` if DOW filters are applied to misaligned sessions. All 3 DOW-filtered sessions (CME_REOPEN, LONDON_METALS, TOKYO_OPEN) are in the aligned set.

### Calendar Filter Data Source

`calendar_filters.py:58-60` — `day_of_week(d)` returns Python `d.weekday()` (0=Mon..6=Sun). Populated into `daily_features.day_of_week` during `build_daily_features.py`.

---

## Phase 1: Triage (cheap, fast)

### T1 — FDR Inclusion Verification

**Question:** Are DOW composite strategies actually included in the BH FDR correction family?

**Method:**
1. Query `experimental_strategies` for all strategy_ids containing `_NOFRI`, `_NOMON`, `_NOTUE`
2. Verify each has a non-null `p_value`
3. Cross-reference with `validated_setups.fdr_significant` and `fdr_adjusted_p`
4. Count: how many DOW hypotheses exist per instrument? What fraction of the total FDR family?

**Pass criteria:** All DOW composite strategies have p-values and appear in the FDR correction. No orphans.

**Expected outcome:** PASS — code at `strategy_validator.py:1105-1143` treats all strategies uniformly.

### T2 — Paired WITH/WITHOUT Comparison

**Question:** Does adding the DOW filter actually improve the strategy vs the base filter?

**Method:** For each (instrument, session, entry_model, rr_target, confirm_bars) tuple that has BOTH a base filter (e.g., ORB_G6) and a DOW variant (e.g., ORB_G6_NOMON):

1. Pull trade-level outcomes for both variants from `orb_outcomes` joined with `daily_features`
2. Compute for each:
   - N (sample size)
   - Win rate ± 95% CI (Wilson interval)
   - ExpR ± 95% CI (bootstrap or t-based)
   - Sharpe ratio
3. Paired comparison:
   - ExpR difference: `ExpR(DOW) - ExpR(base)` with CI
   - Sharpe difference: Jobson-Korkie test (H0: Sharpe_DOW = Sharpe_base)
   - Sample cost: `N_base - N_DOW` (how many days lost)
4. Decision matrix:
   - ExpR improves AND JK significant (p < 0.05) → filter HELPS
   - ExpR improves but JK not significant → filter PLAUSIBLE but unproven
   - ExpR same or worse → filter is NOISE (or harmful)

**Pass criteria:** At least one instrument×session shows statistically significant improvement.

**This is the most important test.** A DOW effect can be "real" (Monday is genuinely worse) but too small to justify the sample size destruction.

### T3 — Sample Size Destruction

**Question:** Does the DOW filter drop any validated strategy below CORE threshold?

**Method:**
1. For each validated strategy using a DOW filter: query N from `validated_setups`
2. Query N for the equivalent base filter strategy (same instrument/session/entry/RR/CB without DOW suffix)
3. Compute: `pct_lost = (N_base - N_dow) / N_base`
4. Flag if N_dow < 100 (dropped below CORE) or N_dow < 30 (dropped to INVALID)

**Pass criteria:** No validated DOW strategy falls below its classification threshold. If it does, the classification is wrong.

### T4 — Brisbane/Exchange DOW Alignment Spot-Check

**Question:** Is the `day_of_week` column in `daily_features` correct for DOW-filtered sessions?

**Method:**
1. For each of CME_REOPEN, LONDON_METALS, TOKYO_OPEN:
   - Pull 10 random rows from `daily_features` with that session's ORB data
   - Extract `trading_day` and `day_of_week`
   - Manually verify: Python `trading_day.weekday()` == stored `day_of_week`
   - Verify the exchange was open on that day (not a holiday incorrectly included)
2. Specifically check dates around DST transitions (March/November) for any off-by-one

**Pass criteria:** 30/30 spot checks match. Zero tolerance — one mismatch means the entire DOW analysis is contaminated.

**Expected outcome:** PASS — `calendar_filters.py:58-60` is trivially correct for aligned sessions, and `validate_dow_filter_alignment()` blocks misaligned ones.

### T5 — Year-by-Year Stability

**Question:** Does the DOW effect persist across years, or is it a calendar artifact?

**Method:** For each DOW filter:
1. Pull all eligible trades, split by year
2. For each year: compute ExpR on the skipped day vs ExpR on included days
3. Check: is the skipped day consistently worse (same sign) across >= 3 of available years?
4. Compute per-year effect size (Cohen's d) to check for outlier-year domination

**Pass criteria:** Skipped day underperforms in >= 60% of years with consistent sign. Effect size reasonably stable (no single year driving > 50% of the aggregate effect).

**Fail criteria:** Effect appears in < 3 years, or one outlier year contributes > 50% of aggregate effect.

---

## Phase 2: Permutation Test (conditional)

**Gate:** Only run if a DOW filter survives ALL of Phase 1 (T1-T5 all pass).

### P1 — DOW Label Shuffle

**Question:** Could the observed DOW effect arise by chance from randomly assigned day labels?

**Method:**
1. For each surviving filter, take the full set of eligible trading days
2. Compute the REAL test statistic: `ExpR(filtered) - ExpR(unfiltered)` (i.e., the improvement from applying the DOW skip)
3. Shuffle: randomly reassign DOW labels (0-4) across trading days, preserving:
   - All other features (ORB size, direction, ATR, etc.)
   - The same number of days per DOW bucket (permute labels, don't resample)
4. Recompute the test statistic under shuffled labels
5. Repeat 5000 times to build the null distribution
6. Rank the real test statistic against the null

**Decision rule:**
- Real statistic outside 95% CI of null → signal candidate (p < 0.05)
- Real statistic outside 90% CI but inside 95% → weak signal, not actionable
- Real statistic inside 90% CI → LIKELY NOISE

**Why 5000 iterations:** p-value resolution of 0.0002. At 1000 iterations, resolution is only 0.001 — too coarse for a 0.05 threshold.

---

## Phase 3: Verdict & Action

### V1 — Microstructure Plausibility

For each filter that passes permutation, state the mechanism:

| Filter | Plausible Mechanism | Testable? |
|--------|-------------------|-----------|
| NOFRI | Weekend positioning unwind, Friday liquidity withdrawal, options expiry effects | Partially — could check if effect concentrates on monthly OPEX Fridays |
| NOMON | Post-weekend institutional ramp-up, Monday gap risk, metals-specific inventory cycle | Partially — could check if Monday gaps are larger |
| NOTUE | Unknown — no obvious Tuesday mechanism for Tokyo metals session | No — absence of mechanism is a red flag |

A filter with no plausible mechanism that passes the permutation test is classified PLAUSIBLE BUT UNPROVEN, not REAL MICROSTRUCTURE.

### V2 — Kill Criteria (forward-looking)

For each surviving filter, define a concrete kill condition:

- **Rolling monitor:** If trailing 12-month ExpR of the skipped day exceeds the mean of included days (i.e., the skip is no longer beneficial), remove the filter
- **Sample threshold:** If cumulative N falls below CORE (100) due to data growth pattern, reclassify
- **Annual review:** At each yearly rebuild, re-run T2 (paired comparison) and T5 (year-by-year stability)

### V3 — Live Portfolio Action

| Outcome | Action |
|---------|--------|
| NOMON fails any Phase 1 test | Remove `MNQ_LONDON_METALS_E2_ORB_G6_NOMON` from `live_config.py` → replace with `ORB_G6` base. Same commit as research artifact. |
| NOMON fails Phase 2 permutation | Same removal as above |
| NOMON passes all phases | Keep in live_config. Document mechanism and kill criteria. |
| NOFRI/NOTUE fail | Remove from `get_session_filters()` grid wiring in `config.py:733-742`. No live exposure, lower urgency. |
| NOFRI/NOTUE pass | Keep in grid. Consider promoting to live_config if instrument×session warrants. |

---

## Verdict Classification

| Classification | Requirements (ALL must hold) |
|---------------|------------------------------|
| **REAL MICROSTRUCTURE** | Permutation p < 0.05 AND year-by-year stable (≥60% years) AND paired JK significant AND plausible mechanism |
| **PLAUSIBLE BUT UNPROVEN** | Permutation p < 0.05 AND (year stability OR paired significance) but NOT both, OR no mechanism |
| **LIKELY NOISE** | Fails permutation OR fails paired comparison OR unstable across years |

Default assumption: **LIKELY NOISE** unless evidence proves otherwise.

---

## Output Artifacts

1. **Conversation:** Results presented at each phase gate before proceeding
2. **Research artifact:** `research/output/DOW_FILTER_STRESS_TEST.md` — full results, verdicts, evidence
3. **Code changes (if needed):** live_config.py and/or config.py modifications in same commit as research artifact
4. **Script (if Phase 2 reached):** `research/research_dow_permutation.py` — reusable permutation test
