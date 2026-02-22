# Break Quality Research Design

**Goal:** Determine if bar-level compression/explosion patterns before and at the ORB break predict trade outcomes for MGC 0900 and 1000.

**Approach:** Research-first — compute metrics from existing bar data, analyze vs outcomes. Only build pipeline features if statistically significant.

---

## Two-Layer Metric Design

### Layer 1: Pre-Break Compression (Universal)

```
pre_break_compression = avg(pre_break_bar_ranges) / orb_range
```

- **Low values** = bars tight relative to ORB = "coiled spring" setup
- **High values** = bars wide/volatile before break = grinding/indecision
- **Lookahead status:** ZERO — all data known before any entry model triggers
- **Works for:** E0 CB1, E0 CB2+, E1, E3 (all entry types)
- **Minimum bars:** Requires >= 2 pre-break bars to compute

### Layer 2: Explosion Ratio (Post-Break)

```
explosion_ratio = break_bar_range / avg(pre_break_bar_ranges)
```

- **High values** = break bar dramatically wider than preceding bars = explosive breakout
- **Low values** = break bar similar to preceding bars = gradual seepage
- **Lookahead status:** ZERO for E1, E3, E0 CB2+ (break bar complete before entry)
- **LOOKAHEAD for E0 CB1** — entry is intra-bar during the break bar; range not known
- **Works for:** E1, E3, E0 CB2+ only as a real-time filter
- **Descriptive for:** E0 CB1 (compute but cannot use as filter)
- **Minimum bars:** Requires >= 1 pre-break bar for denominator

---

## E0 CB1 Lookahead Handling

E0 places a limit order at the ORB level on the confirm bar itself. For CB1, the break bar IS the confirm bar — entry occurs intra-bar before the break bar's full range is known.

| Entry Type | Layer 1 (Pre-Break) | Layer 2 (Explosion) |
|------------|---------------------|---------------------|
| E0 CB1     | Valid filter         | Descriptive only    |
| E0 CB2+    | Valid filter         | Valid filter         |
| E1         | Valid filter         | Valid filter         |
| E3         | Valid filter         | Valid filter         |

If Layer 1 alone is predictive, E0 CB1 strategies (including MGC 0900's best: E0_RR2.5_CB1_ORB_G4_CONT) benefit directly.

---

## Immediate-Break Edge Case

If the break happens on the first bar after ORB formation, there are zero pre-break bars and both metrics are undefined.

**Rules:**
- Layer 1 computable only with >= 2 pre-break bars
- Layer 2 computable only with >= 1 pre-break bar
- Report what percentage of trades get excluded by each threshold

**Immediate-break cohort:** Analyze days where break occurs on bar 1 as their own group. "Break on first bar" is itself a signal worth characterizing — it's the extreme of what FAST5 already captures.

---

## Research Scope

- **Instrument:** MGC
- **Sessions:** 0900, 1000
- **Entry models:** All (E0, E1, E3) — E0 CB1 gets Layer 1 only for filter candidacy
- **DST split:** Mandatory for 0900 (US DST regime); 1000 is DST-clean
- **Data source:** bars_5m + daily_features (break_ts, ORB levels) + orb_outcomes (outcomes)

---

## Analysis Plan

1. For each trade day with a break at 0900/1000:
   - Query bars_5m between ORB end and break_ts
   - Compute pre_break_compression and explosion_ratio
2. Join with orb_outcomes (win/loss/pnl_r)
3. Split analysis:
   - By metric quartile vs outcome rate
   - By entry model (E0 CB1 separate from E1/E3/E0-CB2+)
   - By DST regime (0900 only)
   - By year (stability check)
4. Statistical tests:
   - t-test: metric values for wins vs losses
   - Threshold sweep: test median, Q1, Q3 as filter cutoffs
   - Report exact p-values
5. Sensitivity: +/- 20% around best threshold
6. Immediate-break cohort: separate outcome analysis

## Success Criteria (per RESEARCH_RULES.md)

- p < 0.01 for metric to be actionable
- Must hold across >= 75% of years
- BH FDR correction if testing multiple thresholds (likely 3-5 thresholds)
- Label as "confirmed edge" only if FDR survives

## If Positive: Implementation Path

1. Add `break_bar_range` and `pre_break_avg_range` to `_detect_break()` in `build_daily_features.py`
2. Compute `explosion_ratio` and `pre_break_compression` as stored columns
3. Create composite filter(s) in `config.py`
4. Re-run `build_daily_features.py` to populate columns
5. Add to discovery grid for relevant sessions
