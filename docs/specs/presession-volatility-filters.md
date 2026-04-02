# Pre-Session Volatility Filters — Deployment Spec

## Research Origin
- **Date:** 2026-04-02
- **Scripts:** `scripts/research/scan_presession_features.py` (T1), `scripts/research/scan_presession_t2t8.py` (T2-T8)
- **Hypothesis:** Pre-session consolidation avoidance (skip quiet days)
- **Finding:** OPPOSITE — bigger prior day range = BETTER WR. Volatility begets volatility.

## Literature Grounding

### Verified from local PDFs (text extracted and confirmed)

| Method | Source | Pages | Key content |
|--------|--------|-------|-------------|
| BH FDR (K=9, q=0.05) | Benjamini & Hochberg 1995 | pp. 289-291 Section 3 | Sequential Bonferroni procedure for FDR control |
| Selection bias / trials disclosure | Bailey & de Prado 2014 | pp. 2-5 | "the most important piece of information missing... is the number of trials attempted" |
| Multiple testing haircut | Harvey & Liu 2015 | pp. 12-14 | "the 50% haircut rule of thumb is a serious mistake — actual haircut is nonlinear" |
| T6 null floor (permutation test) | Aronson 2006 Ch5 | pp. 238-242 | Monte Carlo Permutation Method: "randomly pairing rule output values with scrambled market price changes destroys any predictive power." 5000 permutations, p = fraction exceeding observed. Exactly our T6 implementation. |
| Bootstrap for hypothesis testing | Aronson 2006 Ch5 | pp. 234-236 | White's Reality Check extends bootstrap to data-mined rules |

### NOT verified from local PDFs (training memory only)

| Claim | Cited source | Honest status |
|-------|-------------|---------------|
| WFE > 0.50 threshold | "Carver" | READ pp.78-88 = portfolio allocation (WRONG CHAPTER). No WFE discussion found. UNGROUNDED. |
| Volatility clustering / GARCH | "Carver/GARCH literature" | Not found in pages read. UNGROUNDED. |
| p = (b+1)/(m+1) | "Phipson & Smyth 2010" | Not in resources/. UNGROUNDED. Standard permutation formula. |
| Aronson Ch6 data mining bias | "Aronson Ch6" | Only Ch5 read (hypothesis testing methods). Ch6 NOT extracted. |

## 4 Validated Signals

### 1. PDR_HIGH — prev_day_range/atr >= Q3 threshold

**Where:** MGC LONDON_METALS, MGC EUROPE_FLOW, MNQ EUROPE_FLOW

| Metric | LONDON_METALS MGC | EUROPE_FLOW MGC | EUROPE_FLOW MNQ |
|--------|-------------------|-----------------|-----------------|
| WR Spread (Q5-Q1) | +8.6% | +8.0% | +7.6% |
| T3 WFE | 0.66 | 0.52 | 1.32* |
| T6 null p (BH) | 0.003 PASS | 0.008 PASS | 0.017 PASS |
| T7 per-year | 9/10 (90%) | 7/10 (70%) | 8/10 (80%) |
| T8 cross-inst | 3/3 same sign | 2/3 | 2/3 |

*MNQ EUROPE_FLOW WFE=1.32 flagged SUSPECT — OOS outperformed IS. Monitor.

**Direction:** HIGH_BETTER — trade when prev day was volatile, skip when quiet.
**Mechanism:** GARCH clustering. Yesterday's high ATR predicts today's continuation of volatile conditions. ORB breakouts need momentum to resolve profitably.

### 2. GAP_HIGH — abs(gap_open_points)/atr >= Q3 threshold

**Where:** MGC CME_REOPEN only

| Metric | CME_REOPEN MGC |
|--------|----------------|
| WR Spread (Q5-Q1) | +9.2% |
| T3 WFE | 0.68 |
| T4 sensitivity | ALL same sign (Q20-Q80) |
| T6 null p (BH) | 0.009 PASS |
| T7 per-year | 7/10 (70%) |
| T8 cross-inst | 3/3 same sign |

**Direction:** HIGH_BETTER — bigger gaps predict better ORB resolution.
**Mechanism:** Gap = overnight information asymmetry. Larger gap = stronger directional conviction at session open = ORB breaks carry through.

## 2 Conditional Signals (monitor, don't deploy)

### took_pdh × US_DATA_1000 (MES + MNQ)

- Cross-instrument consistent (3/3 same sign, +5.8-5.9% spread)
- T6 null: PASS (p=0.008-0.016)
- **BUT:** T3 WFE > 1.89 (SUSPECT LEAKAGE), T5 family FAIL (2/8 sessions for MES), T7 ERA_DEPENDENT for MNQ
- **Action:** Track forward performance. Do NOT deploy until WFE normalizes.

## 3 Killed Signals

- overnight_range/atr × NYSE_CLOSE × MES — T6 FAIL (p=0.114)
- took_pdl × NYSE_CLOSE × MES — T6 FAIL (p=0.140)
- prev_day_range/atr × NYSE_OPEN × MNQ — T3 FAIL (OOS sign flip), BH FAIL

## Implementation Plan

### New filter names (for `config.py ALL_FILTERS`):

```
PDR_G60  — prev_day_range / atr_20 >= 0.60 (approx Q3, calibrate per session)
PDR_G80  — prev_day_range / atr_20 >= 0.80 (approx Q4)
GAP_G10  — abs(gap_open_points) / atr_20 >= 0.10 (calibrate for CME_REOPEN MGC)
GAP_G20  — abs(gap_open_points) / atr_20 >= 0.20
```

### Thresholds (from actual data distributions):

```
MNQ prev_day_range: Q1=67, med=165, Q3=290, atr_20 med=197
  → PDR_G60 = prev_day_range/atr >= 0.60 (approx bottom of Q3)
  → PDR_G80 = prev_day_range/atr >= 0.80 (approx median)

MGC (need to query — different distribution)

gap_open_points: median=0, Q3=0.5 for MNQ
  → Gap filter only meaningful for CME_REOPEN (session after overnight gap)
  → Threshold needs per-instrument calibration
```

### Where to apply:

| Filter | Sessions | Instruments |
|--------|----------|-------------|
| PDR_G60/G80 | LONDON_METALS, EUROPE_FLOW | MGC, MNQ |
| GAP_G10/G20 | CME_REOPEN | MGC |

### What NOT to do:

- Do NOT apply PDR filters to NYSE_OPEN MNQ (T3 FAIL — OOS sign flipped)
- Do NOT apply overnight_range filters to any session (3 KILLED, 0 VALIDATED)
- Do NOT apply to sessions not tested (no extrapolation)
- Do NOT combine with existing G-filters without re-running T1-T8 on the combo

### Pipeline changes needed:

1. **`trading_app/config.py`** — Add PDR_G60, PDR_G80, GAP_G10, GAP_G20 to ALL_FILTERS
2. **`trading_app/config.py`** — Add to `get_filters_for_grid()` for applicable sessions only
3. **`pipeline/build_daily_features.py`** — No changes needed (prev_day_range and gap_open_points already computed)
4. **`trading_app/outcome_builder.py`** — Filter evaluation logic for new filter types
5. **Discovery grid run** — Full rebuild with new filters for LONDON_METALS, EUROPE_FLOW, CME_REOPEN

### Acceptance criteria:

- [ ] New filters compute correctly (prev_day_range/atr_20 >= threshold)
- [ ] Filter only applied to specified session × instrument combos
- [ ] Existing strategies unaffected (no regression)
- [ ] Discovery produces new experimental strategies with PDR/GAP filters
- [ ] Drift check passes
- [ ] No lookahead (prev_day_range and gap_open_points are strictly prior-day)
