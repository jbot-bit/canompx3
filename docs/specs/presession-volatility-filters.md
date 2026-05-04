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

## Implementation (DONE — Apr 2 2026)

### Filter names deployed in `config.py ALL_FILTERS`:

| Filter | Class | Threshold | Pass rate |
|--------|-------|-----------|-----------|
| `PDR_R080` | `PrevDayRangeNormFilter` | prev_day_range/atr >= 0.80 | ~45% |
| `PDR_R105` | `PrevDayRangeNormFilter` | prev_day_range/atr >= 1.05 | ~40% |
| `PDR_R125` | `PrevDayRangeNormFilter` | prev_day_range/atr >= 1.25 | ~25% |
| `GAP_R005` | `GapNormFilter` | abs(gap)/atr >= 0.005 | ~50% |
| `GAP_R015` | `GapNormFilter` | abs(gap)/atr >= 0.015 | ~25% |

### Thresholds (calibrated from actual data, all instruments consistent):

```
prev_day_range/atr_20 quantiles (MGC/MNQ/MES all within 0.02):
  Q40=0.83  Q50=0.94  Q60=1.05  Q75=1.25  Q80=1.35

MGC abs(gap_open_points)/atr_20:
  Q40=0.004  Q50=0.005  Q60=0.008  Q75=0.016  Q80=0.024
```

### Routing in `get_filters_for_grid()`:

| Filter | Sessions | Instruments |
|--------|----------|-------------|
| PDR_R080/R105/R125 | LONDON_METALS, EUROPE_FLOW | MGC, MNQ |
| GAP_R005/R015 | CME_REOPEN | MGC |

### What NOT to do:

- Do NOT apply PDR filters to NYSE_OPEN MNQ (T3 FAIL — OOS sign flipped)
- Do NOT apply overnight_range filters to any session (3 KILLED, 0 VALIDATED)
- Do NOT apply to sessions not tested (no extrapolation)
- Do NOT combine with existing G-filters without re-running T1-T8 on the combo

### What was changed:

1. **`trading_app/config.py`** — `PrevDayRangeNormFilter` + `GapNormFilter` classes, 5 instances in `ALL_FILTERS`, session routing in `get_filters_for_grid()`
2. **`pipeline/build_daily_features.py`** — No changes (prev_day_range and gap_open_points already computed)
3. **`trading_app/outcome_builder.py`** — No changes (filter evaluation via `matches_row` inherited from StrategyFilter)

### Remaining:

- [ ] Discovery grid run with new filters for LONDON_METALS, EUROPE_FLOW, CME_REOPEN
- [ ] Behavioral tests for new filter classes (review finding #2)
- [ ] Validate discovered strategies through full pipeline (validator → edge families)
