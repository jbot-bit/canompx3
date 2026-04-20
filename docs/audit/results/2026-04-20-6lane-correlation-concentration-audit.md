# 6-Lane Correlation + Mechanism-Concentration Audit (2026-04-20)

**Branch:** `research/six-lane-correlation-concentration-audit`
**Script:** `research/audit_6lane_correlation_concentration.py`
**Data window:** 2019-05-06 → 2026-04-16 (1,789 Brisbane trading days)
**Deployed lanes:** 6 (MNQ, session-spread)

## The question this replaced

Previous framing ("audit the 38-lane allocator tail — is #6 → #7 cutoff well-calibrated?") was flagged as tunnel-visioned (docs/audit/results → Reframing, 2026-04-20). Replaced with:

> **"What is the correlation structure across the 6 deployed lanes, and what
> single mechanism could draw down them all simultaneously?"**

Reason for reframe: the allocator optimised per-lane score (ExpR × Sharpe), not portfolio-level correlation structure. If deployed lanes turned out to be highly correlated, nominal Sharpe would overstate portfolio resilience — and "deploy a 7th lane" would be the wrong next move if the real risk was a 7× correlated-drawdown mechanism.

## Literature grounding

- **Carver 2015 Systematic Trading Ch 11** (pp 165-175, direct PDF extract 183-193):
  - Semi-automatic-trader allocation formula: 100% / max_concurrent_bets (p169)
  - Subsystem correlation for dynamic trading systems ≈ 0.7 × instrument return correlation (p167-168)
  - Instrument diversification multiplier D **hard cap 2.5** (p170) — explicit rationale: "in a crisis such as the 2008 crash, correlations tend to jump higher exposing us to serious losses — an example of unpredictable risk"
  - Effective independent bets N_eff = D² for equal weights (p170)
  - Extract already indexed at `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` (Ch 9-10 body); Ch 11 extract should be added in follow-up.

## Method

1. Load 6 deployed lanes from `docs/runtime/lane_allocation.json`.
2. For each lane: JOIN `daily_features` × `orb_outcomes` on (trading_day, symbol, orb_minutes) + outcome keys; apply canonical `trading_app.config.ALL_FILTERS[filter_type].matches_row()` per row (**no re-encoding** — rule #4 institutional-rigor).
3. Build wide matrix: rows = trading_day, cols = strategy_id, cells = `pnl_r` (NaN where filter didn't fire that day, treated as 0.0 for portfolio-accounting correlation).
4. Pearson correlation matrix on the full-history wide matrix.
5. Falsification battery (to guard against NaN-as-0 artefacts, regime bias, and crisis-correlation-jump):
   - **F1** per-year correlation 2019-2026
   - **F2** conditional correlation on days where both lanes traded (removes NaN-as-0 influence)
   - **F3** stress windows: COVID crash 2020-Feb-15 → 2020-May-15, 2022 bear H1, 2024-Aug vol spike

## Structural concentration flags (pre-audit)

| Axis | Concentration |
|---|---|
| Instrument | **MNQ × 6** (100%) |
| Entry model | **E2 × 6** (100%) |
| Filter | ORB_G5 × 3 (50%), COST_LT12 × 2 (33%), ATR_P50 × 1 |
| RR target | RR1.5 × 5 (83%), RR1.0 × 1 |
| Session region | ASIA × 2, US × 2, EUROPE × 1, EUROPE/US × 1 |
| Aperture | 5-min × 4, 15-min × 2 |

First-order read: looks heavily concentrated (same instrument, same entry model, filter family dominated by range-quality-gate). Behavioural audit below contradicts first-order read.

## Results — full-history correlation matrix

|  | EUROPE_FLOW | SINGAPORE | COMEX_SETTLE | NYSE_OPEN | TOKYO_OPEN | US_DATA_1000 |
|---|---|---|---|---|---|---|
| EUROPE_FLOW_ORB_G5 | 1.000 | 0.009 | -0.008 | 0.039 | 0.036 | 0.002 |
| SINGAPORE_ATR_P50 | | 1.000 | 0.028 | -0.053 | 0.026 | -0.033 |
| COMEX_SETTLE_ORB_G5 | | | 1.000 | 0.026 | 0.025 | 0.023 |
| NYSE_OPEN_COST_LT12 | | | | 1.000 | -0.020 | 0.004 |
| TOKYO_OPEN_COST_LT12 | | | | | 1.000 | -0.011 |
| US_DATA_1000_ORB_G5 | | | | | | 1.000 |

**Off-diagonal statistics (15 pairs):**
- mean +0.006 | median +0.009 | max +0.039 | min -0.053
- **0 pairs ≥ 0.30**

## Carver Ch 11 diversification metrics

| Metric | Value | Interpretation |
|---|---|---|
| Portfolio sigma (equal weights) | 0.4146 | Low |
| Diversification multiplier D | **2.412** | Near Carver's 2.5 hard cap |
| Effective independent bets N_eff = D² | **5.82** | 97% efficiency vs 6 nominal |
| Fragility ratio (nominal / N_eff) | 1.03 | Essentially no fragility |
| PC1 share of variance | **18.0%** | No dominant factor |
| Eigenvalues (desc) | [1.08, 1.05, 1.02, 0.99, 0.96, 0.90] | Near-identity spectrum → maximally uncorrelated |

## Falsification battery

### F1 — per-year correlation stability

| Year | N_days | mean_corr | max_corr | N_eff | PC1 |
|---|---|---|---|---|---|
| 2019 | 169 | +0.001 | 0.117 | 5.98 | 19.6% |
| 2020 | 258 | +0.021 | 0.141 | 5.42 | 21.4% |
| 2021 | 258 | -0.010 | 0.063 | 6.33 | 19.0% |
| 2022 | 258 | +0.010 | 0.081 | 5.73 | 19.7% |
| 2023 | 257 | +0.000 | 0.084 | 5.99 | 20.4% |
| 2024 | 259 | +0.027 | 0.197 | 5.30 | 21.3% |
| 2025 | 258 | -0.026 | 0.067 | 6.88 | 20.1% |
| 2026 | 72 | +0.043 | 0.287 | 4.95 | 25.7% |

No year's mean pairwise correlation exceeds 0.05. No year's max pairwise correlation reaches 0.30. N_eff stays in [4.95, 6.88]. Uncorrelatedness is stable across the full 7-year window.

### F2 — conditional correlation (both lanes traded same day)

Same result after removing NaN-as-0: mean +0.005, max +0.042, min -0.074, 0 pairs ≥ 0.30. The full-history result was not an artefact of 0-padding.

### F3 — stress-period correlation (Carver's crisis-jump warning test)

| Stress window | N | mean_corr | max_corr | N_eff | PC1 |
|---|---|---|---|---|---|
| COVID crash 2020-02-15 → 2020-05-15 | 64 | +0.003 | 0.278 | 5.92 | 23.6% |
| 2022 bear H1 | 149 | -0.001 | 0.074 | 6.04 | 20.5% |
| 2024-08 vol spike | 22 | -0.070 | 0.334 | 9.21 | 31.5% |

Crisis correlations do **NOT** jump. The 2024-08 max of 0.334 (above the 0.30 flag threshold) is a one-pair spike in a 22-day window, not a book-wide regime change; N_eff actually spiked to 9.21 (artefact of small N). COVID peak max was 0.278, also not a book-wide jump.

**This falsifies Carver's crisis-correlation-jump warning for this specific book.** Reason (structural): the 6 lanes trade **temporally non-overlapping sessions** of the same instrument — NYSE session crisis cannot price-infect TOKYO session until the news cycle propagates. Mechanism concentration exists in structural axes (filter, entry model) but not in the temporal dimension that drives correlated losses.

## Verdict on the original hypothesis

**HYPOTHESIS:** 6/6 MNQ + 6/6 E2 + 3/6 ORB_G5 concentration → correlated-drawdown risk that nominal portfolio Sharpe overstates.

**REFUTED.** Observed mean pairwise correlation +0.006; crisis-period max never reached the book-wide jump threshold. 6 nominal lanes deliver 5.82 effective independent bets across all regimes tested.

## Implications (what this changes)

1. **The "deploy a 7th lane for diversification" case collapses.** Current D = 2.412 is already near Carver's 2.5 hard cap. Adding a 7th MNQ lane on an 8th session might push to D ≈ 2.5, but marginal effective-bet gain is <1 and we'd need to re-size existing 6 lanes down. Zero leverage.

2. **Next EV is per-lane quality, not lane count.** Portfolio Sharpe scales as avg_lane_Sharpe × √N_eff. With N_eff already at 5.82, doubling to 11.64 would need 12-14 genuinely uncorrelated lanes (hard, and hit Carver cap). Doubling avg_lane_Sharpe from ~0.8 to ~1.0 is the easier lever.

3. **Risk model can use independence assumption.** Daily-VaR and drawdown modelling should treat the 6 lanes as near-independent (Corr matrix supplied for reference; still load the empirical C rather than assuming I).

4. **Crisis-correlation-jump warning (Carver p170) is backtest-refuted for this book.** Should be re-verified under live/shadow regime: add a 30-day rolling pairwise correlation monitor. If any pair exceeds 0.30 rolling for > 10 days → regime-change alarm, investigate before it becomes a drawdown event.

5. **Allocator validated on portfolio structure.** The allocator's scoring function didn't see correlation, but the session-spread it chose coincidentally achieved near-optimal diversification. This is evidence for **session-spread as a first-class allocator criterion** (even independent of per-lane score) in future additions.

## Non-scope / deferred

- **Carver Ch 11 literature extract** — the Ch 9-10 extract exists at `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`. Ch 11 body has been read from PDF (pages 183-193) for this audit but not yet written as a dedicated extract file. Follow-up: write `carver_2015_ch11_portfolios.md` referenced from this audit for future citation chains.
- **Live / shadow rolling-correlation monitor** — to verify crisis-jump refutation holds forward. To be wired into `trading_app/` monitor layer.
- **Per-lane Sharpe improvement research** — flagged as highest-leverage next-EV path by this audit's conclusion #2; not executed here.

## Reproducibility

```bash
cd <repo-root>
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db \
PYTHONPATH=. PYTHONIOENCODING=utf-8 \
python research/audit_6lane_correlation_concentration.py
```

Artefacts written to `research/output/`:
- `6lane_correlation_matrix.csv`
- `6lane_correlation_by_year.csv`
- `6lane_conditional_correlation_pairs.csv`

All read-only against canonical `gold.db`; no DB writes.
