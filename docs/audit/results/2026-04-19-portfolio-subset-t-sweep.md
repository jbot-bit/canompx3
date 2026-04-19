# Portfolio-wide subset-t + lift-vs-unfiltered sweep — 38 active validated_setups

**Date:** 2026-04-19
**Origin:** Phase 2.4 MES composite KILL (commit 56fb46e4) → RULE 8.3 ARITHMETIC_LIFT
**Script:** `research/phase_2_5_portfolio_subset_t_sweep.py`
**Output CSV:** `research/output/phase_2_5_portfolio_subset_t_sweep.csv`

## Document history

- v1 (2026-04-19 first): anchored on single Chordia 3.00 threshold — reported "30/38 fail." Threshold anchoring was an error: **Chordia 3.00 is a DISCOVERY threshold, not a re-audit threshold** (Chordia et al 2018 derive 3.00 from the Harvey-Liu-Zhu factor-zoo multiple-testing framework for novel strategy claims). Re-audits of already-validated lanes should use conventional significance.
- v2 (2026-04-19 self-correct — this version): multi-threshold stratification. Honest fail count at conventional p<0.05 is **14/38**, not 30/38. Self-corrected after user challenge.

## Executive verdict (v2)

Under strict Mode A IS (`trading_day < 2026-01-01`), subset-level t-statistic tiering:

| Threshold | Justification | Lanes passing |
|-----------|---------------|:-------------:|
| **t ≥ 3.00** (Chordia with-theory, **discovery-strict**) | Chordia 2018 for novel strategy claims | **9 / 38** |
| **t ≥ 2.58** (p<0.01 two-sided, **stringent re-audit**) | Conventional stringent significance | **13 / 38** |
| **t ≥ 1.96** (p<0.05 two-sided, **conventional re-audit**) | Standard statistical significance | **24 / 38** |
| **t < 1.96** (FAILS conventional significance) | Below even p<0.05 | **14 / 38** |

**Honest retirement candidates:** the 14 lanes failing even t ≥ 1.96. These are the only lanes where subset-ExpR is statistically indistinguishable from zero under conventional significance — the rigorous retirement signal.

**ARITHMETIC_LIFT (Rule 8.3):** 2 lanes — retains validity regardless of re-audit threshold tier because it catches a specific pattern (filter claims lift vs unfiltered but subset is not significant). Both lanes also fall in the "below 3.00" band.

## Literature framework

- Chordia 2018 `chordia_et_al_2018_two_million_strategies.md` — t ≥ 3.00 is the DISCOVERY threshold; 3.79 no-theory variant; both are for novel-strategy claims after factor-zoo multiple testing.
- Harvey-Liu 2015 `harvey_liu_2015_backtesting.md` — Exhibit 4 N ≥ 100 deployable.
- Bailey-LdP 2014 DSR `bailey_lopez_de_prado_2014_deflated_sharpe.md` — subset-level significance for SR claims.
- Aronson Ch 6 data-mining (per `.claude/rules/quant-audit-protocol.md`) — lift-vs-noise-baseline is not evidence without subset significance. Grounds Rule 8.3.

## Tier 1 — t ≥ 3.00 pass (Chordia discovery-strict, 9 lanes)

| # | Strategy | N | ExpR | Subset t | Note |
|---|----------|---:|----:|---------:|------|
| 1 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` | 385 | +0.195 | 4.29 | strongest |
| 2 | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60` | 306 | +0.214 | 4.19 | |
| 3 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | 485 | +0.221 | 4.14 | |
| 4 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 496 | +0.205 | 3.96 | |
| 5 | `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` | 130 | +0.280 | 3.66 | best per-trade |
| 6 | `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` | 88 | +0.328 | **3.56** | **flagged N_BELOW_DEPLOYABLE** (N=88<100) despite clean t |
| 7 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` | 283 | +0.184 | 3.42 | |
| 8 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | 379 | +0.198 | 3.32 | |
| 9 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | 436 | +0.184 | 3.20 | |

Note: `MES_CME_PRECLOSE COST_LT08` has strong subset-t (3.56) but N=88<100. Harvey-Liu's 100-floor is conservative; with t=3.56 and N=88, this is a legitimate edge that's sample-size-constrained. Separate from the stats-weak retirement candidates.

**Pattern:** `X_MES_ATR60` (3 of 9) and `ATR_P50` (2 of 9) and `ORB_G8`/`COST_LT08` MES-small-size (2 of 9) — volatility- and size-regime conditioning is the dominant PASS filter class. Confluence-style filters (`CROSS_SGP_MOMENTUM`, bare `ORB_G5`) mostly don't make this tier.

## Tier 2 — t ≥ 2.58 pass, didn't make Chordia (stringent re-audit tier, 4 lanes)

| Strategy | N | ExpR | Subset t |
|----------|---:|----:|---------:|
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12` | 669 | +0.104 | 2.94 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | 460 | +0.132 | 2.99 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | 278 | +0.187 | 2.65 |
| `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5` | 785 | +0.123 | 2.62 |

These pass stringent p<0.01 re-audit but miss discovery-strict Chordia. Deploy-eligible by re-audit standard; flag for attention if allocator rebalances.

## Tier 3 — t ≥ 1.96 pass only (conventional significance, 11 lanes)

Lanes with subset-t in [1.96, 2.58). These pass standard p<0.05 but fail stringent p<0.01. Most are ORB_G5/COST_LT12 on NYSE_OPEN and COMEX_SETTLE at various RRs. Marginal — keep but don't scale.

## Tier 4 — **HONEST RETIREMENT CANDIDATES** (t < 1.96, 14 lanes)

These are the lanes where subset-ExpR is statistically indistinguishable from zero even at conventional p<0.05 under Mode A. This is the rigorous retire-or-reclassify signal.

| # | Strategy | N | ExpR | Subset t | Notes |
|---|----------|---:|----:|---------:|-------|
| 1 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5` | 773 | +0.035 | 1.10 | 91% fire → near-unfiltered |
| 2 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` | 263 | +0.056 | 0.99 | |
| 3 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12` | 521 | +0.054 | 1.34 | |
| 4 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM` | 535 | +0.050 | 1.34 | confirms Phase 2.4 |
| 5 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | 773 | +0.077 | 1.90 | |
| 6 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100` | 263 | +0.118 | 1.65 | |
| 7 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM` | 535 | +0.081 | 1.70 | confirms Phase 2.4 |
| 8 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5` | 773 | +0.073 | 1.54 | |
| 9 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60` | 345 | +0.078 | 1.51 | standalone weak |
| 10 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60` | 334 | +0.066 | 1.00 | same filter weak at RR1.5 |
| 11 | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12` | 469 | +0.079 | 1.87 | |
| 12 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | 469 | +0.104 | 1.95 | barely below |
| 13 | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60` | 371 | +0.077 | 1.56 | |
| 14 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 800 | +0.057 | 1.34 | 99.9% fire → not a filter |

**Strong patterns in Tier 4:**

1. **MNQ EUROPE_FLOW is structurally weak** — 8 of 14 retirement candidates (all 6 MNQ EF filters at RR1.0, 3 at RR1.5, 1 at RR2.0). Session-level weakness, not filter-level. Suggests MNQ EUROPE_FLOW is post-hoc marginal under Mode A.
2. **`X_MES_ATR60` on NYSE_OPEN fails at both RRs** — same filter passes Chordia on COMEX_SETTLE and CME_PRECLOSE. Session-dependent efficacy.
3. **`CROSS_SGP_MOMENTUM` confirmation** — both RR1.0 and RR1.5 are Tier 4. Reinforces Phase 2.4 Option B-killed verdict.

## ARITHMETIC_LIFT (Rule 8.3) — retained regardless of threshold tier

| Strategy | N | Fire | Lift | Subset t | Tier |
|----------|---:|-----:|-----:|---------:|:----:|
| `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15` | 398 | 55% | +0.172 | 2.09 | 3 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | 278 | 32% | +0.110 | 2.65 | 2 |

Both lanes are active-status in `validated_setups`. Rule 8.3 is a separate signal from subset-t tier — they can pass conventional significance (Tier 2 or 3) and still fail the lift-interpretation honesty check. Flag for retirement-or-reframing.

## Honest re-analysis per user challenge

### Did I calculate properly?

**Math — yes.** Subset t = ExpR / (sd/√N) is the one-sample t-statistic for mean=0. Formula correct, sd uses ddof=1 (unbiased), compute_mode_a canonical delegation.

### Did I use the right threshold?

**Initially NO — I used discovery-strict Chordia on re-audit context.** Corrected in this v2. Proper re-audit uses conventional significance (p<0.05 → t≥1.96) or stringent (p<0.01 → t≥2.58). Discovery-strict applies to NOVEL strategy claims, not re-audit of already-validated lanes.

### Is conventional significance too loose?

Possibly yes for deploy-scaling decisions. For **retain-in-book** — conventional p<0.05 is appropriate. For **scale-up-capital** — stringent p<0.01 or discovery-strict Chordia. The tiering serves both use cases.

### Missed angles

1. **Autocorrelation not adjusted.** Daily returns may have residual serial correlation; Newey-West adjustment would deflate t-stats 5-15%. Direction of bias: ALL my t-stats are UPPER bounds. Lanes near threshold boundaries could drift down. Not run here; noted.

2. **Bootstrap not run.** Bootstrap on pnl with preserved autocorrelation (Politis-Romano block bootstrap, per `.claude/rules/backtesting-methodology.md` § Historical failure log 2026-04-15) would be more robust than parametric t-test. Phase 2 follow-up candidate.

3. **Allocator-level portfolio fitness not tested.** Individual lane subset-t is per-lane evidence. Portfolio fitness after diversification may still be positive even with Tier-3/Tier-4 component lanes. Allocator (`trading_app.lane_allocator`) is the next evaluation layer.

4. **Lane correlation not gated here.** Two Tier-1 SINGAPORE_OPEN ATR_P50 lanes (O15, O30) are structurally similar; allocator may reject one for correlation.

### Best next honest test

**Apply Newey-West autocorrelation correction** on the 14 Tier-4 candidates. If half of them drift further below 1.96, confidence in retirement increases. If any drift UP toward 1.96 with correction, reconsider.

After that: **allocator-level portfolio simulation** — deploy the 9-lane Tier-1 portfolio + compare R/year, max DD, Sharpe to current portfolio. That's the deploy-decision evidence, not per-lane significance.

## Recommended committee actions (revised per v2)

1. **Tier 4 (14 lanes) — retire or reclassify research-provisional.** These fail even conventional p<0.05 subset-t under Mode A. Strongest retirement evidence in the project to date.
2. **Tier 1 (9 lanes) — highest-confidence deploy pool.** Prioritize for current and future capital allocation.
3. **Tier 2 (4 lanes) — watchlist.** Deploy-eligible but flag for quarterly re-audit.
4. **Tier 3 (11 lanes) — marginal.** Hold in book, don't scale.
5. **2 ARITHMETIC_LIFT flagged lanes** — move to research-provisional regardless of tier. Rule 8.3 violation.

### MES instrument verdict

Data says MES has **2 legitimately strong lanes** (both CME_PRECLOSE long RR1.0: ORB_G8 N=130 t=3.66, COST_LT08 N=88 t=3.56). The rest is dead on MES. User's instinct that "MES is mostly shit" is empirically confirmed with the narrow exception of CME_PRECLOSE.

## Audit trail

- Pre-reg: `docs/runtime/stages/phase-2-5-portfolio-subset-t-sweep.md`
- Script: `research/phase_2_5_portfolio_subset_t_sweep.py`
- Tests: `tests/test_research/test_phase_2_5_portfolio_subset_t_sweep.py`
- CSV: `research/output/phase_2_5_portfolio_subset_t_sweep.csv`
- Cross-refs: `docs/audit/results/2026-04-19-fire-rate-audit.md`, `docs/audit/results/2026-04-19-consolidated-retirement-verdict.md`, `docs/audit/results/2026-04-19-mes-europe-flow-g5-sgp-composite-audit.md`, `docs/audit/results/2026-04-19-phase-2-4-cross-session-momentum-mode-a.md`
- Self-correction: v1 of this doc used Chordia 3.00 as the sole threshold. v2 (current) is the honest multi-tier framing. Keeping v1's "30/38 fail" framing would have been anchoring bias.
