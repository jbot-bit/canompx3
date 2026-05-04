# Phase B candidate evidence — clean comprehensive-scan survivors

**Date:** 2026-04-28  
**Plan:** `docs/plans/2026-04-28-edge-extraction-phased-plan.md` Phase B  
**Scan basis:** `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md` (clean rebuild post-E2-LA-fix)  
**Holdout:** Mode A strict (`trading_day < 2026-01-01`)

## Scope

Apply Phase 0 deploy-gate evidence stack (C5 DSR, C6 WFE, C7 N, C8 dir-match w/ power floor, C9 era stability)
plus B6 lane-correlation matrix (Carver Ch11) to the 4 mechanism-grounded clean candidates from the rebuilt
comprehensive scan. dow_thu / is_monday / is_friday survivors are NOT in scope (no mechanism per Aronson EBTA Ch6).

## Methodology citations

- **DSR formula:** Bailey-LdP 2014 Eq.2 (`docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`)
- **Effective-N:** Bailey-LdP 2014 Eq.9 with ρ̂=0.5 default (project standard, Bailey example p.9-10)
- **OOS power floor:** Phase 0 Amendment 3.2 + `feedback_oos_power_floor.md` — N_OOS<50 → UNVERIFIED
- **Era stability:** Phase 0 Criterion 9 — no era ExpR < -0.05 with N_era >= 20
- **Mechanism grounding:** Chan 2013 Ch7 (intraday momentum), Fitschen 2013 Ch3 (path-of-least-resistance), Carver 2015 Ch9-10 (vol-targeting)
- **Scratch policy:** include-as-zero per `docs/specs/outcome_builder_scratch_eod_mtm.md` (canonical Stage 5 fix)

## Per-candidate results

### B-MES-EUR — MES EUROPE_FLOW O15 RR1.0 long + ovn_range_pct_GT80

**Mechanism:** Chan Ch7 + Fitschen Ch3 (intraday continuation premium after high overnight participation)

**Sample:**
- N_IS = 186, N_OOS = 9, N_OFF_IS = 702
- ExpR_IS_on = 0.1434, ExpR_OOS_on = 0.2331, Δ_IS = 0.2459, Δ_OOS = 0.2721
- Sharpe_ann_IS = 0.887, Sharpe_ann_OOS = — (UNVERIFIED if N_OOS < 50)
- Skewness γ̂₃ = -0.556, Kurtosis γ̂₄ = 1.331

**Phase 0 gates:**
- **C5 DSR (Pathway A discovery, M=K_family=1850):** DSR_FAIL — DSR = 0.0003 vs threshold ≥ 0.95, SR0 = 0.4273 per-trade (ρ̂=0.5, N̂=926, V[SR_n]_per_trade=0.0175, SR_per_trade=0.1658)
- **C5 DSR (Pathway B K=1, theory-citation path per Amendment 3.0):** DSR_PB_PASS — DSR_PB = 0.9845 vs threshold ≥ 0.95
- **C6 WFE:** WFE_UNVERIFIED — WFE = — (Sharpe_OOS / Sharpe_IS); UNVERIFIED if N_OOS < 50
- **C7 N:** C7_PASS — N_IS = 186 vs threshold ≥ 100
- **C8 dir-match (power-floor aware):** C8_UNVERIFIED_LOWPOWER — dir_match = None, Cohen's d = —, Δ_IS sign = +, Δ_OOS sign = +
- **C9 era stability:** C9_PASS — eligible eras (N≥20) = 6, eras with ExpR < -0.05 = 0

**Per-year breakdown (IS-on, N≥1):**

| Year | N | ExpR |
|---:|---:|---:|
| 2019 | 12 | +0.0034 |
| 2020 | 37 | +0.2761 |
| 2021 | 23 | +0.0948 |
| 2022 | 24 | +0.1699 |
| 2023 | 28 | +0.1464 |
| 2024 | 29 | +0.1738 |
| 2025 | 33 | +0.0310 |

**B6 lane-correlation vs deployed 6 lanes (Pearson on shared trading_day pnl_r):**

| Deployed lane | Correlation |
|---|---:|
| EUROPE_FLOW ORB_G5 | +0.362 |
| SINGAPORE ATR_P50 15m | +0.012 |
| COMEX_SETTLE ORB_G5 | -0.065 |
| NYSE_OPEN COST_LT12 | -0.083 |
| TOKYO COST_LT12 | -0.023 |
| US_DATA_1000 ORB_G5 15m | -0.037 |

**Overall verdict:** **PATHWAY_B_ELIGIBLE**

### B-MES-LON — MES LONDON_METALS O30 RR2.0 long + ovn_range_pct_GT80

**Mechanism:** Chan Ch7 + Fitschen Ch3

**Sample:**
- N_IS = 183, N_OOS = 12, N_OFF_IS = 717
- ExpR_IS_on = 0.2417, ExpR_OOS_on = 0.4407, Δ_IS = 0.3409, Δ_OOS = 0.6702
- Sharpe_ann_IS = 0.944, Sharpe_ann_OOS = — (UNVERIFIED if N_OOS < 50)
- Skewness γ̂₃ = 0.219, Kurtosis γ̂₄ = 1.074

**Phase 0 gates:**
- **C5 DSR (Pathway A discovery, M=K_family=1850):** DSR_FAIL — DSR = 0.0002 vs threshold ≥ 0.95, SR0 = 0.4312 per-trade (ρ̂=0.5, N̂=926, V[SR_n]_per_trade=0.0178, SR_per_trade=0.1780)
- **C5 DSR (Pathway B K=1, theory-citation path per Amendment 3.0):** DSR_PB_PASS — DSR_PB = 0.9928 vs threshold ≥ 0.95
- **C6 WFE:** WFE_UNVERIFIED — WFE = — (Sharpe_OOS / Sharpe_IS); UNVERIFIED if N_OOS < 50
- **C7 N:** C7_PASS — N_IS = 183 vs threshold ≥ 100
- **C8 dir-match (power-floor aware):** C8_UNVERIFIED_LOWPOWER — dir_match = None, Cohen's d = —, Δ_IS sign = +, Δ_OOS sign = +
- **C9 era stability:** C9_PASS — eligible eras (N≥20) = 6, eras with ExpR < -0.05 = 0

**Per-year breakdown (IS-on, N≥1):**

| Year | N | ExpR |
|---:|---:|---:|
| 2019 | 8 | +0.3870 |
| 2020 | 36 | +0.2679 |
| 2021 | 21 | +0.3615 |
| 2022 | 31 | +0.2988 |
| 2023 | 25 | +0.0865 |
| 2024 | 27 | +0.5131 |
| 2025 | 35 | -0.0397 |

**B6 lane-correlation vs deployed 6 lanes (Pearson on shared trading_day pnl_r):**

| Deployed lane | Correlation |
|---|---:|
| EUROPE_FLOW ORB_G5 | +0.061 |
| SINGAPORE ATR_P50 15m | +0.168 |
| COMEX_SETTLE ORB_G5 | -0.176 |
| NYSE_OPEN COST_LT12 | +0.061 |
| TOKYO COST_LT12 | -0.021 |
| US_DATA_1000 ORB_G5 15m | -0.055 |

**Overall verdict:** **PATHWAY_B_ELIGIBLE**

### B-MNQ-NYC — MNQ NYSE_CLOSE O5 RR1.0 long + ovn_range_pct_GT80

**Mechanism:** Chan Ch7 + Fitschen Ch3

**Sample:**
- N_IS = 160, N_OOS = 10, N_OFF_IS = 580
- ExpR_IS_on = 0.2187, ExpR_OOS_on = 0.6467, Δ_IS = 0.2027, Δ_OOS = 0.4030
- Sharpe_ann_IS = 1.533, Sharpe_ann_OOS = — (UNVERIFIED if N_OOS < 50)
- Skewness γ̂₃ = -0.694, Kurtosis γ̂₄ = 1.971

**Phase 0 gates:**
- **C5 DSR (Pathway A discovery, M=K_family=1850):** DSR_FAIL — DSR = 0.0445 vs threshold ≥ 0.95, SR0 = 0.4547 per-trade (ρ̂=0.5, N̂=926, V[SR_n]_per_trade=0.0198, SR_per_trade=0.3049)
- **C5 DSR (Pathway B K=1, theory-citation path per Amendment 3.0):** DSR_PB_PASS — DSR_PB = 0.9997 vs threshold ≥ 0.95
- **C6 WFE:** WFE_UNVERIFIED — WFE = — (Sharpe_OOS / Sharpe_IS); UNVERIFIED if N_OOS < 50
- **C7 N:** C7_PASS — N_IS = 160 vs threshold ≥ 100
- **C8 dir-match (power-floor aware):** C8_UNVERIFIED_LOWPOWER — dir_match = None, Cohen's d = —, Δ_IS sign = +, Δ_OOS sign = +
- **C9 era stability:** C9_PASS — eligible eras (N≥20) = 5, eras with ExpR < -0.05 = 0

**Per-year breakdown (IS-on, N≥1):**

| Year | N | ExpR |
|---:|---:|---:|
| 2019 | 8 | +0.3429 |
| 2020 | 30 | +0.1143 |
| 2021 | 27 | +0.3312 |
| 2022 | 16 | +0.3651 |
| 2023 | 21 | +0.1496 |
| 2024 | 30 | +0.3178 |
| 2025 | 28 | +0.0488 |

**B6 lane-correlation vs deployed 6 lanes (Pearson on shared trading_day pnl_r):**

| Deployed lane | Correlation |
|---|---:|
| EUROPE_FLOW ORB_G5 | -0.018 |
| SINGAPORE ATR_P50 15m | +0.062 |
| COMEX_SETTLE ORB_G5 | +0.044 |
| NYSE_OPEN COST_LT12 | +0.022 |
| TOKYO COST_LT12 | -0.041 |
| US_DATA_1000 ORB_G5 15m | +0.082 |

**Overall verdict:** **PATHWAY_B_ELIGIBLE**

### B-MNQ-COX — MNQ COMEX_SETTLE O5 RR1.0 long + garch_vol_pct_GT70

**Mechanism:** Carver Ch9-10 (vol-targeting / forecast-vol-conditioned execution)

**Sample:**
- N_IS = 199, N_OOS = 17, N_OFF_IS = 680
- ExpR_IS_on = 0.2453, ExpR_OOS_on = 0.1344, Δ_IS = 0.2286, Δ_OOS = 0.2538
- Sharpe_ann_IS = 1.692, Sharpe_ann_OOS = — (UNVERIFIED if N_OOS < 50)
- Skewness γ̂₃ = -0.676, Kurtosis γ̂₄ = 1.460

**Phase 0 gates:**
- **C5 DSR (Pathway A discovery, M=K_family=2700):** DSR_FAIL — DSR = 0.0810 vs threshold ≥ 0.95, SR0 = 0.3832 per-trade (ρ̂=0.5, N̂=1350, V[SR_n]_per_trade=0.0132, SR_per_trade=0.2746)
- **C5 DSR (Pathway B K=1, theory-citation path per Amendment 3.0):** DSR_PB_PASS — DSR_PB = 0.9998 vs threshold ≥ 0.95
- **C6 WFE:** WFE_UNVERIFIED — WFE = — (Sharpe_OOS / Sharpe_IS); UNVERIFIED if N_OOS < 50
- **C7 N:** C7_PASS — N_IS = 199 vs threshold ≥ 100
- **C8 dir-match (power-floor aware):** C8_UNVERIFIED_LOWPOWER — dir_match = None, Cohen's d = —, Δ_IS sign = +, Δ_OOS sign = +
- **C9 era stability:** C9_PASS — eligible eras (N≥20) = 4, eras with ExpR < -0.05 = 0

**Per-year breakdown (IS-on, N≥1):**

| Year | N | ExpR |
|---:|---:|---:|
| 2020 | 10 | +0.3313 |
| 2021 | 23 | +0.1927 |
| 2022 | 70 | +0.2472 |
| 2023 | 7 | +0.3180 |
| 2024 | 48 | +0.1006 |
| 2025 | 41 | +0.4073 |

**B6 lane-correlation vs deployed 6 lanes (Pearson on shared trading_day pnl_r):**

| Deployed lane | Correlation |
|---|---:|
| EUROPE_FLOW ORB_G5 | -0.005 |
| SINGAPORE ATR_P50 15m | +0.037 |
| COMEX_SETTLE ORB_G5 | +0.773 (overlap concern) |
| NYSE_OPEN COST_LT12 | +0.100 |
| TOKYO COST_LT12 | +0.054 |
| US_DATA_1000 ORB_G5 15m | -0.023 |

**Overall verdict:** **PATHWAY_B_ELIGIBLE**

## Aggregate verdict

| Candidate | C5 DSR (Path A) | C5 DSR (Path B K=1) | C6 WFE | C7 N | C8 dir | C9 era | Overall |
|---|---|---|---|---|---|---|---|
| B-MES-EUR | DSR_FAIL | DSR_PB_PASS | WFE_UNVERIFIED | C7_PASS | C8_UNVERIFIED_LOWPOWER | C9_PASS | **PATHWAY_B_ELIGIBLE** |
| B-MES-LON | DSR_FAIL | DSR_PB_PASS | WFE_UNVERIFIED | C7_PASS | C8_UNVERIFIED_LOWPOWER | C9_PASS | **PATHWAY_B_ELIGIBLE** |
| B-MNQ-NYC | DSR_FAIL | DSR_PB_PASS | WFE_UNVERIFIED | C7_PASS | C8_UNVERIFIED_LOWPOWER | C9_PASS | **PATHWAY_B_ELIGIBLE** |
| B-MNQ-COX | DSR_FAIL | DSR_PB_PASS | WFE_UNVERIFIED | C7_PASS | C8_UNVERIFIED_LOWPOWER | C9_PASS | **PATHWAY_B_ELIGIBLE** |

## Verdict

- CANDIDATE_READY (clears Pathway A discovery DSR): **0** of 4
- PATHWAY_B_ELIGIBLE (theory-citation path): **4** of 4 — eligible for Phase D Pathway B K=1 pre-reg
- RESEARCH_SURVIVOR (clears C7+C9 only): 0
- KILL: 0

Phase D pre-regs proceed only for CANDIDATE_READY cells, AFTER user explicit go.

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/phase_b_candidate_evidence_v1.py
```

- DB: `pipeline.paths.GOLD_DB_PATH`
- Holdout: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` (2026-01-01)
- DSR Eq.2 + Eq.9: `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`
- Sharpe annualisation: trades/year ≈ N / IS-window-years

## Caveats / limitations

- **K_family approximated** — `compute_family_sharpe_variance()` uses a fixed M proxy (1850 overnight, 2700 volatility) and V[SR_n]=0.5 (Bailey example). True per-family Sharpe variance requires re-emitting the comprehensive scan with per-cell Sharpe stored. **Effect on DSR:** generally conservative; if true V is lower, DSR PASS is more likely than reported.
- **ρ̂=0.5 default** — Bailey example value. True ρ̂ for our family of trial cells could be higher (highly correlated overnight features) → N̂ smaller → DSR easier to pass; or lower → N̂ larger → DSR harder. Sensitivity analysis deferred.
- **Cohen's d power calculation** — uses approximate noncentral-t via t-distribution shift; true `nc_t_cdf` not in scipy.stats by default in all versions.
- **OOS power floor** — N_OOS<50 → UNVERIFIED is the legitimate verdict; cells park until Q3-2026 (more OOS data accrues).
- **Lane-correlation** — restricted to shared trading_day pnl_r; does NOT capture intra-day timing or position overlap.
- **Phase B is NOT confirmation** — Pathway B K=1 pre-reg (Phase D) is the legitimate confirmatory step.

## Not done by this script

- No bootstrap null-floor (T6) re-run — already done in earlier audit pass for these 4 cells (all p<0.005)
- No Pathway B pre-reg — Phase D
- No capital authorisation — Phase E
- No write to `validated_setups` / `lane_allocation` / `live_config`
- No MGC LONDON_METALS short — held until Phase C instrument-family discipline lands
