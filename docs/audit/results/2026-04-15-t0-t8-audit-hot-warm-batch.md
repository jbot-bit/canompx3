# T0-T8 Batch Audit — All HOT + WARM Mega-Exploration Survivors

**Date:** 2026-04-15
**Source:** `docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md`
**Excludes:** P1, P2, P3 (already audited in `2026-04-15-t0-t8-audit-o5-patterns.md`)
**Audit protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5

## Summary Table

**Verdict totals:** VALIDATED=2, CONDITIONAL=17, KILL_DOWNGRADE=8, INFO_HEAVY=0

| Cell | Verdict | P/F/I | N_on | ExpR_on | Expected_Sign | Key FAILs |
|------|---------|-------|------|---------|---------------|-----------|
| MES_NYSE_OPEN_O30_RR2.0_short_F2_NEAR_PDL_15 | **CONDITIONAL** | 6P/1F/2I | 73 | -0.670 | negative | T3 |
| MGC_SINGAPORE_OPEN_O15_RR1.5_long_F3_NEAR_PIVOT_15 | **CONDITIONAL** | 7P/1F/1I | 244 | -0.294 | negative | T3 |
| MNQ_CME_REOPEN_O5_RR2.0_long_F3_NEAR_PIVOT_50 | **KILL_DOWNGRADE** | 6P/2F/1I | 296 | -0.003 | positive | T3,T7 |
| MNQ_US_DATA_830_O5_RR1.0_long_F3_NEAR_PIVOT_50 | **CONDITIONAL** | 7P/1F/1I | 661 | +0.064 | positive | T3 |
| MNQ_US_DATA_830_O5_RR2.0_long_F3_NEAR_PIVOT_50 | **KILL_DOWNGRADE** | 6P/2F/1I | 642 | +0.014 | positive | T3,T7 |
| MNQ_NYSE_OPEN_O30_RR2.0_short_F2_NEAR_PDL_30 | **CONDITIONAL** | 7P/1F/1I | 130 | -0.380 | negative | T3 |
| MNQ_US_DATA_1000_O5_RR1.0_long_F3_NEAR_PIVOT_50 | **CONDITIONAL** | 7P/1F/1I | 637 | -0.034 | negative | T3 |
| MNQ_US_DATA_1000_O5_RR2.0_long_F3_NEAR_PIVOT_50 | **CONDITIONAL** | 7P/1F/1I | 604 | -0.070 | negative | T3 |
| MNQ_COMEX_SETTLE_O5_RR1.0_long_F6_INSIDE_PDR | **CONDITIONAL** | 6P/1F/2I | 452 | -0.030 | negative | T3 |
| MNQ_NYSE_CLOSE_O5_RR1.0_long_F3_NEAR_PIVOT_15 | **CONDITIONAL** | 5P/1F/3I | 57 | -0.239 | negative | T3 |
| MNQ_NYSE_CLOSE_O5_RR2.0_long_F3_NEAR_PIVOT_15 | **CONDITIONAL** | 5P/1F/3I | 39 | -0.700 | negative | T3 |
| MNQ_NYSE_CLOSE_O5_RR2.0_long_F3_NEAR_PIVOT_30 | **CONDITIONAL** | 6P/1F/2I | 73 | -0.535 | negative | T3 |
| MNQ_BRISBANE_1025_O15_RR1.5_short_F3_NEAR_PIVOT_30 | **CONDITIONAL** | 6P/1F/2I | 607 | -0.115 | negative | T3 |
| MNQ_BRISBANE_1025_O15_RR2.0_short_F3_NEAR_PIVOT_30 | **VALIDATED** | 7P/0F/2I | 607 | -0.124 | negative | — |
| MES_CME_REOPEN_O5_RR1.5_long_F1_NEAR_PDH_15 | **KILL_DOWNGRADE** | 5P/2F/2I | 102 | +0.059 | positive | T3,T7 |
| MES_CME_REOPEN_O5_RR2.0_long_F1_NEAR_PDH_50 | **KILL_DOWNGRADE** | 6P/2F/1I | 211 | -0.150 | positive | T3,T7 |
| MES_CME_REOPEN_O15_RR2.0_long_F3_NEAR_PIVOT_50 | **KILL_DOWNGRADE** | 5P/2F/2I | 234 | -0.228 | positive | T3,T7 |
| MES_TOKYO_OPEN_O5_RR1.0_long_F3_NEAR_PIVOT_30 | **CONDITIONAL** | 7P/1F/1I | 658 | -0.176 | negative | T3 |
| MES_US_DATA_830_O30_RR1.0_short_F2_NEAR_PDL_30 | **VALIDATED** | 8P/0F/1I | 253 | -0.246 | negative | — |
| MES_NYSE_OPEN_O15_RR2.0_short_F2_NEAR_PDL_50 | **KILL_DOWNGRADE** | 6P/2F/1I | 333 | -0.183 | negative | T3,T4 |
| MES_NYSE_OPEN_O30_RR1.0_short_F2_NEAR_PDL_15 | **CONDITIONAL** | 6P/1F/2I | 94 | -0.236 | negative | T3 |
| MES_NYSE_OPEN_O30_RR1.5_short_F2_NEAR_PDL_15 | **CONDITIONAL** | 6P/1F/2I | 81 | -0.383 | negative | T3 |
| MES_US_DATA_1000_O15_RR1.0_short_F2_NEAR_PDL_30 | **CONDITIONAL** | 7P/1F/1I | 217 | -0.104 | negative | T3 |
| MES_COMEX_SETTLE_O5_RR1.5_long_F3_NEAR_PIVOT_50 | **CONDITIONAL** | 7P/1F/1I | 539 | -0.201 | negative | T3 |
| MES_NYSE_CLOSE_O5_RR1.0_short_F1_NEAR_PDH_15 | **KILL_DOWNGRADE** | 5P/2F/2I | 58 | -0.457 | negative | T3,T8 |
| MES_NYSE_CLOSE_O5_RR2.0_short_F1_NEAR_PDH_15 | **KILL_DOWNGRADE** | 5P/2F/2I | 46 | -0.765 | negative | T3,T8 |
| MGC_SINGAPORE_OPEN_O15_RR2.0_long_F3_NEAR_PIVOT_15 | **CONDITIONAL** | 7P/1F/1I | 243 | -0.254 | negative | T3 |

---

## Per-Cell Detail

### MES_NYSE_OPEN_O30_RR2.0_short_F2_NEAR_PDL_15
**Scope:** MES | NYSE_OPEN | O30 | RR2.0 | short | signal=F2_NEAR_PDL_15 SHORT on NYSE_OPEN MES (mega t_cl=-5.83)
**N_total:** 556 | **N_on_signal:** 73

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.095 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.09540899978089314, 'gap_r015_fire': -0.0584595073075421, 'atr70_fire': 0.020071349056172128, 'ovn80_fire': -0.04762896722078397} |
| T1 T1_wr_monotonicity | WR_spread=0.259 (on=0.116 off=0.375) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=69 ExpR=-0.670 WR=0.116 σ=0.92 | **INFO** | exploratory-only (N < 100) |
| T3 T3_oos_wfe | N_OOS=4 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[-0.662, -0.734, -0.249] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.670 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 6/6 in expected direction | **PASS** | 6/6 years (100%) matching expected direction; yr={np.int32(2019): None, np.int32(2020): -1.0, np.int32(2021): -0.7662833333333333, np.int32(2022): -0.63504375, np.int32(2023): -0.5682384615384616, np.int32(2024): -0.8148866666666666, np.int32(2025): -0.042933333333333344} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.365 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

### MGC_SINGAPORE_OPEN_O15_RR1.5_long_F3_NEAR_PIVOT_15
**Scope:** MGC | SINGAPORE_OPEN | O15 | RR1.5 | long | signal=F3_NEAR_PIVOT_15 LONG on SINGAPORE_OPEN MGC (mega t_cl=-4.27)
**N_total:** 503 | **N_on_signal:** 244

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.257 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.25706054801650435, 'gap_r015_fire': -0.0841483972636574, 'atr70_fire': 0.018078865353276692, 'ovn80_fire': -0.23053083207430083} |
| T1 T1_wr_monotonicity | WR_spread=0.176 (on=0.345 off=0.521) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=226 ExpR=-0.294 WR=0.345 σ=0.98 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=-0.93 IS_SR=-0.30 OOS_SR=0.28 N_OOS_on=18 | **FAIL** | WFE=-0.93 < 0.5 OVERFIT |
| T4 T4_sensitivity | deltas=[-0.256, -0.401, -0.323] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.294 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 4/4 in expected direction | **PASS** | 4/4 years (100%) matching expected direction; yr={np.int32(2022): -0.3562916666666667, np.int32(2023): -0.3095153846153846, np.int32(2024): -0.22977297297297297, np.int32(2025): -0.3236490196078431} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.149 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 7 PASS, 1 FAIL, 1 INFO
### Verdict: **CONDITIONAL**

---

### MNQ_CME_REOPEN_O5_RR2.0_long_F3_NEAR_PIVOT_50
**Scope:** MNQ | CME_REOPEN | O5 | RR2.0 | long | signal=F3_NEAR_PIVOT_50 LONG on CME_REOPEN MNQ (mega t_cl=+3.28)
**N_total:** 376 | **N_on_signal:** 296

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.182 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.18194038923506417, 'gap_r015_fire': 0.1262221912824975, 'atr70_fire': -0.04535815322150207, 'ovn80_fire': -0.17731780685724025} |
| T1 T1_wr_monotonicity | WR_spread=0.180 (on=0.367 off=0.187) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=281 ExpR=-0.003 WR=0.367 σ=1.32 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=-209.84 IS_SR=-0.00 OOS_SR=0.43 N_OOS_on=14 | **FAIL** | WFE=-209.84 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | deltas=[0.196, 0.486, 0.361] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0020 ExpR_obs=-0.003 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 3/7 in expected direction | **FAIL** | 3/7 years (43%) matching expected direction; yr={np.int32(2019): -0.23801052631578945, np.int32(2020): -0.04760000000000001, np.int32(2021): -0.09742391304347828, np.int32(2022): 0.07238863636363639, np.int32(2023): 0.21548181818181816, np.int32(2024): -0.1645065217391304, np.int32(2025): 0.14294594594594595} |
| T8 T8_cross_instrument | twin=MES Δ=+0.302 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 6 PASS, 2 FAIL, 1 INFO
### Verdict: **KILL_DOWNGRADE**

---

### MNQ_US_DATA_830_O5_RR1.0_long_F3_NEAR_PIVOT_50
**Scope:** MNQ | US_DATA_830 | O5 | RR1.0 | long | signal=F3_NEAR_PIVOT_50 LONG on US_DATA_830 MNQ (mega t_cl=+3.39)
**N_total:** 858 | **N_on_signal:** 661

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.305 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.30543417857476163, 'gap_r015_fire': -0.10336530650047157, 'atr70_fire': 0.04767517780138482, 'ovn80_fire': -0.24524952061900357} |
| T1 T1_wr_monotonicity | WR_spread=0.161 (on=0.590 off=0.429) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=642 ExpR=+0.064 WR=0.590 σ=0.89 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=-3.74 IS_SR=0.07 OOS_SR=-0.27 N_OOS_on=18 | **FAIL** | WFE=-3.74 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | deltas=[0.116, 0.274, 0.485] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.064 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 6/7 in expected direction | **PASS** | 6/7 years (86%) matching expected direction; yr={np.int32(2019): -0.09124615384615385, np.int32(2020): 0.07989361702127659, np.int32(2021): 0.053086734693877556, np.int32(2022): 0.00046464646464647406, np.int32(2023): 0.15270106382978724, np.int32(2024): 0.04243085106382977, np.int32(2025): 0.1637877551020408} |
| T8 T8_cross_instrument | twin=MES Δ=+0.147 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 7 PASS, 1 FAIL, 1 INFO
### Verdict: **CONDITIONAL**

---

### MNQ_US_DATA_830_O5_RR2.0_long_F3_NEAR_PIVOT_50
**Scope:** MNQ | US_DATA_830 | O5 | RR2.0 | long | signal=F3_NEAR_PIVOT_50 LONG on US_DATA_830 MNQ (mega t_cl=+3.03)
**N_total:** 835 | **N_on_signal:** 642

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.302 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.30171945275256823, 'gap_r015_fire': -0.1016211815371917, 'atr70_fire': 0.04932605315289719, 'ovn80_fire': -0.2442894132366403} |
| T1 T1_wr_monotonicity | WR_spread=0.131 (on=0.376 off=0.244) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=623 ExpR=+0.014 WR=0.376 σ=1.31 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=-14.54 IS_SR=0.01 OOS_SR=-0.16 N_OOS_on=18 | **FAIL** | WFE=-14.54 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | deltas=[0.133, 0.34, 0.525] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.014 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 3/7 in expected direction | **FAIL** | 3/7 years (43%) matching expected direction; yr={np.int32(2019): -0.06892031250000005, np.int32(2020): 0.09707849462365593, np.int32(2021): -0.12847157894736844, np.int32(2022): -0.00014387755102041044, np.int32(2023): 0.19464222222222224, np.int32(2024): -0.02340112359550561, np.int32(2025): 0.012723404255319147} |
| T8 T8_cross_instrument | twin=MES Δ=+0.114 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 6 PASS, 2 FAIL, 1 INFO
### Verdict: **KILL_DOWNGRADE**

---

### MNQ_NYSE_OPEN_O30_RR2.0_short_F2_NEAR_PDL_30
**Scope:** MNQ | NYSE_OPEN | O30 | RR2.0 | short | signal=F2_NEAR_PDL_30 SHORT on NYSE_OPEN MNQ (mega t_cl=-3.38)
**N_total:** 463 | **N_on_signal:** 130

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.173 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.1729363335807328, 'gap_r015_fire': -0.05672730398091058, 'atr70_fire': -0.07277458945824483, 'ovn80_fire': 0.04430875851336455} |
| T1 T1_wr_monotonicity | WR_spread=0.155 (on=0.213 off=0.368) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=127 ExpR=-0.380 WR=0.213 σ=1.20 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | N_OOS=3 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[-0.339, -0.457, -0.416] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0040 ExpR_obs=-0.380 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 6/7 in expected direction | **PASS** | 6/7 years (86%) matching expected direction; yr={np.int32(2019): -0.2913125, np.int32(2020): -0.7532583333333333, np.int32(2021): -0.6179043478260869, np.int32(2022): -0.5763285714285714, np.int32(2023): -0.198272, np.int32(2024): 0.14408888888888888, np.int32(2025): -0.3179769230769231} |
| T8 T8_cross_instrument | twin=MES Δ=-0.180 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 7 PASS, 1 FAIL, 1 INFO
### Verdict: **CONDITIONAL**

---

### MNQ_US_DATA_1000_O5_RR1.0_long_F3_NEAR_PIVOT_50
**Scope:** MNQ | US_DATA_1000 | O5 | RR1.0 | long | signal=F3_NEAR_PIVOT_50 LONG on US_DATA_1000 MNQ (mega t_cl=-3.86)
**N_total:** 914 | **N_on_signal:** 637

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.162 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.16183923812308046, 'gap_r015_fire': -0.04969478123420383, 'atr70_fire': 0.002285083954811248, 'ovn80_fire': -0.09226048570441453} |
| T1 T1_wr_monotonicity | WR_spread=0.127 (on=0.511 off=0.639) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=618 ExpR=-0.034 WR=0.511 σ=0.95 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=3.62 IS_SR=-0.04 OOS_SR=-0.13 N_OOS_on=18 | **FAIL** | WFE=3.62 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | deltas=[-0.102, -0.252, -0.226] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.034 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 5/7 in expected direction | **PASS** | 5/7 years (71%) matching expected direction; yr={np.int32(2019): 0.08305090909090908, np.int32(2020): -0.10335599999999998, np.int32(2021): -0.06808775510204082, np.int32(2022): -0.015222093023255816, np.int32(2023): -0.06529885057471263, np.int32(2024): 0.0570367816091954, np.int32(2025): -0.0647} |
| T8 T8_cross_instrument | twin=MES Δ=-0.206 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 7 PASS, 1 FAIL, 1 INFO
### Verdict: **CONDITIONAL**

---

### MNQ_US_DATA_1000_O5_RR2.0_long_F3_NEAR_PIVOT_50
**Scope:** MNQ | US_DATA_1000 | O5 | RR2.0 | long | signal=F3_NEAR_PIVOT_50 LONG on US_DATA_1000 MNQ (mega t_cl=-3.06)
**N_total:** 874 | **N_on_signal:** 604

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.164 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.16424827189026492, 'gap_r015_fire': -0.04388403559059773, 'atr70_fire': 0.002872076915038906, 'ovn80_fire': -0.09138286698252501} |
| T1 T1_wr_monotonicity | WR_spread=0.101 (on=0.329 off=0.430) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=587 ExpR=-0.070 WR=0.329 σ=1.33 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=3.81 IS_SR=-0.05 OOS_SR=-0.20 N_OOS_on=16 | **FAIL** | WFE=3.81 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | deltas=[-0.191, -0.298, -0.234] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0030 ExpR_obs=-0.070 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 5/7 in expected direction | **PASS** | 5/7 years (71%) matching expected direction; yr={np.int32(2019): 0.24225283018867927, np.int32(2020): -0.024484693877551023, np.int32(2021): -0.12104065934065933, np.int32(2022): -0.08585975609756098, np.int32(2023): -0.1573309523809524, np.int32(2024): 0.06051666666666668, np.int32(2025): -0.2695694736842105} |
| T8 T8_cross_instrument | twin=MES Δ=-0.155 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 7 PASS, 1 FAIL, 1 INFO
### Verdict: **CONDITIONAL**

---

### MNQ_COMEX_SETTLE_O5_RR1.0_long_F6_INSIDE_PDR
**Scope:** MNQ | COMEX_SETTLE | O5 | RR1.0 | long | signal=F6_INSIDE_PDR LONG on COMEX_SETTLE MNQ (mega t_cl=-3.57)
**N_total:** 909 | **N_on_signal:** 452

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.164 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.1642291723800077, 'gap_r015_fire': 0.06419559393132654, 'atr70_fire': 0.05640590579936708, 'ovn80_fire': -0.03241496285292737} |
| T1 T1_wr_monotonicity | WR_spread=0.110 (on=0.536 off=0.646) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=433 ExpR=-0.030 WR=0.536 σ=0.91 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=3.36 IS_SR=-0.03 OOS_SR=-0.11 N_OOS_on=17 | **FAIL** | WFE=3.36 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0020 ExpR_obs=-0.030 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 5/7 in expected direction | **PASS** | 5/7 years (71%) matching expected direction; yr={np.int32(2019): -0.04895555555555555, np.int32(2020): -0.08331076923076924, np.int32(2021): -0.2559428571428571, np.int32(2022): -0.03241833333333334, np.int32(2023): 0.08398367346938777, np.int32(2024): -0.04283802816901407, np.int32(2025): 0.18595205479452057} |
| T8 T8_cross_instrument | twin=MES Δ=-0.082 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

### MNQ_NYSE_CLOSE_O5_RR1.0_long_F3_NEAR_PIVOT_15
**Scope:** MNQ | NYSE_CLOSE | O5 | RR1.0 | long | signal=F3_NEAR_PIVOT_15 LONG on NYSE_CLOSE MNQ (mega t_cl=-3.08)
**N_total:** 419 | **N_on_signal:** 57

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.066 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.06598442647404484, 'gap_r015_fire': 0.04066746650810913, 'atr70_fire': -0.015970780884268135, 'ovn80_fire': -0.04195808299486868} |
| T1 T1_wr_monotonicity | WR_spread=0.226 (on=0.418 off=0.645) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=55 ExpR=-0.239 WR=0.418 σ=0.91 | **INFO** | exploratory-only (N < 100) |
| T3 T3_oos_wfe | N_OOS=2 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[-0.427, -0.414, -0.248] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0020 ExpR_obs=-0.239 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 4/6 in expected direction | **INFO** | 4/6 years (67%) matching expected direction; yr={np.int32(2019): None, np.int32(2020): 0.010199999999999956, np.int32(2021): -0.08628750000000002, np.int32(2022): -0.7736125, np.int32(2023): -0.17248181818181815, np.int32(2024): 0.13247499999999998, np.int32(2025): -0.3584555555555555} |
| T8 T8_cross_instrument | twin=MES Δ=-0.217 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 5 PASS, 1 FAIL, 3 INFO
### Verdict: **CONDITIONAL**

---

### MNQ_NYSE_CLOSE_O5_RR2.0_long_F3_NEAR_PIVOT_15
**Scope:** MNQ | NYSE_CLOSE | O5 | RR2.0 | long | signal=F3_NEAR_PIVOT_15 LONG on NYSE_CLOSE MNQ (mega t_cl=-3.91)
**N_total:** 251 | **N_on_signal:** 39

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.086 (gap_r015_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.07390128923483612, 'gap_r015_fire': 0.08580223745791941, 'atr70_fire': -0.05838644671281738, 'ovn80_fire': -0.0375158023487193} |
| T1 T1_wr_monotonicity | WR_spread=0.243 (on=0.108 off=0.351) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=37 ExpR=-0.700 WR=0.108 σ=0.88 | **INFO** | exploratory-only (N < 100) |
| T3 T3_oos_wfe | N_OOS=2 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | [nan, -0.6588064222638481, -0.5211261761761762] | **INFO** | insufficient N at some theta |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0020 ExpR_obs=-0.700 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 3/4 in expected direction | **PASS** | 3/4 years (75%) matching expected direction; yr={np.int32(2019): None, np.int32(2020): 0.09092000000000003, np.int32(2021): None, np.int32(2022): -1.0, np.int32(2023): -0.5926714285714285, np.int32(2024): None, np.int32(2025): -0.5983857142857143} |
| T8 T8_cross_instrument | twin=MES Δ=-0.290 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 5 PASS, 1 FAIL, 3 INFO
### Verdict: **CONDITIONAL**

---

### MNQ_NYSE_CLOSE_O5_RR2.0_long_F3_NEAR_PIVOT_30
**Scope:** MNQ | NYSE_CLOSE | O5 | RR2.0 | long | signal=F3_NEAR_PIVOT_30 LONG on NYSE_CLOSE MNQ (mega t_cl=-3.48)
**N_total:** 251 | **N_on_signal:** 73

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.070 (gap_r015_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.06291735503288141, 'gap_r015_fire': 0.07041074669537928, 'atr70_fire': -0.03759820107813167, 'ovn80_fire': 0.0234114092008415} |
| T1 T1_wr_monotonicity | WR_spread=0.201 (on=0.171 off=0.373) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=70 ExpR=-0.535 WR=0.171 σ=1.03 | **INFO** | exploratory-only (N < 100) |
| T3 T3_oos_wfe | N_OOS=3 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[-0.488, -0.555, -0.468] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.535 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 7/7 in expected direction | **PASS** | 7/7 years (100%) matching expected direction; yr={np.int32(2019): -0.26542857142857146, np.int32(2020): -0.41139285714285717, np.int32(2021): -1.0, np.int32(2022): -1.0, np.int32(2023): -0.71487, np.int32(2024): -0.4335333333333333, np.int32(2025): -0.13866153846153847} |
| T8 T8_cross_instrument | twin=MES Δ=-0.204 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

### MNQ_BRISBANE_1025_O15_RR1.5_short_F3_NEAR_PIVOT_30
**Scope:** MNQ | BRISBANE_1025 | O15 | RR1.5 | short | signal=F3_NEAR_PIVOT_30 SHORT on BRISBANE_1025 MNQ (mega t_cl=-3.04)
**N_total:** 844 | **N_on_signal:** 607

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.447 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.4472725686583623, 'gap_r015_fire': -0.11899222820509049, 'atr70_fire': -0.07899682159485791, 'ovn80_fire': -0.17717934805999358} |
| T1 T1_wr_monotonicity | WR_spread=0.097 (on=0.403 off=0.500) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=585 ExpR=-0.115 WR=0.403 σ=1.08 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=-0.64 IS_SR=-0.11 OOS_SR=0.07 N_OOS_on=22 | **FAIL** | WFE=-0.64 < 0.5 OVERFIT |
| T4 T4_sensitivity | deltas=[-0.094, -0.243, -0.215] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0050 ExpR_obs=-0.115 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 6/7 in expected direction | **PASS** | 6/7 years (86%) matching expected direction; yr={np.int32(2019): -0.35566268656716415, np.int32(2020): -0.1901857142857143, np.int32(2021): 0.1238945054945055, np.int32(2022): -0.004734042553191516, np.int32(2023): -0.20875959595959598, np.int32(2024): -0.14641818181818184, np.int32(2025): -0.1049022988505747} |
| T8 T8_cross_instrument | N_on_twin=0 N_off_twin=0 | **INFO** | twin MES N insufficient |

**Test counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

### MNQ_BRISBANE_1025_O15_RR2.0_short_F3_NEAR_PIVOT_30
**Scope:** MNQ | BRISBANE_1025 | O15 | RR2.0 | short | signal=F3_NEAR_PIVOT_30 SHORT on BRISBANE_1025 MNQ (mega t_cl=-3.15)
**N_total:** 844 | **N_on_signal:** 607

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.447 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.44727256865836246, 'gap_r015_fire': -0.11899222820509031, 'atr70_fire': -0.07899682159485798, 'ovn80_fire': -0.17717934805999358} |
| T1 T1_wr_monotonicity | WR_spread=0.105 (on=0.333 off=0.439) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=585 ExpR=-0.124 WR=0.333 σ=1.25 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=0.73 IS_SR=-0.10 OOS_SR=-0.07 N_OOS_on=22 | **PASS** | WFE=0.73 healthy, sign match |
| T4 T4_sensitivity | deltas=[-0.122, -0.31, -0.235] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0030 ExpR_obs=-0.124 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 5/7 in expected direction | **PASS** | 5/7 years (71%) matching expected direction; yr={np.int32(2019): -0.33508507462686565, np.int32(2020): -0.285, np.int32(2021): 0.1446197802197802, np.int32(2022): -0.1288382978723404, np.int32(2023): -0.23693232323232322, np.int32(2024): -0.11301298701298701, np.int32(2025): 0.009924137931034487} |
| T8 T8_cross_instrument | N_on_twin=0 N_off_twin=0 | **INFO** | twin MES N insufficient |

**Test counts:** 7 PASS, 0 FAIL, 2 INFO
### Verdict: **VALIDATED**

---

### MES_CME_REOPEN_O5_RR1.5_long_F1_NEAR_PDH_15
**Scope:** MES | CME_REOPEN | O5 | RR1.5 | long | signal=F1_NEAR_PDH_15 LONG on CME_REOPEN MES (mega t_cl=+3.30)
**N_total:** 387 | **N_on_signal:** 102

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.123 (ovn80_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.03726117130266971, 'gap_r015_fire': -0.017916251774884494, 'atr70_fire': 0.029676047413709123, 'ovn80_fire': -0.12305906872519418} |
| T1 T1_wr_monotonicity | WR_spread=0.185 (on=0.520 off=0.336) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=98 ExpR=+0.059 WR=0.520 σ=1.03 | **INFO** | exploratory-only (N < 100) |
| T3 T3_oos_wfe | N_OOS=4 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[0.353, 0.345, 0.301] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0020 ExpR_obs=+0.059 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 3/7 in expected direction | **FAIL** | 3/7 years (43%) matching expected direction; yr={np.int32(2019): 0.2899285714285714, np.int32(2020): 0.6822764705882354, np.int32(2021): -0.12610714285714283, np.int32(2022): -0.2342785714285714, np.int32(2023): -0.03830833333333331, np.int32(2024): -0.23565, np.int32(2025): 0.07926111111111112} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.307 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 5 PASS, 2 FAIL, 2 INFO
### Verdict: **KILL_DOWNGRADE**

---

### MES_CME_REOPEN_O5_RR2.0_long_F1_NEAR_PDH_50
**Scope:** MES | CME_REOPEN | O5 | RR2.0 | long | signal=F1_NEAR_PDH_50 LONG on CME_REOPEN MES (mega t_cl=+3.30)
**N_total:** 359 | **N_on_signal:** 211

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.366 (ovn80_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.28077332376440045, 'gap_r015_fire': -0.08146171360172969, 'atr70_fire': 0.024110706378817157, 'ovn80_fire': -0.3659686469302553} |
| T1 T1_wr_monotonicity | WR_spread=0.135 (on=0.347 off=0.212) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=199 ExpR=-0.150 WR=0.347 σ=1.18 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=-0.55 IS_SR=-0.13 OOS_SR=0.07 N_OOS_on=12 | **FAIL** | WFE=-0.55 < 0.5 OVERFIT |
| T4 T4_sensitivity | deltas=[0.283, 0.293, 0.168] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0080 ExpR_obs=-0.150 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 2/7 in expected direction | **FAIL** | 2/7 years (29%) matching expected direction; yr={np.int32(2019): -0.025192307692307677, np.int32(2020): 0.181571052631579, np.int32(2021): -0.23746538461538458, np.int32(2022): -0.6712548387096775, np.int32(2023): 0.10062666666666668, np.int32(2024): -0.25342, np.int32(2025): -0.15956451612903225} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.199 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 6 PASS, 2 FAIL, 1 INFO
### Verdict: **KILL_DOWNGRADE**

---

### MES_CME_REOPEN_O15_RR2.0_long_F3_NEAR_PIVOT_50
**Scope:** MES | CME_REOPEN | O15 | RR2.0 | long | signal=F3_NEAR_PIVOT_50 LONG on CME_REOPEN MES (mega t_cl=+3.25)
**N_total:** 267 | **N_on_signal:** 234

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.203 (ovn80_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.17741609992953936, 'gap_r015_fire': 0.027447723618413834, 'atr70_fire': 0.03480290983474036, 'ovn80_fire': -0.20265863370866394} |
| T1 T1_wr_monotonicity | WR_spread=0.207 (on=0.307 off=0.100) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=218 ExpR=-0.228 WR=0.307 σ=1.17 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=-0.65 IS_SR=-0.19 OOS_SR=0.13 N_OOS_on=16 | **FAIL** | WFE=-0.65 < 0.5 OVERFIT |
| T4 T4_sensitivity | [0.30762437908496737, 0.5100536391437309, nan] | **INFO** | insufficient N at some theta |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0120 ExpR_obs=-0.228 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 2/7 in expected direction | **FAIL** | 2/7 years (29%) matching expected direction; yr={np.int32(2019): -0.44317333333333336, np.int32(2020): 0.2201925, np.int32(2021): -0.33094705882352937, np.int32(2022): -0.6249388888888888, np.int32(2023): 0.27871875, np.int32(2024): -0.565406896551724, np.int32(2025): -0.32920312500000004} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.491 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 5 PASS, 2 FAIL, 2 INFO
### Verdict: **KILL_DOWNGRADE**

---

### MES_TOKYO_OPEN_O5_RR1.0_long_F3_NEAR_PIVOT_30
**Scope:** MES | TOKYO_OPEN | O5 | RR1.0 | long | signal=F3_NEAR_PIVOT_30 LONG on TOKYO_OPEN MES (mega t_cl=-3.58)
**N_total:** 882 | **N_on_signal:** 658

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.505 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.504778501146448, 'gap_r015_fire': -0.16804292019834108, 'atr70_fire': -0.0006177171918742749, 'ovn80_fire': -0.1941914390961365} |
| T1 T1_wr_monotonicity | WR_spread=0.126 (on=0.529 off=0.654) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=630 ExpR=-0.176 WR=0.529 σ=0.79 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=0.32 IS_SR=-0.22 OOS_SR=-0.07 N_OOS_on=26 | **FAIL** | WFE=0.32 < 0.5 OVERFIT |
| T4 T4_sensitivity | deltas=[-0.088, -0.241, -0.255] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.176 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 7/7 in expected direction | **PASS** | 7/7 years (100%) matching expected direction; yr={np.int32(2019): -0.08539199999999997, np.int32(2020): -0.2521137931034482, np.int32(2021): -0.049725, np.int32(2022): -0.1533318181818182, np.int32(2023): -0.21863402061855666, np.int32(2024): -0.26240921052631583, np.int32(2025): -0.21531121495327102} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.075 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 7 PASS, 1 FAIL, 1 INFO
### Verdict: **CONDITIONAL**

---

### MES_US_DATA_830_O30_RR1.0_short_F2_NEAR_PDL_30
**Scope:** MES | US_DATA_830 | O30 | RR1.0 | short | signal=F2_NEAR_PDL_30 SHORT on US_DATA_830 MES (mega t_cl=-3.11)
**N_total:** 843 | **N_on_signal:** 253

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.204 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.20406654927770457, 'gap_r015_fire': -0.06910075413853202, 'atr70_fire': -0.03793800826195295, 'ovn80_fire': -0.016593654020595823} |
| T1 T1_wr_monotonicity | WR_spread=0.123 (on=0.415 off=0.538) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=241 ExpR=-0.246 WR=0.415 σ=0.89 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=0.84 IS_SR=-0.28 OOS_SR=-0.23 N_OOS_on=12 | **PASS** | WFE=0.84 healthy, sign match |
| T4 T4_sensitivity | deltas=[-0.212, -0.213, -0.176] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0020 ExpR_obs=-0.246 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 7/7 in expected direction | **PASS** | 7/7 years (100%) matching expected direction; yr={np.int32(2019): -0.25370000000000004, np.int32(2020): -0.4759652173913043, np.int32(2021): -0.20472903225806452, np.int32(2022): -0.13335384615384613, np.int32(2023): -0.41699534883720923, np.int32(2024): -0.2734428571428571, np.int32(2025): -0.0745921052631579} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.114 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 8 PASS, 0 FAIL, 1 INFO
### Verdict: **VALIDATED**

---

### MES_NYSE_OPEN_O15_RR2.0_short_F2_NEAR_PDL_50
**Scope:** MES | NYSE_OPEN | O15 | RR2.0 | short | signal=F2_NEAR_PDL_50 SHORT on NYSE_OPEN MES (mega t_cl=-3.26)
**N_total:** 744 | **N_on_signal:** 333

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.273 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.2727704255608356, 'gap_r015_fire': -0.0998560241371448, 'atr70_fire': -0.024875506896758923, 'ovn80_fire': -0.06768955542281359} |
| T1 T1_wr_monotonicity | WR_spread=0.115 (on=0.292 off=0.407) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=322 ExpR=-0.183 WR=0.292 σ=1.27 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=-1.47 IS_SR=-0.14 OOS_SR=0.21 N_OOS_on=11 | **FAIL** | WFE=-1.47 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | deltas=[-0.33, -0.326, -0.069] | **FAIL** | adjacent-theta magnitude < 25% primary — knife-edge |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.183 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 6/7 in expected direction | **PASS** | 6/7 years (86%) matching expected direction; yr={np.int32(2019): 0.03411904761904762, np.int32(2020): -0.5100942857142856, np.int32(2021): -0.08308749999999998, np.int32(2022): -0.2923868852459016, np.int32(2023): -0.29903728813559327, np.int32(2024): -0.02027254901960783, np.int32(2025): -0.025172340425531934} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.233 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 6 PASS, 2 FAIL, 1 INFO
### Verdict: **KILL_DOWNGRADE**

---

### MES_NYSE_OPEN_O30_RR1.0_short_F2_NEAR_PDL_15
**Scope:** MES | NYSE_OPEN | O30 | RR1.0 | short | signal=F2_NEAR_PDL_15 SHORT on NYSE_OPEN MES (mega t_cl=-3.20)
**N_total:** 715 | **N_on_signal:** 94

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.084 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.08429456310057584, 'gap_r015_fire': -0.02516607699403683, 'atr70_fire': 0.03080491398823486, 'ovn80_fire': -0.02766305704258265} |
| T1 T1_wr_monotonicity | WR_spread=0.182 (on=0.400 off=0.582) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=90 ExpR=-0.236 WR=0.400 σ=0.94 | **INFO** | exploratory-only (N < 100) |
| T3 T3_oos_wfe | N_OOS=4 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[-0.225, -0.34, -0.14] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.236 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 5/7 in expected direction | **PASS** | 5/7 years (71%) matching expected direction; yr={np.int32(2019): 0.2544666666666667, np.int32(2020): -0.36541666666666667, np.int32(2021): -0.5638076923076923, np.int32(2022): -0.15804347826086956, np.int32(2023): -0.05372777777777776, np.int32(2024): -0.7481733333333334, np.int32(2025): 0.2828111111111111} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.160 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

### MES_NYSE_OPEN_O30_RR1.5_short_F2_NEAR_PDL_15
**Scope:** MES | NYSE_OPEN | O30 | RR1.5 | short | signal=F2_NEAR_PDL_15 SHORT on NYSE_OPEN MES (mega t_cl=-3.84)
**N_total:** 624 | **N_on_signal:** 81

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.075 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.07493961439839819, 'gap_r015_fire': -0.05901015958770982, 'atr70_fire': 0.012139068266903856, 'ovn80_fire': -0.03597403495942308} |
| T1 T1_wr_monotonicity | WR_spread=0.210 (on=0.260 off=0.470) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=77 ExpR=-0.383 WR=0.260 σ=1.05 | **INFO** | exploratory-only (N < 100) |
| T3 T3_oos_wfe | N_OOS=4 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[-0.258, -0.495, -0.199] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0020 ExpR_obs=-0.383 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 5/6 in expected direction | **PASS** | 5/6 years (83%) matching expected direction; yr={np.int32(2019): None, np.int32(2020): -0.20676666666666668, np.int32(2021): -0.6113833333333334, np.int32(2022): -0.3281833333333334, np.int32(2023): -0.11503124999999997, np.int32(2024): -0.8457399999999999, np.int32(2025): 0.028728571428571405} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.294 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

### MES_US_DATA_1000_O15_RR1.0_short_F2_NEAR_PDL_30
**Scope:** MES | US_DATA_1000 | O15 | RR1.0 | short | signal=F2_NEAR_PDL_30 SHORT on US_DATA_1000 MES (mega t_cl=-3.07)
**N_total:** 776 | **N_on_signal:** 217

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.195 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.19546076361801973, 'gap_r015_fire': -0.06845010563143207, 'atr70_fire': -0.013729793119522677, 'ovn80_fire': -0.01957181002666133} |
| T1 T1_wr_monotonicity | WR_spread=0.126 (on=0.483 off=0.609) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=211 ExpR=-0.104 WR=0.483 σ=0.93 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | N_OOS=6 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[-0.207, -0.227, -0.152] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.104 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 7/7 in expected direction | **PASS** | 7/7 years (100%) matching expected direction; yr={np.int32(2019): -0.10717857142857147, np.int32(2020): -0.04507142857142857, np.int32(2021): -0.07802352941176469, np.int32(2022): -0.15370555555555557, np.int32(2023): -0.07623958333333335, np.int32(2024): -0.20593571428571428, np.int32(2025): -0.05940333333333335} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.097 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 7 PASS, 1 FAIL, 1 INFO
### Verdict: **CONDITIONAL**

---

### MES_COMEX_SETTLE_O5_RR1.5_long_F3_NEAR_PIVOT_50
**Scope:** MES | COMEX_SETTLE | O5 | RR1.5 | long | signal=F3_NEAR_PIVOT_50 LONG on COMEX_SETTLE MES (mega t_cl=-3.36)
**N_total:** 923 | **N_on_signal:** 539

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.195 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.19454154790146633, 'gap_r015_fire': -0.08463138927158859, 'atr70_fire': 0.02696389774092551, 'ovn80_fire': -0.1292666260674729} |
| T1 T1_wr_monotonicity | WR_spread=0.112 (on=0.390 off=0.501) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=521 ExpR=-0.201 WR=0.390 σ=1.01 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=1.71 IS_SR=-0.20 OOS_SR=-0.34 N_OOS_on=17 | **FAIL** | WFE=1.71 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | deltas=[-0.216, -0.244, -0.227] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.201 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 7/7 in expected direction | **PASS** | 7/7 years (100%) matching expected direction; yr={np.int32(2019): -0.3380462962962963, np.int32(2020): -0.34547317073170736, np.int32(2021): -0.39388625, np.int32(2022): -0.18227076923076924, np.int32(2023): -0.05571492537313433, np.int32(2024): -0.12223975903614456, np.int32(2025): -0.010019999999999994} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.146 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 7 PASS, 1 FAIL, 1 INFO
### Verdict: **CONDITIONAL**

---

### MES_NYSE_CLOSE_O5_RR1.0_short_F1_NEAR_PDH_15
**Scope:** MES | NYSE_CLOSE | O5 | RR1.0 | short | signal=F1_NEAR_PDH_15 SHORT on NYSE_CLOSE MES (mega t_cl=-3.25)
**N_total:** 367 | **N_on_signal:** 58

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.118 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.11816249849680698, 'gap_r015_fire': 0.002135967453942446, 'atr70_fire': 0.028281505316635223, 'ovn80_fire': -0.03423604894262095} |
| T1 T1_wr_monotonicity | WR_spread=0.233 (on=0.321 off=0.554) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=56 ExpR=-0.457 WR=0.321 σ=0.80 | **INFO** | exploratory-only (N < 100) |
| T3 T3_oos_wfe | N_OOS=2 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[-0.341, -0.4, -0.212] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.457 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 6/6 in expected direction | **PASS** | 6/6 years (100%) matching expected direction; yr={np.int32(2019): -0.36686, np.int32(2020): -0.6940833333333334, np.int32(2021): -0.7078272727272727, np.int32(2022): -0.3773090909090909, np.int32(2023): -0.03169999999999999, np.int32(2024): None, np.int32(2025): -0.3766857142857143} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.014 | **FAIL** | sign_ok=False mag_ok=False |

**Test counts:** 5 PASS, 2 FAIL, 2 INFO
### Verdict: **KILL_DOWNGRADE**

---

### MES_NYSE_CLOSE_O5_RR2.0_short_F1_NEAR_PDH_15
**Scope:** MES | NYSE_CLOSE | O5 | RR2.0 | short | signal=F1_NEAR_PDH_15 SHORT on NYSE_CLOSE MES (mega t_cl=-3.23)
**N_total:** 251 | **N_on_signal:** 46

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.136 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.13584598911351922, 'gap_r015_fire': 0.09383075327578728, 'atr70_fire': 0.05498363000366677, 'ovn80_fire': -0.04920313624760917} |
| T1 T1_wr_monotonicity | WR_spread=0.172 (on=0.091 off=0.263) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=44 ExpR=-0.765 WR=0.091 σ=0.75 | **INFO** | exploratory-only (N < 100) |
| T3 T3_oos_wfe | N_OOS=2 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[-0.405, -0.44, -0.331] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0060 ExpR_obs=-0.765 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 4/4 in expected direction | **PASS** | 4/4 years (100%) matching expected direction; yr={np.int32(2019): None, np.int32(2020): -1.0, np.int32(2021): -1.0, np.int32(2022): -0.47478999999999993, np.int32(2023): None, np.int32(2024): None, np.int32(2025): -0.7837818181818182} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.073 | **FAIL** | sign_ok=False mag_ok=True |

**Test counts:** 5 PASS, 2 FAIL, 2 INFO
### Verdict: **KILL_DOWNGRADE**

---

### MGC_SINGAPORE_OPEN_O15_RR2.0_long_F3_NEAR_PIVOT_15
**Scope:** MGC | SINGAPORE_OPEN | O15 | RR2.0 | long | signal=F3_NEAR_PIVOT_15 LONG on SINGAPORE_OPEN MGC (mega t_cl=-3.50)
**N_total:** 500 | **N_on_signal:** 243

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.260 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.25958656105337535, 'gap_r015_fire': -0.08061726930050195, 'atr70_fire': 0.016118448568283938, 'ovn80_fire': -0.23014144771207456} |
| T1 T1_wr_monotonicity | WR_spread=0.130 (on=0.307 off=0.436) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=225 ExpR=-0.254 WR=0.307 σ=1.13 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=-1.35 IS_SR=-0.22 OOS_SR=0.30 N_OOS_on=18 | **FAIL** | WFE=-1.35 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | deltas=[-0.183, -0.361, -0.316] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.254 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 4/4 in expected direction | **PASS** | 4/4 years (100%) matching expected direction; yr={np.int32(2022): -0.22755000000000003, np.int32(2023): -0.25356615384615383, np.int32(2024): -0.23959452054794522, np.int32(2025): -0.29446274509803927} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.114 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 7 PASS, 1 FAIL, 1 INFO
### Verdict: **CONDITIONAL**

---
