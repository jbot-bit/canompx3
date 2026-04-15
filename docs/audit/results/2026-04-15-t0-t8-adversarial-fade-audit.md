# T0-T8 Adversarial Fade Audit

**Date:** 2026-04-15
**Hypothesis:** SKIP signals are latent FADE signals. Take the opposite direction.
**Source cells:** HOT + WARM from mega-exploration with negative delta_is
**Protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5

## Verdict Summary

| Verdict | Count |
|---------|-------|
| FADE_FAILS — no positive ExpR | 13 |
| FADE_WEAK — missing core tests | 9 |
| FADE_CONDITIONAL | 1 |

## Summary Table (sorted by fade ExpR)

| Fade Cell | Orig Dir | ExpR (fade) | Verdict | N_on | Key FAILs |
|-----------|----------|-------------|---------|------|-----------|
| FADE_MES_NYSE_CLOSE_O5_RR1.0_long_F1_NEAR_PDH_15 | short | +0.155 | **FADE_WEAK — missing core tests** | 58 | T3,T6 |
| FADE_MNQ_US_DATA_1000_O5_RR2.0_short_F3_NEAR_PIVOT_50 | long | +0.153 | **FADE_WEAK — missing core tests** | 554 | T3,T4,T6,T8 |
| FADE_MNQ_US_DATA_1000_O5_RR1.0_short_F3_NEAR_PIVOT_50 | long | +0.113 | **FADE_WEAK — missing core tests** | 570 | T1,T3,T4,T6,T8 |
| FADE_MNQ_NYSE_CLOSE_O5_RR1.0_short_F3_NEAR_PIVOT_15 | long | +0.098 | **FADE_WEAK — missing core tests** | 67 | T3,T4,T6 |
| FADE_MES_NYSE_CLOSE_O5_RR1.5_long_F1_NEAR_PDH_15 | short | +0.080 | **FADE_WEAK — missing core tests** | 39 | T3,T6 |
| FADE_MNQ_BRISBANE_1025_O15_RR2.0_long_F3_NEAR_PIVOT_30 | short | +0.053 | **FADE_WEAK — missing core tests** | 665 | T3,T6 |
| FADE_MNQ_COMEX_SETTLE_O5_RR1.0_short_F6_INSIDE_PDR | long | +0.043 | **FADE_WEAK — missing core tests** | 391 | T3,T6,T7,T8 |
| FADE_MES_NYSE_OPEN_O15_RR2.0_long_F2_NEAR_PDL_50 | short | +0.043 | **FADE_WEAK — missing core tests** | 312 | T3,T4,T6,T8 |
| FADE_MNQ_NYSE_CLOSE_O5_RR1.5_short_F3_NEAR_PIVOT_15 | long | +0.019 | **FADE_WEAK — missing core tests** | 52 | T3,T4,T6,T7 |
| FADE_MES_US_DATA_830_O30_RR1.0_long_F2_NEAR_PDL_30 | short | +0.019 | **FADE_CONDITIONAL** | 220 | T3 |
| FADE_MNQ_BRISBANE_1025_O15_RR1.5_long_F3_NEAR_PIVOT_30 | short | -0.009 | **FADE_FAILS — no positive ExpR** | 665 | T1,T3,T6,T7 |
| FADE_MNQ_NYSE_CLOSE_O5_RR2.0_short_F3_NEAR_PIVOT_30 | long | -0.018 | **FADE_FAILS — no positive ExpR** | 87 | T3,T4,T6,T7,T8 |
| FADE_MES_US_DATA_1000_O15_RR1.0_long_F2_NEAR_PDL_30 | short | -0.028 | **FADE_FAILS — no positive ExpR** | 235 | T3,T6,T7,T8 |
| FADE_MES_NYSE_OPEN_O30_RR1.0_long_F2_NEAR_PDL_15 | short | -0.045 | **FADE_FAILS — no positive ExpR** | 81 | T1,T3,T6,T8 |
| FADE_MES_NYSE_CLOSE_O5_RR2.0_long_F1_NEAR_PDH_15 | short | -0.051 | **FADE_FAILS — no positive ExpR** | 32 | T3,T6 |
| FADE_MGC_SINGAPORE_OPEN_O15_RR1.5_short_F3_NEAR_PIVOT_15 | long | -0.082 | **FADE_FAILS — no positive ExpR** | 218 | T3,T6,T8 |
| FADE_MGC_SINGAPORE_OPEN_O15_RR2.0_short_F3_NEAR_PIVOT_15 | long | -0.111 | **FADE_FAILS — no positive ExpR** | 217 | T3,T6,T8 |
| FADE_MNQ_NYSE_CLOSE_O5_RR2.0_short_F3_NEAR_PIVOT_15 | long | -0.143 | **FADE_FAILS — no positive ExpR** | 44 | T3,T6,T8 |
| FADE_MES_TOKYO_OPEN_O5_RR1.0_short_F3_NEAR_PIVOT_30 | long | -0.152 | **FADE_FAILS — no positive ExpR** | 691 | T1,T4,T6,T7,T8 |
| FADE_MES_COMEX_SETTLE_O5_RR1.5_short_F3_NEAR_PIVOT_50 | long | -0.180 | **FADE_FAILS — no positive ExpR** | 452 | T3,T6,T7 |
| FADE_MES_NYSE_OPEN_O30_RR1.5_long_F2_NEAR_PDL_15 | short | -0.193 | **FADE_FAILS — no positive ExpR** | 71 | T3,T4,T6,T7,T8 |
| FADE_MES_NYSE_OPEN_O30_RR2.0_long_F2_NEAR_PDL_15 | short | -0.280 | **FADE_FAILS — no positive ExpR** | 63 | T1,T3,T4,T6,T7,T8 |
| FADE_MNQ_NYSE_OPEN_O30_RR2.0_long_F2_NEAR_PDL_30 | short | -0.302 | **FADE_FAILS — no positive ExpR** | 109 | T3,T4,T6,T7,T8 |

---

## Per-Cell Detail (fade validated / conditional only)

### FADE_MES_US_DATA_830_O30_RR1.0_long_F2_NEAR_PDL_30
**Fade scope:** MES | US_DATA_830 | O30 | RR1.0 | long
**Signal:** CAST((ABS((d.orb_US_DATA_830_high + d.orb_US_DATA_830_low)/2.0 - d.prev_day_low) / d.atr_20 < 0.3) AS INTEGER)...
**N_total:** 855 | **N_on_signal:** 220
**Fade ExpR:** +0.019

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.234 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.23419509593851898, 'gap_r015_fire': -0.09566719960111375, 'atr70_fire': 0.03933202255341369, 'ovn80_fire': -0.014761644026445165} |
| T1 T1_wr_monotonicity | WR_spread=0.061 (on=0.566 off=0.505) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=212 ExpR=+0.019 WR=0.566 σ=0.90 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | N_OOS=8 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[0.166, 0.109, 0.054] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0470 ExpR_obs=+0.019 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 4/7 in expected direction | **INFO** | 4/7 years (57%) matching expected direction; yr={np.int32(2019): -0.0037473684210526277, np.int32(2020): -0.21176666666666666, np.int32(2021): 0.061993939393939386, np.int32(2022): -0.12495625000000002, np.int32(2023): 0.06061463414634147, np.int32(2024): 0.010646666666666664, np.int32(2025): 0.29738333333333333} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.102 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

### Verdict: **FADE_CONDITIONAL**

---
