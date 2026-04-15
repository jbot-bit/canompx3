# T0-T8 Audit — O5 Prior-Day Level Patterns

**Date:** 2026-04-15
**Source patterns:** mega-exploration survivors at O5 only
**Audit protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5

## P1_NEAR_PIVOT_LONG_NYSE_CLOSE
**Description:** F3 NEAR_PIVOT_15 LONG on NYSE_CLOSE MNQ — strongest cross-session negative
**Scope:** MNQ | NYSE_CLOSE | O5 | RR1.5 | long | expected_sign=negative
**N_total:** 304 | **N_on_signal:** 41

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.069 (gap_r015_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.05187909031079035, 'gap_r015_fire': 0.06877859024325467, 'atr70_fire': -0.03896794897082899, 'ovn80_fire': -0.03383678495660189} |
| T1 T1_wr_monotonicity | WR_spread=0.340 (on=0.154 off=0.494) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=39 ExpR=-0.642 WR=0.154 σ=0.85 | **INFO** | exploratory-only (N < 100) |
| T3 T3_oos_wfe | N_OOS=2 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | [nan, -0.7590223924813566, -0.43929889380530973] | **INFO** | insufficient N at some theta |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.642 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 5/5 in expected direction | **PASS** | 5/5 years (100%) matching expected direction; yr={np.int32(2019): None, np.int32(2020): -0.09089999999999998, np.int32(2021): None, np.int32(2022): -1.0, np.int32(2023): -0.40895, np.int32(2024): -0.52758, np.int32(2025): -0.6653285714285715} |
| T8 T8_cross_instrument | twin=MES Δ=-0.296 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 5 PASS, 1 FAIL, 3 INFO

### Verdict: **CONDITIONAL** — one fail, acceptable if non-load-bearing (e.g., OOS N thin). Pre-reg with explicit fail annotation.

---

## P2_NEAR_PDH_SHORT_NYSE_CLOSE_MES
**Description:** F1 NEAR_PDH_15 SHORT on NYSE_CLOSE MES — cross-RR negative
**Scope:** MES | NYSE_CLOSE | O5 | RR1.5 | short | expected_sign=negative
**N_total:** 282 | **N_on_signal:** 47

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.142 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.14162447911486983, 'gap_r015_fire': 0.06252818913170463, 'atr70_fire': 0.04920678313051228, 'ovn80_fire': -0.021369953193578997} |
| T1 T1_wr_monotonicity | WR_spread=0.234 (on=0.133 off=0.368) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=45 ExpR=-0.716 WR=0.133 σ=0.73 | **INFO** | exploratory-only (N < 100) |
| T3 T3_oos_wfe | N_OOS=2 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[-0.436, -0.505, -0.241] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0020 ExpR_obs=-0.716 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 4/4 in expected direction | **PASS** | 4/4 years (100%) matching expected direction; yr={np.int32(2019): None, np.int32(2020): -1.0, np.int32(2021): -0.77786, np.int32(2022): -0.36779, np.int32(2023): None, np.int32(2024): None, np.int32(2025): -0.8198181818181819} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.104 | **FAIL** | sign_ok=False mag_ok=True |

**Test counts:** 5 PASS, 2 FAIL, 2 INFO

### Verdict: **KILL / DOWNGRADE** — multiple failures. Do not deploy as-is. See fail details for labels.

---

## P3_BELOW_PDL_LONG_US_DATA_1000_MNQ
**Description:** F5 BELOW_PDL LONG on US_DATA_1000 MNQ — cross-RR positive
**Scope:** MNQ | US_DATA_1000 | O5 | RR1.0 | long | expected_sign=positive
**N_total:** 914 | **N_on_signal:** 144

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.114 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.1141321344429128, 'gap_r015_fire': -0.05015148096073722, 'atr70_fire': -0.005115609077046531, 'ovn80_fire': 0.05009126913849453} |
| T1 T1_wr_monotonicity | WR_spread=0.168 (on=0.691 off=0.523) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=136 ExpR=+0.326 WR=0.691 σ=0.89 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | N_OOS=8 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.326 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 7/7 in expected direction | **PASS** | 7/7 years (100%) matching expected direction; yr={np.int32(2019): 0.28408, np.int32(2020): 0.48053076923076926, np.int32(2021): 0.025253333333333336, np.int32(2022): 0.658917857142857, np.int32(2023): 0.011984615384615363, np.int32(2024): 0.4424, np.int32(2025): 0.3016375} |
| T8 T8_cross_instrument | twin=MES Δ=+0.210 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Test counts:** 6 PASS, 1 FAIL, 2 INFO

### Verdict: **CONDITIONAL** — one fail, acceptable if non-load-bearing (e.g., OOS N thin). Pre-reg with explicit fail annotation.

---
