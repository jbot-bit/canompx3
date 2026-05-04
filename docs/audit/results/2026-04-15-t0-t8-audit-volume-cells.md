# T0-T8 Audit — Volume Cells from 2026-04-15 Scans

**Date:** 2026-04-15
**Source scans:** comprehensive-deployed-lane-scan.md + volume-confluence-scan.md
**Protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5

## V1_MES_COMEX_SETTLE_O5_RR1.0_short_REL_VOL_HIGH
**Description:** MES COMEX_SETTLE O5 RR1.0 SHORT rel_vol > P67 — BH-global t=+4.89 (rel_vol P67=1.867 on MES IS)
**Scope:** MES | COMEX_SETTLE | O5 | RR1.0 | short | expected=positive
**N_total:** 789 | **N_on_signal:** 299

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.167 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.16729102382077296, 'gap_r015_fire': 0.04523422365410289, 'atr70_fire': -0.06292438756890144, 'ovn80_fire': 0.06336315493708249} |
| T1 T1_wr_monotonicity | WR_spread=0.153 (on=0.628 off=0.476) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=288 ExpR=+0.077 WR=0.628 σ=0.84 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=3.93 IS_SR=0.09 OOS_SR=0.36 N_OOS_on=11 | **FAIL** | WFE=3.93 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.077 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 5/7 in expected direction | **PASS** | 5/7 years (71%) matching expected direction; yr={np.int32(2019): -0.29231818181818175, np.int32(2020): 0.012033333333333318, np.int32(2021): 0.06136078431372547, np.int32(2022): 0.20843611111111113, np.int32(2023): 0.2053239130434783, np.int32(2024): -0.06083750000000001, np.int32(2025): 0.2927918918918919} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.314 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

## V2_MES_TOKYO_OPEN_O5_RR1.5_long_REL_VOL_HIGH
**Description:** MES TOKYO_OPEN O5 RR1.5 LONG rel_vol > P67 — BH-global t=+4.46 (rel_vol P67=2.351 on MES IS)
**Scope:** MES | TOKYO_OPEN | O5 | RR1.5 | long | expected=positive
**N_total:** 882 | **N_on_signal:** 287

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.282 (ovn80_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.1572140711459432, 'gap_r015_fire': 0.09962835505249565, 'atr70_fire': 0.003204695235768622, 'ovn80_fire': 0.28212628913315596} |
| T1 T1_wr_monotonicity | WR_spread=0.142 (on=0.536 off=0.395) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=276 ExpR=+0.092 WR=0.536 σ=1.03 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=2.19 IS_SR=0.09 OOS_SR=0.20 N_OOS_on=11 | **FAIL** | WFE=2.19 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.092 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 4/7 in expected direction | **INFO** | 4/7 years (57%) matching expected direction; yr={np.int32(2019): -0.010252941176470557, np.int32(2020): 0.31350217391304347, np.int32(2021): -0.0170625, np.int32(2022): 0.1363767441860465, np.int32(2023): -0.030668085106382945, np.int32(2024): 0.16331785714285713, np.int32(2025): 0.07730000000000004} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.285 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 5 PASS, 1 FAIL, 3 INFO
### Verdict: **CONDITIONAL**

---

## V3_MNQ_SINGAPORE_OPEN_O5_RR1.0_short_REL_VOL_HIGH
**Description:** MNQ SINGAPORE_OPEN O5 RR1.0 SHORT rel_vol > P67 — BH-global t=+4.27 (rel_vol P67=2.314 on MNQ IS)
**Scope:** MNQ | SINGAPORE_OPEN | O5 | RR1.0 | short | expected=positive
**N_total:** 866 | **N_on_signal:** 298

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.270 (ovn80_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.13344794137725116, 'gap_r015_fire': 0.1110127216391377, 'atr70_fire': 0.007481241710248353, 'ovn80_fire': 0.27028533884638256} |
| T1 T1_wr_monotonicity | WR_spread=0.133 (on=0.653 off=0.520) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=285 ExpR=+0.139 WR=0.653 σ=0.84 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=3.28 IS_SR=0.17 OOS_SR=0.54 N_OOS_on=13 | **FAIL** | WFE=3.28 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.139 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 5/7 in expected direction | **PASS** | 5/7 years (71%) matching expected direction; yr={np.int32(2019): -0.033649999999999985, np.int32(2020): 0.3154818181818182, np.int32(2021): 0.01096666666666668, np.int32(2022): 0.3159825, np.int32(2023): 0.2693972972972973, np.int32(2024): -0.039088372093023245, np.int32(2025): 0.12804878048780488} |
| T8 T8_cross_instrument | twin=MES Δ=+0.219 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

## V4_MNQ_COMEX_SETTLE_O5_RR1.5_short_REL_VOL_HIGH_AND_F6
**Description:** MNQ COMEX_SETTLE O5 RR1.5 SHORT rel_vol×F6 — confluence t=+3.51, Δ_OOS=+0.276 (rel_vol P67=1.788 on MNQ IS)
**Scope:** MNQ | COMEX_SETTLE | O5 | RR1.5 | short | expected=positive
**N_total:** 798 | **N_on_signal:** 154

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.169 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.16926718655513828, 'gap_r015_fire': 0.016272673332238294, 'atr70_fire': -0.053566371245498234, 'ovn80_fire': 0.024008175380533753} |
| T1 T1_wr_monotonicity | WR_spread=0.157 (on=0.591 off=0.433) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=149 ExpR=+0.358 WR=0.591 σ=1.14 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | N_OOS=5 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.358 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 7/7 in expected direction | **PASS** | 7/7 years (100%) matching expected direction; yr={np.int32(2019): 0.8982, np.int32(2020): 0.03369999999999999, np.int32(2021): 0.44517272727272733, np.int32(2022): 0.1188235294117647, np.int32(2023): 0.49479032258064515, np.int32(2024): 0.20883333333333334, np.int32(2025): 0.63803} |
| T8 T8_cross_instrument | twin=MES Δ=+0.278 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

## M1_MGC_LONDON_METALS_O5_RR1.0_short_REL_VOL_HIGH
**Description:** MGC LONDON_METALS O5 RR1.0 SHORT rel_vol > P67 — BH-global t=+4.78 (strongest MGC) (rel_vol P67=2.036 on MGC IS)
**Scope:** MGC | LONDON_METALS | O5 | RR1.0 | short | expected=positive
**N_total:** 479 | **N_on_signal:** 165

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.225 (ovn80_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.07713196668852985, 'gap_r015_fire': -0.03632444508687559, 'atr70_fire': 0.04314911891767667, 'ovn80_fire': 0.22544434438938418} |
| T1 T1_wr_monotonicity | WR_spread=0.201 (on=0.711 off=0.510) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=149 ExpR=+0.125 WR=0.711 σ=0.73 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=0.33 IS_SR=0.17 OOS_SR=0.06 N_OOS_on=16 | **FAIL** | WFE=0.33 < 0.5 OVERFIT |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.125 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 4/4 in expected direction | **PASS** | 4/4 years (100%) matching expected direction; yr={np.int32(2022): 0.32507142857142857, np.int32(2023): 0.0182258064516129, np.int32(2024): 0.14560238095238096, np.int32(2025): 0.05809375} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.164 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

## M2_MGC_LONDON_METALS_O5_RR1.5_short_REL_VOL_HIGH
**Description:** MGC LONDON_METALS O5 RR1.5 SHORT rel_vol > P67 — cross-RR family check (rel_vol P67=2.036 on MGC IS)
**Scope:** MGC | LONDON_METALS | O5 | RR1.5 | short | expected=positive
**N_total:** 479 | **N_on_signal:** 165

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.225 (ovn80_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.07713196668852985, 'gap_r015_fire': -0.03632444508687559, 'atr70_fire': 0.04314911891767667, 'ovn80_fire': 0.22544434438938418} |
| T1 T1_wr_monotonicity | WR_spread=0.152 (on=0.544 off=0.392) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=149 ExpR=+0.080 WR=0.544 σ=1.01 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=1.82 IS_SR=0.08 OOS_SR=0.15 N_OOS_on=16 | **FAIL** | WFE=1.82 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.080 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 3/4 in expected direction | **PASS** | 3/4 years (75%) matching expected direction; yr={np.int32(2022): 0.2539857142857143, np.int32(2023): -0.04032258064516129, np.int32(2024): 0.0758047619047619, np.int32(2025): 0.061406249999999996} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.260 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

## M3_MGC_NYSE_OPEN_O5_RR1.5_short_BB_RATIO_HIGH
**Description:** MGC NYSE_OPEN O5 RR1.5 SHORT bb_vol_ratio > P67 — alt-feature, t=+3.47 (bb_ratio P67=0.2982 on MGC IS)
**Scope:** MGC | NYSE_OPEN | O5 | RR1.5 | short | expected=positive
**N_total:** 492 | **N_on_signal:** 177

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.067 (atr70_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.038700441995583, 'gap_r015_fire': 0.009802055353508402, 'atr70_fire': -0.06657340086129959, 'ovn80_fire': -0.008427530965448808} |
| T1 T1_wr_monotonicity | WR_spread=0.165 (on=0.538 off=0.373) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=169 ExpR=+0.138 WR=0.538 σ=1.06 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | N_OOS=8 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0020 ExpR_obs=+0.138 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 3/4 in expected direction | **PASS** | 3/4 years (75%) matching expected direction; yr={np.int32(2022): -0.135628, np.int32(2023): 0.2768928571428571, np.int32(2024): 0.0023740740740740554, np.int32(2025): 0.32666470588235297} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.021 | **FAIL** | sign_ok=False mag_ok=False |

**Counts:** 5 PASS, 2 FAIL, 2 INFO
### Verdict: **KILL_DOWNGRADE**

---

## M4_MGC_US_DATA_1000_O5_RR1.0_short_BB_RATIO_HIGH
**Description:** MGC US_DATA_1000 O5 RR1.0 SHORT bb_vol_ratio > P67 — cross-session, t=+3.15 (bb_ratio P67=0.2288 on MGC IS)
**Scope:** MGC | US_DATA_1000 | O5 | RR1.0 | short | expected=positive
**N_total:** 437 | **N_on_signal:** 158

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.086 (atr70_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.0020786898870180036, 'gap_r015_fire': -0.08528235352697673, 'atr70_fire': 0.08585139409843216, 'ovn80_fire': 0.0030739033086306356} |
| T1 T1_wr_monotonicity | WR_spread=0.162 (on=0.652 off=0.491) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=141 ExpR=+0.134 WR=0.652 σ=0.84 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=2.18 IS_SR=0.16 OOS_SR=0.35 N_OOS_on=16 | **FAIL** | WFE=2.18 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0020 ExpR_obs=+0.134 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 3/4 in expected direction | **PASS** | 3/4 years (75%) matching expected direction; yr={np.int32(2022): 0.22357499999999994, np.int32(2023): 0.28791052631578945, np.int32(2024): -0.207895, np.int32(2025): 0.2757441860465116} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.057 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---
