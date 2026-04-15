# T0-T8 Audit — Volume Cells from 2026-04-15 Scans

**Date:** 2026-04-15
**Source scans:** comprehensive-deployed-lane-scan.md + volume-confluence-scan.md
**Protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5

## V1_MES_COMEX_SETTLE_O5_RR1.0_short_REL_VOL_HIGH
**Description:** MES COMEX_SETTLE O5 RR1.0 SHORT with rel_vol > P67 — BH-global survivor t=+4.89 (rel_vol threshold P67=1.867)
**Scope:** MES | COMEX_SETTLE | O5 | RR1.0 | short | expected=positive
**N_total:** 789 | **N_on_signal:** 299

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.167 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.16729102382077304, 'gap_r015_fire': 0.04523422365410286, 'atr70_fire': -0.06292438756890145, 'ovn80_fire': 0.06336315493708249} |
| T1 T1_wr_monotonicity | WR_spread=0.153 (on=0.628 off=0.476) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=288 ExpR=+0.077 WR=0.628 σ=0.84 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=3.93 IS_SR=0.09 OOS_SR=0.36 N_OOS_on=11 | **FAIL** | WFE=3.93 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.077 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 5/7 in expected direction | **PASS** | 5/7 years (71%) matching expected direction; yr={np.int32(2019): -0.29231818181818175, np.int32(2020): 0.012033333333333318, np.int32(2021): 0.06136078431372547, np.int32(2022): 0.20843611111111113, np.int32(2023): 0.20532391304347827, np.int32(2024): -0.06083749999999998, np.int32(2025): 0.2927918918918919} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.314 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

## V2_MES_TOKYO_OPEN_O5_RR1.5_long_REL_VOL_HIGH
**Description:** MES TOKYO_OPEN O5 RR1.5 LONG with rel_vol > P67 — BH-global survivor t=+4.46 (rel_vol threshold P67=2.351)
**Scope:** MES | TOKYO_OPEN | O5 | RR1.5 | long | expected=positive
**N_total:** 882 | **N_on_signal:** 287

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.282 (ovn80_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.1572140711459432, 'gap_r015_fire': 0.09962835505249555, 'atr70_fire': 0.0032046952357686403, 'ovn80_fire': 0.2821262891331559} |
| T1 T1_wr_monotonicity | WR_spread=0.142 (on=0.536 off=0.395) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=276 ExpR=+0.092 WR=0.536 σ=1.03 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=2.19 IS_SR=0.09 OOS_SR=0.20 N_OOS_on=11 | **FAIL** | WFE=2.19 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.092 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 4/7 in expected direction | **INFO** | 4/7 years (57%) matching expected direction; yr={np.int32(2019): -0.010252941176470557, np.int32(2020): 0.3135021739130435, np.int32(2021): -0.0170625, np.int32(2022): 0.13637674418604653, np.int32(2023): -0.030668085106382956, np.int32(2024): 0.16331785714285713, np.int32(2025): 0.07730000000000004} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.285 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 5 PASS, 1 FAIL, 3 INFO
### Verdict: **CONDITIONAL**

---

## V3_MNQ_SINGAPORE_OPEN_O5_RR1.0_short_REL_VOL_HIGH
**Description:** MNQ SINGAPORE_OPEN O5 RR1.0 SHORT with rel_vol > P67 — BH-global survivor t=+4.27 (rel_vol threshold P67=2.314)
**Scope:** MNQ | SINGAPORE_OPEN | O5 | RR1.0 | short | expected=positive
**N_total:** 866 | **N_on_signal:** 298

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.270 (ovn80_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.13344794137725116, 'gap_r015_fire': 0.11101272163913754, 'atr70_fire': 0.0074812417102484265, 'ovn80_fire': 0.27028533884638256} |
| T1 T1_wr_monotonicity | WR_spread=0.133 (on=0.653 off=0.520) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=285 ExpR=+0.139 WR=0.653 σ=0.84 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=3.28 IS_SR=0.17 OOS_SR=0.54 N_OOS_on=13 | **FAIL** | WFE=3.28 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.139 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 5/7 in expected direction | **PASS** | 5/7 years (71%) matching expected direction; yr={np.int32(2019): -0.033649999999999985, np.int32(2020): 0.3154818181818182, np.int32(2021): 0.01096666666666668, np.int32(2022): 0.3159825, np.int32(2023): 0.2693972972972973, np.int32(2024): -0.039088372093023245, np.int32(2025): 0.12804878048780485} |
| T8 T8_cross_instrument | twin=MES Δ=+0.219 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---

## V4_MNQ_COMEX_SETTLE_O5_RR1.5_short_REL_VOL_HIGH_AND_F6
**Description:** MNQ COMEX_SETTLE O5 RR1.5 SHORT with rel_vol > P67 AND F6_INSIDE_PDR — confluence survivor t=+3.51, Δ_OOS=+0.276 (rel_vol threshold P67=1.788)
**Scope:** MNQ | COMEX_SETTLE | O5 | RR1.5 | short | expected=positive
**N_total:** 798 | **N_on_signal:** 154

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.169 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.16926718655513842, 'gap_r015_fire': 0.01627267333223833, 'atr70_fire': -0.053566371245498214, 'ovn80_fire': 0.0240081753805338} |
| T1 T1_wr_monotonicity | WR_spread=0.157 (on=0.591 off=0.433) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=149 ExpR=+0.358 WR=0.591 σ=1.14 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | N_OOS=5 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.358 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 7/7 in expected direction | **PASS** | 7/7 years (100%) matching expected direction; yr={np.int32(2019): 0.8982, np.int32(2020): 0.03369999999999999, np.int32(2021): 0.44517272727272733, np.int32(2022): 0.11882352941176468, np.int32(2023): 0.49479032258064515, np.int32(2024): 0.20883333333333334, np.int32(2025): 0.63803} |
| T8 T8_cross_instrument | twin=MES Δ=+0.278 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---
