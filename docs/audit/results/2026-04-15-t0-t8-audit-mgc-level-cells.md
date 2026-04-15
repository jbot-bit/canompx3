# T0-T8 Audit — 4 New MGC Level Cells

**Date:** 2026-04-15
**Source:** `docs/audit/results/2026-04-15-mgc-level-scan.md` promising list
**Protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5

**MGC-specific caveats (no bias applied):**
- Data window 2022-06 to 2026-04 (~3.8 years) — fewer years for T7 per-year than MES/MNQ
- OOS window 3 months (same as all instruments) — T3 thin at low fire rates
- T8 cross-instrument twin defaults to MNQ in existing code (equity vs gold — wrong asset class)
- 2026 MGC regime shift flagged on prior M1 cell (WFE=0.33) — monitor

## L1_MGC_LONDON_METALS_O30_RR1.5_long_F2_NEAR_PDL_30
**Description:** MGC LONDON_METALS O30 RR1.5 LONG F2_NEAR_PDL_30 — Δ_OOS=+1.046 strongest MGC TAKE
**Scope:** MGC | LONDON_METALS | O30 | RR1.5 | long | expected=positive
**N_total:** 494 | **N_on_signal:** 161

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.189 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.18921056017033214, 'gap_r015_fire': -0.10454653848130757, 'atr70_fire': -0.01596441493215696, 'ovn80_fire': -0.04878420695285238} |
| T1 T1_wr_monotonicity | WR_spread=0.144 (on=0.548 off=0.404) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=155 ExpR=+0.204 WR=0.548 σ=1.10 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | N_OOS=6 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | deltas=[0.432, 0.311, 0.213] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0030 ExpR_obs=+0.204 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 3/4 in expected direction | **PASS** | 3/4 years (75%) matching expected direction; yr={np.int32(2022): -0.1068206896551724, np.int32(2023): 0.13762075471698107, np.int32(2024): 0.2886578947368421, np.int32(2025): 0.4698742857142858} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.076 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 7 PASS, 1 FAIL, 1 INFO
### Verdict: **CONDITIONAL**

---

## L2_MGC_EUROPE_FLOW_O30_RR1.5_short_F2_NEAR_PDL_30
**Description:** MGC EUROPE_FLOW O30 RR1.5 SHORT F2_NEAR_PDL_30 — Δ_OOS=+0.612
**Scope:** MGC | EUROPE_FLOW | O30 | RR1.5 | short | expected=positive
**N_total:** 464 | **N_on_signal:** 162

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.225 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': -0.22479702709082083, 'gap_r015_fire': -0.0593469059140478, 'atr70_fire': -0.04127864593894861, 'ovn80_fire': -0.07008427979066269} |
| T1 T1_wr_monotonicity | WR_spread=0.146 (on=0.517 off=0.370) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=151 ExpR=+0.117 WR=0.517 σ=1.09 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=4.06 IS_SR=0.11 OOS_SR=0.44 N_OOS_on=11 | **FAIL** | WFE=4.06 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | deltas=[0.082, 0.304, 0.16] | **PASS** | signs match, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0030 ExpR_obs=+0.117 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 3/4 in expected direction | **PASS** | 3/4 years (75%) matching expected direction; yr={np.int32(2022): -0.030902857142857138, np.int32(2023): 0.1950659574468085, np.int32(2024): 0.14962903225806454, np.int32(2025): 0.1307578947368421} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.247 | **FAIL** | sign_ok=False mag_ok=True |

**Counts:** 6 PASS, 2 FAIL, 1 INFO
### Verdict: **KILL_DOWNGRADE**

---

## L3_MGC_NYSE_OPEN_O15_RR1.0_short_F6_INSIDE_PDR
**Description:** MGC NYSE_OPEN O15 RR1.0 SHORT F6_INSIDE_PDR — Δ_OOS=+0.190 TAKE
**Scope:** MGC | NYSE_OPEN | O15 | RR1.0 | short | expected=positive
**N_total:** 433 | **N_on_signal:** 254

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.189 (pdr_r105_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.18864866860765864, 'gap_r015_fire': -0.06190717703063837, 'atr70_fire': 0.03197491842211078, 'ovn80_fire': -0.1458563754291148} |
| T1 T1_wr_monotonicity | WR_spread=0.150 (on=0.611 off=0.461) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=239 ExpR=+0.103 WR=0.611 σ=0.89 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=1.48 IS_SR=0.12 OOS_SR=0.17 N_OOS_on=15 | **FAIL** | WFE=1.48 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0040 ExpR_obs=+0.103 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 3/4 in expected direction | **PASS** | 3/4 years (75%) matching expected direction; yr={np.int32(2022): -0.10247435897435898, np.int32(2023): 0.18574848484848486, np.int32(2024): 0.13678688524590163, np.int32(2025): 0.10988219178082191} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.092 | **FAIL** | sign_ok=False mag_ok=True |

**Counts:** 5 PASS, 2 FAIL, 2 INFO
### Verdict: **KILL_DOWNGRADE**

---

## L4_MGC_TOKYO_OPEN_O30_RR2.0_long_F6_INSIDE_PDR
**Description:** MGC TOKYO_OPEN O30 RR2.0 LONG F6_INSIDE_PDR — Δ_OOS=-0.239 SKIP
**Scope:** MGC | TOKYO_OPEN | O30 | RR2.0 | long | expected=negative
**N_total:** 524 | **N_on_signal:** 463

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.161 (ovn80_fire) | **PASS** | no tautology with deployed filters; correlations={'pdr_r105_fire': 0.02515402412937819, 'gap_r015_fire': -0.16093803092401215, 'atr70_fire': -0.04043735999621401, 'ovn80_fire': -0.16148428110052326} |
| T1 T1_wr_monotonicity | WR_spread=0.190 (on=0.370 off=0.560) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=433 ExpR=-0.104 WR=0.370 σ=1.18 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=-2.74 IS_SR=-0.09 OOS_SR=0.24 N_OOS_on=30 | **FAIL** | WFE=-2.74 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature — no theta grid |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.104 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 4/4 in expected direction | **PASS** | 4/4 years (100%) matching expected direction; yr={np.int32(2022): -0.32679473684210525, np.int32(2023): -0.044307874015748025, np.int32(2024): -0.07848045112781955, np.int32(2025): -0.08943965517241383} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.120 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL**

---
