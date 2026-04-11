# ML V3 — Pooled RR-Stratified Meta-Label Postmortem

**Run started:** 2026-04-11T00:14:00.456315+00:00
**Git commit:** `a775604443d633c85d0e464e50985409ddf07518`
**Hypothesis SHA (v2):** `b7d5c1c8db111891`
**Data hash:** `adc3bd36d5628cce`
**Holdout end (locked):** 2026-04-06

**Active strategies (post G3):** 29
**Dropped by G2 (train-window ExpR <= 0):** 0

## VERDICT: DEAD (0/3 survived)

### Trial v3_rr10 (RR=1.0)

- Training rows: 11247
- Training strategies: 11
- Baseline train ExpR: 0.0929R
- Baseline train WR: 0.5859
- CPCV mean AUC: 0.5018
- CPCV mean Brier: 0.2505
- Youden J threshold: 0.5717
- Null A p-value (shuffled labels): 0.4378
- Feature importance (MDA):
  - orb_volume_norm: 0.2456
  - orb_size_norm: 0.1927
  - atr_20_pct: 0.1779
  - orb_pre_velocity_norm: 0.1682
  - gap_open_points_norm: 0.1599
  - prior_sessions_broken: 0.0558

**Holdout (593 rows):**
- Baseline holdout ExpR: 0.1229R
- ML holdout ExpR: -0.6687R
- Paired bootstrap lower 95% CI (ML - baseline): -0.2157R
- Baseline net dollars: $8144.85
- ML net dollars: $-905.29
- Dollar lift: $-9050.14

**Per-strategy holdout:**
| strategy_id | n_trades | baseline_expR | ml_expR | trades_taken |
|---|---|---|---|---|
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` | 7 | 0.0660 | -0.0614 | 2 |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` | 11 | 0.0117 | -0.0614 | 2 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12` | 61 | 0.0954 | -1.0000 | 4 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | 63 | 0.1141 | -1.0000 | 4 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` | 60 | 0.1384 | -1.0000 | 4 |
| `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12` | 63 | 0.1302 | 0.0000 | 0 |
| `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5` | 67 | 0.1410 | 0.0000 | 0 |
| `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` | 64 | 0.1945 | 0.0000 | 0 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 66 | 0.1327 | 0.0000 | 0 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | 66 | 0.1327 | 0.0000 | 0 |
| `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12` | 65 | 0.0515 | 0.8768 | 1 |

**TRIAL v3_rr10: DEAD**
  - C1 BH FDR adjusted p=0.7313 >= 0.05
  - C2 proxy (CPCV AUC=0.5018 < 0.52, walk-forward deferred)
  - C3/C4 paired bootstrap lift lower CI=-0.2157 <= 0
  - C8 dollar lift=$-9050.14 <= $0
  - C9 per-strategy local loss: 5 strategies go negative under ML gate: ['MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08', 'MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8', 'MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12', 'MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5', 'MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100']

---

### Trial v3_rr15 (RR=1.5)

- Training rows: 10972
- Training strategies: 9
- Baseline train ExpR: 0.1026R
- Baseline train WR: 0.4765
- CPCV mean AUC: 0.4991
- CPCV mean Brier: 0.2507
- Youden J threshold: 0.5288
- Null A p-value (shuffled labels): 0.4876
- Feature importance (MDA):
  - orb_volume_norm: 0.2443
  - orb_pre_velocity_norm: 0.1981
  - orb_size_norm: 0.1924
  - gap_open_points_norm: 0.1675
  - atr_20_pct: 0.1627
  - prior_sessions_broken: 0.0351

**Holdout (571 rows):**
- Baseline holdout ExpR: 0.1845R
- ML holdout ExpR: 0.0116R
- Paired bootstrap lower 95% CI (ML - baseline): -0.2717R
- Baseline net dollars: $8632.87
- ML net dollars: $-640.43
- Dollar lift: $-9273.30

**Per-strategy holdout:**
| strategy_id | n_trades | baseline_expR | ml_expR | trades_taken |
|---|---|---|---|---|
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | 61 | 0.0494 | -0.3100 | 7 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | 58 | 0.1037 | -0.1950 | 6 |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12` | 63 | 0.2632 | 0.0939 | 13 |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | 67 | 0.2857 | 0.1721 | 14 |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100` | 64 | 0.3460 | 0.1721 | 14 |
| `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` | 63 | 0.0926 | -1.0000 | 2 |
| `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5` | 63 | 0.0926 | -1.0000 | 2 |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | 65 | 0.2062 | 0.0422 | 16 |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | 67 | 0.2002 | 0.0992 | 17 |

**TRIAL v3_rr15: DEAD**
  - C1 BH FDR adjusted p=0.7313 >= 0.05
  - C2 proxy (CPCV AUC=0.4991 < 0.52, walk-forward deferred)
  - C3/C4 paired bootstrap lift lower CI=-0.2717 <= 0
  - C8 dollar lift=$-9273.30 <= $0
  - C9 per-strategy local loss: 4 strategies go negative under ML gate: ['MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5', 'MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100', 'MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12', 'MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5']

---

### Trial v3_rr20 (RR=2.0)

- Training rows: 4674
- Training strategies: 3
- Baseline train ExpR: 0.0788R
- Baseline train WR: 0.3994
- CPCV mean AUC: 0.4765
- CPCV mean Brier: 0.2510
- Youden J threshold: 0.4751
- Null A p-value (shuffled labels): 0.9403
- Feature importance (MDA):
  - orb_size_norm: 0.2086
  - orb_volume_norm: 0.2036
  - orb_pre_velocity_norm: 0.1926
  - atr_20_pct: 0.1814
  - gap_open_points_norm: 0.1641
  - prior_sessions_broken: 0.0498

**Holdout (194 rows):**
- Baseline holdout ExpR: 0.1649R
- ML holdout ExpR: 0.1850R
- Paired bootstrap lower 95% CI (ML - baseline): -0.0240R
- Baseline net dollars: $1556.04
- ML net dollars: $1960.76
- Dollar lift: $404.72

**Per-strategy holdout:**
| strategy_id | n_trades | baseline_expR | ml_expR | trades_taken |
|---|---|---|---|---|
| `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5` | 60 | 0.0426 | 0.1374 | 55 |
| `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5` | 67 | 0.2104 | 0.1617 | 65 |
| `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5` | 67 | 0.2289 | 0.2475 | 66 |

**TRIAL v3_rr20: DEAD**
  - C1 BH FDR adjusted p=0.9403 >= 0.05
  - C2 proxy (CPCV AUC=0.4765 < 0.52, walk-forward deferred)
  - C3/C4 paired bootstrap lift lower CI=-0.0240 <= 0

---

## Narrative

All trials DEAD under pre-registered kill criteria. This is consistent with the prior ML V1/V2 verdicts (memory references: ml_phase0_final.md, ml_v2_final_verdict.md, ml_institutional_audit_p1.md) that ML meta-labeling over validated ORB strategies does not provide net lift sufficient to justify the infrastructure. Stage 4 should delete trading_app/ml/ and update Blueprint NO-GO registry.

## Caveats

- The walk-forward multi-cut (Criterion 6, Amendment A12) was approximated in this run via CPCV AUC > 0.52 as a C2 proxy. A true 5-fold walk-forward (4yr train / 6mo test) is deferred to a follow-up run if any trial survived C1/C3/C4/C7/C8/C9.
- The full Bailey-LdP 2014 DSR formula (Criterion 5) was approximated via the paired bootstrap lower 95% CI > 0 test (C4). A proper DSR with skewness/kurtosis corrections is deferred to a follow-up run.
- The Chordia t-statistic (Criterion 4) was not computed independently; the paired bootstrap already provides a 95% CI check which is a stricter version of the same null-hypothesis test.
- Sensitivity analysis (D6, active + retired pool) is noted as a follow-up if any trial survived.