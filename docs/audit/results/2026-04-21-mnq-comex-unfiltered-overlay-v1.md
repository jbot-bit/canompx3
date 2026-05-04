# MNQ COMEX_SETTLE unfiltered overlay family v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-21-mnq-comex-unfiltered-overlay-v1.yaml` (LOCKED, commit_sha=0cdd40f4)
**Script:** `research/mnq_comex_unfiltered_overlay_v1.py`
**Lane:** `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_UNFILTERED`
**Observed family K:** `2`

## Verdict

`PARK`

Family verdict applies only to this prereg's two overlay hypotheses on the exact unfiltered COMEX lane.

## Baseline lane truth

- IS rows: `1658`
- OOS rows: `69`
- Scratch rows total: `25`
- Scratch-inclusive baseline IS ExpR: `0.0658`
- Scratch-inclusive baseline OOS ExpR: `0.0055`
- One-sample IS t / p vs 0: `t=2.367` `p=0.0180`

## Frozen IS-only thresholds

- `rel_vol_HIGH_Q3`: `1.8011`
- `overnight_range_pct_HIGH_Q3`: `69.0252`

## Family results

| Hypothesis | Threshold | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | Delta_OOS | raw_p | q_family | years_pos | dir_match_oos | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| H01_RELVOL_HIGH_Q3 | 1.8011 | 546 | 1112 | 25 | 0.2309 | -0.0153 | 0.2463 | -0.0182 | 0.0000 | 0.0001 | 6 | False | **PARK** |
| H02_OVERNIGHT_RANGE_HIGH_Q3 | 69.0252 | 541 | 1117 | 27 | 0.1146 | 0.0421 | 0.0724 | -0.1468 | 0.2257 | 0.2257 | 4 | False | **KILL** |

## Decision notes

- `H01_RELVOL_HIGH_Q3` survives the IS gates but parks because the OOS delta flips sign on usable OOS sample.
- `H02_OVERNIGHT_RANGE_HIGH_Q3` fails the prereg raw-p gate and is killed.

## Scratch-inclusive vs resolved-only comparison

| Hypothesis | ExpR_on_scratch0_IS | ExpR_on_resolved_IS | ExpR_off_scratch0_IS | ExpR_off_resolved_IS |
|---|---:|---:|---:|---:|
| H01_RELVOL_HIGH_Q3 | 0.2309 | 0.2370 | -0.0153 | -0.0155 |
| H02_OVERNIGHT_RANGE_HIGH_Q3 | 0.1146 | 0.1165 | 0.0421 | 0.0426 |

## Yearly IS delta by hypothesis

### H01_RELVOL_HIGH_Q3

| Year | N_on | N_off | Delta_IS | Eligible |
|---:|---:|---:|---:|---|
| 2019 | 58 | 106 | -0.0535 | True |
| 2020 | 91 | 158 | 0.4287 | True |
| 2021 | 84 | 167 | 0.3660 | True |
| 2022 | 63 | 187 | 0.0557 | True |
| 2023 | 86 | 162 | 0.4898 | True |
| 2024 | 86 | 163 | 0.1092 | True |
| 2025 | 78 | 169 | 0.2045 | True |

### H02_OVERNIGHT_RANGE_HIGH_Q3

| Year | N_on | N_off | Delta_IS | Eligible |
|---:|---:|---:|---:|---|
| 2019 | 34 | 130 | -0.1006 | True |
| 2020 | 95 | 154 | 0.3002 | True |
| 2021 | 83 | 168 | 0.0448 | True |
| 2022 | 77 | 173 | -0.0574 | True |
| 2023 | 81 | 167 | -0.2267 | True |
| 2024 | 87 | 162 | 0.1743 | True |
| 2025 | 84 | 163 | 0.1305 | True |

## Outputs

- Row-level CSV: `docs/audit/results/2026-04-21-mnq-comex-unfiltered-overlay-v1-rows.csv`
