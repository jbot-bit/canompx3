# MNQ EUROPE_FLOW unfiltered pre-break context family v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml` (LOCKED, commit_sha=0cdd40f4)
**Script:** `research/mnq_l1_europe_flow_prebreak_context_v1.py`
**Lane:** `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_UNFILTERED`
**Observed family K:** `2`

## Verdict

`KILL`

Family verdict applies only to this prereg's two overlay hypotheses on the exact unfiltered L1 lane.

## Baseline lane truth

- IS rows: `1718`
- OOS rows: `72`
- Unfiltered baseline IS ExpR: `0.0432`
- Unfiltered baseline OOS ExpR: `0.2928`
- One-sample IS t / p vs 0: `t=1.613` `p=0.1069`

## Frozen IS-only thresholds

- `pre_velocity_HIGH_Q3`: `0.5000`
- `rel_vol_HIGH_Q3`: `1.9206`

## Family results

| Hypothesis | Threshold | Fire% | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | Delta_OOS | raw_p | q_family | years_pos | dir_match_oos | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| H01_PRE_VELOCITY_HIGH_Q3 | 0.5000 | 0.3201 | 540 | 1178 | 33 | 0.0665 | 0.0324 | 0.0341 | 0.1130 | 0.5578 | 0.5578 | 5 | True | **KILL** |
| H02_RELVOL_HIGH_Q3 | 1.9206 | 0.3318 | 565 | 1153 | 29 | 0.1286 | 0.0013 | 0.1273 | -0.3959 | 0.0268 | 0.0537 | 6 | False | **KILL** |

## Decision notes

- This runner tests magnitude overlays only. It does not retest the null direction-alignment framing from PR #72.
- Any hypothesis outside the prereg fire-rate operating band is killed even if the raw p-value is small.

## Yearly IS delta by hypothesis

### H01_PRE_VELOCITY_HIGH_Q3

| Year | N_on | N_off | Delta_IS | Eligible |
|---:|---:|---:|---:|---|
| 2019 | 27 | 144 | 0.3462 | True |
| 2020 | 75 | 181 | 0.0252 | True |
| 2021 | 73 | 186 | -0.1568 | True |
| 2022 | 106 | 152 | 0.1943 | True |
| 2023 | 81 | 177 | 0.0546 | True |
| 2024 | 87 | 172 | 0.0279 | True |
| 2025 | 91 | 166 | -0.2290 | True |

### H02_RELVOL_HIGH_Q3

| Year | N_on | N_off | Delta_IS | Eligible |
|---:|---:|---:|---:|---|
| 2019 | 67 | 104 | 0.0141 | True |
| 2020 | 91 | 165 | 0.2491 | True |
| 2021 | 97 | 162 | 0.1431 | True |
| 2022 | 81 | 177 | 0.1327 | True |
| 2023 | 74 | 184 | -0.1317 | True |
| 2024 | 89 | 170 | 0.3154 | True |
| 2025 | 66 | 191 | 0.2300 | True |

## Outputs

- Row-level CSV: `docs/audit/results/2026-04-21-l1-europe-flow-pre-break-context-prereg-rows.csv`
