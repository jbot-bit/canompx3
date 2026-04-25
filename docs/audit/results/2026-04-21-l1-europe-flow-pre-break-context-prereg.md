# MNQ EUROPE_FLOW unfiltered pre-break context family v1

**Generated:** 2026-04-23T00:02:12+00:00
**Pre-reg:** `docs/audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml` (LOCKED, commit_sha=0cdd40f4)
**Script:** `research/l1_europe_flow_pre_break_context_scan.py`
**Lane:** `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_UNFILTERED`
**IS window:** `trading_day < 2026-01-01`
**OOS window:** `trading_day >= 2026-01-01` (descriptive, no tuning)

## Overall Verdict: **KILL**

Decision contract from the prereg:

- `CONTINUE`: at least one hypothesis survives `K=2` family BH-FDR and all IS gates, with no OOS sign flip when OOS `N_on >= 5`.
- `PARK`: an IS signal exists but OOS remains directional-only or does not confirm cleanly.
- `KILL`: zero family survivors.

## Family Summary

| Hypothesis | Feature | P67 | N_on_IS | Fire_IS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | BH | Years+ | N_on_OOS | Fire_OOS | Delta_OOS | dir_match | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|---|
| H01_PRE_VELOCITY_HIGH_Q3 | pre_velocity_HIGH_Q3 | 0.5000 | 540 | 31.4% | +0.0665 | +0.0324 | +0.0341 | 0.5578 | 0.5578 | FAIL | 4 | 33 | 45.8% | +0.1130 | YES | **KILL** |
| H02_RELVOL_HIGH_Q3 | rel_vol_HIGH_Q3 | 1.9206 | 565 | 32.9% | +0.1286 | +0.0013 | +0.1273 | 0.0268 | 0.0537 | FAIL | 6 | 29 | 40.3% | -0.3959 | NO | **KILL** |

## Fire-Rate Sanity

- Prereg target operating band: `5%-95%`.
- `pre_velocity_HIGH_Q3` fire rate: IS `31.4%`, OOS `45.8%`.
- `rel_vol_HIGH_Q3` fire rate: IS `32.9%`, OOS `40.3%`.

## Interpretation

- The prereg family closes cleanly under the frozen two-feature scan.
- This does not justify reopening banned break-bar or ATR-normalized variants.

### pre_velocity_HIGH_Q3 yearly on-signal breakdown

| Year | N_on | ExpR_on | WinRate_on |
|---|---:|---:|---:|
| 2019 | 27 | +0.1325 | 55.6% |
| 2020 | 75 | -0.0030 | 44.0% |
| 2021 | 73 | -0.0917 | 41.1% |
| 2022 | 106 | +0.2129 | 52.8% |
| 2023 | 81 | +0.1500 | 51.9% |
| 2024 | 87 | +0.0572 | 47.1% |
| 2025 | 91 | -0.0046 | 42.9% |

### rel_vol_HIGH_Q3 yearly on-signal breakdown

| Year | N_on | ExpR_on | WinRate_on |
|---|---:|---:|---:|
| 2019 | 67 | -0.1505 | 41.8% |
| 2020 | 91 | +0.1398 | 51.6% |
| 2021 | 97 | +0.1104 | 49.5% |
| 2022 | 81 | +0.1895 | 50.6% |
| 2023 | 74 | +0.0186 | 45.9% |
| 2024 | 89 | +0.2456 | 55.1% |
| 2025 | 66 | +0.3142 | 56.1% |

## Outputs

- `research/output/l1_europe_flow_pre_break_context_rows.csv`
- `research/output/l1_europe_flow_pre_break_context_summary.csv`
