# MNQ Profile Candidate Proposal

**Date:** 2026-05-11
**Profile:** `topstep_50k_mnq_auto`
**Live impact:** None. No DB, schema, broker, validated-setups, or `lane_allocation.json` mutation.

## Scope

This classifies the 177 `CONTROLLED_LIVE_PILOT_CANDIDATE` MNQ rows from `docs/audit/results/2026-05-11-mnq-all-active-deployability.json` into the queue-required profile-construction taxonomy.

This is a paper/sandbox-only proposal gate. A `PASS_*` row is not live authorization; it still carries controlled-pilot slippage/event-tail status and must pass operator preflight before any allocation mutation.

## Grounding / Authority

- Multiple-testing and strict replay gate: `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` extracted from `resources/Two_Million_Trading_Strategies_FDR.pdf`.
- Selection-bias and correlated-trial caution: `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` extracted from `resources/deflated-sharpe.pdf`.
- Portfolio add/replace and correlation discipline: `docs/institutional/literature/carver_2015_ch11_portfolios.md` extracted from `resources/Robert Carver - Systematic Trading.pdf`.
- Live drift monitoring requirement: `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` extracted from `resources/real_time_strategy_monitoring_cusum.pdf`.
- Prop-firm profile constraints: `resources/prop-firm-official-rules.md` plus `trading_app/prop_profiles.py`.
- Research/deployment separation: `docs/institutional/research_pipeline_contract.md` and `docs/institutional/pre_registered_criteria.md`.

## Classification Counts

| Decision | Count |
|---|---:|
| `PASS_ADD` | 0 |
| `PASS_REPLACE` | 0 |
| `PARK` | 50 |
| `KILL` | 118 |
| `ARCHITECTURE_REQUIRED` | 9 |

## Proposed Paper/Sandbox Change Set

No `PASS_ADD` or `PASS_REPLACE` candidates survived the profile-construction gate.

## Main Blockers

| Blocker | Count |
|---|---:|
| `dedupe_dominated_variant` | 118 |
| `chordia_missing` | 45 |
| `non_default_stop_multiplier` | 9 |
| `runtime_sr_not_evaluated` | 2 |
| `already_selected` | 2 |
| `chordia_fail_both` | 1 |

## Highest-Signal Parked Rows

| Candidate | Reason | Chordia | Add dAnnR | Add dSharpe | Corr |
|---|---|---|---:|---:|---|
| `MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_DIR_LONG_O30` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | Already selected in the current active profile; no proposal change. | `PASS_CHORDIA` | +0.0 | +0.000 | `False` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | Already selected in the current active profile; no proposal change. | `PASS_CHORDIA` | +0.0 | +0.000 | `False` |
| `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_TOKYO_OPEN_E2_RR2.5_CB1_ORB_VOL_4K_O15` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_US_DATA_1000_E2_RR2.5_CB1_VWAP_MID_ALIGNED` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT10` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT10_O15` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_COST_LT12_O30` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P50` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_US_DATA_1000_E2_RR2.5_CB1_CROSS_NYSE_MOMENTUM` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ORB_VOL_4K_O30` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT12_O15` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_COST_LT15_O15` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_COST_LT12` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ORB_G8_O15` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT15` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R080` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G8` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P30` | Allocator Chordia gate does not permit deploy (MISSING). | `MISSING` | - | - | - |

## Verdict

No direct allocation mutation is authorized by this report. Rows parked by `runtime_sr_not_evaluated` may only enter a PROVISIONAL bootstrap patch if a pre-promotion Criterion 12 SR evaluation returns `CONTINUE`, `NO_DATA`, or a code-backed `watch` review. An unreviewed SR `ALARM` remains a hard deployment block. Rows parked by missing Chordia audit need exact-lane strict replay before any profile proposal; dominated duplicates should not be re-opened unless the selected head fails under a newer audit.

## Reproduction

```bash
./.venv-wsl/bin/python research/mnq_profile_candidate_proposal_2026_05_11.py
# Optional patch emission path; still fails closed on unreviewed SR ALARM:
./.venv-wsl/bin/python research/mnq_profile_candidate_proposal_2026_05_11.py --bootstrap-runtime-control
```
