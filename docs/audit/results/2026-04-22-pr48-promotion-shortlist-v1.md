# PR48 promotion shortlist v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-22-pr48-promotion-shortlist-v1.yaml`
**Pre-reg commit SHA:** `5c39f142`
**Canonical layers:** `daily_features`, `orb_outcomes`
**Scope:** O5 x E2 x CB1 x RR1.5
**Candidates:** `MES q45 executable`, `MGC continuous executable`, `MES+MGC duo`, `MNQ continuous executable shadow add-on`
**Primary tests:** daily dollar delta vs the declared parent comparator, BH FDR at family `K=4` on IS.
**Sacred OOS window:** `2026-01-01` onward
**Latest canonical trading day:** `2026-04-16`

## IS shortlist tests

| candidate | mean_daily_delta_$ | t | p_two_tailed | bh_survives | direction_positive |
|---|---:|---:|---:|:---:|:---:|
| MES:q45_exec | +17.98 | +8.642 | 0.0000 | Y | Y |
| MGC:cont_exec | +15.29 | +4.188 | 0.0000 | Y | Y |
| DUO:mes_q45_plus_mgc_cont_exec | +25.92 | +9.050 | 0.0000 | Y | Y |
| MNQ:shadow_addon | +44.71 | +5.604 | 0.0000 | Y | Y |

## OOS direction checks

| candidate | IS sign | OOS sign | OOS mean_daily_delta_$ | direction_match |
|---|---:|---:|---:|:---:|
| MES:q45_exec | + | + | +17.95 | Y |
| MGC:cont_exec | + | + | +51.10 | Y |
| DUO:mes_q45_plus_mgc_cont_exec | + | + | +68.34 | Y |
| MNQ:shadow_addon | + | + | +7.27 | Y |

## Candidate-level metrics

| instrument | era | role | policy_ev_per_opp_r | daily_total_$ | daily_max_dd_$ |
|---|---|---|---:|---:|---:|
| MES | IS | parent | -0.1160 | -25,591 | +28,541 |
| MES | OOS | parent | -0.0902 | -1,349 | +2,031 |
| MES | IS | q45_exec | -0.0055 | +5,768 | +6,560 |
| MES | OOS | q45_exec | -0.0047 | -57 | +1,641 |
| MGC | IS | parent | -0.1346 | -17,739 | +22,483 |
| MGC | OOS | parent | +0.0695 | +13,494 | +1,984 |
| MGC | IS | cont_exec | -0.0854 | -3,505 | +16,306 |
| MGC | OOS | cont_exec | +0.1152 | +17,122 | +2,426 |
| MNQ | IS | cont_exec_shadow | +0.0823 | +79,311 | +3,821 |
| MNQ | OOS | cont_exec_shadow | +0.0579 | +531 | +4,331 |

## Combo metrics

| combo | era | total_$ | mean_daily_$ | max_dd_$ | best_day_$ | worst_day_$ |
|---|---|---:|---:|---:|---:|---:|
| parent_duo | IS | -43,330 | -24.63 | +50,114 | +1,354 | -1,055 |
| candidate_duo | IS | +2,263 | +1.29 | +20,897 | +1,831 | -991 |
| candidate_trio | IS | +81,574 | +45.98 | +13,098 | +3,701 | -2,331 |
| parent_duo | OOS | +12,145 | +168.68 | +1,869 | +1,726 | -1,226 |
| candidate_duo | OOS | +17,066 | +237.02 | +3,121 | +3,760 | -1,785 |
| candidate_trio | OOS | +17,596 | +241.04 | +6,510 | +4,758 | -2,286 |

## Interpretation guardrails

- This is a shortlist-action study, not a fresh discovery scan.
- Dollar deltas use canonical `risk_dollars` translation from `entry_price`, `stop_price`, and instrument cost specs.
- MNQ remains a shadow add-on unless it improves the MES/MGC duo on fresh OOS, not just IS.
- OOS from 2026-01-01 to 2026-04-16 is thin; use it to decide continue vs shadow, not full deployment.
