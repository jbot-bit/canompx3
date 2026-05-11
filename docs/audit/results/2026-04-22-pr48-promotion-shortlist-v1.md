# PR48 promotion shortlist v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-22-pr48-promotion-shortlist-v1.yaml`
**Pre-reg commit SHA:** `5c39f142`
**Canonical layers:** `daily_features`, `orb_outcomes`
**Scope:** O5 x E2 x CB1 x RR1.5
**Candidates:** `MES q45 executable`, `MGC continuous executable`, `MES+MGC duo`, `MNQ continuous executable shadow add-on`
**Primary tests:** daily dollar delta vs the declared parent comparator, BH FDR at family `K=4` on IS.
**Sacred OOS window:** `2026-01-01` onward
**Latest canonical trading day:** `2026-05-07`

## IS shortlist tests

| candidate | mean_daily_delta_$ | t | p_two_tailed | bh_survives | direction_positive |
|---|---:|---:|---:|:---:|:---:|
| MES:q45_exec | +16.19 | +8.219 | 0.0000 | Y | Y |
| MGC:cont_exec | +16.60 | +4.935 | 0.0000 | Y | Y |
| DUO:mes_q45_plus_mgc_cont_exec | +24.76 | +9.257 | 0.0000 | Y | Y |
| MNQ:shadow_addon | +56.39 | +7.446 | 0.0000 | Y | Y |

## OOS direction checks

| candidate | IS sign | OOS sign | OOS mean_daily_delta_$ | direction_match |
|---|---:|---:|---:|:---:|
| MES:q45_exec | + | + | +20.48 | Y |
| MGC:cont_exec | + | + | +28.76 | Y |
| DUO:mes_q45_plus_mgc_cont_exec | + | + | +47.29 | Y |
| MNQ:shadow_addon | + | + | +38.87 | Y |

## Candidate-level metrics

| instrument | era | role | policy_ev_per_opp_r | daily_total_$ | daily_max_dd_$ |
|---|---|---|---:|---:|---:|
| MES | IS | parent | -0.1019 | -17,215 | +21,201 |
| MES | OOS | parent | -0.0863 | -1,500 | +2,337 |
| MES | IS | q45_exec | -0.0015 | +13,502 | +4,967 |
| MES | OOS | q45_exec | +0.0040 | +323 | +1,629 |
| MGC | IS | parent | -0.1302 | -16,789 | +22,540 |
| MGC | OOS | parent | +0.0406 | +12,575 | +3,457 |
| MGC | IS | cont_exec | -0.0801 | -120 | +14,673 |
| MGC | OOS | cont_exec | +0.0743 | +14,962 | +4,059 |
| MNQ | IS | cont_exec_shadow | +0.0843 | +108,668 | +3,775 |
| MNQ | OOS | cont_exec_shadow | +0.0799 | +3,537 | +3,666 |

## Combo metrics

| combo | era | total_$ | mean_daily_$ | max_dd_$ | best_day_$ | worst_day_$ |
|---|---|---:|---:|---:|---:|---:|
| parent_duo | IS | -34,004 | -17.77 | +42,999 | +1,567 | -1,054 |
| candidate_duo | IS | +13,383 | +6.99 | +15,795 | +1,885 | -1,070 |
| candidate_trio | IS | +122,051 | +63.34 | +9,005 | +3,786 | -2,329 |
| parent_duo | OOS | +11,075 | +124.44 | +4,618 | +1,947 | -1,401 |
| candidate_duo | OOS | +15,285 | +171.74 | +4,057 | +3,750 | -1,694 |
| candidate_trio | OOS | +18,821 | +206.83 | +4,941 | +4,794 | -2,233 |

## Interpretation guardrails

- This is a shortlist-action study, not a fresh discovery scan.
- Dollar deltas use canonical `risk_dollars` translation from `entry_price`, `stop_price`, and instrument cost specs.
- MNQ remains a shadow add-on unless it improves the MES/MGC duo on fresh OOS, not just IS.
- OOS from 2026-01-01 to 2026-04-16 is thin; use it to decide continue vs shadow, not full deployment.

## Verdict

`CONTINUE` as a bounded shortlist for conditional-role translation. The recovered winners are alive as research-level allocator / filter candidates, but this report alone is not a validated-shelf or live-routing promotion.

## Reproduction

- `python3 -m py_compile research/pr48_promotion_shortlist_v1.py`
- `./.venv-wsl/bin/python research/pr48_promotion_shortlist_v1.py`

## Caveats

- Latest canonical day in this replay is fixed by the current DB state, not by this script.
- OOS remains thin and should not be used to re-specify the shortlist.
- Current repo allocation consumes validated standalone lanes, not these conditional-role outputs directly.
