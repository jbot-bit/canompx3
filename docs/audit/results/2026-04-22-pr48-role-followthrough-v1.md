# PR48 role follow-through v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-22-pr48-role-followthrough-v1.yaml`
**Pre-reg commit SHA:** `7e3500e0`
**Canonical layers:** `daily_features`, `orb_outcomes`
**Scope:** `MNQ, MES, MGC` x O5 x E2 x CB1 x RR1.5
**Primary test:** daily policy delta versus parent for `Q4+Q5` and `continuous_desired`, BH FDR at family `K=6`.
**Diagnostics:** drawdown, dollar translation, executable drag, and max-open-lots proxy.
**Sacred OOS window:** `2026-01-01` onward.
**Latest canonical trading day:** `2026-05-07`

## MNQ

### IS daily delta tests

| candidate | mean_daily_delta_r | t | p_two_tailed | bh_survives | direction_positive |
|---|---:|---:|---:|:---:|:---:|
| q45_filter | +0.0715 | +1.202 | 0.2294 | N | Y |
| continuous_desired | +0.2749 | +9.844 | 0.0000 | Y | Y |

### Role metrics

| era | role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp_r | capital_normalized_ev_r | daily_total_r | daily_max_dd_r | daily_total_dollars | daily_max_dd_dollars |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | parent | 19922 | 1.000 | +0.0412 | +0.0412 | +0.0412 | +820.36 | +166.24 | +75,028 | +2,861 |
| IS | q45_filter | 7966 | 0.400 | +0.1200 | +0.0480 | +0.1200 | +956.20 | +35.50 | +56,168 | +2,493 |
| IS | continuous_desired | 19922 | 1.000 | +0.0412 | +0.0674 | +0.0674 | +1342.31 | +120.40 | +95,531 | +3,125 |
| OOS | parent | 919 | 1.000 | +0.0827 | +0.0827 | +0.0827 | +75.96 | +12.46 | +4,819 | +2,147 |
| OOS | q45_filter | 381 | 0.415 | +0.1281 | +0.0531 | +0.1281 | +48.80 | +9.63 | +3,216 | +1,980 |
| OOS | continuous_desired | 919 | 1.000 | +0.0827 | +0.0906 | +0.0893 | +83.23 | +14.15 | +5,003 | +2,925 |

### OOS direction match

| candidate | IS daily delta sign | OOS daily delta sign | OOS mean_daily_delta_r | direction_match |
|---|---:|---:|---:|:---:|
| q45_filter | + | - | -0.3122 | N |
| continuous_desired | + | + | +0.0835 | Y |

### Executable drag and concurrency diagnostics

- Continuous desired -> executable proxy policy EV drag: IS `+0.0169R/opp`, OOS `-0.0107R/opp`
- Continuous executable proxy total dollars: IS `$+108,668`, OOS `$+3,537`
- Q4+Q5 max open contracts/lots in IS: 4 contracts / 1 lots
- Continuous executable max open contracts/lots in IS: 7 contracts / 1 lots

## MES

### IS daily delta tests

| candidate | mean_daily_delta_r | t | p_two_tailed | bh_survives | direction_positive |
|---|---:|---:|---:|:---:|:---:|
| q45_filter | +0.9606 | +17.541 | 0.0000 | Y | Y |
| continuous_desired | +0.2981 | +11.881 | 0.0000 | Y | Y |

### Role metrics

| era | role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp_r | capital_normalized_ev_r | daily_total_r | daily_max_dd_r | daily_total_dollars | daily_max_dd_dollars |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | parent | 18151 | 1.000 | -0.1019 | -0.1019 | -0.1019 | -1849.69 | +1854.79 | -17,215 | +21,201 |
| IS | q45_filter | 7261 | 0.400 | -0.0038 | -0.0015 | -0.0038 | -27.43 | +171.41 | +13,502 | +4,967 |
| IS | continuous_desired | 18151 | 1.000 | -0.1019 | -0.0708 | -0.0708 | -1284.26 | +1284.69 | -2,444 | +13,031 |
| OOS | parent | 873 | 1.000 | -0.0863 | -0.0863 | -0.0863 | -75.35 | +78.58 | -1,500 | +2,337 |
| OOS | q45_filter | 356 | 0.408 | +0.0097 | +0.0040 | +0.0097 | +3.47 | +25.08 | +323 | +1,629 |
| OOS | continuous_desired | 873 | 1.000 | -0.0863 | -0.0568 | -0.0559 | -49.60 | +58.12 | -792 | +2,715 |

### OOS direction match

| candidate | IS daily delta sign | OOS daily delta sign | OOS mean_daily_delta_r | direction_match |
|---|---:|---:|---:|:---:|
| q45_filter | + | + | +0.8856 | Y |
| continuous_desired | + | + | +0.2893 | Y |

### Executable drag and concurrency diagnostics

- Continuous desired -> executable proxy policy EV drag: IS `+0.0204R/opp`, OOS `+0.0192R/opp`
- Continuous executable proxy total dollars: IS `$+6,108`, OOS `$-256`
- Q4+Q5 max open contracts/lots in IS: 3 contracts / 1 lots
- Continuous executable max open contracts/lots in IS: 7 contracts / 1 lots

## MGC

### IS daily delta tests

| candidate | mean_daily_delta_r | t | p_two_tailed | bh_survives | direction_positive |
|---|---:|---:|---:|:---:|:---:|
| q45_filter | +0.9266 | +13.320 | 0.0000 | Y | Y |
| continuous_desired | +0.2382 | +7.600 | 0.0000 | Y | Y |

### Role metrics

| era | role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp_r | capital_normalized_ev_r | daily_total_r | daily_max_dd_r | daily_total_dollars | daily_max_dd_dollars |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | parent | 8082 | 1.000 | -0.1302 | -0.1302 | -0.1302 | -1052.13 | +1078.57 | -16,789 | +22,540 |
| IS | q45_filter | 3229 | 0.400 | -0.0377 | -0.0151 | -0.0377 | -121.84 | +222.00 | +5,293 | +4,038 |
| IS | continuous_desired | 8082 | 1.000 | -0.1302 | -0.1006 | -0.1007 | -812.99 | +884.36 | -7,447 | +18,028 |
| OOS | parent | 681 | 1.000 | +0.0406 | +0.0406 | +0.0406 | +27.68 | +29.59 | +12,575 | +3,457 |
| OOS | q45_filter | 241 | 0.354 | +0.1456 | +0.0515 | +0.1456 | +35.09 | +12.35 | +8,814 | +1,709 |
| OOS | continuous_desired | 681 | 1.000 | +0.0406 | +0.0639 | +0.0655 | +43.53 | +27.81 | +14,666 | +3,455 |

### OOS direction match

| candidate | IS daily delta sign | OOS daily delta sign | OOS mean_daily_delta_r | direction_match |
|---|---:|---:|---:|:---:|
| q45_filter | + | + | +0.0893 | Y |
| continuous_desired | + | + | +0.1910 | Y |

### Executable drag and concurrency diagnostics

- Continuous desired -> executable proxy policy EV drag: IS `+0.0205R/opp`, OOS `+0.0104R/opp`
- Continuous executable proxy total dollars: IS `$-120`, OOS `$+14,962`
- Q4+Q5 max open contracts/lots in IS: 3 contracts / 1 lots
- Continuous executable max open contracts/lots in IS: 7 contracts / 1 lots

## Interpretation guardrails

- Primary inference is on daily policy delta versus parent, not selected-trade mean.
- Continuous desired sizing is the research object; executable drag is a diagnostic, not a separate promoted winner.
- OOS from 2026-01-01 to 2026-04-16 is still thin and should be treated as direction/implementation monitoring only.

## Verdict

`CONTINUE` for the bounded conditional-role follow-through. The measured daily-delta evidence supports further translation work, but this remains research-level evidence rather than a direct live-promotion decision.

## Reproduction

- `python3 -m py_compile research/pr48_role_followthrough_v1.py`
- `./.venv-wsl/bin/python research/pr48_role_followthrough_v1.py`

## Caveats

- Executable drag is a diagnostic overlay, not a separate promoted winner.
- OOS remains a monitor, not a tuning surface.
- This report does not create a native validation/allocation bridge by itself.
