# PR48 role follow-through v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-22-pr48-role-followthrough-v1.yaml`
**Pre-reg commit SHA:** `7e3500e0`
**Canonical layers:** `daily_features`, `orb_outcomes`
**Scope:** `MNQ, MES, MGC` x O5 x E2 x CB1 x RR1.5
**Primary test:** daily policy delta versus parent for `Q4+Q5` and `continuous_desired`, BH FDR at family `K=6`.
**Diagnostics:** drawdown, dollar translation, executable drag, and max-open-lots proxy.
**Sacred OOS window:** `2026-01-01` onward.
**Latest canonical trading day:** `2026-04-16`

## MNQ

### IS daily delta tests

| candidate | mean_daily_delta_r | t | p_two_tailed | bh_survives | direction_positive |
|---|---:|---:|---:|:---:|:---:|
| q45_filter | +0.1157 | +1.812 | 0.0702 | N | Y |
| continuous_desired | +0.2842 | +9.444 | 0.0000 | Y | Y |

### Role metrics

| era | role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp_r | capital_normalized_ev_r | daily_total_r | daily_max_dd_r | daily_total_dollars | daily_max_dd_dollars |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | parent | 17828 | 1.000 | +0.0365 | +0.0365 | +0.0365 | +650.97 | +177.68 | +54,112 | +3,467 |
| IS | q45_filter | 7131 | 0.400 | +0.1196 | +0.0478 | +0.1196 | +852.92 | +42.48 | +41,132 | +2,646 |
| IS | continuous_desired | 17828 | 1.000 | +0.0365 | +0.0643 | +0.0644 | +1147.14 | +131.39 | +69,687 | +3,599 |
| OOS | parent | 771 | 1.000 | +0.0589 | +0.0589 | +0.0589 | +45.45 | +12.98 | +2,083 | +2,555 |
| OOS | q45_filter | 320 | 0.415 | +0.0851 | +0.0353 | +0.0851 | +27.25 | +12.30 | +1,409 | +2,268 |
| OOS | continuous_desired | 771 | 1.000 | +0.0589 | +0.0652 | +0.0643 | +50.29 | +14.80 | +2,006 | +3,199 |

### OOS direction match

| candidate | IS daily delta sign | OOS daily delta sign | OOS mean_daily_delta_r | direction_match |
|---|---:|---:|---:|:---:|
| q45_filter | + | - | -0.2493 | N |
| continuous_desired | + | + | +0.0663 | Y |

### Executable drag and concurrency diagnostics

- Continuous desired -> executable proxy policy EV drag: IS `+0.0179R/opp`, OOS `-0.0073R/opp`
- Continuous executable proxy total dollars: IS `$+79,311`, OOS `$+531`
- Q4+Q5 max open contracts/lots in IS: 4 contracts / 1 lots
- Continuous executable max open contracts/lots in IS: 6 contracts / 1 lots

## MES

### IS daily delta tests

| candidate | mean_daily_delta_r | t | p_two_tailed | bh_survives | direction_positive |
|---|---:|---:|---:|:---:|:---:|
| q45_filter | +1.0151 | +17.299 | 0.0000 | Y | Y |
| continuous_desired | +0.2992 | +11.113 | 0.0000 | Y | Y |

### Role metrics

| era | role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp_r | capital_normalized_ev_r | daily_total_r | daily_max_dd_r | daily_total_dollars | daily_max_dd_dollars |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | parent | 16014 | 1.000 | -0.1160 | -0.1160 | -0.1160 | -1857.91 | +1863.52 | -25,591 | +28,541 |
| IS | q45_filter | 6407 | 0.400 | -0.0137 | -0.0055 | -0.0137 | -87.65 | +193.67 | +5,768 | +6,560 |
| IS | continuous_desired | 16014 | 1.000 | -0.1160 | -0.0834 | -0.0834 | -1336.10 | +1337.04 | -13,926 | +21,051 |
| OOS | parent | 702 | 1.000 | -0.0902 | -0.0902 | -0.0902 | -63.34 | +62.75 | -1,349 | +2,031 |
| OOS | q45_filter | 296 | 0.422 | -0.0112 | -0.0047 | -0.0112 | -3.30 | +22.93 | -57 | +1,641 |
| OOS | continuous_desired | 702 | 1.000 | -0.0902 | -0.0600 | -0.0583 | -42.10 | +47.06 | -801 | +2,571 |

### OOS direction match

| candidate | IS daily delta sign | OOS daily delta sign | OOS mean_daily_delta_r | direction_match |
|---|---:|---:|---:|:---:|
| q45_filter | + | + | +0.8338 | Y |
| continuous_desired | + | + | +0.2950 | Y |

### Executable drag and concurrency diagnostics

- Continuous desired -> executable proxy policy EV drag: IS `+0.0212R/opp`, OOS `+0.0280R/opp`
- Continuous executable proxy total dollars: IS `$-7,443`, OOS `$-64`
- Q4+Q5 max open contracts/lots in IS: 3 contracts / 1 lots
- Continuous executable max open contracts/lots in IS: 6 contracts / 1 lots

## MGC

### IS daily delta tests

| candidate | mean_daily_delta_r | t | p_two_tailed | bh_survives | direction_positive |
|---|---:|---:|---:|:---:|:---:|
| q45_filter | +0.9562 | +13.074 | 0.0000 | Y | Y |
| continuous_desired | +0.2416 | +7.197 | 0.0000 | Y | Y |

### Role metrics

| era | role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp_r | capital_normalized_ev_r | daily_total_r | daily_max_dd_r | daily_total_dollars | daily_max_dd_dollars |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | parent | 7444 | 1.000 | -0.1346 | -0.1346 | -0.1346 | -1001.61 | +1028.37 | -17,739 | +22,483 |
| IS | q45_filter | 2974 | 0.400 | -0.0374 | -0.0150 | -0.0375 | -111.38 | +209.43 | +3,903 | +4,496 |
| IS | continuous_desired | 7444 | 1.000 | -0.1346 | -0.1043 | -0.1044 | -776.71 | +840.49 | -9,509 | +18,655 |
| OOS | parent | 601 | 1.000 | +0.0695 | +0.0695 | +0.0695 | +41.80 | +18.40 | +13,494 | +1,984 |
| OOS | q45_filter | 222 | 0.369 | +0.1941 | +0.0717 | +0.1941 | +43.09 | +11.14 | +8,824 | +1,650 |
| OOS | continuous_desired | 601 | 1.000 | +0.0695 | +0.1013 | +0.1021 | +60.88 | +19.42 | +15,937 | +2,366 |

### OOS direction match

| candidate | IS daily delta sign | OOS daily delta sign | OOS mean_daily_delta_r | direction_match |
|---|---:|---:|---:|:---:|
| q45_filter | + | + | +0.0182 | Y |
| continuous_desired | + | + | +0.2688 | Y |

### Executable drag and concurrency diagnostics

- Continuous desired -> executable proxy policy EV drag: IS `+0.0190R/opp`, OOS `+0.0139R/opp`
- Continuous executable proxy total dollars: IS `$-3,505`, OOS `$+17,122`
- Q4+Q5 max open contracts/lots in IS: 3 contracts / 1 lots
- Continuous executable max open contracts/lots in IS: 6 contracts / 1 lots

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
