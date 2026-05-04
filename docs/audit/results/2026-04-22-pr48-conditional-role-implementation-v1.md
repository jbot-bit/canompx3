# PR48 conditional-role implementation v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-22-pr48-conditional-role-implementation-v1.yaml`
**Pre-reg commit SHA:** `24a36006`
**Canonical layers:** `daily_features`, `orb_outcomes`
**Scope:** `MNQ, MES, MGC` x O5 x E2 x CB1 x RR1.5 with IS-frozen role rules.
**Sacred OOS window:** `2026-01-01` onward (monitor only; thin window).

**Latest canonical trading day in orb_outcomes:** `2026-04-16`

## MNQ

- Range: `2019-05-13` to `2026-04-16`; total N=18599, IS N=17828, OOS N=771, lanes=24

### IS

- Rank slope: N=17828, beta=+0.27775, t=+9.589
| role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp | avg_weight | capital_normalized_ev |
|---|---:|---:|---:|---:|---:|---:|
| parent | 17828 | 1.000 | +0.0365 | +0.0365 | 1.000 | +0.0365 |
| q45_filter | 7131 | 0.400 | +0.1196 | +0.0478 | 0.400 | +0.1196 |
| q5_filter | 3571 | 0.200 | +0.1613 | +0.0323 | 0.200 | +0.1613 |
| continuous_sizer | 17828 | 1.000 | +0.0365 | +0.0643 | 1.000 | +0.0644 |

- Quintiles: Q1: N=3576, avg=-0.0671, Q2: N=3562, avg=-0.0212, Q3: N=3559, avg=+0.0320, Q4: N=3560, avg=+0.0778, Q5: N=3571, avg=+0.1613
- Q5 minus Q1 mean spread: +0.2284R

### OOS

- Rank slope: N=771, beta=+0.14433, t=+0.964
| role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp | avg_weight | capital_normalized_ev |
|---|---:|---:|---:|---:|---:|---:|
| parent | 771 | 1.000 | +0.0589 | +0.0589 | 1.000 | +0.0589 |
| q45_filter | 320 | 0.415 | +0.0851 | +0.0353 | 0.415 | +0.0851 |
| q5_filter | 137 | 0.178 | +0.0096 | +0.0017 | 0.178 | +0.0096 |
| continuous_sizer | 771 | 1.000 | +0.0589 | +0.0652 | 1.015 | +0.0643 |

- Quintiles: Q1: N=135, avg=+0.0157, Q2: N=141, avg=+0.0353, Q3: N=175, avg=+0.0635, Q4: N=183, avg=+0.1417, Q5: N=137, avg=+0.0096
- Q5 minus Q1 mean spread: -0.0061R

## MES

- Range: `2019-05-13` to `2026-04-16`; total N=16716, IS N=16014, OOS N=702, lanes=22

### IS

- Rank slope: N=16014, beta=+0.33025, t=+11.802
| role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp | avg_weight | capital_normalized_ev |
|---|---:|---:|---:|---:|---:|---:|
| parent | 16014 | 1.000 | -0.1160 | -0.1160 | 1.000 | -0.1160 |
| q45_filter | 6407 | 0.400 | -0.0137 | -0.0055 | 0.400 | -0.0137 |
| q5_filter | 3207 | 0.200 | +0.0117 | +0.0023 | 0.200 | +0.0117 |
| continuous_sizer | 16014 | 1.000 | -0.1160 | -0.0834 | 1.000 | -0.0834 |

- Quintiles: Q1: N=3210, avg=-0.2566, Q2: N=3201, avg=-0.1532, Q3: N=3196, avg=-0.1428, Q4: N=3200, avg=-0.0391, Q5: N=3207, avg=+0.0117
- Q5 minus Q1 mean spread: +0.2682R

### OOS

- Rank slope: N=702, beta=+0.36543, t=+2.492
| role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp | avg_weight | capital_normalized_ev |
|---|---:|---:|---:|---:|---:|---:|
| parent | 702 | 1.000 | -0.0902 | -0.0902 | 1.000 | -0.0902 |
| q45_filter | 296 | 0.422 | -0.0112 | -0.0047 | 0.422 | -0.0112 |
| q5_filter | 155 | 0.221 | +0.1123 | +0.0248 | 0.221 | +0.1123 |
| continuous_sizer | 702 | 1.000 | -0.0902 | -0.0600 | 1.028 | -0.0583 |

- Quintiles: Q1: N=118, avg=-0.1987, Q2: N=136, avg=-0.1760, Q3: N=152, avg=-0.0832, Q4: N=141, avg=-0.1469, Q5: N=155, avg=+0.1123
- Q5 minus Q1 mean spread: +0.3111R

## MGC

- Range: `2022-06-20` to `2026-04-16`; total N=8045, IS N=7444, OOS N=601, lanes=18

### IS

- Rank slope: N=7444, beta=+0.29975, t=+7.541
| role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp | avg_weight | capital_normalized_ev |
|---|---:|---:|---:|---:|---:|---:|
| parent | 7444 | 1.000 | -0.1346 | -0.1346 | 1.000 | -0.1346 |
| q45_filter | 2974 | 0.400 | -0.0374 | -0.0150 | 0.400 | -0.0374 |
| q5_filter | 1493 | 0.201 | +0.0150 | +0.0030 | 0.201 | +0.0150 |
| continuous_sizer | 7444 | 1.000 | -0.1346 | -0.1043 | 1.000 | -0.1044 |

- Quintiles: Q1: N=1497, avg=-0.2296, Q2: N=1487, avg=-0.2025, Q3: N=1486, avg=-0.1651, Q4: N=1481, avg=-0.0903, Q5: N=1493, avg=+0.0150
- Q5 minus Q1 mean spread: +0.2446R

### OOS

- Rank slope: N=601, beta=+0.42276, t=+2.519
| role | selected_n | trade_share | selected_avg_r | policy_ev_per_opp | avg_weight | capital_normalized_ev |
|---|---:|---:|---:|---:|---:|---:|
| parent | 601 | 1.000 | +0.0695 | +0.0695 | 1.000 | +0.0695 |
| q45_filter | 222 | 0.369 | +0.1941 | +0.0717 | 0.369 | +0.1941 |
| q5_filter | 95 | 0.158 | +0.2626 | +0.0415 | 0.158 | +0.2626 |
| continuous_sizer | 601 | 1.000 | +0.0695 | +0.1013 | 0.992 | +0.1021 |

- Quintiles: Q1: N=96, avg=-0.0261, Q2: N=145, avg=-0.0225, Q3: N=138, avg=+0.0325, Q4: N=127, avg=+0.1428, Q5: N=95, avg=+0.2626
- Q5 minus Q1 mean spread: +0.2888R

## Interpretation guardrails

- `selected_avg_r` is not enough. Conditional roles are judged on `policy_ev_per_opp` first.
- `capital_normalized_ev` is reported for the continuous sizer so a weight map is not mistaken for a binary filter.
- OOS from 2026-01-01 to latest canonical day is monitoring only; use it for direction and implementation sanity, not retuning.

## Verdict

`CONTINUE` as role-aware evidence. The implementation study supports treating PR48 as a conditional-role question first, not forcing it into standalone-or-dead framing.

## Reproduction

- `python3 -m py_compile research/pr48_conditional_role_implementation_v1.py`
- `./.venv-wsl/bin/python research/pr48_conditional_role_implementation_v1.py`

## Caveats

- This study is descriptive role evidence, not a promotion gate by itself.
- The frozen quintile mapping is IS-defined and OOS-monitored only.
- Current repo runtime surfaces do not yet consume these role outputs natively.
