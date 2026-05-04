# MNQ US_DATA_1000 F5_BELOW_PDL long single-cell confirmation v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-21-mnq-us-data-1000-f5-below-pdl-v1.yaml` (LOCKED, commit_sha=0cdd40f4)
**Script:** `research/mnq_us_data_1000_f5_below_pdl_v1.py`

## Verdict

`PARK`

## Locked single-cell metrics

- Full sample rows: `916`
- IS total rows: `881`
- OOS total rows: `32`
- Full sample on-signal rows: `144`
- IS on-signal rows: `136`
- IS fire rate: `0.1544`
- IS ExpR on-signal: `0.3258`
- IS WR on-signal: `0.6912`
- IS delta (on - off): `0.3370`
- IS Welch t / p: `t=4.018` `p=0.0001`
- IS on-signal one-sample t: `4.265`
- Positive IS years: `7` / `7`
- OOS on-signal N: `8`
- OOS delta (on - off): `0.0797`
- OOS dir match: `True`
- OOS/IS ratio: `-0.0745`
- Era min on-signal ExpR (N>=50): `NA`

## T0/T1/T2/T3/T6/T7/T8 audit table

| Test | Value | Status | Detail |
|---|---|---|---|
| T0_tautology | max |corr|=0.113 (pdr_r105_fire) | PASS | correlations={'pdr_r105_fire': -0.11326963530397866, 'gap_r015_fire': -0.05187963853431432, 'atr70_fire': -0.006307303289695921, 'ovn80_fire': 0.048738275892488386} |
| T1_wr_monotonicity | WR_spread=0.168 (on=0.691 off=0.523) | PASS | ExpR_spread=0.337 |
| T2_is_baseline | N=136 ExpR=0.326 WR=0.691 | PASS | deployable N gate |
| T3_oos_wfe | N_OOS=8 | FAIL | insufficient OOS N for WFE (< 10) |
| T6_null_floor | p=0.0010 ExpR_obs=0.326 | PASS | 1000 shuffles |
| T7_per_year | 7/7 in expected direction | PASS | yr={2019: 0.28408, 2020: 0.48053076923076926, 2021: 0.025253333333333336, 2022: 0.658917857142857, 2023: 0.011984615384615363, 2024: 0.4424, 2025: 0.3016375} |
| T8_cross_instrument | twin=MES Δ=0.205 | PASS | sign match, mag>=0.05 |

## Decision notes

- This is a single locked Pathway-B cell, not a family rescan.
- Thin OOS remains a park condition, not a rescue argument.
- Pre-reg provenance note: the locked `expected_n_total_is=914` field does not match current authoritative raw counts. Current repo state is `916` full rows, `881` IS rows, and `32` OOS rows; the locked `136` IS on-signal baseline still matches.

## Outputs

- Row-level CSV: `docs/audit/results/2026-04-21-mnq-us-data-1000-f5-below-pdl-v1-rows.csv`
