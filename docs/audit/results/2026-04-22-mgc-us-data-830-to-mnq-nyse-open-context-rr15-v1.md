# MGC US_DATA_830 size context on MNQ NYSE_OPEN quality RR1.5 v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-22-mgc-us-data-830-to-mnq-nyse-open-context-rr15-v1.yaml` (LOCKED, commit_sha=c88adc0a)
**Script:** `research/mgc_us_data_830_to_mnq_nyse_open_context_rr15_v1.py`

## Verdict

`KILL`

## Locked single-cell metrics

- Threshold (`source_orb_size_norm` Q3 on IS): `0.216849`
- Sample window: `2022-06-14` -> `2026-04-16`
- Full sample rows: `951`
- IS total rows: `883`
- OOS total rows: `68`
- Baseline target-lane IS ExpR: `0.1432`
- Baseline target-lane OOS ExpR: `0.0846`
- Full sample on-signal rows: `226`
- IS on-signal rows: `221`
- IS fire rate: `0.2503`
- IS ExpR on-signal: `0.2517`
- IS WR on-signal: `0.5158`
- IS delta (on - off): `0.1447`
- IS Welch t / p: `t=1.535` `p=0.1257`
- IS on-signal one-sample t: `3.078`
- Positive IS years: `4` / `4`
- OOS on-signal N: `5`
- OOS delta (on - off): `0.4258`
- OOS dir match: `True`
- OOS/IS ratio: `1.9034`

## Chronology audit

- All checked rows safe: `True`
- Trading days checked: `951`
- Minimum source-end to target-start gap (minutes): `55.0`
- Sample UTC windows:
  - `2022-06-14: source=[2022-06-14 12:30:00+00:00, 2022-06-14 12:35:00+00:00) target=[2022-06-14 13:30:00+00:00, 2022-06-14 13:35:00+00:00) gap_min=55.0`
  - `2024-05-07: source=[2024-05-07 12:30:00+00:00, 2024-05-07 12:35:00+00:00) target=[2024-05-07 13:30:00+00:00, 2024-05-07 13:35:00+00:00) gap_min=55.0`
  - `2026-04-16: source=[2026-04-16 12:30:00+00:00, 2026-04-16 12:35:00+00:00) target=[2026-04-16 13:30:00+00:00, 2026-04-16 13:35:00+00:00) gap_min=55.0`

## T0/T1/T2/T3/T6/T7 audit table

| Test | Value | Status | Detail |
|---|---|---|---|
| T0_tautology | max |corr|=0.149 (pdr_r105_fire) | PASS | correlations={'pdr_r105_fire': 0.14855041040384867, 'gap_r015_fire': -0.039906977275811516, 'atr70_fire': -0.10191039200437772, 'ovn80_fire': -0.015122456615121613} |
| T1_wr_monotonicity | WR_spread=0.058 (on=0.516 off=0.458) | PASS | ExpR_spread=0.145 |
| T2_is_baseline | N=221 ExpR=0.252 WR=0.516 | PASS | deployable N gate |
| T3_oos_wfe | N_OOS=5 | FAIL | insufficient OOS N for WFE (< 10) |
| T6_null_floor | p=0.0599 ExpR_obs=0.252 | FAIL | 1000 shuffles |
| T7_per_year | 4/4 in expected direction | PASS | yr={2019: nan, 2020: nan, 2021: nan, 2022: 0.4002624999999999, 2023: 0.25031647058823525, 2024: 0.23240615384615385, 2025: 0.10427741935483875} |

## Decision notes

- This is a single locked cross-asset chronology path, not a family sweep.
- The source feature is trade-time-knowable and fully resolved before the target lane opens.
- The usable sample starts in 2022 because this path requires same-day MGC source rows; this is a source-availability restriction, not a target-lane filter choice.
- If this path fails, it fails on effect size under the frozen prereg gates, not on chronology.

## Outputs

- Row-level CSV: `docs/audit/results/2026-04-22-mgc-us-data-830-to-mnq-nyse-open-context-rr15-v1-rows.csv`
