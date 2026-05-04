# MGC US_DATA_830 size context on MNQ NYSE_OPEN quality v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-22-mgc-us-data-830-to-mnq-nyse-open-context-v1.yaml` (LOCKED, commit_sha=264bbda6)
**Script:** `research/mgc_us_data_830_to_mnq_nyse_open_context_v1.py`

## Verdict

`KILL`

## Locked single-cell metrics

- Threshold (`source_orb_size_norm` Q3 on IS): `0.214843`
- Sample window: `2022-06-14` -> `2026-04-16`
- Full sample rows: `974`
- IS total rows: `903`
- OOS total rows: `66`
- Baseline target-lane IS ExpR: `0.1308`
- Baseline target-lane OOS ExpR: `0.1327`
- Full sample on-signal rows: `231`
- IS on-signal rows: `226`
- IS fire rate: `0.2503`
- IS ExpR on-signal: `0.1603`
- IS WR on-signal: `0.5973`
- IS delta (on - off): `0.0393`
- IS Welch t / p: `t=0.536` `p=0.5922`
- IS on-signal one-sample t: `2.524`
- Positive IS years: `4` / `4`
- OOS on-signal N: `5`
- OOS delta (on - off): `0.4784`
- OOS dir match: `True`
- OOS/IS ratio: `3.5856`

## Chronology audit

- All checked rows safe: `True`
- Trading days checked: `974`
- Minimum source-end to target-start gap (minutes): `55.0`
- Sample UTC windows:
  - `2022-06-14: source=[2022-06-14 12:30:00+00:00, 2022-06-14 12:35:00+00:00) target=[2022-06-14 13:30:00+00:00, 2022-06-14 13:35:00+00:00) gap_min=55.0`
  - `2024-05-15: source=[2024-05-15 12:30:00+00:00, 2024-05-15 12:35:00+00:00) target=[2024-05-15 13:30:00+00:00, 2024-05-15 13:35:00+00:00) gap_min=55.0`
  - `2026-04-16: source=[2026-04-16 12:30:00+00:00, 2026-04-16 12:35:00+00:00) target=[2026-04-16 13:30:00+00:00, 2026-04-16 13:35:00+00:00) gap_min=55.0`

## T0/T1/T2/T3/T6/T7 audit table

| Test | Value | Status | Detail |
|---|---|---|---|
| T0_tautology | max |corr|=0.144 (pdr_r105_fire) | PASS | correlations={'pdr_r105_fire': 0.14413909751490397, 'gap_r015_fire': -0.047103164125648404, 'atr70_fire': -0.10618702410920362, 'ovn80_fire': -0.0023845752080634544} |
| T1_wr_monotonicity | WR_spread=0.018 (on=0.597 off=0.579) | INFO | ExpR_spread=0.039 |
| T2_is_baseline | N=226 ExpR=0.160 WR=0.597 | PASS | deployable N gate |
| T3_oos_wfe | N_OOS=5 | FAIL | insufficient OOS N for WFE (< 10) |
| T6_null_floor | p=0.2987 ExpR_obs=0.160 | FAIL | 1000 shuffles |
| T7_per_year | 4/4 in expected direction | PASS | yr={2019: nan, 2020: nan, 2021: nan, 2022: 0.3151186046511628, 2023: 0.09196823529411763, 2024: 0.17734848484848484, 2025: 0.098609375} |

## Decision notes

- This is a single locked cross-asset chronology path, not a family sweep.
- The source feature is trade-time-knowable and fully resolved before the target lane opens.
- The usable sample starts in 2022 because this path requires same-day MGC source rows; this is a source-availability restriction, not a target-lane filter choice.
- If this path fails, it fails on effect size under the frozen prereg gates, not on chronology.

## Outputs

- Row-level CSV: `docs/audit/results/2026-04-22-mgc-us-data-830-to-mnq-nyse-open-context-v1-rows.csv`
