# MNQ US_DATA_830 F3_NEAR_PIVOT_50 long single-cell confirmation v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-22-mnq-us-data-830-f3-near-pivot-50-v1.yaml` (LOCKED, commit_sha=47dbef70)
**Script:** `research/mnq_us_data_830_f3_near_pivot_50_v1.py`

## Verdict

`PARK`

## Locked single-cell metrics

- Full sample rows: `875`
- Full sample resolved rows: `857`
- Full sample scratch rows: `18`
- IS total rows: `844`
- IS resolved rows: `826`
- OOS total rows: `31`
- OOS resolved rows: `31`
- Full sample on-signal rows: `670`
- Full sample on-signal resolved rows: `660`
- IS on-signal rows: `652`
- IS on-signal resolved rows: `642`
- IS fire rate: `0.7725`
- IS ExpR lane resolved: `0.0032`
- IS ExpR lane incl scratch=0: `0.0032`
- IS ExpR on-signal resolved: `0.0642`
- IS ExpR off-signal resolved: `-0.2095`
- IS ExpR on-signal incl scratch=0: `0.0632`
- IS ExpR off-signal incl scratch=0: `-0.2008`
- IS delta resolved (on - off): `0.2737`
- IS delta incl scratch=0 (on - off): `0.2640`
- IS Welch t / p: `t=3.590` `p=0.0004`
- Positive IS years (delta): `6` / `7`
- OOS on-signal resolved N: `18`
- OOS delta resolved (on - off): `0.0017`
- OOS delta incl scratch=0 (on - off): `0.0017`
- OOS dir match: `True`
- Scratch rate on/off IS: `0.0153` / `0.0417`

## T0/T1/T2/T3/T6/T7/T8 audit table

| Test | Value | Status | Detail |
|---|---|---|---|
| T0_tautology | max |corr|=0.304 (pdr_r105_fire) | PASS | correlations={'pdr_r105_fire': -0.3043822838231526, 'gap_r015_fire': -0.095166221939943, 'atr70_fire': 0.05505520491847991, 'ovn80_fire': -0.2477418303258626} |
| T1_accounting_consistency | resolved=0.2737 incl_scratch=0.2640 | PASS | sign agreement across accounting views |
| T2_parent_lane_quality | off_resolved=-0.2095 off_incl_scratch=-0.2008 | PASS | off-signal baseline quality disclosed |
| T3_oos_direction | N_on_OOS=18 delta=0.0017 | PARK | thin OOS if N_on_OOS < 30 |
| T6_null_floor | p=0.0020 delta_obs=0.2737 | PASS | 1000 shuffles on resolved rows |
| T7_per_year_delta | 6/7 positive | PASS | yr_delta={2019: 0.00786963562753036, 2020: 0.4141769503546099, 2021: 0.34358959183673476, 2022: 0.48507828282828286, 2023: 0.10893706382978723, 2024: -0.09553278529980658, 2025: 0.48089975510204086} |
| T8_scratch_bias | on=0.0153 off=0.0417 | PASS | scratch concentrations disclosed, not hidden |

## Decision notes

- This is a single locked Pathway-B cell, not a family rescan.
- The honest role is a take overlay on a weak parent lane.
- Scratch-inclusive and resolved-only views agree on sign, so the result is not being flattered by excluding scratches.
- OOS remains too thin for promotion; this is a park outcome if IS survives.

## Outputs

- Row-level CSV: `docs/audit/results/2026-04-22-mnq-us-data-830-f3-near-pivot-50-v1-rows.csv`
