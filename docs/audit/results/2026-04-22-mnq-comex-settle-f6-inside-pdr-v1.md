# MNQ COMEX_SETTLE F6_INSIDE_PDR long single-cell confirmation v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-22-mnq-comex-settle-f6-inside-pdr-v1.yaml` (LOCKED, commit_sha=a9856dbd)
**Script:** `research/mnq_comex_settle_f6_inside_pdr_v1.py`

## Verdict

`PARK`

## Locked single-cell metrics

- Full sample rows: `911`
- Full sample resolved rows: `907`
- Full sample scratch rows: `4`
- IS total rows: `879`
- IS resolved rows: `876`
- OOS total rows: `32`
- OOS resolved rows: `31`
- Full sample on-signal rows: `454`
- Full sample on-signal resolved rows: `450`
- IS on-signal rows: `436`
- IS on-signal resolved rows: `433`
- IS fire rate: `0.4960`
- IS ExpR lane resolved: `0.0688`
- IS ExpR lane incl scratch=0: `0.0686`
- IS ExpR on-signal resolved: `-0.0296`
- IS ExpR off-signal resolved: `0.1651`
- IS ExpR on-signal incl scratch=0: `-0.0294`
- IS ExpR off-signal incl scratch=0: `0.1651`
- IS delta resolved (on - off): `-0.1947`
- IS delta incl scratch=0 (on - off): `-0.1945`
- IS Welch t / p: `t=-3.238` `p=0.0013`
- Negative IS years (delta): `6` / `7`
- OOS on-signal resolved N: `17`
- OOS delta resolved (on - off): `-0.3202`
- OOS delta incl scratch=0 (on - off): `-0.3143`
- OOS dir match: `True`
- Scratch rate on/off IS: `0.0069` / `0.0000`

## T0/T1/T2/T3/T6/T7/T8 audit table

| Test | Value | Status | Detail |
|---|---|---|---|
| T0_tautology | max |corr|=0.166 (pdr_r105_fire) | PASS | correlations={'pdr_r105_fire': 0.16552776776570918, 'gap_r015_fire': 0.06424798527006714, 'atr70_fire': 0.04657761466323256, 'ovn80_fire': -0.05177538032694389} |
| T1_accounting_consistency | resolved=-0.1947 incl_scratch=-0.1945 | PASS | sign agreement across accounting views |
| T2_parent_lane_quality | off_resolved=0.1651 off_incl_scratch=0.1651 | PASS | off-signal baseline remains positive |
| T3_oos_direction | N_on_OOS=17 delta=-0.3202 | PARK | thin OOS if N_on_OOS < 30 |
| T6_null_floor | p=0.0010 delta_obs=-0.1947 | PASS | 1000 shuffles on resolved rows |
| T7_per_year_delta | 6/7 negative | PASS | yr_delta={2019: -0.0513828282828283, 2020: -0.18216851570964246, 2021: -0.38813317972350225, 2022: -0.31152268115942033, 2023: -0.09548211600429642, 2024: -0.2698216347263911, 2025: 0.0014170547945205614} |
| T8_scratch_bias | on=0.0069 off=0.0000 | INFO | scratch concentrations disclosed, not hidden |

## Decision notes

- This is a single locked Pathway-B cell, not a family rescan.
- The honest role is an avoid overlay on a positive COMEX parent lane.
- Scratch-inclusive and resolved-only views agree on sign, so the result is not being flattered by excluding scratches.
- OOS remains too thin for promotion; this is a park outcome if IS survives.

## Outputs

- Row-level CSV: `docs/audit/results/2026-04-22-mnq-comex-settle-f6-inside-pdr-v1-rows.csv`
