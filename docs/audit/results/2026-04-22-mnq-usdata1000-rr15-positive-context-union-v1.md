# MNQ US_DATA_1000 RR1.5 positive-context union v1

Date: 2026-04-22  
Branch: `wt-codex-mnq-hiroi-scan`  
Hypothesis: `docs/audit/hypotheses/2026-04-22-mnq-usdata1000-rr15-positive-context-union-v1.yaml`

## Question

Does the already-promoted `PD_GO_LONG` prior-day geometry union also survive on
the adjacent `MNQ US_DATA_1000 O5 E2 RR1.5 long` parent lane?

Exact family definition:

- `PD_GO_LONG`
- long break direction only
- ORB midpoint is in downside displacement context:
  - `orb_mid < prev_day_low`
  - OR `|orb_mid - prev_day_low| / atr_20 < 0.15`
- OR ORB midpoint is outside the prior-day congestion regime:
  - NOT strictly inside prior-day range
  - NOT within `0.50 ATR-20` of prior-day pivot

## Grounding

- Canonical truth layers only: `orb_outcomes`, `daily_features`
- Holdout discipline: `2026-01-01`
- This is not a fresh feature search. It is the exact transfer test of the
  already-promoted `PD_GO_LONG` family from the adjacent RR1.0 parent lane to
  RR1.5 on the same session and direction.

## Read-only grounding

Pre-bridge union check on `MNQ US_DATA_1000 O5 E2 RR1.5 long`:

- `union_on`
  - `IS: N=356, ExpR=+0.2078, WinRate=0.4944, Total_R=+72.1118`
  - `OOS: N=16, ExpR=+0.2964, WinRate=0.5000, Total_R=+4.4467`
- `union_off`
  - `IS: N=537, ExpR=-0.0352`
  - `OOS: N=20, ExpR=-0.2699`

The union remains the positive side of the lane; the excluded residual is
negative in both IS and OOS.

## Cheap gate

`research/phase4_candidate_precheck.py`:

- `hypothesis_sha=d5d3df102fe19a15391cfcaf3b4e2a60691d76a15b909afe0d3399d3e3e0b21d`
- `accepted_raw_trials=1`
- `experimental_rows_with_sha=0`
- `validated_rows_from_sha=0`
- exact combo:
  - `orb_label='US_DATA_1000'`
  - `filter_key='PD_GO_LONG'`
  - `entry_model='E2'`
  - `rr_target=1.5`
  - `confirm_bars=1`
  - `stop_multiplier=1.0`
  - `n_preholdout_outcomes=330`

## Discovery write

`trading_app/strategy_discovery.py --instrument MNQ --orb-minutes 5 --holdout-date 2026-01-01 --hypothesis-file ...`

Result:

- strategy id: `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG`
- `sample_size=321`
- `entry_signals=330`
- `expectancy_r=+0.2222`
- `win_rate=0.5109`
- `p_value=0.000896`
- `fdr_adjusted_p_discovery=0.000896`
- `is_canonical=True`

## Validator outcome

`python -m trading_app.strategy_validator --instrument MNQ`

Outcome:

- Criterion 8: `N_oos=15 < 30` -> Pathway A permissive pass-through
- Walkforward: `9/11` windows positive
- aggregate OOS ExpR: `+0.2553`
- `WFE=1.3983`
- validator status: `PASSED`
- validation notes:
  - `Phase 3: 1 non-waivable negative year(s) (2025), within 75% threshold`

Promoted row in `validated_setups`:

- `strategy_id='MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG'`
- `status='active'`
- `validation_pathway='family'`
- `wf_passed=True`
- `wf_windows=11`
- `oos_exp_r=+0.2553`
- `dsr_score=0.6223150344377846` (informational only)

## Verdict

`PD_GO_LONG` also survives on the adjacent RR1.5 parent lane and promotes.

This upgrades the conclusion from “one good lane” to “same mechanism family
works on both `US_DATA_1000 RR1.0` and `US_DATA_1000 RR1.5` long parents under
the exact same pre-trade geometry union.”
