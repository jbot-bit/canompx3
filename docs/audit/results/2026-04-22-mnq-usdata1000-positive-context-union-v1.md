# MNQ US_DATA_1000 positive-context union v1

Date: 2026-04-22  
Branch: `wt-codex-mnq-hiroi-scan`  
Hypothesis: `docs/audit/hypotheses/2026-04-22-mnq-usdata1000-positive-context-union-v1.yaml`

## Question

Should the two already-validated positive prior-day geometry families on
`MNQ US_DATA_1000 O5 E2 RR1.0 long` be collapsed into one broader positive
regime instead of staying as separate overlapping filters?

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
- This is not a fresh broad scan. It is the bounded union of two already
  validated positive family encodings:
  - `PD_DISPLACE_LONG`
  - `PD_CLEAR_LONG`

## Read-only partition grounding

Parent lane partition before bridge:

- `both`
  - `IS: N=89, ExpR=+0.3927`
  - `OOS: N=8, ExpR=-0.0243`
- `pd_clear_only`
  - `IS: N=148, ExpR=+0.1092`
  - `OOS: N=5, ExpR=+0.5489`
- `pd_displace_only`
  - `IS: N=119, ExpR=+0.1135`
  - `OOS: N=3, ExpR=+0.9697`
- `residual`
  - `IS: N=537, ExpR=-0.0499`
  - `OOS: N=20, ExpR=-0.3175`

Union check:

- `union_on`
  - `IS: N=356, ExpR=+0.1811, WinRate=0.6096, Total_R=+63.3928`
  - `OOS: N=16, ExpR=+0.2993, WinRate=0.6250, Total_R=+4.4898`
- `union_off`
  - `IS: N=537, ExpR=-0.0499`
  - `OOS: N=20, ExpR=-0.3175`

The union isolates the positive part of the parent lane; the excluded residual
state is negative in both IS and OOS.

## Cheap gate

`research/phase4_candidate_precheck.py`:

- `hypothesis_sha=87bb3d73f3cd1fb7f89eb47bdd0ea2c7056abc4063653f35f7d1e13aaf8b5d85`
- `accepted_raw_trials=1`
- `experimental_rows_with_sha=0`
- `validated_rows_from_sha=0`
- exact combo:
  - `orb_label='US_DATA_1000'`
  - `filter_key='PD_GO_LONG'`
  - `entry_model='E2'`
  - `rr_target=1.0`
  - `confirm_bars=1`
  - `stop_multiplier=1.0`
  - `n_preholdout_outcomes=330`

## Discovery write

`trading_app/strategy_discovery.py --instrument MNQ --orb-minutes 5 --holdout-date 2026-01-01 --hypothesis-file ...`

Result:

- strategy id: `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG`
- `sample_size=324`
- `entry_signals=330`
- `expectancy_r=+0.1934`
- `win_rate=0.6235`
- `p_value=0.000181`
- `fdr_adjusted_p_discovery=0.000181`
- `is_canonical=True`

## Validator outcome

`python -m trading_app.strategy_validator --instrument MNQ`

Outcome:

- Criterion 8: `N_oos=15 < 30` -> Pathway A permissive pass-through
- Walkforward: `9/11` windows positive
- aggregate OOS ExpR: `+0.2176`
- `WFE=1.2609`
- validator status: `PASSED`
- validation notes:
  - `Phase 3: 1 non-waivable negative year(s) (2023), within 75% threshold`

Promoted row in `validated_setups`:

- `strategy_id='MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG'`
- `status='active'`
- `validation_pathway='family'`
- `wf_passed=True`
- `wf_windows=11`
- `oos_exp_r=+0.2176`
- `dsr_score=0.7548853893624123` (informational only)

## Verdict

`PD_GO_LONG` survives the full bridge and promotes.

This closes the local partition on the live-adjacent `MNQ US_DATA_1000 O5 E2
RR1.0 long` parent lane more cleanly than the earlier exact-cell work:

- two broader positive families survive individually
- their bounded union also survives
- the excluded residual state is negative in both IS and OOS

The next honest move is likely another MNQ parent lane, not another narrower
cut on this exact lane.
