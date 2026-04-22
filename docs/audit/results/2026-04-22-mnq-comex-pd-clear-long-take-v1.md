# MNQ COMEX_SETTLE PD_CLEAR_LONG take v1

Date: 2026-04-22  
Branch: `wt-codex-mnq-hiroi-scan`  
Hypothesis: `docs/audit/hypotheses/2026-04-22-mnq-comex-pd-clear-long-take-v1.yaml`

## Question

Does the `PD_CLEAR_LONG` prior-day geometry family transfer cleanly to the
next unsolved MNQ parent lane:

- `MNQ COMEX_SETTLE O5 E2 RR1.0 long`

This is an exact transfer test of the already-promoted MNQ prior-day geometry
family, not a fresh feature search.

## Grounding

- Canonical truth layers only: `orb_outcomes`, `daily_features`
- Structural start: `2020-01-01`
- Holdout discipline: `2026-01-01`
- Mechanism:
  - prior-day congestion is hostile to ORB continuation
  - on this session, the read-only transfer board showed the positive side is
    specifically `PD_CLEAR_LONG`, not the broader `PD_GO_LONG`, because
    downside displacement is not positive here

## Read-only transfer-board grounding

`docs/audit/results/2026-04-22-mnq-geometry-transfer-board-v1.md`:

- `COMEX_SETTLE RR1.0 long PD_CLEAR_LONG`
  - `N_on_IS=303`
  - `ExpR_on_IS=+0.1841`
  - `ExpR_off_IS=+0.0137`
  - `delta_IS=+0.1704`
  - `BH=0.0243`
  - `N_on_OOS=15`
  - `delta_OOS=+0.2336`
  - `same_sign_oos=True`

Component split on the same lane:

- `PD_CLEAR_LONG`
  - positive both IS and OOS
- `PD_DISPLACE_LONG`
  - weak / negative
- conclusion:
  - this lane wants the bounded non-congestion side, not the broader union

## Cheap gate

`research/phase4_candidate_precheck.py`:

- `hypothesis_sha=560459105e707209e8f70c5c9afb1eb60a8f0dc1fe2ea59006eae0b12c90d3ea`
- `accepted_raw_trials=1`
- `experimental_rows_with_sha=0`
- `validated_rows_from_sha=0`
- exact combo:
  - `orb_label='COMEX_SETTLE'`
  - `filter_key='PD_CLEAR_LONG'`
  - `entry_model='E2'`
  - `rr_target=1.0`
  - `confirm_bars=1`
  - `stop_multiplier=1.0`
  - `n_preholdout_outcomes=303`

## Discovery write

`trading_app/strategy_discovery.py --instrument MNQ --orb-minutes 5 --holdout-date 2026-01-01 --hypothesis-file ...`

Result:

- strategy id: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG`
- `sample_size=303`
- `entry_signals=303`
- `expectancy_r=+0.1841`
- `win_rate=0.6469`
- `p_value=0.000267`
- `fdr_adjusted_p_discovery=0.000267`
- `is_canonical=True`

## Validator outcome

`python -m trading_app.strategy_validator --instrument MNQ`

Outcome:

- Criterion 8: `N_oos=15 < 30` -> Pathway A permissive pass-through
- Walkforward: `10/11` windows positive
- aggregate OOS ExpR: `+0.2212`
- `WFE=1.9858`
- validator status: `PASSED`

Promoted row in `validated_setups`:

- `strategy_id='MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG'`
- `status='active'`
- `validation_pathway='family'`
- `wf_passed=True`
- `wf_windows=11`
- `oos_exp_r=+0.2212`
- `dsr_score=0.7409016818636894` (informational only)

## Adversarial audit

Hard-truth checks on the exact long-only parent universe:

- parent surface:
  - `IS: N=787, ExpR=+0.0793`
  - `OOS: N=34, ExpR=+0.0016`
- `PD_CLEAR_LONG`:
  - `IS: N_on=303, ExpR_on=+0.1841`
  - `IS off-side: N_off=484, ExpR_off=+0.0137`
  - `delta_IS=+0.1704`
  - `OOS: N_on=15, ExpR_on=+0.1321`
  - `OOS off-side: N_off=19, ExpR_off=-0.1015`
  - `delta_OOS=+0.2336`

This is therefore:

- a positive take/allocator overlay on a still-positive parent lane in-sample
- not a “parent is dead, filter saves it” story
- not a universal or always-on session truth

Year-by-year delta on the exact long-only parent lane:

- `2020: -0.0526`
- `2021: +0.4271`
- `2022: +0.1861`
- `2023: +0.1895`
- `2024: +0.3386`
- `2025: -0.0581`
- `2026 OOS: +0.2336`

So the family is:

- clearly not just one hot year
- but also not universal across every year
- vulnerable to at least two reverse years so far: `2020` and `2025`

ATR-20 quartile deltas on pre-2026 long-only parent:

- `Q1: +0.1321`
- `Q2: +0.2267`
- `Q3: +0.2791`
- `Q4: +0.1112`

This does **not** look like a pure “only when MNQ is hot” proxy.

Family-component check on the same lane:

- `both`: `N_IS=121`, `ExpR=+0.1651`
- `clear_only`: `N_IS=182`, `ExpR=+0.1967`
- `displace_only`: `N_IS=87`, `ExpR=-0.0210`
- `residual`: `N_IS=397`, `ExpR=+0.0213`

OOS component check:

- `both`: `N=6`, `ExpR=-0.0372`
- `clear_only`: `N=9`, `ExpR=+0.2450`
- `displace_only`: `N=2`, `ExpR=-0.0472`
- `residual`: `N=17`, `ExpR=-0.1079`

That is the strongest evidence that the selected family member is the right one:

- `PD_CLEAR_LONG` is the positive transfer state
- `PD_DISPLACE_LONG` is not
- blindly transferring the broader `PD_GO_LONG` union would have been weaker

Implementation audit:

- no lookahead defect found in `PrevDayGeometryFilter`
- no discovery/validator mismatch found for this family
- one tooling issue **was** found and fixed before acceptance:
  - `research/mnq_geometry_transfer_board_v1.py` now derives masks from
    canonical `ALL_FILTERS` instead of re-encoding the logic by hand

## Verdict

`PD_CLEAR_LONG` survives the full bridge on `MNQ COMEX_SETTLE O5 E2 RR1.0 long`
and promotes.

This is the first successful parent-lane transfer after the `US_DATA_1000`
geometry family work and confirms the correct non-tunnel pattern:

- reuse the same mechanism class
- do not assume the same family shape transfers everywhere
- let the transfer board pick the exact family member the new lane actually wants

Hard truth:

- this is a real transfer
- it is **not** a universal regime
- it carries thin-OOS risk and two negative IS years
- it should be treated as a session-specific validated family, not as proof that
  the whole MNQ book has been solved
