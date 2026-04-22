# MNQ US_DATA_1000 clear-of-congestion take v1

Date: 2026-04-22  
Branch: `wt-codex-mnq-hiroi-scan`  
Hypothesis: `docs/audit/hypotheses/2026-04-22-mnq-usdata1000-clear-of-congestion-take-v1.yaml`

## Question

Does the positive side of the bounded MNQ prior-day congestion family survive
the full bridge as a promotable take overlay on the live-adjacent parent lane
`MNQ US_DATA_1000 O5 E2 RR1.0 long`?

Exact family definition:

- `PD_CLEAR_LONG`
- long break direction only
- ORB midpoint is **not** strictly inside the prior-day range
- ORB midpoint is **not** within `0.50 ATR-20` of the prior-day pivot

This is the explicit take-side complement of the read-only
`AVOID_CONGESTION = F3_NEAR_PIVOT_50 OR F6_INSIDE_PDR` family from
`docs/audit/results/2026-04-22-mnq-prior-day-family-board-v1.md`.

## Grounding

- Canonical truth layers only: `orb_outcomes`, `daily_features`
- Holdout discipline: `2026-01-01`
- Mechanism grounding:
  - prior-day congestion / two-sided auction context is hostile to ORB
    continuation
  - the research claim here is the positive complement, not the negative state
    itself

## Read-only family grounding

From the bounded family board:

- `AVOID_CONGESTION` on `MNQ US_DATA_1000 O5 E2 RR1.0 long`
  - `N_on_IS=649`
  - `ExpR_on_IS=-0.0216`
  - `ExpR_off_IS=+0.2155`
  - `delta_IS=-0.2371`
  - `BH=0.0040`
  - `N_on_OOS=22`
  - `delta_OOS=-0.3967`
  - `same_sign_oos=True`

Explicit complement check before bridge:

- `PD_CLEAR_LONG` equivalent (`NOT congestion`)
  - `IS: N=238, ExpR=+0.2182, WinRate=0.6261, Total_R=+50.8514`
  - `OOS: N=13, ExpR=+0.1962, WinRate=0.6154, Total_R=+2.5504`

Overlap vs already-promoted `PD_DISPLACE_LONG` on the same lane:

- total parent IS rows: `881`
- `PD_DISPLACE_LONG`: `205`
- congestion family: `649`
- intersection: `118`
- Jaccard: `0.1603`
- `pd_only`: `87`, `ExpR=+0.3927`
- `congest_only`: `531`, `ExpR=-0.0516`

This is not just the same state under a new label.

## Cheap gate

`research/phase4_candidate_precheck.py`:

- `hypothesis_sha=e93af94608b68dd0433f20cad66973cdca6e1cd571811bbebb0226464f2ee875`
- `accepted_raw_trials=1`
- `experimental_rows_with_sha=0`
- `validated_rows_from_sha=0`
- exact combo:
  - `orb_label='US_DATA_1000'`
  - `filter_key='PD_CLEAR_LONG'`
  - `entry_model='E2'`
  - `rr_target=1.0`
  - `confirm_bars=1`
  - `stop_multiplier=1.0`
  - `n_preholdout_outcomes=216`

## Discovery write

`trading_app/strategy_discovery.py --instrument MNQ --orb-minutes 5 --holdout-date 2026-01-01 --hypothesis-file ...`

Result:

- strategy id: `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG`
- `sample_size=211`
- `entry_signals=216`
- `expectancy_r=+0.2270`
- `win_rate=0.6398`
- `p_value=0.000356`
- `fdr_adjusted_p_discovery=0.000356`
- `is_canonical=True`

## Validator outcome

`python -m trading_app.strategy_validator --instrument MNQ`

Outcome:

- Criterion 8: `N_oos=13 < 30` -> Pathway A permissive pass-through
- Walkforward: `6/7` windows positive
- aggregate OOS ExpR: `+0.2926`
- `WFE=1.4639`
- validator status: `PASSED`
- validation notes:
  - `Phase 3: 1 non-waivable negative year(s) (2020), within 75% threshold`

Promoted row in `validated_setups`:

- `strategy_id='MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG'`
- `status='active'`
- `validation_pathway='family'`
- `wf_passed=True`
- `wf_windows=7`
- `oos_exp_r=+0.2926`
- `dsr_score=0.8612828720460417` (informational only)

## Verdict

`PD_CLEAR_LONG` survives the full bridge and promotes.

This is the second broader MNQ prior-day family on the same live-adjacent
parent lane to survive full discovery + validator, and it does so more cleanly
than the earlier exact-cell avoid attempts.
