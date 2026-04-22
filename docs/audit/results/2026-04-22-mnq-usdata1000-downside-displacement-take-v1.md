# 2026-04-22 MNQ US_DATA_1000 Downside-Displacement Take v1

Exact Phase-4 bridge result for:

- `MNQ`
- `US_DATA_1000`
- `O5`
- `E2`
- `RR1.0`
- `CB1`
- `PD_DISPLACE_LONG`
- role: `TAKE`

Hypothesis file:

- `docs/audit/hypotheses/2026-04-22-mnq-usdata1000-downside-displacement-take-v1.yaml`
- SHA: `7e3c4f5cdd8977e0ae458e2cf5540b611ed3a7656d641cc2966d813974154e4b`

## Family-board grounding

This candidate came from the bounded read-only family board:

- `docs/audit/results/2026-04-22-mnq-prior-day-family-board-v1.md`

Top family-board row:

- lane: `MNQ US_DATA_1000 O5 E2 RR1.0 long`
- family: `TAKE_DOWNSIDE_DISPLACEMENT`
- members: `F2_NEAR_PDL_15 OR F5_BELOW_PDL`
- `delta_IS=+0.2491`
- `delta_OOS=+0.3188`
- `BH=0.0040`

## Cheap precheck

Read-only Phase 4 precheck:

- `accepted_raw_trials=1`
- `experimental_rows_with_sha=0` before write
- `validated_rows_from_sha=0` before write
- exact accepted combo:
  - `orb_label=US_DATA_1000`
  - `filter_type=PD_DISPLACE_LONG`
  - `entry_model=E2`
  - `rr_target=1.0`
  - `confirm_bars=1`
  - `stop_multiplier=1.0`
  - `n_preholdout_outcomes=195`

## Discovery write

Discovery was run on canonical `daily_features + orb_outcomes` with `holdout_date=2026-01-01`.

Final raw acceptance:

- `Phase 4 safety net: 1/1 raw trials accepted by scope predicate`

Written experimental row:

- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG`
- `sample_size`: `192`
- `entry_signals`: `195`
- `expectancy_r`: `+0.2396`
- `win_rate`: `0.6458`
- `p_value`: `0.000314`
- `fdr_adjusted_p_discovery`: `0.000314`
- `is_canonical`: `True`

## Validator outcome

Validator run:

- `python -m trading_app.strategy_validator --instrument MNQ`

Observed path:

- Criterion 8 permissive pass-through: `N_oos=10 < 30`
- Walk-forward: `PASS`
  - `5/7` windows positive
  - `agg_ExpR=+0.1922`
  - `WFE=0.7552`
- Stratified session FDR:
  - `US_DATA_1000 K=39`
  - row survived

Final promoted row in `validated_setups`:

- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG`
- `status`: `active`
- `promoted_from`: `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG`
- `wfe`: `0.7552`
- `oos_exp_r`: `+0.1922`
- `validation_pathway`: `family`
- `wf_passed`: `True`
- `wf_windows`: `7`

Informational only:

- `dsr_score=0.8933` (below the 0.95 cross-check threshold, but DSR remains cross-check only per current doctrine)

## Verdict

This broader family encoding survived where the narrower exact-cell bridge candidates failed.

Why it matters:

- it is broader than the exact-cell `F5_BELOW_PDL` and `F3_NEAR_PIVOT_50` snipes
- it preserved strong canonical read-only economics
- it cleared the actual discovery + validator bridge into `validated_setups`

Current truth:

- **broader MNQ prior-day downside-displacement family = live-adjacent survivor**
- **narrow exact-cell variants in the same neighborhood remain more fragile**
