# 2026-04-22 MNQ US_DATA_1000 Near-Pivot-50 Avoid v1

Exact Phase-4 bridge result for:

- `MNQ`
- `US_DATA_1000`
- `O5`
- `E2`
- `RR1.0`
- `CB1`
- `F3_NEAR_PIVOT_50`
- role: `AVOID`

Hypothesis file:

- `docs/audit/hypotheses/2026-04-22-mnq-usdata1000-near-pivot-50-avoid-v1.yaml`
- SHA: `9a754e6fd2a702a0167c94495dd152a8262a4c293c595b5c94955cfa9f5e7d0d`

## Discovery Write

Discovery was run on canonical `daily_features + orb_outcomes` with `holdout_date=2026-01-01`.

Final raw acceptance:

- `Phase 4 safety net: 1/1 raw trials accepted by scope predicate`

Written experimental row:

- `strategy_id`: `MNQ_US_DATA_1000_E2_RR1.0_CB1_F3_NEAR_PIVOT_50`
- `sample_size`: `563`
- `entry_signals`: `566`
- `expectancy_r`: `-0.0459`
- `win_rate`: `0.5009`
- `p_value`: `0.253982`
- `fdr_adjusted_p_discovery`: `0.253982`
- `is_canonical`: `True`

Interpretation:

- This is an exact negative on-state on the parent lane, consistent with the read-only candidate-board framing.
- The row is research-real enough to write into `experimental_strategies`.
- It is not statistically strong on its own as a standalone discovered strategy, which is expected for an `AVOID` overlay role.

## Validator Outcome

Validator run:

- `python -m trading_app.strategy_validator --instrument MNQ`

Result:

- `REJECTED`
- `validation_notes`: `criterion_9: era 2020-2022 ExpR=-0.0645 < -0.05 (N=284)`

Promotion status:

- `validated_setups`: no row written
- exact bridge outcome: `NO PROMOTION`

## Operational Note

During this bridge, direct-file execution of `trading_app/strategy_discovery.py` from an isolated worktree was found to be import-path sensitive. The branch now anchors the repo root onto `sys.path` so direct-file execution and `python -m trading_app.strategy_discovery` resolve against the active checkout and produce the same Phase-4 result.
