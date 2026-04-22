# MNQ TOKYO_OPEN COST_LT08 take v1

Date: 2026-04-22  
Branch: `wt-codex-mnq-hiroi-scan`  
Hypothesis: `docs/audit/hypotheses/2026-04-22-mnq-tokyo-costlt08-take-v1.yaml`

## Question

Does the tighter cost-ratio threshold survive on:

- `MNQ TOKYO_OPEN O5 E2 RR1.5 long`

This is **not** a new feature-class discovery. It is an exact bridge on the
strongest unsolved row inside the next alive MNQ mechanism class after the
prior-day geometry work.

## Grounding

- Canonical truth layers only: `orb_outcomes`, `daily_features`
- Structural start: `2020-01-01`
- Holdout discipline: `2026-01-01`
- Comparator:
  - exact on-state vs exact off-side on the `MNQ TOKYO_OPEN RR1.5 long` parent
    lane
- mechanism:
  - lower estimated friction relative to ORB risk should improve realized
    expectancy on a momentum-friendly opening-break parent lane

## Why this row

The broader mechanism-class board showed the next alive non-geometry class was
`cost_ratio`, with the strongest exact unsolved row:

- `MNQ TOKYO_OPEN RR1.5 COST_LT08`

This was selected only after checking that:

- it was not already in `validated_setups`
- it was stronger than the looser active threshold family members on the same
  lane
- the lane-level shape remained broad enough under year and ATR splits

Existing validated overlap on the same lane:

- `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12`
  - already active

So the question here is whether the tighter threshold survives as a better
exact row, not whether cost-ratio exists at all.

## Read-only grounding

Exact long-only parent lane:

- `IS parent`
  - `N=1551`
  - `ExpR=+0.0831`
- `OOS parent`
  - `N=72`
  - `ExpR=+0.1489`

`COST_LT08`:

- `IS`
  - `N_on=427`
  - `ExpR_on=+0.2037`
  - `ExpR_off=+0.0374`
  - `delta_IS=+0.1663`
- `OOS`
  - `N_on=60`
  - `ExpR_on=+0.2692`
  - `ExpR_off=-0.4525`
  - `delta_OOS=+0.7217`

Year-by-year deltas:

- `2020: +0.1809`
- `2021: +0.0911`
- `2022: +0.2426`
- `2023: -0.2892`
- `2024: +0.2771`
- `2025: +0.2111`
- `2026 OOS: +0.7217`

ATR-20 quartile deltas on pre-2026:

- `Q1: +0.2008`
- `Q2: +0.1037`
- `Q3: +0.3566`
- `Q4: +0.0796`

Comparison to active looser threshold family on the same lane:

- `COST_LT12`
  - `delta_IS=+0.1132`
  - `delta_OOS=+0.1477`
- `COST_LT08`
  - `delta_IS=+0.1663`
  - `delta_OOS=+0.7217`

This makes `COST_LT08` a real upgrade candidate, not just a duplicate name.

## Cheap gate

`research/phase4_candidate_precheck.py`:

- `hypothesis_sha=fdf262d36f5590033cc5fc3bc985329bdf357e09c883c9e346057306803622ce`
- `accepted_raw_trials=1`
- `experimental_rows_with_sha=0`
- `validated_rows_from_sha=0`
- exact combo:
  - `orb_label='TOKYO_OPEN'`
  - `filter_key='COST_LT08'`
  - `entry_model='E2'`
  - `rr_target=1.5`
  - `confirm_bars=1`
  - `stop_multiplier=1.0`
  - `n_preholdout_outcomes=427`

## Discovery write

`trading_app/strategy_discovery.py --instrument MNQ --orb-minutes 5 --holdout-date 2026-01-01 --hypothesis-file ...`

Result:

- strategy id: `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`
- `sample_size=427`
- `entry_signals=427`
- `expectancy_r=+0.2037`
- `win_rate=0.5105`
- `p_value=0.000362`
- `fdr_adjusted_p_discovery=0.000362`
- `is_canonical=True`

## Validator outcome

`python -m trading_app.strategy_validator --instrument MNQ`

Outcome:

- walkforward: `5/8` windows positive
- aggregate OOS ExpR: `+0.2233`
- `WFE=0.9752`
- validator status: `PASSED`

Promoted row in `validated_setups`:

- `strategy_id='MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08'`
- `status='active'`
- `validation_pathway='family'`
- `wf_passed=True`
- `wf_windows=8`
- `oos_exp_r=+0.2233`
- `dsr_score=0.5087662631825595` (informational only)

## Hard truth

This survives, but the caveats matter:

- this is an **upgrade within the cost-ratio family**, not a new mechanism
- one reverse year exists (`2023`)
- OOS off-side is only `N=12`
- DSR does not clear the project’s higher comfort threshold

So the right interpretation is:

- valid exact row
- economically stronger than the active looser threshold on this lane
- still a conditional, lane-specific improvement rather than a universal
  threshold law

## Verdict

`MNQ TOKYO_OPEN O5 E2 RR1.5 COST_LT08` survives the full bridge and promotes.

This is the next live MNQ mechanism-family winner after the prior-day geometry
program, and it broadens the active search surface beyond geometry without
reopening dead paths like ML or blind stacked confluence.
