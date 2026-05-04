# PR48 MGC shadow-only overlay contract design

**Date:** 2026-04-23  
**Scope:** choose the smallest honest contract for carrying frozen `MGC:cont_exec` into the repo as a `shadow_only` profile-local conditional overlay.

## Grounding

- Upstream truth:
  - `docs/audit/results/2026-04-23-pr48-mgc-cont-exec-bounded-translation.md`
  - `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
  - `docs/audit/hypotheses/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.yaml`
  - `research/output/pr48_mes_mgc_sizer_rule_breakpoints_v1.csv`
- Runtime / operator surfaces inspected:
  - `trading_app/derived_state.py`
  - `trading_app/prop_profiles.py`
  - `trading_app/prop_portfolio.py`
  - `trading_app/pre_session_check.py`
  - `trading_app/live/bot_dashboard.py`
  - `trading_app/paper_trade_logger.py`
  - `trading_app/db_manager.py`
  - `docs/specs/research_modes_and_lineage.md`

## Exact question

What is the smallest honest **shadow-only** contract that:

1. preserves the frozen PR48 `MGC:cont_exec` sizing truth
2. is called automatically by the runtime/operator surfaces
3. does not pretend to be a standalone lane
4. does not silently mutate execution size

## Design decision

Use a **two-part contract**:

1. a checked-in **static overlay spec registry**
2. a daily **derived-state shadow envelope**

Do **not** use:

- `validated_setups`
- `lane_allocator.py`
- `lane_allocation.json`
- the existing late-applied `ExecutionEngine.size_multiplier`
- `paper_trades` as the primary state carrier

## Why this wins

### 1. It matches the true object

The frozen PR48 result is not a single strategy lane. It is:

- instrument `MGC`
- `orb_minutes=5`
- `entry_model=E2`
- `confirm_bars=1`
- `rr_target=1.5`
- nine sessions
- both directions
- per-lane rel-vol breakpoints and a frozen size map

So the carrier must be a role-tagged conditional overlay object, not a lane row.

### 2. It fits existing repo contracts

The repo already has a canonical self-invalidating state pattern in
`trading_app/derived_state.py`:

- profile fingerprint
- lane ids
- DB identity
- code fingerprint
- freshness envelope

That is the correct substrate for a shadow-only overlay because it:

- can fail closed
- can be read by preflight / dashboard surfaces
- does not require DB schema mutation
- does not pretend the object is deployable execution truth

### 3. It avoids abusing `paper_trades`

`paper_trades` is an execution journal keyed by `(strategy_id, trading_day)`.
That is the wrong primary carrier for a broad conditional overlay:

- it assumes a concrete strategy id
- it implies a trade-like object
- it is better used later for explicit shadow logging, not as the first-class contract

### 4. It is automatically callable without execution drift

The static spec can be resolved at startup, and the derived-state file can be:

- generated or refreshed before session checks
- shown on the dashboard
- used as operator truth for shadow recommendations

without changing entries, contracts, or risk checks.

## Chosen contract

### A. Static overlay spec registry

Recommended new code surface:

- `trading_app/conditional_overlays.py`

Recommended canonical object:

- `ConditionalOverlaySpec`

Recommended required fields:

- `overlay_id`
- `profile_id`
- `mode` = `shadow_only`
- `role` = `allocator`
- `instrument` = `MGC`
- `orb_minutes` = `5`
- `entry_model` = `E2`
- `confirm_bars` = `1`
- `rr_target` = `1.5`
- `sessions`
- `directions`
- `feature_family` = `rel_vol_session`
- `breakpoint_artifact_path`
- `size_map`
- `holdout_frozen_from`
- `notes`

The spec is static, checked-in, and references frozen artifacts only.

### B. Daily derived-state envelope

Recommended state type:

- `conditional_overlay_shadow`

Recommended path pattern:

- `data/state/conditional_overlay_<profile_id>_<overlay_id>.json`

Recommended canonical envelope:

- built with `build_state_envelope(...)`
- validated with `validate_state_envelope(...)`

Recommended payload fields:

- `overlay_id`
- `profile_id`
- `mode`
- `as_of_date`
- `frozen_artifact_sha`
- `status`
  - `ready`
  - `unscored`
  - `invalid`
- `rows`
  - one per `(session, direction)`
  - `feature_value`
  - `bucket`
  - `size_multiplier`
  - `reason`
- `summary`
  - active sessions today
  - coverage
  - any invalid/missing feature warnings

This is operator/runtime shadow truth, not research truth.

## Runtime read path

Phase 1 readers only:

1. `pre_session_check.py`
   - display overlay availability and any invalid/unscored status
   - no gate change

2. `live/bot_dashboard.py`
   - show overlay state alongside the owning profile
   - clearly marked `shadow_only`

3. optional derived-state inspection surfaces
   - reuse the existing state-envelope validation style

Phase 1 non-readers:

- `execution_engine.py`
- `risk_manager.py`
- `lane_allocator.py`
- `validated_setups`

That keeps the first implementation bounded and honest.

## Shadow logging policy

Phase 1:

- no `paper_trades` writes
- no synthetic strategy ids
- no fake “would have traded 1.5x” execution rows

Reason:

- the current repo has no first-class contract for non-lane shadow sizing events
- writing them as trades too early would blur operator state with execution state

Phase 2, if Phase 1 proves useful:

- add explicit shadow logging with a distinct `execution_source`
- only after the overlay contract itself is stable

## Fail-closed rules

If any of the following fail, the overlay is **inactive**, not silently guessed:

1. profile mismatch
2. profile fingerprint mismatch
3. lane-id mismatch
4. DB identity mismatch
5. code fingerprint mismatch
6. stale freshness window
7. missing breakpoint artifact
8. missing required session feature for the current day
9. invalid bucket assignment

Required degraded states:

- `invalid` if the contract itself cannot be trusted
- `unscored` if today’s feature value is absent
- never “best effort”

## Blast radius

### Phase 1 bounded implementation

Likely touched files:

- `trading_app/conditional_overlays.py` new
- `trading_app/derived_state.py` reuse only, ideally no semantic change
- one small builder / loader utility
- `trading_app/pre_session_check.py`
- `trading_app/live/bot_dashboard.py`

### Deferred deliberately

- `trading_app/execution_engine.py`
- `trading_app/risk_manager.py`
- `trading_app/paper_trade_logger.py`
- `trading_app/db_manager.py`
- database schema

That is the core reason this is the right next step: bounded surface, high signal,
no fake execution claims.

## Alternatives rejected

### Reuse the old `topstep_50k` MGC lane

Rejected because it is the wrong parent:

- one session only
- different RR/filter object
- already a standalone-style lane surface

### Reuse `ExecutionEngine.size_multiplier`

Rejected because it is execution-time sizing, applied too late, and already
known to be dishonest for this branch.

### Store the overlay in `validated_setups`

Rejected because `validated_setups` is a standalone deployable shelf, not a
conditional overlay registry.

## Verdict / Decision

**Verdict:** `CONTINUE`

Continue with one bounded implementation stage for a **shadow-only derived-state
overlay contract**, not runtime sizing and not a schema-wide conditional-system
rebuild.

## Recommended next step

Use:

- `docs/runtime/stages/pr48-mgc-shadow-only-overlay-contract.md`

That stage should implement only:

- static overlay spec
- derived-state envelope builder/validator
- pre-session/dashboard read path

and stop there.

## Caveats

- This design does not make `MGC:cont_exec` live-ready.
- This design does not resolve later execution sizing semantics.
- This design does not reopen research or retune the frozen map.

## Reproduction

- Read:
  - `docs/audit/results/2026-04-23-pr48-mgc-cont-exec-bounded-translation.md`
  - `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
  - `docs/audit/hypotheses/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.yaml`
  - `research/output/pr48_mes_mgc_sizer_rule_breakpoints_v1.csv`
- Inspect:
  - `trading_app/derived_state.py`
  - `trading_app/prop_profiles.py`
  - `trading_app/prop_portfolio.py`
  - `trading_app/pre_session_check.py`
  - `trading_app/live/bot_dashboard.py`
  - `trading_app/paper_trade_logger.py`
  - `trading_app/db_manager.py`
