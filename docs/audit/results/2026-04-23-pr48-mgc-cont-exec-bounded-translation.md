# PR48 MGC `cont_exec` bounded translation

**Date:** 2026-04-23  
**Stage:** `docs/runtime/stages/pr48-mgc-cont-exec-bounded-translation.md`  
**Scope:** determine whether the frozen `MGC:cont_exec` PR48 sizing map can be carried into the current runtime as a bounded, honest, profile-local object.

## Grounding

- Research truth:
  - `docs/audit/results/2026-04-23-pr48-conditional-role-validation-translation.md`
  - `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
  - `docs/audit/hypotheses/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.yaml`
  - `research/output/pr48_mes_mgc_sizer_rule_breakpoints_v1.csv`
- Runtime surfaces inspected:
  - `trading_app/prop_profiles.py`
  - `trading_app/prop_portfolio.py`
  - `trading_app/portfolio.py`
  - `trading_app/execution_engine.py`
  - `trading_app/risk_manager.py`
  - `trading_app/live_config.py`

## Exact question

What is the smallest honest bridge that can carry the frozen `MGC:cont_exec`
size map into a profile-local, non-standalone runtime surface?

## Measured repo truth

### 1. The frozen PR48 object is broad, not a single-lane lane override

The locked PR48 sizer was frozen on the unfiltered canonical parent:

- instrument: `MGC`
- `orb_minutes=5`
- `entry_model=E2`
- `confirm_bars=1`
- `rr_target=1.5`
- sessions: `CME_REOPEN`, `COMEX_SETTLE`, `EUROPE_FLOW`, `LONDON_METALS`,
  `NYSE_OPEN`, `SINGAPORE_OPEN`, `TOKYO_OPEN`, `US_DATA_1000`, `US_DATA_830`
- directions: both `long` and `short`
- threshold surface: per-lane `(session x direction)` rel-vol quintiles

Measured parent coverage from canonical `orb_outcomes`:

- `2022-06-13` through `2026-04-16`
- `N=8086` canonical rows

This is not a single strategy ID and not a one-session sleeve.

### 2. The only existing MGC profile-local surface is the wrong parent

`trading_app/prop_profiles.py` currently exposes one explicit MGC conditional
profile:

- profile: `topstep_50k`
- instrument scope: `MGC` only
- session scope: `TOKYO_OPEN` only
- lane: `MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4_CONT_S075`

That surface is structurally different from the PR48 parent:

- one session vs nine
- one concrete strategy ID vs unfiltered session-direction parent lanes
- `RR2.0 + ORB_G4` vs unfiltered `RR1.5`

So the current profile-local MGC surface is **not** an honest attachment point
for `MGC:cont_exec`.

### 3. The repo still assumes profile lanes are concrete validated strategy IDs

`prop_portfolio.py` and `portfolio.py` resolve profile lanes by:

- reading `DailyLaneSpec.strategy_id`
- looking that strategy up in `validated_setups`
- building a `PortfolioStrategy` with `max_contracts=1`

There is no native profile object for:

- a cross-session conditional sizer
- per-lane frozen breakpoints
- a role-tagged `allocator` / `sizer`
- a non-standalone parent population

### 4. The current size-multiplier path is not an honest bridge

`execution_engine.py` has a `size_multiplier` field on `ActiveTrade`, but it is
currently used as a late execution overlay and applied **after** base contract
count has already been computed and clamped.

That creates two measured problems for `MGC:cont_exec`:

1. `0.5x` is not honest at `1` contract:
   - `max(1, int(1 * 0.5)) = 1`
   - so the supposed reduction is a no-op

2. `1.5x` / `2.0x` can bypass the original base-lane cap:
   - base contracts are clamped before the multiplier
   - then `trade.contracts` is scaled afterward
   - so a lane built with `max_contracts=1` can still become `2`

That means the existing multiplier path is not a safe carrier for a frozen
institutional sizing map.

### 5. Topstep scaling-plan checks are still projected as a 1-contract entry

`risk_manager.py` Topstep F-1 enforcement currently projects the new entry with
`new_contracts=1`.

That is fine for today’s common 1-contract lanes, but it is not an honest
control surface for a post-cap sizer overlay that may try to move the same
entry to `2` contracts.

So even if the execution-engine multiplier path were reused, it would still
misalign with the scaling-plan guard.

## Candidate target surfaces

### Option A — report-only profile note

**Verdict:** too weak

This preserves documentation but does not actually translate the frozen rule
into a runtime object. It is not a bridge; it is a reminder.

### Option B — reuse the current runtime `size_multiplier`

**Verdict:** invalid

This is the tempting shortcut, but it is structurally wrong for this branch:

- `0.5x` is a no-op at 1 contract
- `1.5x` / `2.0x` can bypass the base lane cap
- Topstep scaling checks still assume a 1-contract projected entry

This would create a false appearance of an implemented sizer while violating
the repo’s own runtime honesty standard.

### Option C — one new profile-local conditional config surface, shadow-only first

**Verdict:** correct design target, but requires redesign before implementation

The smallest honest target is a new profile-local conditional overlay object
with:

- explicit `role = allocator` or `sizer`
- explicit parent scope:
  - instrument `MGC`
  - `orb_minutes=5`
  - `entry_model=E2`
  - `confirm_bars=1`
  - `rr_target=1.5`
  - session list frozen from the prereg
- frozen per-lane breakpoints from
  `research/output/pr48_mes_mgc_sizer_rule_breakpoints_v1.csv`
- frozen size map `{Q1:0.0, Q2:0.5, Q3:1.0, Q4:1.5, Q5:2.0}`
- deployment mode set initially to `shadow_only`

But that is **not** a drop-in use of existing lane or multiplier code. It needs
a dedicated contract.

## Smallest honest bridge

The smallest honest bridge is:

> a new **profile-local conditional overlay contract**, introduced in
> `shadow_only` mode first, with no live contract mutation until sizing is
> applied before caps and before Topstep scaling checks.

That keeps the research truth alive without pretending the current runtime can
already execute it honestly.

## Blast radius

If this redesign is pursued, the real blast radius is:

1. profile/config surface
   - `trading_app/prop_profiles.py`
   - likely one new dataclass or adjacent config registry

2. runtime readers / exposure
   - `trading_app/prop_portfolio.py`
   - `trading_app/pre_session_check.py`
   - `trading_app/live/bot_dashboard.py`
   - `trading_app/derived_state.py`

3. execution contract
   - `trading_app/execution_engine.py`
   - `trading_app/risk_manager.py`

4. shadow logging / monitoring
   - `paper_trades` or equivalent shadow logger surface
   - SR / monitoring surfaces if the overlay is to be observed forward

This is still bounded, but it is not zero-blast-radius.

## Fail-closed requirements

Any honest redesign must fail closed as follows:

1. If the overlay config is missing, malformed, or out of scope for the
   current profile:
   - do not silently fall back to a pseudo-sized execution path
   - treat the overlay as disabled

2. If the required daily feature is missing for the exact session:
   - do not guess a bucket
   - do not reuse another session’s threshold
   - shadow mode may log `unscored`

3. If live execution is ever enabled later:
   - apply target contract count **before** post-sizing caps and broker/firm
     risk checks
   - re-run Topstep scaling-plan projection with the true requested contracts
   - reject the trade if the requested reduction/increase cannot be expressed

## Verdict / Decision

**Verdict:** `REDESIGN`

`MGC:cont_exec` is still alive as a frozen allocator/sizer result, but the
current runtime does not have an honest profile-local carrier for it.

The right next move is not:

- forcing it into `validated_setups`
- reusing the current `size_multiplier` hook
- or attaching it to the old `topstep_50k` MGC lane as if that were the same parent

The right next move is one bounded redesign:

- define a profile-local conditional overlay contract
- ship it in `shadow_only` mode first
- keep live size mutation out until caps and scaling-plan checks are
  re-threaded around the true requested contract count

## Recommended next step

Write one exact design-stage doc for the new `shadow_only` conditional-overlay
contract for `MGC:cont_exec`:

- object shape
- frozen artifact inputs
- runtime read path
- shadow logging output
- fail-closed rules

Do **not** implement execution sizing in the same step.

## Caveats

- This stage does not weaken the PR48 research result. It closes a translation
  gap, not a signal gap.
- This stage does not authorize live deployment.
- This stage does not reopen discovery, and it does not revive the broader
  pooled confluence path.

## Reproduction

- Read:
  - `docs/runtime/stages/pr48-mgc-cont-exec-bounded-translation.md`
  - `docs/audit/results/2026-04-23-pr48-conditional-role-validation-translation.md`
  - `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
  - `docs/audit/hypotheses/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.yaml`
  - `research/output/pr48_mes_mgc_sizer_rule_breakpoints_v1.csv`
- Inspect:
  - `trading_app/prop_profiles.py`
  - `trading_app/prop_portfolio.py`
  - `trading_app/portfolio.py`
  - `trading_app/execution_engine.py`
  - `trading_app/risk_manager.py`
