# Prior-Day Geometry Execution Translation Pre-Audit

Date: 2026-04-23

## Scope

Translate the closed routing audit into the next honest question:

- which positive same-session MNQ prior-day geometry rows are executable under
  the current runtime
- which require architecture / shadow testing first
- which are replacement-only or manual-review only

This is not a new discovery pass. It is a runtime-feasibility split grounded in
current repo behavior.

## Truth Surfaces Used

Canonical trade truth:

- `gold.db::orb_outcomes`
- `gold.db::daily_features`

Runtime / portfolio truth:

- `trading_app/risk_manager.py`
- `trading_app/live/session_orchestrator.py`
- `trading_app/prop_profiles.py`
- `trading_app/prop_portfolio.py`
- `docs/plans/manual-trading-playbook.md`
- `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`

## MEASURED Runtime Constraints

### 1. Same-aperture same-session coexistence is blocked

From `trading_app/risk_manager.py`:

- `RiskLimits.max_per_orb_positions = 1`
- `can_enter(...)` rejects a second entered trade on the same `orb_label`
  and same `orb_minutes`

Implication:

- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG` cannot coexist live with the
  current `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
- any `US_DATA_1000 O5` row cannot coexist with another `US_DATA_1000 O5` row
  if one is already live

### 2. Different-aperture same-session coexistence is not blocked outright

From `trading_app/risk_manager.py`:

- `RiskLimits.max_per_session_positions = 2`
- Check 7 applies a reduced contract factor when a same-session
  different-aperture trade is already active

Implication:

- `US_DATA_1000 O5` candidates are not structurally impossible next to the
  current live `US_DATA_1000 O15` lane
- but the runtime path is explicitly translated, not native: later entries
  require size-down handling

### 3. The live runtime can carry duplicate same-session lane definitions

From `trading_app/prop_profiles.py` and `trading_app/live/session_orchestrator.py`:

- `get_profile_lane_definitions()` preserves full lane identity and does not
  collapse duplicate sessions
- `get_lane_registry()` now keys caps by `(orb_label, instrument)`, not session
  alone
- `session_orchestrator` loads ORB caps using the same `(orb_label, instrument)`
  key shape

Implication:

- duplicate same-session lanes are not automatically a config/parser failure
- the remaining question is behavioral safety, not config impossibility

### 4. The profile-construction path still deduplicates session x instrument

From `trading_app/prop_portfolio.py`:

- `_deduplicate_sessions()` keeps one best strategy per `(orb_label, instrument)`

Implication:

- current portfolio-construction canon still treats same-session duplicates as a
  non-default path
- any live addition of a second same-session lane is therefore a translation
  branch, not a normal portfolio-builder outcome

### 5. Manual-playbook relevance is separate from current auto routing

From `docs/plans/manual-trading-playbook.md`:

- the manual playbook is a separate operating surface
- “not in the current auto profile” is not equivalent to “bad trade”

Implication:

- positive shelf rows that are not auto-routable today should remain visible for
  manual review or later profile use

## Candidate Split

### A. `US_DATA_1000` prior-day geometry rows

Rows:

- `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG`
- `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG`
- `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG`
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG`

Current live collision:

- `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`

Measured classification:

- `CURRENT-RUNTIME COEXISTENCE PLAUSIBLE, BUT NOT YET HONESTLY ROUTABLE`

Why:

- different aperture (`O5` candidate vs `O15` incumbent)
- not blocked by `max_per_orb_positions`
- constrained by `max_per_session_positions = 2`
- requires size-down / sequencing translation because the runtime explicitly
  reduces size for same-session different-aperture overlap
- not a clean “free 7th slot” interpretation

Honest next move:

- narrow shadow-test / execution-translation audit for `US_DATA_1000 O5 + O15`
  coexistence

### B. `COMEX_SETTLE` prior-day geometry row

Row:

- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG`

Current live collision:

- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`

Measured classification:

- `REPLACEMENT-ONLY OR MANUAL-REVIEW ONLY UNDER CURRENT RUNTIME`

Why:

- same aperture (`O5` vs `O5`)
- same-session coexistence is blocked by `max_per_orb_positions = 1`
- routing audit already showed negative replacement deltas vs the incumbent

Honest next move:

- do not pursue as an auto add
- keep visible on shelf / manual review, but do not spend current auto-routing
  capital here

## Verdict

The highest-EV next stage is narrower than “translate all prior-day geometry.”

It is:

1. `US_DATA_1000` same-session cross-aperture execution translation / shadow
   test
2. park `COMEX_SETTLE PD_CLEAR_LONG` as an auto-routing branch unless a new
   role is proposed

## Anti-Bias Guard

Do not let positive research-only add math be described as a live-route win
until the exact same-session execution path is modeled or shadow-tested.
