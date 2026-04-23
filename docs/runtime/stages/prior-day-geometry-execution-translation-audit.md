---
slug: prior-day-geometry-execution-translation-audit
classification: RESEARCH
mode: DESIGN
stage: 1
of: 1
created: 2026-04-23
updated: 2026-04-23
task: Resolve whether the positive same-session MNQ prior-day geometry shelf rows can be translated honestly into executable live behavior without violating current runtime constraints.
---

# Stage: Prior-Day Geometry Execution Translation Audit

## Question

The prior-day geometry branch is no longer a discovery problem and no longer a
simple routing problem.

The five promoted shelf survivors remain positive, but every candidate
collides with an existing same-session live lane. The honest next question is:

> can any of these same-session geometry rows be translated into executable
> live behavior under the current runtime, or do they require a new
> architecture / shadow-test branch first?

## Scope Lock

- MNQ only
- same-session collisions only:
  - `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG`
  - `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG`
  - `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG`
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG`
  - `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG`
- incumbent live lanes:
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`

## Required Read Set

- `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`
- `trading_app/risk_manager.py`
- `trading_app/prop_profiles.py`
- `trading_app/live/session_orchestrator.py`
- `docs/plans/manual-trading-playbook.md`
- canonical `orb_outcomes` / `daily_features`

## Required Questions

1. Execution feasibility
   - Is the candidate blocked by current same-session runtime limits?
   - same aperture?
   - different aperture?
   - same-direction overlap?

2. Role honesty
   - replacement only?
   - conditional veto / allocator?
   - delayed-entry / size-down branch?
   - manual-only watchlist?

3. Translation cost
   - what code/runtime changes would be required?
   - what shadow-test surface is needed before any live-route claim?
   - does the translation preserve pre-trade honesty?

4. Manual playbook relevance
   - if not auto-routable, is the setup still worth preserving explicitly for
     manual review?
   - do not collapse “not in current auto lane” into “not a good trade”

## Acceptance Criteria

1. Separates research-only additivity from executable additivity.
2. States whether the path is:
   - current-runtime feasible
   - architecture-change required
   - manual-playbook only
   - park
3. Does not reopen Track A discovery.
4. Does not rerun consumed hypothesis files.
5. Ends with one explicit next move per candidate family, not generic options.

## Non-goals

- not another prior-day filter search
- not broad profile redesign
- not direct lane allocation edits
- not discretionary story-first chart interpretation
