---
status: active
owner: codex
last_reviewed: 2026-05-02
superseded_by: ""
---

# Topstep Operator Architecture V2

## Scope

Decision memo for the operator architecture around `canompx3` with a
Topstep-centered lifecycle. This memo is a **decision** surface under
`docs/plans/`; it explains the current best framing and the next spike, but it
does not override code/runtime truth or broker policy docs.

## Method

Hard rules used in this pass:

- No lookahead: do not let a strong `Combine/XFA` setup win if it breaks in
  `Live Funded`.
- No hidden averaging: a candidate with a broken stage does not get rescued by
  scores elsewhere.
- No platform loyalty: compare operating models, not brand affinity.
- No rewrite trap: keep ORB/session/filter/risk truth in `canompx3`; do not
  port the core logic into platform scripting unless explicitly justified.
- No false certainty: every conclusion is labeled `MEASURED`, `INFERRED`, or
  `UNSUPPORTED`.

## Local Grounding

- `MEASURED` The dashboard is a bot control + account-management surface; manual
  trading happens on external platforms. Source:
  `docs/superpowers/specs/2026-04-06-dashboard-connections-page-design.md:9`
- `MEASURED` The in-house embedded chart path is parked, and the repo already
  acknowledges that platforms such as NinjaTrader/Sierra/Quantower provide
  chart + custom-indicator + execution surfaces while carrying a rewrite risk.
  Source:
  `docs/superpowers/specs/2026-04-11-dashboard-embedded-chart-signal-overlay.md:139-145`
- `MEASURED` Current `lane_cards` exported from `bot_state.py` are still thin
  and do not carry the richer ORB/signal contract that an external operator
  surface would need. Source: `trading_app/live/bot_state.py:70-120`
- `MEASURED` The parked chart spec explicitly identifies missing
  `lane_card` fields such as `orb_high`, `orb_low`, `stop_price`,
  `target_price`, and `signal_time_utc` as blocking prerequisites. Source:
  `docs/superpowers/specs/2026-04-11-dashboard-embedded-chart-signal-overlay.md:255-261`
- `MEASURED` The live broker abstraction already spans `projectx`,
  `tradovate`, and `rithmic`. Source:
  `trading_app/live/broker_connections.py:23-55`

## External Grounding

- `MEASURED` Topstep new Trading Combines became TopstepX-only on
  2025-07-07, and resets became TopstepX-only on 2025-08-01 for the practical
  new-account path. Source:
  <https://help.topstep.com/en/articles/8284149-tradovate-connection-instructions>
- `MEASURED` TopstepX API supports automation, third-party tools, market data,
  and direct order execution. Source:
  <https://help.topstep.com/en/articles/11187768-topstepx-api-access>
- `MEASURED` Topstep `Live Funded` permits automated strategies generally, but
  explicitly prohibits automated trading through the ProjectX API. Source:
  <https://help.topstep.com/en/articles/10657969-live-funded-account-parameters>
- `MEASURED` TopstepX manual/operator surface has real constraints: no external
  TradingView connection, no custom indicators, one-window DOM behavior.
  Source: <https://help.topstep.com/en/articles/14434175-topstepx>
- `MEASURED` Quantower supports Topstep, ProjectX-powered prop accounts,
  Rithmic, replay, and custom algo/indicator development. Sources:
  <https://help.quantower.com/quantower/connections/connection-to-topstep>,
  <https://help.quantower.com/quantower/connections/connecting-to-projectx>,
  <https://help.quantower.com/quantower/connections/connection-to-rithmic>,
  <https://help.quantower.com/quantower/trading-panels/market-replay>,
  <https://help.quantower.com/quantower/quantower-algo>

## Self-Audit

- `MEASURED` Prior framing over-weighted operator aesthetics and under-weighted
  lifecycle continuity. The repo itself already framed the dashboard as a
  control/account surface, not the main execution home.
- `MEASURED` Prior framing implicitly treated "API exists" as "API is allowed
  across all stages." That was false for `Live Funded`.
- `INFERRED` Prior framing was too close to a platform-selection contest. The
  stronger unit of analysis is the operator model that survives the full
  `Combine -> XFA -> Live` path.
- `MEASURED` Failure containment was under-specified. Wrong-account risk, stale
  state, reject visibility, and bracket collisions are first-order architecture
  concerns here.

## Candidate Set

### C0 — TopstepX-only Manual Baseline

- No sidecar
- No Quantower
- `canompx3` remains research/signal truth only
- Role: null hypothesis / minimum-complexity control

### C1 — TopstepX-core + Read-only `canompx3` Sidecar

- `canompx3` emits ORB/signal truth, warnings, and state
- Human executes in TopstepX
- No order routing from the sidecar

### C2 — TopstepX-core + Manual-assist Sidecar

- Same as C1
- Adds pre-submit operator controls:
  - account/stage confirmation
  - bracket-template validation
  - duplicate-order warnings
  - stale-state and mismatch warnings
- Still no sidecar execution authority

### C3 — TopstepX + Quantower Operator Shell + `canompx3` Brain

- Quantower used as a challenger surface
- Must not absorb the ORB/research truth into platform scripting

### C4 — Live Automation via ProjectX API

- Included explicitly to be tested against current policy
- Expected dead end for end-state `Live Funded` architecture

## Unexplored Edges

1. `INFERRED` Manual-assist is underexplored. There is a likely high-EV middle
   ground between pure manual TopstepX use and full external-operator adoption.
2. `INFERRED` Failure-prevention tooling is underexplored. Wrong-account,
   stale-state, duplicate-order, and bracket-collision prevention likely carry
   more EV than another chart surface.
3. `INFERRED` Replay parity is underexplored. The key question is not prettier
   replay; it is whether the replay routine trains the same behavior used live.

## Unknowns Register

- `UNSUPPORTED` Whether Topstep explicitly endorses Quantower as a clean,
  long-run primary surface for new TopstepX-era traders across all relevant
  lifecycle stages.
- `UNSUPPORTED` Whether `Live Funded` read-only API-driven overlays and safety
  tooling are explicitly blessed, rather than merely not explicitly banned.
- `UNSUPPORTED` Whether Quantower materially reduces operator error rate, as
  opposed to improving subjective comfort only.
- `UNSUPPORTED` Whether the user's real pain is primarily charting/replay, or
  primarily account/risk/workflow safety.

These unknowns are blockers to overconfident conclusions. They must be tested,
not narrated away.

## Regime-Split Matrix

Scale:

- `5` strong
- `3` mixed / unproven
- `1` weak
- `0` blocked

### Regime A — Combine / XFA

| Candidate | Policy | Lifecycle | Integration | Operator EV | Failure | Complexity | View |
|---|---:|---:|---:|---:|---:|---:|---|
| C0 TopstepX-only manual | 5.0 | 5.0 | 5.0 | 2.0 | 2.0 | 5.0 | viable but weak tooling |
| C1 TopstepX + read-only sidecar | 5.0 | 5.0 | 4.5 | 3.0 | 4.0 | 4.0 | strong |
| C2 TopstepX + manual-assist sidecar | 5.0 | 5.0 | 4.0 | 4.0 | 5.0 | 3.5 | strongest hypothesis |
| C3 TopstepX + Quantower shell | 3.5 | 3.5 | 3.5 | 4.5 | 3.0 | 2.5 | useful but more fragile |
| C4 ProjectX API execution path | 4.0 | 1.0 | 2.0 | 4.0 | 2.0 | 2.0 | contaminated by future break |

### Regime B — Live Funded

| Candidate | Policy | Lifecycle | Integration | Operator EV | Failure | Complexity | View |
|---|---:|---:|---:|---:|---:|---:|---|
| C0 TopstepX-only manual | 5.0 | 5.0 | 5.0 | 2.0 | 2.0 | 5.0 | safe baseline |
| C1 TopstepX + read-only sidecar | 4.0 | 5.0 | 4.5 | 3.0 | 4.0 | 4.0 | likely viable |
| C2 TopstepX + manual-assist sidecar | 4.0 | 5.0 | 4.0 | 4.0 | 5.0 | 3.5 | likely best if non-executing |
| C3 TopstepX + Quantower shell | 3.0 | 3.0 | 3.5 | 4.0 | 3.0 | 2.5 | uncertain continuity |
| C4 ProjectX API execution path | 0.0 | 0.0 | 2.0 | 4.0 | 2.0 | 2.0 | dead |

## Minimax Read

Take the weaker stage for each candidate:

- `MEASURED` C4 is dead as an end-state architecture because its weak stage is
  blocked by current `Live Funded` policy.
- `INFERRED` C3 is a challenger only. It may still prove useful, but its
  weakest-stage continuity and complexity profile are materially worse than the
  TopstepX-core sidecar variants.
- `INFERRED` C2 is the strongest current hypothesis because it improves failure
  containment while preserving the Topstep-centered lifecycle and avoiding
  execution automation.
- `INFERRED` C1 is the conservative fallback if manual-assist proves too heavy.
- `MEASURED` C0 remains the correct control and simplicity floor.

## Kill Criteria

### Global

- Kill any candidate that depends on a route Topstep blocks in the stage being
  tested.
- Kill any candidate that requires porting core ORB/session/filter/risk logic
  into platform scripting.
- Kill any candidate that creates two truths instead of one source of truth plus
  a thin projection.

### C0

- Kill if baseline TopstepX-only operation clearly fails obvious safety or
  operator needs that a thin sidecar could solve with bounded complexity.

### C1

- Kill if the read-only sidecar cannot get timely enough state to be trusted.
- Kill if stale-state visibility is not obvious before an execution decision.
- Kill if human use of the sidecar creates pseudo-execution confusion.

### C2

- Kill if manual-assist grows into a second platform rather than a bounded
  safety layer.
- Kill if wrong-account, bracket mismatch, or duplicate-order risk are not
  materially reduced relative to C0.
- Kill if the assistive checks become routine bypass friction rather than real
  protection.

### C3

- Kill if Quantower does not materially reduce at least one of:
  - execution friction
  - operator error rate
  - replay-to-live drift
- Kill if Quantower duplicates symbol/account/risk awareness across too many
  surfaces without measurable safety gain.
- Kill if plugin or indicator work starts absorbing ORB/research truth.
- Kill if the measurable improvement is mostly visual comfort.

### C4

- Kill as an end-state architecture if it depends on ProjectX API automated
  trading in `Live Funded`.

## Spike Plan

### Day 1 — Baseline and Unknowns

- Establish C0 TopstepX baseline across:
  - execution friction
  - wrong-account risk
  - stale-state ambiguity
  - bracket handling
  - replay drift
- Write the unknowns register as a live checklist, not as footnotes.

### Day 2 — C1 Contract Test

- Define the minimal read-only export contract from `canompx3`
- No execution
- No platform scripting
- Test whether ORB/signal/stop-target/safety visibility is enough to be useful

### Day 3 — C2 Safety Tooling Test

- Prototype manual-assist logic on paper / checklist terms first:
  - account confirmation
  - stage confirmation
  - bracket template validation
  - duplicate-order warning
- Test whether these reduce risk without becoming a second platform

### Day 4 — C3 Challenger Test

- Quantower only tests as a challenger against C2
- Measure:
  - whether it reduces friction
  - whether it reduces errors
  - whether replay better matches the live routine
- If not, kill it

### Day 5 — Minimax Decision

- Score each candidate by regime
- Take the weakest-stage view
- Prefer the simpler candidate unless the challenger proves a real operational
  gain
- Output:
  - winner
  - fallback
  - dead paths
  - unresolved unknowns

## Current Verdict

- `MEASURED` C4 is dead as an end-state architecture.
- `MEASURED` Legacy non-TopstepX-first thinking is strategically obsolete for a
  new Topstep-centered path.
- `INFERRED` Current favorite is **C2: TopstepX-core + `canompx3`
  manual-assist sidecar**, if and only if it stays non-executing,
  lifecycle-safe, and bounded.
- `INFERRED` C1 is the conservative fallback.
- `INFERRED` C3 is the challenger, not the presumptive winner.

## Decision Reminder

The unit of choice is not "which platform do we like more." The unit of choice
is "how much operator intelligence should live beside TopstepX without taking
execution authority or creating a second truth surface."
