---
slug: prior-day-geometry-routing-audit
classification: RESEARCH
mode: RESEARCH
stage: 1
of: 1
created: 2026-04-23
updated: 2026-04-23
task: Resolve the five promoted MNQ prior-day geometry shelf survivors into explicit routing decisions instead of treating Track A as open discovery.
---

# Stage: Prior-Day Geometry Routing Audit

## Question

The prior-day bridge signal work is no longer the open problem.

The exact bridge hypotheses were already consumed:

- one exact avoid cell rejected
- five broader geometry rows promoted to the validated shelf
- none currently routed in the active MNQ live profile

The honest open question is now:

> are the promoted prior-day geometry rows additive live candidates, same-session substitutes, or valid shelf rows that should remain unrouted?

## Scope Lock

- MNQ only
- promoted bridge survivors only:
  - `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG`
  - `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG`
  - `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG`
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG`
  - `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG`
- comparator book:
  - current `topstep_50k_mnq_auto` lanes from `docs/runtime/lane_allocation.json`

## Blast Radius

- research-only
- allowed outputs:
  - one result doc under `docs/audit/results/`
  - optional read-only audit script under `research/` if needed
- no writes to:
  - `validated_setups`
  - `edge_families`
  - `lane_allocation.json`
  - live config / profiles

## Required Read Set

- `docs/audit/results/2026-04-23-prior-day-bridge-closure-audit.md`
- `docs/audit/results/2026-04-19-validated-shelf-vs-live-deployment-audit.md`
- `docs/runtime/lane_allocation.json`
- `docs/runtime/decision-ledger.md`
- `trading_app/validated_shelf.py`
- canonical `orb_outcomes` / `daily_features`

## Required Output

For each of the five promoted shelf rows, report:

1. canonical IS / OOS expectancy and trade-count context
2. day-overlap / same-session substitution view versus the current live book
3. portfolio contribution versus the current live MNQ profile
4. explicit decision:
   - `ROUTE_LIVE`
   - `KEEP_ON_SHELF`
   - `PARK_NON_ADDITIVE`

## Acceptance Criteria

1. Does not rerun consumed hypothesis files.
2. Does not reopen broad prior-day discovery.
3. Distinguishes shelf truth from live-routing truth.
4. Ends with explicit per-row routing decisions.
5. Leaves a durable result doc so this branch cannot masquerade as open discovery again.

## Non-goals

- not another Track A prereg
- not another prior-day mega scan
- not automatic deployment
- not a broad MNQ profile redesign
