---
status: active
owner: codex
last_reviewed: 2026-05-03
superseded_by: ""
---

# Live-Readiness Report Design

## Purpose

Add one operator-facing command that answers a narrow question without creating
another truth layer:

> For one profile, what is the current deploy/no-deploy control state based on
> the repo's canonical live-control surfaces?

The command is a read-only aggregator. It must not recode Criterion 11,
Criterion 12, allocator eligibility, or deployment truth. It only composes
those authorities into one reproducible report.

## Authority and Source Routing

Canonical inputs:

- `trading_app/lifecycle_state.py`
  - Criterion 11 gate/report state
  - Criterion 12 SR state
  - per-strategy blocked reasons
- `trading_app/prop_profiles.py`
  - profile lane definitions
- `deployable_validated_setups`
  - validated-active universe for deployed-vs-validated comparison
- `docs/runtime/lane_allocation.json`
  - allocator lane buckets and rebalance provenance

Non-goals:

- no new live-control logic
- no new deployment policy
- no writes to runtime state
- no replacement of `project_pulse.py` or `context_views.py`

## Interface

New command:

- `python scripts/tools/live_readiness_report.py`

Flags:

- `--profile <id>` optional profile override
- `--format text|json|markdown`
- `--out <path>` optional artifact write
- `--allocation-path <path>` test/debug override only

Outputs:

- deployment summary: deployed count, validated-active count, mismatch sets
- Criterion 11 summary
- Criterion 12 summary
- active lanes with merged lifecycle blocked/SR state
- paused/stale allocator rows with reasons
- allocator provenance: source path, rebalance date, trailing window

## Design Choices

1. Use `lifecycle_state` directly rather than `context_views.build_trading_context`.
   `context_views` is a good truth-class template, but it currently omits
   allocator provenance and does not expose a profile-specific interface.
2. Use `lane_allocation.json` as the live-book truth when present.
   Fall back to `prop_profiles` lane definitions only when allocator state is
   absent, and mark those rows as `profile_config`.
3. Keep the report read-only and additive.
   This is an operator surface, not a new state registry.

## Verification

Focused tests must cover:

- allocator-present path with merged lifecycle block reasons
- allocator-missing fallback path
- rendered operator output containing the required sections

Runtime verification:

- run the tool against the active checkout in JSON mode
- run targeted pytest for the new report module
- run targeted lint on the touched files
