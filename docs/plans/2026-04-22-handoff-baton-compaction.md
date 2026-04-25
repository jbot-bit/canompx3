# 2026-04-22 Handoff Baton Compaction

## Problem

`HANDOFF.md` had grown into an 8k-line mixed surface: current baton, stale update
log, durable decision history, and route commentary all in one file. That was
burning startup tokens and, worse, the autonomous MNQ discovery loop spent its
first iteration updating handoff prose instead of advancing a bounded research
candidate.

## Decision

Keep `HANDOFF.md` as a thin cross-tool baton only.

- Current baton stays in the root `HANDOFF.md`
- Full prior baton snapshots are archived under `docs/handoffs/archived/`
- Durable decisions stay in `docs/runtime/decision-ledger.md`
- Durable debt stays in `docs/runtime/debt-ledger.md`
- Design history and active research rationale stay in `docs/plans/`

## Workflow

Use `scripts/tools/compact_handoff.py` when the root baton starts accreting
session history instead of current state.

The tool does three things in one pass:

1. archives the current root `HANDOFF.md`
2. rewrites `HANDOFF.md` into the compact `Last Session` and `Next Steps`
   layout already consumed by `session_preflight.py` and `project_pulse.py`
3. keeps explicit references back to the durable history surfaces

## Guardrails

- Do not cite archived handoffs as runtime truth when canonical code or DB
  exists
- Keep the compact baton short enough to read on every session start
- Preserve the `- **Tool:**`, `- **Date:**`, and `- **Summary:**` lines so
  startup preflight remains compatible
- Preserve `## Next Steps` so `project_pulse.py` can still surface the baton

## Immediate Use

Apply the compaction before resuming the MNQ autonomous discovery loop so the
runner spends iteration budget on bounded candidate discovery and verification,
not handoff housekeeping.
