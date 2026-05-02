# 2026-05-02 Topstep Official-Doc Packet Handover

## Scope

Fresh isolated worktree for the official-documentation decision packet:

- Worktree: `/tmp/canompx3-topstep-official-doc-packet`
- Branch: `codex/topstep-official-doc-packet`

## Read First

1. `docs/plans/active/2026-05/2026-05-02-official-doc-decision-packet.md`
2. `docs/plans/active/2026-05/2026-05-02-operator-export-contract.md`
3. `docs/handoffs/2026-05-02-topstep-operator-v3-handover.md`
4. `trading_app/live/bot_state.py`
5. `trading_app/live/session_orchestrator.py`
6. `trading_app/live/projectx/data_feed.py`
7. `trading_app/live/projectx/order_router.py`
8. `trading_app/live/copy_order_router.py`

## Purpose

Produce the official-doc decision packet that determines whether any external
operator surface is even allowed and clean enough to test.

## Hard Boundary

No platform integration code in this worktree.

If a finding is not grounded in repo-local sources or official vendor docs, it
must be marked `NEEDS VERIFICATION`.

## Immediate Next Step

Build the stage-split evidence packet:

- Topstep / TopstepX policy by stage
- ProjectX API constraints
- candidate platform connection / SDK / replay docs
- candidate-by-candidate unsupported assumptions register

## Current Verdict

- Implementation is still blocked at the external-platform layer.
- Cleared: repo-local canonical export contract and operator-state projection.
- Not cleared:
  - explicit Live allowance for a read-only/assist sidecar
  - explicit lifecycle blessing for Quantower as a TopstepX-era shell
  - proof that any shell can avoid split-brain order/account authority
