# 2026-05-02 Topstep Official-Doc Packet Handover

## Scope

Fresh isolated worktree for the official-documentation decision packet:

- Worktree: `/tmp/canompx3-topstep-official-doc-packet`
- Branch: `codex/topstep-official-doc-packet`

## Read First

1. `docs/plans/active/2026-05/2026-05-02-official-doc-decision-packet.md`
2. `docs/plans/active/2026-05/2026-05-02-topstep-verification-questions.md`
3. `docs/plans/active/2026-05/2026-05-02-operator-export-contract.md`
4. `docs/handoffs/2026-05-02-topstep-operator-v3-handover.md`
5. `trading_app/live/bot_state.py`
6. `trading_app/live/session_orchestrator.py`
7. `trading_app/live/projectx/data_feed.py`
8. `trading_app/live/projectx/order_router.py`
9. `trading_app/live/copy_order_router.py`

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
- use the saved verification-questions doc for any outbound support request

## Current Verdict

- Status: blocked on policy clarification.
- Implementation is still blocked at the external-platform layer.
- Cleared: repo-local canonical export contract and operator-state projection.
- New hard disconfirmation from official Topstep docs:
  - TopstepX accounts cannot be connected to other trading platforms.
  - This materially weakens `TopstepX + Quantower shell` as a same-account
    lifecycle candidate.
- New artifacts prepared on the implementation branch:
  - branch: `codex/passive-sidecar-skeleton`
  - `docs/support-requests/draft-live-readonly-sidecar-policy-clarification.md`
  - `docs/support-requests/topstep-live-readonly-sidecar-send-runbook.md`
  - `trading_app/live/passive_sidecar/`
  - `docs/operator/passive-sidecar-enforcement.md`
  - `docs/operator/deprecated-paths.md`
  - `docs/operator/policy-clearance-checklist.md`
- Not cleared:
  - explicit Live allowance for a read-only/assist sidecar
  - proof that any shell can avoid split-brain order/account authority

## Next Step

1. Get human approval to send the Topstep support-request draft.
2. Obtain written policy clarification on Q1-Q3 from the verification questions doc.
3. Only after written clarification, consider flipping
   `LIVE_PASSIVE_SIDECAR_ALLOWED=true` for controlled non-live testing.

`docs/plans/active/2026-05/2026-05-02-topstep-verification-questions.md`
remains the single source of truth for unresolved policy items.
