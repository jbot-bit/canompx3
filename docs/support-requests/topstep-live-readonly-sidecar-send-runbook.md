# Topstep Live Read-Only Sidecar Send Runbook

This runbook governs the outbound policy-clarification request for the passive,
non-executing TopstepX sidecar.

## Purpose

Send exactly one clean support request to Topstep, preserve the raw evidence,
and avoid contaminating the policy record with paraphrase, scope creep, or
mixed-channel ambiguity.

## Authority chain

1. `docs/plans/active/2026-05/2026-05-02-topstep-verification-questions.md`
   is the sole source of unresolved policy questions.
2. `docs/support-requests/draft-live-readonly-sidecar-policy-clarification.md`
   is the exact outbound wording.
3. Topstep is the primary authority for `Live Funded` permission.
4. ProjectX may clarify protocol capability, but does not overrule Topstep
   policy for `Live Funded`.

## Pre-send checklist

Before sending, verify all of the following:

- the current implementation branch is `codex/passive-sidecar-skeleton`
- the passive sidecar remains blocked by default
- the draft still asks only Q1-Q3
- no additional architecture ideas have been mixed into the request
- the human has approved sending something off-machine

## Channel selection

Preferred order:

1. Topstep support ticket or email that yields a durable written reply
2. Topstep chat only if a full transcript can be copied and saved

Do not open parallel requests on multiple channels on day one.

## Send rules

- send the draft unchanged
- do not add Quantower, Sierra, MotiveWave, replay, trade copier, or future
  automation questions
- do not ask for “general guidance”
- do not let support redirect the request into marketing language
- ask for the explicit response format:
  - `Q1: Yes/No + restrictions`
  - `Q2: Yes/No + restrictions`
  - `Q3: Yes/No + restrictions`

## Evidence capture

Immediately after sending, create:

- `docs/support-requests/topstep-live-readonly-sidecar-outbound-2026-05-02.md`
  if the request is sent today; otherwise use the actual send date

That file must contain:

- channel used
- send timestamp with timezone
- exact body sent
- any ticket/reference number
- any support agent name if shown

When a reply arrives, create:

- `docs/support-requests/topstep-live-readonly-sidecar-response-YYYY-MM-DD.md`

That file must contain:

- full raw reply text
- timestamps
- ticket/reference number
- screenshot/transcript note if the response came via chat
- no interpretation inline beyond minimal provenance notes

## Follow-up rules

Only send a follow-up if one of these is true:

- no reply after a reasonable waiting window
- the reply does not answer Q1-Q3
- the reply is ambiguous marketing copy instead of a policy answer

Any follow-up must:

- quote the original ticket/reference number
- restate only the unanswered question(s)
- avoid introducing new scope

## Stop conditions

Stop and keep the path blocked if:

- Topstep answers with ambiguity instead of policy language
- support refuses to answer in writing
- different Topstep channels conflict
- the answer narrows permission to non-Live stages only

If conflicting written answers arrive, do not choose the convenient one. Save
both and escalate the conflict into the policy-clearance checklist.
