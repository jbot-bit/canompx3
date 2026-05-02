# Passive Sidecar Policy Clearance Checklist

This checklist is the go/no-go gate between "support replied" and "allowed to
flip `LIVE_PASSIVE_SIDECAR_ALLOWED=true` for controlled testing."

## Inputs

Required artifacts:

- `docs/plans/active/2026-05/2026-05-02-topstep-verification-questions.md`
- `docs/support-requests/draft-live-readonly-sidecar-policy-clarification.md`
- raw outbound request record
- raw Topstep written reply

Do not use memory, paraphrase, or chat summaries as evidence.

## Stage 1: provenance check

All must be true:

- the reply is written, not verbal
- the source is Topstep support or another clearly attributable Topstep channel
- the ticket/reference number is captured if available
- the raw response text is saved verbatim in the repo

If any item fails, status remains `BLOCKED`.

## Stage 2: question coverage

All three questions must have an answer:

- Q1 — Live read-only assistive tooling
- Q2 — scope of connected tools in Live
- Q3 — order-authority feedback expectations

If any question is skipped, status remains `BLOCKED`.

## Stage 3: answer quality

Treat as acceptable only if the response is explicit enough to classify.

Acceptable:

- `Yes` with restrictions
- `No`

Not acceptable:

- “should be fine”
- “that sounds okay”
- generic product language
- links without a direct answer
- a restatement of API capability without Live policy interpretation

If any answer is not explicit, status remains `BLOCKED`.

## Stage 4: boundary matching

The approved tool must still match the tool we asked about:

- local-device only
- non-executing
- passive dashboards / alerts / monitoring / reconciliation only
- no placing orders
- no cancelling orders
- no modifying orders

If Topstep's answer changes those boundaries, the current implementation is not
automatically approved. Status becomes `REDESIGN REQUIRED` or `BLOCKED`.

## Stage 5: classification matrix

### Clear

Mark `CLEARED FOR CONTROLLED NON-LIVE TESTING` only if:

- Q1 = yes
- Q2 = yes
- Q3 = yes or yes-with-restrictions compatible with passive monitoring
- no restriction contradicts the current sidecar design

### Blocked

Mark `BLOCKED` if:

- Q1 = no
- Q2 = no
- any answer prohibits passive monitoring in `Live Funded`
- any answer limits permission to non-Live stages only

### Redesign required

Mark `REDESIGN REQUIRED` if:

- the answer is positive but only under constraints the current sidecar does
  not satisfy
- the answer permits some passive functions but not others needed by the
  current design

### Ambiguous

Mark `BLOCKED PENDING FOLLOW-UP` if:

- any answer is vague
- channels conflict
- written language is internally inconsistent

## Stage 6: post-clearance actions

Only after `CLEARED FOR CONTROLLED NON-LIVE TESTING`:

1. update the official-doc handover with the exact classification
2. create a short clearance memo with:
   - `VERIFIED`
   - `INFERRED`
   - `UNSUPPORTED`
3. only then consider enabling `LIVE_PASSIVE_SIDECAR_ALLOWED=true`
4. first testing scope must remain controlled and non-live

## Explicit non-actions

Even with clearance, do not:

- send orders from the sidecar
- add order-cancel or modify paths
- widen the sidecar into copy trading
- treat policy clearance as permission for future automation ideas
