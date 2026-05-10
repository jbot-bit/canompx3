# Terminal Proposal Approval Audit

Use this before approving a large Claude or terminal proposal/design dump.

This is a pre-approval gate, not a full system audit, deployment review, or
post-result sanity pass. Keep it scoped to the proposal.

```text
Stop. Review this proposal before I approve it.

Mode: token-efficient grounded proposal audit.

Goal:
Decide APPROVE / MODIFY / BLOCK without wasting tokens or trusting unsupported claims.

Rules:
- Do not write code.
- Do not broad-scan the repo unless needed.
- Read only:
  1. project authority docs relevant to this task
  2. files the proposal says it will touch
  3. resources/literature needed for methodology claims
- Actually read contents, not filenames/metadata.
- List READ / NOT READ sources.
- If unsupported, say UNSUPPORTED.

Check:
1. Is the problem real?
2. Is the proposed fix the simplest safe fix?
3. Any hidden blast radius?
4. Any stale assumption?
5. Any contradiction with project rules?
6. Any token-heavy / overengineered step?
7. Any safer lower-blast-radius alternative?
8. Are STOP gates clear enough?

Output only:
- Verdict: APPROVE / MODIFY / BLOCK
- Why
- Required changes before approval
- Risks
- Smallest safe next step

Be skeptical, concise, resource-grounded, and fail-closed.
```

## Scope Rules

- Use `MODIFY` for a useful proposal with fixable unsupported claims, unclear
  stop gates, stale assumptions, or avoidable blast radius.
- Use `BLOCK` when approval would authorize unsafe mutation, deployment, hidden
  blast radius, stale truth, missing required canon/resources, or unclear stop
  gates.
- Use `APPROVE` only when the proposal is bounded, grounded, low-blast-radius,
  and has explicit stop gates.
- Read local resources or literature only when the proposal makes methodology,
  statistical, market-structure, or trading-evidence claims.
- Do not treat this prompt as code truth, live trading truth, research truth, or
  runtime state.
