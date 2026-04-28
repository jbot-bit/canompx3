---
status: active
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# 2026-04-23 Multi-Terminal Git Guardrails

## Problem

The repo already has worktree, claim, handoff, and pulse machinery, but the
protection is fragmented. In practice, two recurring failure modes still leak:

1. multi-terminal mutation on the same branch/root causes ambiguous git state
2. research/result artifacts get staged or committed without a durable repo
   home, so the facts exist but the system forgets them

## External Grounding

Official / semi-official references used:

- Git `worktree` docs:
  - linked worktrees exist specifically so one repository can support multiple
    working trees at once
  - `worktree add` refuses to reuse a branch already checked out elsewhere
    unless forced
  - clean linked worktrees should be removed with `git worktree remove`
  - source: `https://git-scm.com/docs/git-worktree`
- Git `githooks` docs:
  - `pre-commit` is the correct hook to inspect current state and abort a
    commit with non-zero exit
  - source: `https://git-scm.com/docs/githooks`
- Git `status` docs:
  - machine parsers should use porcelain / `-z` stable formats
  - source: `https://git-scm.com/docs/git-status`
- GitHub branch docs:
  - feature/topic branches isolate work from other changes
  - source: `https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches`
- Atlassian Rovo worktree mode:
  - isolated worktree session, active lock, and explicit keep/remove behavior
    is a modern operator pattern for AI + multi-session work
  - source: `https://support.atlassian.com/rovo/docs/use-worktree-mode-in-rovo-dev-cli/`

## Decision

Adopt a repo-native flow, not an external terminal manager:

1. keep `session_preflight.py` + claims as the session-start control plane
2. add `session_router.py` at launcher time so a second mutating terminal from
   the main checkout is auto-routed into a managed worktree instead of
   co-editing the same root
3. keep a cheap commit-time `checkpoint_guard.py` for the two leaks that still
   matter at save time:
   - staged result artifacts without a durable closeout surface
   - claim-visible same-branch mutating conflicts
4. wire that guard into `.githooks/pre-commit`

## Why This Design

- An outside app can open tabs, but it cannot understand repo claims, branch
  conflicts, task names, handoff state, or managed worktree metadata.
- The repo already knows how to isolate work correctly; the missing piece was
  auto-routing at launch instead of relying on the operator to remember a
  separate command.
- Commit-time guardrails remain narrow and cheap because the stronger
  concurrency protection now happens before editing starts.

## Current Flow

At session start:

- `scripts/infra/codex-project.sh` calls `session_router.py`
- if the main checkout branch already has a fresh mutating claim, the new
  mutating session is auto-routed into a managed worktree
- if the session already starts inside a linked worktree, do not nest again

At commit time:

If a commit stages `docs/audit/results/**`, it must also stage at least one
durable closeout surface:

- `HANDOFF.md`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/**`

If the repo can prove multiple fresh mutating claims on the same branch at
commit time, the commit is blocked until one session is finished, handed off,
or isolated.

## Edge Cases

- Taskless second terminal: auto-create a timestamped parallel workstream name.
- Launch from existing worktree: keep that root; do not create nested
  worktrees automatically.
- Commit hook conflict detection stays claim-visible only. It does not pretend
  to infer terminals that bypass repo launchers or overwrite the same claim.
