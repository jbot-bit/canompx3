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

---

## 2026-06-05 — Recurrence root cause + compaction trigger (Claude)

**Why this plan came back:** the baton re-accreted to 182 lines / ~18 stacked
`## ...` blocks, despite the header's own "Compact baton only" rule. The plan
file itself was a casualty of the bulk doc-reorg `074e2f83` (2026-04-28, moved
~30 plans into `archive/` subdirs) — recovered here from history (`a1ac60c8`).

**Structural root cause (the part that makes accretion inevitable):** the
`.githooks/post-commit` hook only rewrites the **`## Last Session`** block. Its
regex is:

```python
pattern = r'## Last Session\n.*?(?=\n## (?!Last Session))'
```

That replaces exactly one block — the one between `## Last Session` and the next
`## ` heading. Every **other** `## ...` block (the cross-tool Codex/Claude
follow-ups) is therefore immortal: the hook never prunes them, and each session
that adds a follow-up block grows the file monotonically. The hook is doing its
narrow job correctly; nothing ever does the *wide* job of pruning. So the baton
grows without bound until a human (or this tool) compacts it.

**Compaction trigger (the rule, now explicit):** compact when ANY of:
- `HANDOFF.md` exceeds ~120 lines, OR
- more than ~3 `## ` session/follow-up blocks have accreted beyond
  `## Last Session`, OR
- session start spends visible token budget reading stale follow-ups.

**What "compact" keeps (the intended thin baton):**
- the header (rule / CRITICAL / "Compact baton only")
- the live `## Last Session` block
- (optionally) the single most-recent cross-tool follow-up block if it is still
  actionable next-step context
- the `## Durable References` list

Everything older migrates to `docs/handoffs/archived/<date>-*.md` (move, never
delete — durable decisions are *also* mirrored in `decision-ledger.md`).

**Optional hardening (advisory size-warning):** a post-commit soft warning when
the baton exceeds the line threshold was evaluated 2026-06-05. Decision: **not
wired** — the post-commit hook is on the capital-adjacent commit path and the
existing copy already documents (lines 9-25) how a tracked-file write in that
hook deadlocked a rebase (`feedback_post_commit_hook_mutating_tracked_file_deadlocks_git_rebase`).
Adding even an advisory `HANDOFF.md` write there re-opens that exact failure
class for marginal benefit. The manual/tool trigger above is the chosen
mechanism; `compact_handoff.py` is the one-command fix. If a future recurrence
proves the manual trigger is forgotten too often (n≥2), revisit a **read-only**
warning that prints to stderr without writing any tracked file.
