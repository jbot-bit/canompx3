# Claude Token Hygiene

Keep Claude cheap by default and rigorous by verification, not by running the
highest reasoning tier on every prompt.

## Core Model

- Default to medium reasoning for exploration, routine edits, grep/read work,
  and small localized fixes.
- Escalate to high reasoning when the bug is unclear, the blast radius spans
  multiple subsystems, or the work touches money, live routing, dates, schema,
  concurrency, or deployment safety.
- Reserve the highest-cost review path for deploy readiness, real-capital
  safety, adversarial review, or when two cheaper passes still disagree.

High reasoning can improve difficult reasoning. It does not replace reading the
right files, querying the right truth source, or running verification.

## Cheap Execution, Expensive Verification

Use a two-tier operating model:

1. Cheap execution:
   - keep the prompt narrow
   - read only the smallest relevant context
   - implement with the normal reasoning tier unless the task is already risky
2. Expensive verification:
   - run targeted tests, drift checks, and static gates
   - run a higher-rigor review pass only on the risky paths
   - use a fresh context or independent reviewer when the decision matters

This is usually safer than paying for maximum reasoning on the full session.

## Subagents And Teams

- Subagents can save tokens when a side task would otherwise flood the main
  thread with logs, search results, or broad file dumps.
- Subagents waste tokens when they are used for work the main thread could do
  in one or two reads.
- Keep subagent prompts narrow and returned summaries short.
- Do not use agent teams by default. Each teammate has its own context cost.

Good subagent use:
- scan logs and return only failures
- compare two candidate files and report the behavioral delta
- audit one risky path independently after the main change is done

Bad subagent use:
- explore the whole repo and summarize everything
- spawn multiple workers for a tiny localized fix
- keep teammates idle during a mostly serial task

## Session Habits That Save Real Tokens

- Use a fresh session or `/clear` between unrelated tasks.
- Prefer exact prompts over broad brainstorm dumps.
- Keep startup memory small and stable.
- Keep path-scoped rules path-scoped; do not move everything into always-on
  files.
- Archive stale stage files so orientation stays compact.
- Use CLI or local inspection when that is enough; web or remote lookups should
  be deliberate, not reflexive.
- Commit small logical checkpoints. They usually reduce ambiguity and repeated
  review rather than increasing token usage.

## Worktrees And Private Context

Gitignored repo-root files such as `SOUL.md`, `USER.md`, and `memory/` are a
poor home for required startup context in multi-worktree setups. New worktrees
do not reliably inherit them.

Prefer:

- `~/.claude/CLAUDE.md` for user-level personal preferences that should follow
  you across worktrees
- `CLAUDE.local.md` only for preferences intentionally local to one worktree

Keep private memory concise. If it becomes a life story, every session pays for
it.

## How To Inspect The Current Setup

Run:

```bash
python3 scripts/tools/token_hygiene_report.py
```

What it checks:

- current Claude reasoning defaults
- whether agent teams are enabled
- whether the prompt-tier escalation hook is configured
- startup doc size in characters and lines
- always-on versus path-scoped rule count
- active stage-file count
- whether user-level Claude memory exists

For live session measurement, compare Claude's own context view in a fresh
session before and after touching a high-doctrine path such as `pipeline/`.
