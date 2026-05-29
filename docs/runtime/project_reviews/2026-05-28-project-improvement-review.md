# Project Improvement Review

- Generated: `2026-05-29T00:06:39+10:00`
- Scope: `recent`
- Since ref: `HEAD~10`
- Git context: `OK`
- Branch: `codex/audit-hermesstyle-agent-idea`
- Scanned files: `11`

## Source-of-truth integrity
- No deterministic v1 finding.

## Git/worktree hygiene
- **MEDIUM** `git worktree` (HIGH)
  - Evidence: C:/Users/joshd/.codex/worktrees/4984/canompx3, C:/Users/joshd/.codex/worktrees/9deb/canompx3, C:/Users/joshd/.codex/worktrees/a683/canompx3
  - Why: Dirty sibling worktrees can hide parallel edits or stale context.
  - Patch: Inspect sibling worktree status before broad refactors or commits.
  - Test: Run `git worktree list --porcelain` and targeted sibling `git status` checks.
  - Stop: Stop if sibling work touches the same files.

## Test/CI gaps
- No deterministic v1 finding.

## Research integrity
- **MEDIUM** `RESEARCH_RULES.md` (LOW)
  - Evidence: trading hypothesis language without nearby pre-reg/K accounting token
  - Why: Static pattern risk: hypothesis discussion should route to pre-registration and K accounting.
  - Patch: Convert any proposed trading improvement into a pre-reg draft before testing.
  - Test: Add a fixture or hypothesis-file test if this is an executable research path.
  - Stop: Stop if the hypothesis is already being tested or promoted.

## Code quality
- No deterministic v1 finding.

## Literature/resource grounding
- No deterministic v1 finding.

## Workflow speed
- **LOW** `scripts/tools/context_resolver.py` (HIGH)
  - Evidence: context resolver route not part of v1 scan
  - Why: The current task had no deterministic route during planning; v1 should remain tool-only until report quality is proven.
  - Patch: After several useful reports, add a context_resolver route for this review task.
  - Test: Add resolver tests only in the follow-up route patch.
  - Stop: Stop if the route would duplicate or override canonical Claude/Codex rules.

## Highest-EV next action
- **MEDIUM** `git worktree` (HIGH)
- Rationale: Dirty sibling worktrees can hide parallel edits or stale context.
- Patch: Inspect sibling worktree status before broad refactors or commits.
- Tests: Run `git worktree list --porcelain` and targeted sibling `git status` checks.
- Stop condition: Stop if sibling work touches the same files.

## Verification commands
```powershell
uv run python scripts/tools/project_improvement_review.py
uv run python pipeline/check_drift.py --fast
uv run python -m ruff check . --quiet
```

> Report-only output. Static pattern risks are not semantic proof; verify with targeted commands before claiming fixes.
