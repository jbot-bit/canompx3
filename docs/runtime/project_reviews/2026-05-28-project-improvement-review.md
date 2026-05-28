# Project Improvement Review

- Generated: `2026-05-28T23:26:05+10:00`
- Scope: `recent`
- Since ref: `HEAD~10`
- Git context: `UNKNOWN/BLOCKED CONTEXT`
- Branch: `DETACHED`
- Scanned files: `28`

## State Notes
- fatal: ref HEAD is not a symbolic ref

## Source-of-truth integrity
- No deterministic v1 finding.

## Git/worktree hygiene
- **HIGH** `git` (HIGH)
  - Evidence: UNKNOWN/BLOCKED CONTEXT
  - Why: Git context is degraded; review conclusions may miss branch/workstream intent.
  - Patch: Attach the work to an intentional branch before commit/push work, or keep this as report-only.
  - Test: Run `git status --short --branch` and `git log --oneline -10`.
  - Stop: Stop if you need to commit or publish from a detached or unknown context.
- **MEDIUM** `git worktree` (HIGH)
  - Evidence: C:/Users/joshd/.codex/worktrees/4984/canompx3, C:/Users/joshd/.codex/worktrees/9deb/canompx3, C:/Users/joshd/.codex/worktrees/a683/canompx3
  - Why: Dirty sibling worktrees can hide parallel edits or stale context.
  - Patch: Inspect sibling worktree status before broad refactors or commits.
  - Test: Run `git worktree list --porcelain` and targeted sibling `git status` checks.
  - Stop: Stop if sibling work touches the same files.
- **LOW** `git` (MEDIUM)
  - Evidence: docs/runtime/project_reviews/, scripts/tools/project_improvement_review.py, tests/test_tools/test_project_improvement_review.py
  - Why: Untracked review/runtime artifacts can be accidentally missed or accidentally committed.
  - Patch: Decide whether each artifact is disposable, ignored, or intentionally derived output.
  - Test: Run `git status --short` before staging.
  - Stop: Stop if an artifact is used as evidence but remains untracked.

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
- **MEDIUM** `trading_app/live/bot_dashboard.py` (MEDIUM)
  - Evidence: except Exception:
  - Why: Static pattern risk: broad exception handling may swallow failures without a fail-closed result.
  - Patch: Narrow the exception or return an explicit blocked/error state with evidence.
  - Test: Add a failure-mode unit test for the exception path.
  - Stop: Stop if the broad handler wraps trading, DB, git, or verification state.
- **MEDIUM** `trading_app/live/session_orchestrator.py` (MEDIUM)
  - Evidence: except Exception as e:
  - Why: Static pattern risk: broad exception handling may swallow failures without a fail-closed result.
  - Patch: Narrow the exception or return an explicit blocked/error state with evidence.
  - Test: Add a failure-mode unit test for the exception path.
  - Stop: Stop if the broad handler wraps trading, DB, git, or verification state.
- **MEDIUM** `trading_app/strategy_validator.py` (MEDIUM)
  - Evidence: except Exception as e:
  - Why: Static pattern risk: broad exception handling may swallow failures without a fail-closed result.
  - Patch: Narrow the exception or return an explicit blocked/error state with evidence.
  - Test: Add a failure-mode unit test for the exception path.
  - Stop: Stop if the broad handler wraps trading, DB, git, or verification state.
- **LOW** `trading_app/live/bot_dashboard.py` (LOW)
  - Evidence: .date() # Use Brisbane date — not system local (matters at NYSE_OPEN midnight crossing) sessions = [] for name,
  - Why: Static pattern risk: canonical session/instrument/profile literals can drift when duplicated.
  - Patch: Verify this literal is fixture-only or delegated to the canonical registry.
  - Test: Run the companion tests for the touched module.
  - Stop: Stop if the literal controls production routing or live strategy eligibility.
- **LOW** `trading_app/strategy_validator.py` (LOW)
  - Evidence: : python trading_app/strategy_validator.py --instrument MGC python trading_app/strategy_validator.py --instrument M
  - Why: Static pattern risk: canonical session/instrument/profile literals can drift when duplicated.
  - Patch: Verify this literal is fixture-only or delegated to the canonical registry.
  - Test: Run the companion tests for the touched module.
  - Stop: Stop if the literal controls production routing or live strategy eligibility.

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
- **HIGH** `git` (HIGH)
- Rationale: Git context is degraded; review conclusions may miss branch/workstream intent.
- Patch: Attach the work to an intentional branch before commit/push work, or keep this as report-only.
- Tests: Run `git status --short --branch` and `git log --oneline -10`.
- Stop condition: Stop if you need to commit or publish from a detached or unknown context.

## Verification commands
```powershell
uv run python scripts/tools/project_improvement_review.py
uv run python pipeline/check_drift.py --fast
uv run python -m ruff check . --quiet
```

> Report-only output. Static pattern risks are not semantic proof; verify with targeted commands before claiming fixes.
