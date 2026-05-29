# Project Improvement Review

- Generated: `2026-05-30T02:17:59+10:00`
- Scope: `recent`
- Since ref: `HEAD~10`
- Git context: `OK`
- Branch: `main`
- Scanned files: `66`

## Source-of-truth integrity
- No deterministic v1 finding.

## Git/worktree hygiene
- **MEDIUM** `git worktree` (HIGH)
  - Evidence: C:/Users/joshd/.codex/worktrees/3de6/canompx3, C:/Users/joshd/.codex/worktrees/9deb/canompx3, C:/Users/joshd/.codex/worktrees/c5cd/canompx3
  - Why: Dirty sibling worktrees can hide parallel edits or stale context.
  - Patch: Inspect sibling worktree status before broad refactors or commits.
  - Test: Run `git worktree list --porcelain` and targeted sibling `git status` checks.
  - Stop: Stop if sibling work touches the same files.
- **LOW** `git` (MEDIUM)
  - Evidence: docs/audit/results/2026-05-30-project-review-oos-integrity-audit.md
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
- **MEDIUM** `.claude/hooks/session-start.py` (MEDIUM)
  - Evidence: except Exception:
  - Why: Static pattern risk: broad exception handling may swallow failures without a fail-closed result.
  - Patch: Narrow the exception or return an explicit blocked/error state with evidence.
  - Test: Add a failure-mode unit test for the exception path.
  - Stop: Stop if the broad handler wraps trading, DB, git, or verification state.
- **MEDIUM** `.claude/hooks/shell-canon-guard.py` (MEDIUM)
  - Evidence: except Exception:
  - Why: Static pattern risk: broad exception handling may swallow failures without a fail-closed result.
  - Patch: Narrow the exception or return an explicit blocked/error state with evidence.
  - Test: Add a failure-mode unit test for the exception path.
  - Stop: Stop if the broad handler wraps trading, DB, git, or verification state.
- **MEDIUM** `.claude/hooks/stage-awareness.py` (MEDIUM)
  - Evidence: except Exception:
  - Why: Static pattern risk: broad exception handling may swallow failures without a fail-closed result.
  - Patch: Narrow the exception or return an explicit blocked/error state with evidence.
  - Test: Add a failure-mode unit test for the exception path.
  - Stop: Stop if the broad handler wraps trading, DB, git, or verification state.
- **MEDIUM** `scripts/run_live_session.py` (MEDIUM)
  - Evidence: except Exception:
  - Why: Static pattern risk: broad exception handling may swallow failures without a fail-closed result.
  - Patch: Narrow the exception or return an explicit blocked/error state with evidence.
  - Test: Add a failure-mode unit test for the exception path.
  - Stop: Stop if the broad handler wraps trading, DB, git, or verification state.
- **MEDIUM** `scripts/tools/build_resources_index.py` (MEDIUM)
  - Evidence: except Exception:
  - Why: Static pattern risk: broad exception handling may swallow failures without a fail-closed result.
  - Patch: Narrow the exception or return an explicit blocked/error state with evidence.
  - Test: Add a failure-mode unit test for the exception path.
  - Stop: Stop if the broad handler wraps trading, DB, git, or verification state.
- **MEDIUM** `scripts/tools/session_preflight.py` (MEDIUM)
  - Evidence: except Exception:
  - Why: Static pattern risk: broad exception handling may swallow failures without a fail-closed result.
  - Patch: Narrow the exception or return an explicit blocked/error state with evidence.
  - Test: Add a failure-mode unit test for the exception path.
  - Stop: Stop if the broad handler wraps trading, DB, git, or verification state.
- **MEDIUM** `trading_app/ai/query_agent.py` (MEDIUM)
  - Evidence: except Exception as e:
  - Why: Static pattern risk: broad exception handling may swallow failures without a fail-closed result.
  - Patch: Narrow the exception or return an explicit blocked/error state with evidence.
  - Test: Add a failure-mode unit test for the exception path.
  - Stop: Stop if the broad handler wraps trading, DB, git, or verification state.
- **LOW** `scripts/run_live_session.py` (LOW)
  - Evidence: FIRM + broker creds. INSTRUMENT SELECTION: --instrument MGC Single instrument --all All active
  - Why: Static pattern risk: canonical session/instrument/profile literals can drift when duplicated.
  - Patch: Verify this literal is fixture-only or delegated to the canonical registry.
  - Test: Run the companion tests for the touched module.
  - Stop: Stop if the literal controls production routing or live strategy eligibility.
- **LOW** `scripts/tools/live_readiness_report.py` (LOW)
  - Evidence: gacy_lane_allocation_path() DEFAULT_TELEMETRY_INSTRUMENT = "MNQ" DEFAULT_SIGNALS_DIR = PROJECT_ROOT LIVE_STAGE_PATHS: tuple
  - Why: Static pattern risk: canonical session/instrument/profile literals can drift when duplicated.
  - Patch: Verify this literal is fixture-only or delegated to the canonical registry.
  - Test: Run the companion tests for the touched module.
  - Stop: Stop if the literal controls production routing or live strategy eligibility.

## Literature/resource grounding
- No deterministic v1 finding.

## Workflow speed
- **LOW** `docs/plans/2026-05-03-repo-hygiene-tidy-plan.md` (LOW)
  - Evidence: thout adding a measured note. ### Minimal operator prompt (copy/paste) ```text Run Phase <N> of docs/plans/2026-05-03-repo-hygie
  - Why: Static pattern risk: repeated manual operator prompting may be better captured as a script or skill.
  - Patch: If this workflow has repeated twice, extract the deterministic steps into a small tool or skill.
  - Test: Add a smoke test for any new deterministic tool.
  - Stop: Stop if the manual step controls verification or git hygiene.
- **LOW** `pipeline/check_drift.py` (HIGH)
  - Evidence: file exceeds 500000 bytes
  - Why: The reviewer skipped this file, so absence of findings is not proof the file is clean.
  - Patch: Inspect the skipped file manually if it is in the active workstream.
  - Test: Add a narrow fixture if this file type should be scanned automatically.
  - Stop: Stop if the skipped file is a production or research authority surface.
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
