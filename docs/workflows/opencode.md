# OpenCode Third-POV Workflow

OpenCode is an optional third coding and review perspective for canompx3. It is
not a project authority layer, runtime orchestrator, queue, MCP server, or
replacement for Claude/Codex.

## Authority

OpenCode must follow the same source-of-truth chain as every other agent:

1. Current repo code and canonical data surfaces.
2. `TRADING_RULES.md` for trading semantics.
3. `RESEARCH_RULES.md` and local institutional docs for research claims.
4. `CLAUDE.md` and `.claude/` for repo workflow and guardrails.
5. `CODEX.md` and `.codex/` only as Codex adapter guidance.
6. `HANDOFF.md` and `docs/plans/` for cross-tool session context, never as
   runtime truth when code or DB disagree.

## Modes

### Plan / Read-Only Mode

Use this by default. OpenCode may inspect files, run read-only commands, map
blast radius, and produce a review report. It must not edit files, write DBs,
commit, push, merge, or create durable state.

Required opening checks:

```powershell
git status --short --branch
git diff --name-only
git ls-files -u
git log --oneline -8
python scripts/tools/context_resolver.py --task "<task>" --format markdown
```

### Build Mode

Build mode is allowed only when the operator explicitly asks for it and the work
is inside a managed disposable worktree:

```powershell
python scripts/tools/worktree_manager.py create --tool opencode --name "<task>" --purpose "OpenCode third-POV build"
```

OpenCode must open that worktree, re-run the opening checks there, and keep all
edits scoped to the task. It must not commit, push, merge, or ship the branch.
Claude or Codex reviews and ships any resulting work.

## Stop Conditions

Stop and report instead of continuing when any of these are true:

- `git ls-files -u` prints anything.
- The current worktree has unrelated dirty files.
- The task needs live/capital truth and `project_pulse.py --fast` blocks.
- The task needs drift-clean proof and `pipeline/check_drift.py --fast` blocks.
- The request touches a forbidden path without explicit later approval.
- The source-of-truth chain is ambiguous.
- OpenCode is asked to install external runtimes, vendor repos, create queues,
  create daemons, add MCP servers, write DBs, or bypass Claude/Codex authority.

## Forbidden By Default

OpenCode must not edit these surfaces without a fresh explicit approval and a
Claude/Codex review gate:

- `gold.db`, `*.db`, `*.duckdb`, market-data files, or canonical data artifacts.
- `pipeline/` canonical data code.
- `research/` executable research code.
- `docs/audit/results/`, `docs/audit/hypotheses/`, and pre-registration files.
- `trading_app/live/`, `trading_app/live_config.py`, `trading_app/prop_profiles.py`.
- `trading_app/config.py` and strategy/discovery/validation logic.
- `docs/runtime/lane_allocation*.json` and action queues.

## Evidence Format

OpenCode outputs must include:

- `READ`: exact files and docs read.
- `COMMANDS`: exact commands run and pass/fail status.
- `FINDINGS`: evidence-grounded issues with file paths and line references when
  possible.
- `UNSUPPORTED`: claims it cannot verify from the repo.
- `CHANGES`: only when in build mode, exact files changed and why.
- `STOPPED`: any stop condition hit and the safest next action.

## Commit Gate

When a launcher or operator sets `OPENCODE_AGENT_ACTIVE=1`, the pre-commit hook
runs `scripts/tools/claude_review_deepseek.py`. The legacy name is intentional:
older hooks already reference it. The gate is deterministic and blocks OpenCode
commits that stage protected canompx3 truth surfaces.
