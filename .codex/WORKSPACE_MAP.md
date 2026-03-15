# Codex Workspace Map

Use this as the fast navigation layer for the repo.

## Core Paths

- `pipeline/`: ingestion, aggregation, feature building, drift checks, DB path helpers
- `trading_app/`: outcome builder, validation, live trading, broker adapters, AI utilities
- `scripts/`: audits, reports, infra, rebuild helpers, utilities
- `tests/`: pipeline, trading app, UI, scripts, research coverage
- `docs/`: specs, plans, prompts, strategy docs, architecture references
- `.claude/`: canonical shared rules, commands, agents, hooks, settings
- `.codex/`: Codex-only adapter layer
- `scripts/infra/`: repo-local operator launchers, including Codex wrappers

## First Places To Look

- Current orientation: `.codex/PROJECT_BRIEF.md`
- Current project snapshot: `.codex/CURRENT_STATE.md`
- Highest-signal open work: `.codex/NEXT_STEPS.md`
- Architecture overview: `docs/ARCHITECTURE.md`
- Project roadmap: `ROADMAP.md`
- File inventory: `REPO_MAP.md`
- Existing specs: `docs/specs/`
- Guardian prompts: `docs/prompts/`
- Trading rules: `TRADING_RULES.md`
- Research rules: `RESEARCH_RULES.md`
- MCP declarations: `.mcp.json`

## Hot Paths By Task

### Pipeline work

- `pipeline/`
- `.claude/rules/pipeline-patterns.md`
- `.claude/rules/daily-features-joins.md`
- `.claude/rules/validation-workflow.md`

### Trading runtime or live execution

- `trading_app/live/`
- `trading_app/execution_engine.py`
- `trading_app/live_config.py`
- `TRADING_RULES.md`

### Research and validation

- `trading_app/strategy_discovery.py`
- `trading_app/strategy_validator.py`
- `scripts/walkforward/`
- `RESEARCH_RULES.md`

### Audits and autonomous flows

- `scripts/audits/`
- `docs/prompts/`
- `plugins/ralph-wiggum/README.md`
- `.claude/agents/ralph-*.md`

### Codex operator surfaces

- `.codex/`
- `.codex/PROJECT_BRIEF.md`
- `.codex/CURRENT_STATE.md`
- `.codex/NEXT_STEPS.md`
- `.codex/config.toml`
- `scripts/infra/codex-project.sh`
- `scripts/infra/codex-project-search.sh`
- `scripts/infra/codex-review.sh`

## Execution Defaults

- Prefer `uv run python ...` over bare `python ...` when dependencies matter.
- Prefer `rg` for search and `rg --files` for file discovery.
- Prefer read-only MCP for DB inspection before writing ad hoc SQL.
