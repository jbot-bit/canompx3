# Codex Integrations

This repo already has shared runtime integrations. Codex should use them, not recreate them.

## Primary Rule

Codex uses the same project as Claude Code and defers to Claude's project rules.

- Same repo
- Same files
- Same git history
- Same canonical DB if the project still uses one
- No Codex-only project copy
- No Codex-only database
- No Codex-owned override of Claude integrations or Claude workflow surfaces

## MCP

Declared in `.mcp.json`:

- `repo-state`
  - Shared repo MCP declaration via `python scripts/tools/repo_state_mcp_server.py` (matches the canonical `gold-db` python-direct pattern for Windows/macOS/Linux portability)
  - Codex launchers attach the same server via `bash scripts/infra/run-repo-state-mcp.sh` for WSL profiles
  - Read-only control-plane surface for task routing, startup packets, system context, pulse, and generated context views
  - Appropriate as a default repo-understanding MCP because it does not create a second truth layer
- `research-catalog`
  - Shared repo MCP declaration via `python scripts/tools/research_catalog_mcp_server.py` (matches the canonical `gold-db` python-direct pattern for Windows/macOS/Linux portability)
  - Codex launchers attach the same server via `bash scripts/infra/run-research-catalog-mcp.sh` for WSL profiles
  - Read-only research-grounding surface for local literature extracts, prereg hypotheses, audit results, and bounded catalog search
  - Appropriate as a default research-grounding MCP because it reads committed repo evidence instead of inventing a second state layer
- `gold-db`
  - Shared repo MCP declaration
  - Optional project integration, not required for Codex to work in this repo
- `code-review-graph`
  - Shared repo MCP declaration via `uvx code-review-graph serve`
  - Structural navigation and review aid only, not a project-truth layer
  - Use minimal-detail flows; canonical code and `gold.db` still win on truth

Project-scoped Codex MCP declarations also exist in `.codex/config.toml`:

- `openaiDeveloperDocs`
- `repo-state` is attached by the Codex project/review launchers by default
  - this is the default self-understanding surface for Codex sessions in this repo
- `research-catalog` is attached by the Codex project/review launchers by default
  - this is the default local grounding surface for literature, prereg, and audit-result work
- `gold-db` is deliberately not attached by default
  - use `scripts/infra/codex-project-gold-db.sh` or
    `scripts/infra/codex-project-search-gold-db.sh` when a task actually needs
    the trading DB via MCP
- `code-review-graph` stays repo-scoped in `.mcp.json`
  - keep it opt-in for structural code navigation and review, not routine truth-finding

## Claude Runtime Surfaces Worth Reusing

- `.claude/settings.json`: shared plugin and hook configuration
- `.claude/settings.local.json`: local permissions and enabled MCP-json servers
- `.claude/hooks/`: post-edit and pre-edit safety hooks
- `.claude/agent-memory/`: shared agent memory for reusable workflows

## Local Codex Runtime Facts

- Codex binary: `/home/joshd/.nvm/versions/node/v24.14.0/bin/codex`
- User config: `/home/joshd/.codex/config.toml`
- Current default model: `gpt-5.4`
- Project trust is already set for this repo
- `codex mcp list` now includes `openaiDeveloperDocs`
- Enabled curated plugin in the current local config: `GitHub`
- Disabled curated plugins in the current local config: `gmail`, `google-calendar`, `build-web-apps`

## Current Codex CLI Capabilities Confirmed Locally

- `codex review` for non-interactive code review
- `codex exec` for non-interactive runs
- `codex mcp add|get|list|remove`
- profiles via `-p`
- live web search via `--search`
- `--full-auto` alias for `workspace-write` plus `on-request`

## Important Distinction

Claude and Codex do not share MCP registration automatically.

- Claude reads the repo’s `.mcp.json` plus Claude settings.
- Codex keeps its own MCP registrations in the user Codex config.
- This repo defines project-scoped Codex profiles in `.codex/config.toml`.
- Optional MCP integrations should stay minimal and must not imply a second project or second DB.
- Codex-side integrations must never supersede Claude rules or shared repo behavior.

## Repo Execution Defaults

- Sync environment: `uv sync --frozen`
- Run drift: `uv run python pipeline/check_drift.py`
- Run tests: `uv run python -m pytest tests/ -x -q`
- Run lint: `uv run ruff check pipeline/ trading_app/ ui/ scripts/ tests/`
- Run type checks: `uv run pyright`

## Practical Rule

If Claude already has a working integration surface for this repo, Codex should point at it rather than creating a competing layer.

Minimal-by-default beats clever-by-default. Optional repo MCPs should be
attached only for sessions that need them.

Current intended split:

- default self-understanding: `repo-state`
- default research grounding: `research-catalog`
- explicit trading-data truth: `gold-db`
- explicit structural code navigation: `code-review-graph`
- explicit official-doc lookup: `openaiDeveloperDocs`

Do not mutate Claude integration surfaces unless the user explicitly asks.
