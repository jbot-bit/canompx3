# Codex Supercharge Roadmap

Date: 2026-05-01
Owner surface: Codex layer for `canompx3`
Scope: skills, hooks, MCPs, plugins, and local runtime surfaces that improve
project self-understanding, market/research grounding, and repeatable operator
work without creating a second truth system.

## Goal

Make Codex better at:

- understanding the repo's current state without cold-start wandering
- grounding strategy and market reasoning in canonical local evidence
- using external integrations only when they remove a real manual loop
- staying thin, token-efficient, and subordinate to the canonical Claude layer

## MEASURED Current State

- Repo-local Codex skills were maintained in `.codex/skills/`, but official
  Codex repo-skill discovery is `.agents/skills/`.
  Status: FIXED in this session via thin discovery wrappers in `.agents/skills/`.
- Project-scoped Codex config now includes repo-local WSL hooks for
  `SessionStart` and `UserPromptSubmit`.
  Status: FIXED in this session; hooks inject route/grounding context only when
  needed.
- The review-max profile was still pinned to `gpt-5.1-codex-max`.
  Status: FIXED in this session; `canompx3_max` now points to `gpt-5.3-codex`.
- Live repo MCP declarations are:
  - `repo-state` in `.mcp.json`
  - `research-catalog` in `.mcp.json`
  - `gold-db` in `.mcp.json`
  - `code-review-graph` in `.mcp.json`
  - `openaiDeveloperDocs` in `.codex/config.toml`
- `repo-state` MCP has now landed.
  Status: FIXED in this session via `scripts/tools/repo_state_mcp_server.py`,
  launcher wiring, tests, and `.mcp.json` registration.
- `research-catalog` MCP has now landed.
  Status: FIXED in this session via
  `scripts/tools/research_catalog_mcp_server.py`, launcher wiring, tests, and
  `.mcp.json` registration.
- Current local curated plugin state from `~/.codex/config.toml`:
  - enabled: `GitHub`
  - disabled: `gmail`, `google-calendar`, `build-web-apps`
- Current checkout is a `/mnt/c/...` fallback surface with no `.venv-wsl`.
  WSL mount guard and preflight both fail from this checkout.
  Status: BLOCKER for ideal Codex usage; repo should be operated from a
  WSL-home clone such as `~/canompx3`.

## What To Keep

### ADOPT

- `gold-db` MCP as the canonical live trading/research query surface.
  Why: already wired, read-only, and project-truth aligned.
- `code-review-graph` MCP as a structural navigation/review aid.
  Why: useful for blast radius and code search, but it must remain non-truth.
  Rule: keep `detail_level="minimal"` and never use CRG as market/research truth.
- `openaiDeveloperDocs` as the default external-doc MCP for Codex/OpenAI
  feature changes.
  Why: official-source grounding for unstable Codex/OpenAI config surfaces.
- Curated GitHub plugin.
  Why: already enabled, high-ROI for PR, CI, issue, and review workflows.

## What To Build Next

### BUILD: `repo-state` MCP

Status: LANDED

Use existing local scripts and truth-classed views to expose read-only project
state as tools instead of long prompts.

Primary backing surfaces:

- `scripts/tools/context_resolver.py`
- `scripts/tools/task_route_packet.py`
- `scripts/tools/project_pulse.py`
- `scripts/tools/system_context.py`
- `scripts/tools/context_views.py`

Initial tool set:

- `resolve_task_route(task_text)`
- `get_project_pulse(fast: bool = true)`
- `get_system_context(context_name, action, mode)`
- `get_context_view(view_name)`
- `get_startup_packet(task_text, briefing_level)`

Expected payoff:

- lower cold-start token cost
- better repo self-awareness
- less drift between docs, handoff, queue, and runtime state

### BUILD: `research-catalog` MCP

Status: LANDED

Expose the repo's research canon and audit artifacts through read-only tools.

Primary backing surfaces:

- `docs/institutional/`
- `docs/audit/hypotheses/`
- `docs/audit/results/`
- `scripts/tools/render_context_catalog.py`
- Pinecone snapshot outputs and manifests where they already exist

Initial tool set:

- `list_literature_sources()`
- `get_literature_excerpt(source_id, section_hint)`
- `list_open_hypotheses()`
- `get_audit_result(result_id)`
- `search_research_catalog(query)`

Expected payoff:

- stronger local grounding for strategy claims
- less dependence on stale summaries
- faster truth-check loops for research/readiness work

### BUILD: `strategy-lab` MCP

Expose read-only strategy-validation and deployment-readiness summaries from the
canonical code and DB.

Primary backing surfaces:

- `trading_app/strategy_validator.py`
- `trading_app/strategy_fitness.py`
- `trading_app/lane_allocator.py`
- `docs/runtime/lane_allocation.json`
- `gold-db` as the underlying truth query surface

Initial tool set:

- `get_strategy_readiness(strategy_id)`
- `get_lane_allocation_summary(profile_name)`
- `get_recent_fitness(instrument, rolling_months)`
- `list_promotable_candidates()`

Expected payoff:

- better deploy/readiness triage
- cleaner separation between discovery, validation, and operations questions

### BUILD: repo-local Codex plugin bundle

Package the repo's Codex-facing layer into one reusable local plugin after the
next two MCPs exist.

Bundle target:

- `.agents/skills/` wrappers
- `.codex/hooks/`
- MCP registrations for `openaiDeveloperDocs`, optional `gold-db`, and the new
  local MCPs above

Reason:

- consistent onboarding across worktrees and machines
- less manual setup drift
- cleaner separation between repo-owned Codex tooling and user-global config

## External Integrations To Pilot Carefully

### PILOT: Context7 for dependency / API upgrade tasks

Reason:

- official Codex MCP docs use Context7 as the example developer-doc MCP
- useful for dependency upgrades, library docs, and API surface changes
- does not overlap with `gold-db` or research truth

Constraint:

- use it for dependency/docs upgrade work only
- do not let it become a market/research truth surface

## PARK / KILL

### PARK: MotherDuck MCP

Reason:

- current handoff explicitly says it is out of scope unless the separate eval
  comes back ADOPT
- risk of splitting truth away from canonical local `gold.db`

### KILL: NotebookLM revival for this Codex layer

Reason:

- current repo stance is retired
- local PDFs and project canon are the approved grounding path

### KILL: TradingView puppet / remote-debug MCP

Reason:

- already rejected in `docs/superpowers/specs/2026-04-11-dashboard-embedded-chart-signal-overlay.md`
- no official server, side-project risk, poor truth economics

### PARK: broad market/news connector spray

Reason:

- external context should follow workflow, not vibe
- only add once there is a specific loop such as event-gating, catalyst audit,
  or scheduled macro calendar triage with a clear canonical output

## Implementation Order

1. Move active Codex work to a WSL-home clone and create `.venv-wsl`.
2. Keep the current minimal-by-default MCP stance: `repo-state` for default
   self-understanding, `research-catalog` for default local grounding,
   `openaiDeveloperDocs` for official docs, `gold-db` and CRG by task.
3. Build `strategy-lab` MCP next, using `gold-db` as its truth substrate.
4. Package the resulting repo-owned layer as a local Codex plugin.
5. Pilot Context7 only after the local MCPs above are stable.

## Non-Negotiables

- No second truth DB.
- No MCP that silently outranks canonical code, `gold.db`, or the
  institutional docs.
- No plugin or MCP added just because the platform supports it.
- No market-data or strategy-research connector without an explicit workflow,
  output contract, and truth boundary.
