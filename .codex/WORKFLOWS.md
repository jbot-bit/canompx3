# Codex Workflows

This file captures the repo-specific execution defaults Codex should use after
the thin startup pass.

## First Principle

Codex is another operator on the same project, not a separate project
environment. Claude remains the authority for project workflow and guardrails.
Codex should adapt to that authority, not reshape it.

## When To Open This File

- Execution defaults are unclear
- You need repo-specific verification defaults
- You are choosing between implementation, review, research, or audit routes
- You are improving the Codex layer itself

## Preferred Repo-Local Skills

When the task matches, prefer these repo-local skills:

- `canompx3-workspace` for workspace routing and onboarding
- `canompx3-verify` for post-edit verification
- `canompx3-audit` for structured audit work
- `canompx3-research` for research, analysis, and claim scrutiny
- `canompx3-live-audit` for live-trading/runtime safety audits
- `canompx3-deploy-readiness` for promotion/deployment go-no-go decisions

## Design / Implementation Routing

Before proposing or implementing any non-trivial change:

1. Check `docs/specs/` for an existing spec.
2. If the task touches entry models, read `docs/prompts/ENTRY_MODEL_GUARDIAN.md`.
3. If the task touches pipeline data flow or schema, read `docs/prompts/PIPELINE_DATA_GUARDIAN.md`.
4. For broad design work, use the `/design` logic in `.claude/skills/design/SKILL.md`.

## Research And Data Routing

Route to the correct source:

- Live data, strategy counts, fitness, schema, trade history:
  - Use the repo's canonical sources first
  - If the shared MCP path is intentionally enabled, follow `.claude/rules/mcp-usage.md`
- Live deployment profiles and what is actually routed to accounts:
  - Use `trading_app/prop_profiles.py`
  - Treat `trading_app/live_config.py` as compatibility or deprecated surface unless current code says otherwise
- Project memory, design history, prior findings:
  - Use `.claude/skills/pinecone-assistant/SKILL.md`
- Academic methodology: use local PDFs in `resources/` (BH FDR, walk-forward, deflated Sharpe).

## Verification Defaults

For normal code changes:

- `uv run python pipeline/check_drift.py`
- `uv run python scripts/tools/audit_behavioral.py`
- `uv run python scripts/tools/audit_integrity.py` when pipeline or trading logic changed
- Run targeted tests before broad tests

For large post-edit verification, use the shared logic indexed in:

- `.codex/COMMANDS.md`
- `.codex/AGENTS.md`

## Review Flows

- Day-to-day implementation: Codex app or `scripts/infra/codex-project.sh`
- Narrow scripted jobs: `codex exec`
- Non-interactive review of current changes: `codex review` or
  `scripts/infra/codex-review.sh`
- Interactive work: `scripts/infra/codex-project.sh`
- Interactive with live web search: `scripts/infra/codex-project-search.sh`
- Heavier coding / review profile: `canompx3_max`

## Codex CLI Surfaces Worth Using

Confirmed from the installed CLI:

- `codex review` for non-interactive review
- `codex exec` for non-interactive task execution
- `codex mcp` for managing Codex MCP registrations
- profiles via `-p`
- live web search via `--search`
- sandbox modes: `read-only`, `workspace-write`, `danger-full-access`
- approval modes: `untrusted`, `on-request`, `never`

Use these intentionally. They do not replace or outrank Claude workflow rules.

For the operator split and recommended day-to-day surface choice, see
`docs/reference/codex-operator-handbook.md`.

## Current Project Profiles

Defined in `.codex/config.toml`:

- `canompx3`: default project profile
- `canompx3_search`: live-search variant
- `canompx3_max`: heavier coding/review profile using `gpt-5.1-codex-max`
