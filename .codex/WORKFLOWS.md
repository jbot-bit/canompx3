# Codex Workflows

This file captures the repo-specific workflows Codex should default to after the deeper audit pass.

## First Principle

Codex is another operator on the same project, not a separate project environment.
Claude remains the authority for project workflow and guardrails.

## Orientation Pass

When starting a fresh session or switching into a new repo area, load:

1. `.codex/PROJECT_BRIEF.md`
2. `.codex/CURRENT_STATE.md`
3. `.codex/NEXT_STEPS.md` if the task is about priorities, planning, or "where are we up to?"
4. `.codex/OPENAI_CODEX_STANDARDS.md` if the task is about Codex setup quality or repo-wide Codex improvements

Then load the smallest relevant canonical docs for the task itself.

## Preferred Repo-Local Skills

When the task matches, prefer these repo-local skills:

- `canompx3-workspace` for workspace routing and onboarding
- `canompx3-verify` for post-edit verification
- `canompx3-audit` for structured audit work
- `canompx3-research` for research, analysis, and claim scrutiny

## Feature Or Design Work

Before proposing or implementing any non-trivial change:

1. Check `docs/specs/` for an existing spec.
2. If the task touches entry models, read `docs/prompts/ENTRY_MODEL_GUARDIAN.md`.
3. If the task touches pipeline data flow or schema, read `docs/prompts/PIPELINE_DATA_GUARDIAN.md`.
4. For broad design work, use the `/4t` logic in `.claude/skills/4t/SKILL.md`.

## Research And Data Routing

Route to the correct source:

- Live data, strategy counts, fitness, schema, trade history:
  - Use the repo's canonical sources first
  - If the shared MCP path is intentionally enabled, follow `.claude/rules/mcp-usage.md`
- Project memory, design history, prior findings:
  - Use `.claude/rules/pinecone-assistant.md`
- Academic methodology:
  - Use local PDFs in `resources/`
  - Follow `.claude/rules/notebooklm.md`

NotebookLM is retired in this repo. Do not route methodology work there.

## Verification Defaults

For normal code changes:

- `uv run python pipeline/check_drift.py`
- `uv run python scripts/tools/audit_behavioral.py`
- `uv run python scripts/tools/audit_integrity.py` when pipeline or trading logic changed
- Run targeted tests before broad tests

For large post-edit verification, use the shared logic indexed in `.codex/COMMANDS.md` and `.codex/AGENTS.md`.

## Review Flows

- Interactive work: `scripts/infra/codex-project.sh`
- Interactive with live web search: `scripts/infra/codex-project-search.sh`
- Non-interactive review of current changes: `scripts/infra/codex-review.sh`
- Heavier coding / review profile: `canompx3_max`

## Current Codex-Native Surfaces Worth Using

Confirmed from the installed CLI:

- `codex review` for non-interactive review
- `codex exec` for non-interactive task execution
- `codex mcp` for managing Codex MCP registrations
- profiles via `-p`
- live web search via `--search`
- sandbox modes: `read-only`, `workspace-write`, `danger-full-access`
- approval modes: `untrusted`, `on-request`, `never`

Use these intentionally. They do not replace or outrank Claude workflow rules.

## Current Project Profiles

Defined in `.codex/config.toml`:

- `canompx3`: default project profile
- `canompx3_search`: live-search variant
- `canompx3_max`: heavier coding/review profile using `gpt-5.1-codex-max`
