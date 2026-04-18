# Codex Maintenance Automations

These are Codex-only automation templates for maintaining the Codex operator
layer. They are intentionally report-only.

## Rules

- Do not edit repo-tracked files.
- Do not edit `CLAUDE.md`, `.claude/`, `claude.bat`, or Claude-owned settings.
- Output recommendations only.
- If a recommendation affects both tools, point to `HANDOFF.md` or
  `docs/plans/`.

## Weekly Operator Health Review

- Goal:
  - inspect the Codex operator layer for stale docs, broken setup assumptions,
    launcher drift, and WSL/app workflow regressions
- Suggested cadence:
  - weekly
- Prompt:
  - review `CODEX.md`, `docs/reference/codex-operator-handbook.md`,
    `docs/reference/codex-claude-operator-setup.md`, `.codex/config.toml`,
    and `scripts/infra/codex_*` / `scripts/infra/codex-*`
  - verify the documented primary path is still WSL-home clone plus Codex app
  - verify native Windows is still documented as fallback-only
  - report stale instructions, broken commands, duplicated guidance, and any
    new friction
  - do not edit files; recommend the smallest durable fix only

## Repeated-Friction Review

- Goal:
  - turn repeated Codex mistakes into the right durable fix
- Suggested cadence:
  - weekly or after a run of frustrating sessions
- Prompt:
  - review recent Codex friction and classify each issue as one of:
    `AGENTS.md`, skill, `.codex/config.toml`, script/launcher, or no change
  - recommend only Codex-owned fixes unless the user explicitly requested
    shared-layer changes
  - do not edit files; produce a short action list with rationale
