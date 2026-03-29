# Codex Command Routing

The shared slash-command equivalents live in `.claude/skills/`. Use them as task recipes.

## High-Value Commands

- `discover/SKILL.md`: structured read-first exploration
- `health-check/SKILL.md`: repo health and validation sweep
- `validate-instrument/SKILL.md`: instrument validation workflow
- `rebuild-outcomes/SKILL.md`: outcome rebuild workflow
- `quant-debug/SKILL.md`: debugging trading logic or validation issues
- `quant-tdd/SKILL.md`: test-driven implementation workflow
- `quant-verify/SKILL.md`: verification workflow for quant changes
- `blast-radius/SKILL.md`: change impact analysis
- `verify-done/SKILL.md`: done-definition enforcement (or dispatch `.claude/agents/verify-complete.md`)
- `audit/SKILL.md`: audit flows (full, quick, phase)
- `ralph/SKILL.md`: autonomous Ralph loop entrypoint

## Codex-Native CLI Shortcuts

- Interactive project session:
  - `scripts/infra/codex-project.sh`
- Interactive project session with live web search:
  - `scripts/infra/codex-project-search.sh`
- Max-capability review profile:
  - `scripts/infra/codex-review.sh`
- Review current changes:
  - `codex -p canompx3_max review --uncommitted`
- Non-interactive task execution:
  - `codex exec`
- Non-interactive review:
  - `codex review`

## Selection Heuristic

- Start with `discover/SKILL.md` when the repo area is unclear.
- Use `blast-radius/SKILL.md` before edits that touch shared interfaces.
- Use `quant-verify/SKILL.md` or `verify-done/SKILL.md` before closing any material implementation task.
- Use `health-check/SKILL.md` when the right validation command is unclear.

## Codex Note

Treat these as shared runbooks. Keep new Codex-specific command guidance here only when it is about routing, not duplicated command content.
