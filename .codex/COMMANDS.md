# Codex Command Routing

The shared slash-command equivalents live in `.claude/commands/`. Use them as task recipes.

## High-Value Commands

- `discover.md`: structured read-first exploration
- `health-check.md`: repo health and validation sweep
- `validate-instrument.md`: instrument validation workflow
- `rebuild-outcomes.md`: outcome rebuild workflow
- `quant-debug.md`: debugging trading logic or validation issues
- `quant-tdd.md`: test-driven implementation workflow
- `quant-verify.md`: verification workflow for quant changes
- `blast-radius.md`: change impact analysis
- `verify-complete.md`: done-definition enforcement
- `audit.md`, `audit-quick.md`, `audit-phase.md`: audit flows
- `ralph.md`: autonomous Ralph loop entrypoint

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

- Start with `discover.md` when the repo area is unclear.
- Use `blast-radius.md` before edits that touch shared interfaces.
- Use `quant-verify.md` or `verify-complete.md` before closing any material implementation task.
- Use `health-check.md` when the right validation command is unclear.

## Codex Note

Treat these as shared runbooks. Keep new Codex-specific command guidance here only when it is about routing, not duplicated command content.
