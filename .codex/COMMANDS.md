# Codex Command Routing

The shared slash-command equivalents live in `.claude/skills/`. Use them as task recipes.

## High-Value Commands

- `discover/SKILL.md`: discovery front door for new edge ideas and hypothesis triage
- Natural-language discovery requests route through
  `docs/institutional/research_pipeline_contract.md` and `discover/SKILL.md`;
  Codex should run the internal prereg tooling itself instead of asking the
  user to remember scripts.
- Internal prereg guard: `scripts/infra/prereg-loop.sh --hypothesis-file <yaml>`
  inspects a locked prereg's execution route.
- Internal prereg execution: `scripts/infra/prereg-loop.sh --hypothesis-file <yaml> --execute`
  runs a locked prereg through the correct branch (`experimental_strategies` for
  `standalone_edge`, bounded runner for `conditional_role`).
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
- Interactive project data session with `gold-db` MCP:
  - `scripts/infra/codex-project-gold-db.sh`
  - `scripts/infra/codex-project-search-gold-db.sh`
- Max-capability review profile:
  - `scripts/infra/codex-review.sh`
- Capital-at-risk review profile:
  - `scripts/infra/codex-capital-review.sh`
- Review current changes:
  - `codex -p canompx3_max review --uncommitted`
- Non-interactive task execution:
  - `codex exec`
- Non-interactive review:
  - `codex review`

## Selection Heuristic

- Do not wait for the user to name the exact command or skill if the intended
  workflow is already clear.
- Start with `discover/SKILL.md` when the user is exploring a new edge, noisy
  idea, chart read, or hypothesis direction.
- Once a prereg exists, use the internal prereg front door instead of manually
  assembling `strategy_discovery.py` or bounded-runner commands. Do not expose
  script syntax as the required user workflow.
- Use `blast-radius/SKILL.md` before edits that touch shared interfaces.
- Use `quant-verify/SKILL.md` or `verify-done/SKILL.md` before closing any material implementation task.
- Use `health-check/SKILL.md` when the right validation command is unclear.
- Use `canompx3-capital-review` or `scripts/infra/codex-capital-review.sh`
  when review scope can affect real capital, deployment, live execution,
  research truth, credentials, external inputs, CI, or supply chain.

## Codex Note

Treat these as shared runbooks. Keep new Codex-specific command guidance here only when it is about routing, not duplicated command content.
