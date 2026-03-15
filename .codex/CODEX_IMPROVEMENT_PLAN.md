# Codex Improvement Plan

This is the practical backlog implied by the official OpenAI Codex guidance and the current state of this repo.

## Priority 1: Durable Guidance Quality

- Keep `AGENTS.md` short, accurate, and operational.
- Add or refine rules only after repeated real friction.
- Keep repo-wide rules in `AGENTS.md`, not scattered through `.codex/`.

## Priority 2: Better Skill Coverage

Likely repo-local skills to add next:

- live-trading audit skill
- validation / verification skill
- research-scrutiny skill
- operator-status or monitoring skill
- safe rebuild / audit pass skill

Rule:

- if the same workflow needs the same steering more than once, convert it into a skill

Current progress:

- Added `canompx3-verify`
- Added `canompx3-audit`
- Added `canompx3-research`

## Priority 3: Keep Context Thin But Smart

- Maintain the existing orientation pack:
  - `.codex/PROJECT_BRIEF.md`
  - `.codex/CURRENT_STATE.md`
  - `.codex/NEXT_STEPS.md`
- Keep these as summaries, not mirrors of the canonical docs.
- Review them when the project meaningfully shifts.

## Priority 4: External Context Only When Justified

Add MCP only when it removes a real manual loop. Candidate future MCP categories:

- official docs
- issue tracking
- CI / PR review surfaces
- production or monitoring surfaces

Do not add tools just because they exist.

## Priority 5: Future Automation

Only automate after a workflow is stable manually.

Good future automation candidates:

- audit summaries
- CI failure triage
- change review summaries
- recurring project-health checks

## Review Cadence

When Codex friction appears:

1. Ask whether this is a static rule, a repeatable workflow, or missing external context.
2. Update the right layer:
   - static rule -> `AGENTS.md`
   - repeatable workflow -> skill
   - external changing context -> MCP
   - runtime behavior -> `.codex/config.toml`
3. Record the durable lesson in memory.
