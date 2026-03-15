# OpenAI Codex Standards

This file distills official OpenAI Codex guidance into repo-specific rules for building stronger project awareness and better long-term Codex performance.

Claude remains the canonical authority for this repo. This file exists to keep the Codex layer aligned with official OpenAI guidance, not to create a second rule system.

## Primary Sources

- AGENTS guidance:
  - https://developers.openai.com/codex/guides/agents-md/
- Best practices:
  - https://developers.openai.com/codex/learn/best-practices/
- Customization model:
  - https://developers.openai.com/codex/concepts/customization/
- Skills:
  - https://developers.openai.com/codex/skills/
- Config basics:
  - https://developers.openai.com/codex/config-basic/
- Advanced config:
  - https://developers.openai.com/codex/config-advanced/
- Config reference:
  - https://developers.openai.com/codex/config-reference/

## OpenAI's Core Model

OpenAI's recommended order is:

1. Durable repo guidance in `AGENTS.md`
2. Repo and user config in `config.toml`
3. Skills for repeatable workflows
4. MCP for external context
5. Automations for stable recurring work

These layers are complementary, not competing.

## Rules For This Repo

### 1. Keep `AGENTS.md` small and practical

Use `AGENTS.md` for:

- build, test, lint, and verification commands
- repo layout and routing guidance
- constraints and do-not rules
- what "done" means

Do not use it as a giant project encyclopedia.

### 2. Fix repeated mistakes by updating durable guidance

Per OpenAI guidance, when Codex makes the same mistake more than once:

- update `AGENTS.md` if the rule is repo-wide
- add nested guidance only if the rule is local to a subtree
- add a skill if the fix is really a repeatable workflow rather than a static rule

### 3. Prefer progressive disclosure over startup bloat

OpenAI explicitly recommends keeping startup guidance concise and moving richer workflows into skills.

For this repo:

- keep startup docs thin
- use orientation summaries for fast awareness
- load deeper docs only when the task actually needs them
- avoid copying large portions of canonical Claude docs into `.codex/`

### 4. Use skills for repeatable procedures

If a workflow keeps reappearing, it should become a skill rather than a long repeated prompt.

Good candidates here:

- audit flows
- live-trading safety reviews
- strategy-validation routines
- rebuild / verification routines
- research-output review against `RESEARCH_RULES.md`

### 5. Use MCP only for real external loops

Official guidance says MCP should be used when context lives outside the repo or changes frequently.

For this repo:

- do not add MCP servers just because they are possible
- add them only when they remove a real repeated manual loop
- keep the MCP set small and high-value

### 6. Keep config at the right layer

OpenAI recommends:

- user defaults in `~/.codex/config.toml`
- project-specific behavior in `.codex/config.toml`
- one-off changes via CLI overrides

For this repo, project config should stay focused on:

- profiles
- sandbox and approval defaults
- small repo-specific runtime settings

It should not become a second instruction system.

### 7. Plan and verify, especially in large repos

OpenAI's best-practices guidance emphasizes:

- clear goal
- relevant context
- constraints
- explicit done criteria

And then:

- tests
- checks
- review
- confirmation of final behavior

This aligns with the existing Claude workflow and should remain the default Codex operating style.

## Practical Standards For "Superbrain" Behavior

To improve Codex project awareness without degrading quality:

- Keep one thin repo-orientation layer in `.codex/`
- Keep canonical truth in `CLAUDE.md`, `TRADING_RULES.md`, `RESEARCH_RULES.md`, and the real code
- Encode recurring workflows as skills
- Encode recurring static rules in `AGENTS.md`
- Avoid repeating large docs across multiple files
- Prefer fewer, better summaries over more files
- Review and refine guidance after repeated friction

## Anti-Patterns

These go against the official guidance and should be avoided:

- gigantic startup docs
- multiple competing authority layers
- copying canonical docs into `.codex/`
- adding MCP integrations with no repeated workflow behind them
- keeping repeatable workflows as ad hoc prompts forever
- turning unstable workflows into automations too early

## Repo Implication

The correct target state for this repo is:

- Claude stays canonical
- Codex stays thin but well-oriented
- skills handle repeatable jobs
- config handles runtime defaults
- optional MCP handles external systems only when justified

That is the highest-signal way to make Codex more capable here without creating drift.
