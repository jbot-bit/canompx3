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
- `canompx3-capital-review` for thorough capital-at-risk review orchestration
  across code, live, deploy, research, evidence, security, threat-model, and
  supply-chain surfaces
- shared `.claude/agents/evidence-auditor.md` when the task is really claim scrutiny or anti-bias review

## Intent-To-Workflow Routing

Codex should route from intent, not from exact wording.

- If the user asks fuzzily for a review, hardening pass, deployment check,
  readiness check, verification, or cleanup of a specific risk, map the ask to
  the nearest valid workflow instead of waiting for the exact skill or command
  name.
- Use the smallest strong route that covers the request:
  - `canompx3-audit` for evidence-first audits and repo-health findings
  - `canompx3-verify` for post-edit verification and done-definition checks
  - `canompx3-live-audit` for runtime, broker, session, or live-control safety
  - `canompx3-deploy-readiness` for go/no-go promotion decisions
  - `canompx3-capital-review` when the ask is broad, adversarial, anti-bias,
    "real capital", "thorough AF", or spans multiple risk surfaces
  - `evidence-auditor.md` for separate-context scrutiny of claims, results, and readiness narratives
  - `research-methodologist.md` / `canompx3_reviewer` for research, OOS, holdout, FDR, DSR, MinBTL, edge, significance, or methodology claims
  - `live-risk-auditor.md` / `canompx3_reviewer` for live, broker, webhook, prop-profile, order execution, kill/flatten, account-routing, or runtime-control work
  - `test-coverage-scout.md` / `canompx3_reviewer` for missing tests, stale tests, coverage, or exact pytest-target selection
  - `canompx3_worker` for one scope-locked implementation task after planning and ownership are clear
  - shared `.claude/skills/` recipes when the task is really a shared command
    flow
- If multiple routes fit, prefer the path with the strongest verification and
  the lowest blast radius, then state the mapping briefly.
- Only ask for clarification when the ambiguity is load-bearing.

## Agent Auto-Routing Hints

The user should not need to remember agent names. When a prompt contains the intent signals below, route automatically:

| Signal | Agent route |
| --- | --- |
| research, backtest, validation, OOS, holdout, FDR, DSR, MinBTL, edge, significant | `research-methodologist` plus `canompx3_reviewer` for separate-context critique |
| live, broker, webhook, account, prop profile, order, execution, kill, flatten, risk | `live-risk-auditor` plus `canompx3_reviewer` |
| coverage, missing tests, stale tests, pytest, verification uncertainty | `test-coverage-scout` |
| non-trivial implementation after plan approval | `canompx3_worker` or Claude `executor`, one writer per owned scope |
| broad review with independent concerns | spawn `canompx3_explorer`, `canompx3_reviewer`, and the relevant specialist; synthesize in the lead |

## Codex Plugin Routing

Use Codex plugins as helpers, not project authority. `CLAUDE.md`, `.claude/`,
`RESEARCH_RULES.md`, `TRADING_RULES.md`, `HANDOFF.md`, and `docs/plans/`
remain canonical.

- `Superpowers`: useful for process discipline. Prefer
  `superpowers:systematic-debugging` for bugs, `superpowers:dispatching-parallel-agents`
  for independent parallel investigations, `superpowers:requesting-code-review`
  for review passes, and `superpowers:verification-before-completion` before
  completion claims. Do not let it create unmanaged durable truth under
  `docs/superpowers/`; use `docs/plans/` and `HANDOFF.md`. Do not use generic
  worktree behavior instead of `scripts/infra/codex-worktree.sh`.
- `GitHub`: useful for PR, review-comment, CI, and publishing workflows. Use it
  only after local repo state and canompx3 verification are clear.
- `CodeRabbit`: useful as an extra external-style review lens on PRs. Treat its
  output as claims to verify, not authority.
- `OpenAI Developer Docs`: use for current OpenAI/Codex/API behavior.
- `CB Insights`, `Stripe`, `Gmail`, and `Google Calendar`: not core canompx3
  development tools. Use only when a task explicitly needs company-market
  research, payments, or personal/admin context.

## Design / Implementation Routing

Before proposing or implementing any non-trivial change:

1. Check `docs/specs/` for an existing spec.
2. If the task touches entry models, read `docs/prompts/ENTRY_MODEL_GUARDIAN.md`.
3. If the task touches pipeline data flow or schema, read `docs/prompts/PIPELINE_DATA_GUARDIAN.md`.
4. For broad design work, use the `/design` logic in `.claude/skills/design/SKILL.md`.

## Terminal Proposal Approval Audit

When Claude or a terminal session emits a large proposal/design plan for user
approval, use `docs/prompts/TERMINAL_PROPOSAL_APPROVAL_AUDIT.md` before
recommending approval.

- Keep the review read-only and scoped to the proposal.
- Read only relevant authority docs, files the proposal says it will touch, and
  local resources/literature needed for methodology or trading-evidence claims.
- List `READ` / `NOT READ` sources and mark unsupported claims as `UNSUPPORTED`.
- Decide `APPROVE` / `MODIFY` / `BLOCK`; default to `MODIFY` for fixable
  grounding gaps and `BLOCK` for unsafe mutation, hidden blast radius, stale
  truth, or unclear stop gates.
- Do not substitute this prompt for a full system audit, deploy-readiness
  review, post-result sanity pass, or research validation workflow.

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
- For claim-heavy work, downgrade conclusions unless they are explicitly tagged or supportable as `MEASURED`; `INFERRED` and `UNSUPPORTED` are valid end states, not failures.
- For research pipeline status, use
  `docs/institutional/research_pipeline_contract.md`: discovered, confirmed,
  validated, deployed, and executed are separate claims.
- Route research/runtime questions to one primary option before acting:
  `standalone_discovery`, `conditional_role`, `confirmation`,
  `deployment_readiness`, or `operations`.
- For new-edge discovery, noisy ideas, chart reads, or hypothesis triage, use
  `docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md` as the front door before
  any scan, prereg, or implementation talk.
- The human-facing interface is natural language. When the user asks to find,
  test, classify, validate, or audit an idea, Codex should run the repo tooling
  internally and report the result rather than asking the user to call scripts.
- Once a prereg exists, use the prereg front door internally to inspect the
  route before execution. Execute only after confirming the branch is correct:
  - `standalone_edge` -> `experimental_strategies` -> validator -> `validated_setups`
  - `conditional_role` -> bounded runner/result doc -> explicit role decision
- For `confirmation`, `deployment_readiness`, and `operations` questions, do
  not use the prereg front door as the authority. Use the validator / validated
  shelf, deployment profile checks, or runtime / `paper_trades` evidence
  respectively.
- For institutional research review prompts, prefer the compact runtime rubric in `docs/prompts/INSTITUTIONAL_RESEARCH_REVIEW_MINI.md` instead of re-pasting long prompt blocks.

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
- Capital-at-risk review of current changes:
  `scripts/infra/codex-capital-review.sh`
- Interactive work: `scripts/infra/codex-project.sh`
- Interactive with live web search: `scripts/infra/codex-project-search.sh`
- Heavier coding / review profile: `canompx3_power`

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

For WSL launcher sessions, define these in user-level `~/.codex/config.toml`:

- `canompx3`: default project profile
- `canompx3_search`: live-search variant
- `canompx3_power`: heavier coding/review profile
