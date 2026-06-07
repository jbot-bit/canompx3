# OpenAI Agent-Ops Upgrade Plan for canompx3

- **Date:** 2026-06-07
- **Class:** decision / implementation roadmap
- **Status:** active plan; not live/runtime truth
- **Owner:** Codex + Claude cross-tool workflow
- **Scope:** Codex/Claude operating layer, OpenAI documentation/Cookbook alignment, agent-eval discipline, repo memory/handoff persistence, and optional tool integrations
- **Non-scope:** live trading behavior, broker credentials, research-result promotion, database writes, model fine-tuning, or replacing the canonical Claude layer

## Authority and alignment

This plan is subordinate to `CLAUDE.md`, `.claude/`, `AGENTS.md`, `CODEX.md`, `TRADING_RULES.md`, and `RESEARCH_RULES.md`. It is a roadmap for improving how our AI operators plan, review, verify, persist, and automate work in this repo. It does not create new trading truth, research truth, or live deployment authority.

The durable implementation state should live here and in follow-up stage docs under `docs/plans/`. The compact session baton belongs in `HANDOFF.md`. Long-term operator preference/memory belongs in `MEMORY.md` only as a short pointer, not as a duplicate of this plan.

## Inputs reviewed

### Official OpenAI / Cookbook inputs

- OpenAI model optimization track: start with evals and only move to prompt/model/fine-tune/distillation changes when measured failure data justifies it.
  <https://developers.openai.com/tracks/model-optimization>
- OpenAI Agents guide: useful for owned agent applications with tools, handoffs, guardrails, tracing, and state; not a reason to replace existing Codex/Claude workflows wholesale.
  <https://developers.openai.com/api/docs/guides/agents>
- OpenAI Computer Use guide: only suitable here for isolated, least-privilege, human-gated, mocked UI QA; never for live broker/control paths without a separate explicit approval plan.
  <https://developers.openai.com/api/docs/guides/tools-computer-use>
- OpenAI Skills + Agents SDK blog: package repeatable procedures as skills with clear descriptions and progressive disclosure.
  <https://developers.openai.com/blog/skills-agents-sdk>
- OpenAI Skyscanner Codex + JetBrains MCP blog: IDE problem inspection and predefined run configs can shorten the edit/test loop when used as optional tooling, not canonical truth.
  <https://developers.openai.com/blog/skyscanner-codex-jetbrains-mcp>
- OpenAI Cookbook repository: practical examples for agents, evals, MCP, structured outputs, coding agents, and Responses/API workflows.
  <https://github.com/openai/openai-cookbook>
- Cookbook evaluation flywheel: annotate failures, build graders, change prompts/tools, and repeat.
  <https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/Building_resilient_prompts_using_an_evaluation_flywheel.md>
- Cookbook MCP eval notebook: grade not only final answers, but also whether the expected tools/MCP servers were used.
  <https://developers.openai.com/cookbook/examples/evaluation/use-cases/mcp_eval_notebook>
- Cookbook structured outputs intro: use strict schemas for artifacts that need deterministic validation.
  <https://cookbook.openai.com/examples/structured_outputs_intro>
- Cookbook coding-agent example: shell/apply_patch/web_search/MCP patterns are powerful, but must be isolated and approval-gated.
  <https://cookbook.openai.com/examples/build_a_coding_agent_with_gpt-5.1>
- Cookbook GPT-5.1 prompting / Codex prompting guidance: use clear tool-use expectations, eval-informed prompt changes, and concise user-facing summaries.
  <https://cookbook.openai.com/examples/gpt-5/gpt5-1_prompting_guide>
  <https://cookbook.openai.com/examples/gpt-5/gpt-5-1-codex-max_prompting_guide>

### User-provided architecture-prompt inputs

The user's architecture prompt list should be integrated as a canompx3-specific design-review playbook, not copied as generic advice. The useful prompt classes are:

1. **Design scalable architecture for X** — use for non-trivial new workflows, dashboards, MCP servers, or orchestration surfaces.
2. **Break down modules required for X** — use before code when blast radius spans scripts, hooks, docs, tests, MCP, and CI.
3. **What design patterns fit this use case?** — use for plugins, adapters, strategy interfaces, broker boundaries, and skill/workflow routing.
4. **Suggest a microservices/module split for this monolith** — adapt as “what repo modules should remain separate, and what must not be merged?”
5. **Design an API contract for this feature** — use for MCP tools, dashboard endpoints, structured artifacts, and local agent interfaces.
6. **Review this architecture for bottlenecks** — use for CI shards, drift checks, slow research runners, dashboard refresh paths, and agent cold-start context.
7. **Add security best practices** — use for GitHub Actions, MCP tools, Computer Use, browser automation, credentials, and live-control surfaces.
8. **What data models should this feature use?** — use for eval fixtures, grader outputs, task-route artifacts, and PR review schemas.
9. **Suggest CI/CD and DevOps setup** — use for read-only Codex PR review, report-only eval jobs, and action labels.
10. **How would you document this system for developers?** — use for onboarding, plan files, skill docs, and operator runbooks.
11. **Chain prompts iteratively** — design → modules → contracts → security → DevOps → docs → bottleneck review → smallest diff.

## Design principle

Treat Codex as an architecture/review partner, not just a code generator. However, do not let generalized architecture advice override this repo's capital-at-risk guardrails. The target behavior is:

1. Ask architecture-shaped questions before large edits.
2. Ground answers in repo files, official docs, and measurable evals.
3. Produce task stubs and smallest-diff implementation slices.
4. Persist decisions in `docs/plans/` and compact baton state in `HANDOFF.md`.
5. Verify with static checks, tests, drift/preflight gates, and eval graders before automation expands.

## Ranked roadmap by ROI

### 0. Persist plan and memory pointers

- **ROI:** very high
- **Diff size:** tiny
- **Risk:** low
- **Adopt now:** yes

This plan itself is the durable project record. `HANDOFF.md` should carry a compact pointer to it, and `MEMORY.md` should carry only a short long-term note that the user's preferred Codex posture is architecture/review/eval partner rather than autocomplete.

**Implementation tasks**

1. Keep this file as the canonical plan.
2. Add a compact session entry to `HANDOFF.md`.
3. Add a concise memory entry to `MEMORY.md`.
4. Do not duplicate the full plan in handoff or memory.

### 1. WSL-portable Codex MCP configuration and audit

- **ROI:** very high
- **Diff size:** small
- **Risk:** medium if tool paths are changed broadly; low if scoped to `.codex/config.toml`
- **Adopt now:** yes

The repo already has WSL-safe MCP runner scripts for repo-state, research-catalog, strategy-lab, and gold-db. Codex should use those from the WSL-home operating surface instead of depending on Windows-absolute paths. MCP usage should also become auditable, following the Cookbook MCP eval pattern.

**Implementation tasks**

1. Add Codex-scoped MCP entries in `.codex/config.toml` for:
   - `repo-state` → `scripts/infra/run-repo-state-mcp.sh`
   - `research-catalog` → `scripts/infra/run-research-catalog-mcp.sh`
   - `strategy-lab` → `scripts/infra/run-strategy-lab-mcp.sh`
   - `gold-db` → `scripts/infra/run-gold-db-mcp.sh`
2. Leave `.mcp.json` Windows/Claude entries alone unless a separate task explicitly includes them.
3. Add `scripts/tools/codex_mcp_config_audit.py` to verify:
   - expected MCP names exist in Codex config,
   - runner scripts exist,
   - Codex WSL MCP entries do not point at `C:/Users/...`,
   - missing scripts fail closed.
4. Add `tests/test_tools/test_codex_mcp_config_audit.py`.
5. Manual verification: start Codex from the WSL-home clone, run `/mcp`, and call a read-only repo-state/project-pulse tool.

### 2. Agent-ops eval flywheel

- **ROI:** very high
- **Diff size:** medium
- **Risk:** low
- **Adopt now:** yes

OpenAI's model-optimization/Cookbook guidance makes evals the control loop. For this repo, the first evals should grade agent behavior, not model intelligence: did it route to the right skill, read the right files, use official docs for OpenAI claims, respect live-safety rules, emit task stubs, and persist handoff/memory when appropriate?

**Implementation tasks**

1. Create `docs/ai/evals/codex_agent_ops_cases.yaml`.
2. Seed at least 20 cases:
   - startup routing,
   - read-only QA review,
   - code implementation,
   - live-trading audit,
   - deploy readiness,
   - research claim review,
   - OpenAI docs/API upgrade,
   - post-change verification,
   - PR review,
   - memory/handoff persistence,
   - architecture design proposal,
   - API/MCP contract design,
   - bottleneck review,
   - security review,
   - documentation plan.
3. Each case should include:
   - `id`,
   - `prompt`,
   - `expected_skill`,
   - `required_sources`,
   - `required_output_markers`,
   - `forbidden_actions`,
   - `expected_mcp_tools`,
   - `risk_class`,
   - `expected_architecture_prompt_class` when applicable.
4. Add `scripts/tools/codex_agent_ops_eval_report.py` as a deterministic report-only renderer.
5. Add tests for schema validity, duplicate IDs, missing required fields, and empty prompt cases.

### 3. Deterministic graders for source/tool discipline

- **ROI:** high
- **Diff size:** medium
- **Risk:** low
- **Adopt now:** yes

The most valuable grader is not “was the answer pretty?” It is “did the agent use the source/tool it was required to use?” This guards against stale memory, unsupported OpenAI claims, and skipping MCP project context.

**Implementation tasks**

1. Add `scripts/tools/codex_agent_ops_graders.py` with pure-Python graders:
   - `grade_required_markers`,
   - `grade_forbidden_actions`,
   - `grade_expected_tools`,
   - `grade_required_citations`,
   - `grade_architecture_review_coverage`.
2. Add fixture outputs/traces under `tests/fixtures/codex_agent_ops/`.
3. Fail when:
   - OpenAI/Codex answers lack official source references,
   - repo-routing answers omit `HANDOFF.md` / `CODEX.md` where required,
   - MCP-required cases lack expected MCP trace evidence,
   - design prompts omit modules/contracts/security/testing/docs when requested,
   - live-risk cases suggest unsafe actions.
4. Add tests under `tests/test_tools/test_codex_agent_ops_graders.py`.

### 4. Architecture-prompt playbook for canompx3 design gates

- **ROI:** high
- **Diff size:** small
- **Risk:** low
- **Adopt now:** yes

The user's prompt list should become a repo-specific prompt playbook. It should be used before non-trivial design/implementation work and as a reviewer checklist after proposals. This is the cleanest way to turn “Codex as architect” into consistent practice without increasing startup bloat.

**Implementation tasks**

1. Create `docs/prompts/codex_architecture_review_playbook.md`.
2. Include these reusable prompt chains:
   - **New feature chain:** design architecture → module split → API/data contracts → security → CI/verification → docs → bottleneck review.
   - **Refactor chain:** current module responsibilities → coupling boundaries → design patterns → migration stages → regression tests → rollback.
   - **MCP/tool chain:** tool contract → input/output schema → permission model → failure mode → eval grader → docs.
   - **Live-safety chain:** control boundary → fail-closed behavior → credential/account isolation → human approval gate → audit artifacts.
   - **Research chain:** hypothesis contract → data model → leakage checks → statistical controls → result artifact → deployment block.
3. Add a short rule: architecture prompts are aids for planning; canonical repo rules and measured evidence win.
4. Link this playbook from the OpenAI agent-ops plan and, in a later smallest-diff change, from `CODEX.md` or `.codex/OPENAI_CODEX_STANDARDS.md`.

### 5. Structured-output schemas for agent artifacts

- **ROI:** high
- **Diff size:** medium
- **Risk:** low
- **Adopt now:** yes

Markdown is good for humans, but JSON schemas make PR reviews, evals, and audit reports machine-checkable. Structured outputs should support, not replace, human-readable summaries.

**Implementation tasks**

1. Add `docs/ai/schemas/agent_finding.schema.json`.
2. Add `docs/ai/schemas/agent_run_summary.schema.json`.
3. Include fields for:
   - severity,
   - claim type (`MEASURED`, `INFERRED`, `UNSUPPORTED`),
   - affected paths,
   - premise/trace/evidence/verdict,
   - task stub,
   - residual risk,
   - skills used,
   - MCP tools used,
   - commands/checks run,
   - blocked reasons.
4. Add `scripts/tools/validate_agent_artifact.py`.
5. Add valid/invalid fixture tests.

### 6. Read-only Codex PR review pilot

- **ROI:** high
- **Diff size:** medium
- **Risk:** medium; keep read-only and label-gated
- **Adopt now:** yes, gated

OpenAI's Codex GitHub Action should first be used for review comments, not auto-fixes. The repo's capital/research/live-risk posture makes read-only review the right first pilot.

**Implementation tasks**

1. Create `.github/codex/prompts/canompx3-pr-review.md`.
2. Add `.github/workflows/codex-pr-review.yml` using the official Codex GitHub Action.
3. Trigger only on pull requests with a label such as `codex-review`.
4. Require read-only review of:
   - correctness bugs,
   - live-trading risk,
   - research leakage,
   - stale volatile facts,
   - fail-open behavior,
   - missing tests,
   - documentation drift.
5. Require findings to include `MEASURED` / `INFERRED` / `UNSUPPORTED` and task stubs.
6. Do not allow auto-patching in the pilot.

### 7. canompx3 OpenAI-docs integration skill

- **ROI:** medium-high
- **Diff size:** small
- **Risk:** low
- **Adopt now:** yes

This exact workflow will recur: crawl official OpenAI docs/Cookbook, map to repo constraints, rank by ROI, produce task stubs, and persist the plan. The system `openai-docs` skill handles official source lookup; a repo skill should handle canompx3 routing and persistence.

**Implementation tasks**

1. Create `.codex/skills/canompx3-openai-docs/SKILL.md`.
2. Create `.agents/skills/canompx3-openai-docs/SKILL.md` as a thin wrapper.
3. Require the skill to:
   - use official OpenAI docs/Cookbook first,
   - cite official sources,
   - map suggestions to existing repo authority files,
   - prefer smallest diff,
   - produce ranked ROI task stubs,
   - persist to `docs/plans/` and update `HANDOFF.md` when writing is allowed.
4. Do not duplicate large OpenAI docs in the skill.

### 8. Safe coding-agent sandbox pattern

- **ROI:** medium-high
- **Diff size:** small
- **Risk:** low if docs-only; higher if automation launches shell tools without gates
- **Adopt now:** yes as documentation

The Cookbook coding-agent pattern is useful, but canompx3 must constrain it to managed worktrees and repo preflight.

**Implementation tasks**

1. Create `docs/workflows/codex-coding-agent-sandbox.md`.
2. State that Cookbook-style shell/apply_patch/MCP agents may only mutate inside:
   - `scripts/infra/codex-worktree.sh open <task>`, or
   - disposable temp workspaces.
3. Require before mutation:
   - `python3 scripts/infra/codex_local_env.py doctor --platform wsl`,
   - `python scripts/tools/session_preflight.py`,
   - `git log --oneline -10`,
   - re-read files to be touched.
4. Forbid broker/live commands, destructive filesystem operations, credential handling, and broad dependency installs without explicit scope.
5. Require `canompx3-verify` mapping for final checks.

### 9. Agent failure-mode taxonomy

- **ROI:** medium
- **Diff size:** small
- **Risk:** low
- **Adopt now:** yes

A taxonomy lets the eval flywheel classify recurring failures and decide whether the fix belongs in `AGENTS.md`, `CODEX.md`, a skill, a hook, a script, an MCP server, or a test.

**Implementation tasks**

1. Create `docs/ai/agent_failure_taxonomy.md`.
2. Seed categories:
   - `startup_context_miss`,
   - `authority_boundary_violation`,
   - `missing_task_stub`,
   - `unsupported_claim`,
   - `stale_volatile_fact`,
   - `wrong_mcp_tool`,
   - `overbroad_diff`,
   - `unsafe_live_path`,
   - `test_claim_without_evidence`,
   - `memory_not_persisted`,
   - `duplicate_skill_routing`,
   - `windows_wsl_path_confusion`,
   - `architecture_review_skipped`,
   - `api_contract_missing`,
   - `security_review_missing`,
   - `documentation_plan_missing`.
3. For each category define:
   - symptom,
   - prevention layer,
   - grader candidate,
   - example remediation.

### 10. Skill discovery de-duplication audit

- **ROI:** medium
- **Diff size:** small/medium
- **Risk:** low
- **Adopt now:** yes

The repo currently exposes canompx3 skills through `.agents/skills/` wrappers and `.codex/skills/` canonical bodies. The intended discovery surface should be explicit and auditable so duplicate skill metadata does not waste context or confuse routing.

**Implementation tasks**

1. Add `scripts/tools/codex_skill_inventory.py`.
2. Report:
   - skill name,
   - discovery path,
   - canonical path,
   - duplicate names,
   - missing wrapper/canonical target.
3. Add `tests/test_tools/test_codex_skill_inventory.py`.
4. Update `.codex/skills/README.md` with the rule: one discoverable wrapper per skill; one canonical body.

### 11. Refresh OpenAI standards index

- **ROI:** medium
- **Diff size:** small
- **Risk:** low
- **Adopt now:** yes

`.codex/OPENAI_CODEX_STANDARDS.md` should list the official docs and Cookbook pages used by this plan so future agents do not rely on memory.

**Implementation tasks**

1. Add a dated `2026-06-07 official docs and Cookbook crawl` section.
2. Group sources by:
   - Codex: AGENTS.md, skills, MCP, GitHub Action, config,
   - Agents SDK: agents, tools, guardrails, evals, observability,
   - Cookbook: coding agent, eval flywheel, MCP eval, structured outputs, prompt optimization,
   - Safety: Computer Use isolation and human-gated high-impact actions,
   - Architecture prompting: user-provided prompt classes adapted to canompx3.
3. Label each group:
   - adopt now,
   - pilot only,
   - defer,
   - forbidden without explicit approval.

### 12. Optional JetBrains MCP workflow

- **ROI:** medium
- **Diff size:** small
- **Risk:** low if optional/operator-local
- **Adopt now:** optional

JetBrains MCP can improve edit/test loops by letting Codex inspect IDE problem state and invoke predefined run configs. It should remain optional and never become a source of trading truth.

**Implementation tasks**

1. Create `docs/workflows/codex-jetbrains-mcp.md`.
2. Scope it as optional/operator-local.
3. Permit:
   - IDE problem inspection,
   - predefined lint/test run configs,
   - code navigation.
4. Forbid:
   - live broker launch,
   - credential handling,
   - bypassing session preflight,
   - treating IDE diagnostics as canonical market/research truth.
5. Link from `.codex/INTEGRATIONS.md`.

### 13. Computer Use dashboard QA pilot, mocked only

- **ROI:** medium
- **Diff size:** plan-only first
- **Risk:** high if mis-scoped; low if mocked-only
- **Adopt now:** plan only

Computer Use can inspect UI flows, but the dashboard can reach live-control surfaces. First use must be mocked dashboard QA with no broker credentials or real accounts.

**Implementation tasks**

1. Create `docs/plans/2026-06-07-computer-use-dashboard-qa-pilot.md`.
2. Scope to mocked localhost dashboard/API flows only.
3. Require:
   - isolated browser profile,
   - no broker credentials,
   - no real accounts,
   - no live session runner,
   - screenshots/appshots as artifacts.
4. First scenarios:
   - first-viewport operator safety,
   - disabled live button while blockers exist,
   - stale account/profile warning,
   - mobile topbar wrapping,
   - hold-to-confirm with mocked backend only.
5. Any expansion beyond mocked QA requires separate explicit approval.

### 14. Defer Agents SDK orchestration until evals and MCP are stable

- **ROI:** medium-low now
- **Diff size:** medium
- **Risk:** medium
- **Adopt now:** no, defer

Agents SDK is useful when we own a custom agent app, but this repo already has Codex/Claude sessions, skills, hooks, launchers, and MCP. The first SDK pilot should be report-only eval orchestration, not a replacement operating layer.

**Prerequisites**

1. WSL MCP config works.
2. Agent-ops eval suite exists.
3. Structured artifact schemas exist.
4. Read-only PR review pilot exists.
5. Failure taxonomy has at least one round of measured cases.

### 15. Defer fine-tuning/distillation until measured failure data exists

- **ROI:** low now
- **Diff size:** large
- **Risk:** high if premature
- **Adopt now:** no, defer

Prompt/skill/MCP/eval improvements are cheaper and safer than fine-tuning. Fine-tuning or distillation should require a stable eval set and repeated measurable failures not solved by smaller changes.

**Prerequisites**

1. At least 50 labeled failure examples.
2. Stable structured-output schemas.
3. Deterministic graders for source/tool discipline.
4. A measurable cost/latency/reliability target.
5. Evidence that prompting, skills, MCP, and model selection are insufficient.

## Execution roadmap

### Sprint 1 — persist and unblock tool context

1. Keep this plan in `docs/plans/`.
2. Update `HANDOFF.md` and `MEMORY.md` compactly.
3. Implement WSL-portable Codex MCP config.
4. Add MCP config audit and tests.
5. Refresh `.codex/OPENAI_CODEX_STANDARDS.md`.

### Sprint 2 — make agent behavior measurable

1. Add agent-ops eval cases.
2. Add deterministic eval report script.
3. Add source/tool graders.
4. Add agent failure taxonomy.
5. Add structured-output schemas and validator.

### Sprint 3 — integrate architecture prompting and safe automation

1. Add `docs/prompts/codex_architecture_review_playbook.md`.
2. Add canompx3 OpenAI-docs integration skill.
3. Add skill inventory/de-dup audit.
4. Add read-only Codex PR review pilot.
5. Add coding-agent sandbox doc.

### Sprint 4 — optional/pilot integrations

1. Optional JetBrains MCP docs.
2. Mocked Computer Use dashboard QA plan.
3. Report-only Agents SDK eval orchestration design.
4. Revisit prompt/model optimization only after eval evidence exists.

## ROI table

| Rank | Item | ROI | Diff size | Risk | Adopt now? |
|---:|---|---:|---:|---:|---|
| 0 | Persist plan + handoff + memory pointer | Very high | Tiny | Low | Yes |
| 1 | WSL-portable Codex MCP config | Very high | Small | Medium | Yes |
| 2 | Agent-ops eval flywheel | Very high | Medium | Low | Yes |
| 3 | Source/tool graders | High | Medium | Low | Yes |
| 4 | Architecture-prompt playbook | High | Small | Low | Yes |
| 5 | Structured-output schemas | High | Medium | Low | Yes |
| 6 | Read-only Codex PR review | High | Medium | Medium | Yes, gated |
| 7 | canompx3 OpenAI-docs integration skill | Medium-high | Small | Low | Yes |
| 8 | Coding-agent sandbox doc | Medium-high | Small | Low | Yes |
| 9 | Failure taxonomy | Medium | Small | Low | Yes |
| 10 | Skill de-dup audit | Medium | Small/medium | Low | Yes |
| 11 | OpenAI standards refresh | Medium | Small | Low | Yes |
| 12 | JetBrains MCP docs | Medium | Small | Low | Optional |
| 13 | Computer Use dashboard QA | Medium | Plan-only first | High if mis-scoped | Mocked only |
| 14 | Agents SDK orchestration | Medium-low now | Medium | Medium | Defer |
| 15 | Fine-tuning/distillation | Low now | Large | High | Defer |

## Acceptance criteria for this plan

- The plan is saved under `docs/plans/`.
- `HANDOFF.md` points to the plan and the first implementation step.
- `MEMORY.md` contains only a compact long-term preference/pointer.
- No live trading, broker, DB write, or research-promotion behavior changes are introduced by this plan.
- Future implementation follows smallest-diff staged tasks with tests/checks for each stage.
