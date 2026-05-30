# Codex Plugin Routing

This file is the project-level contract for when Codex should use installed
plugins while working on canompx3.

## Core Rule

Plugins are assistants, not authority. Project truth still resolves through
code, `gold.db`, repo MCPs, committed docs, runtime artifacts, GitHub, and
`HANDOFF.md`.

Default truth surfaces:

- Repo state and startup truth: `repo-state`
- Research/literature/prereg/audit grounding: `research-catalog`
- Strategy and deployment readiness: `strategy-lab`
- Trading data and historical results: `gold-db`
- PR/CI/review state: GitHub

No plugin may create a second canonical database, decision ledger, strategy
registry, allocation source, or live-control plane.

## Automatic Routing

- GitHub: PRs, issues, review comments, CI status, merge/publish state.
- Browser or Playwright: local dashboard, generated HTML, screenshots, console
  errors, and UI acceptance proof.
- Chrome: authenticated/profile-dependent browser state when Browser is not
  enough.
- CodeRabbit: extra PR review lens. Treat output as claims to verify.
- Codex Security and semgrep: broker/live/auth/webhook/secrets/supply-chain
  changes, generated code, and security-sensitive diffs.
- Superpowers: debugging, TDD, verification, planning, and parallel read-only
  investigations. canompx3 gates still win.
- OpenAI Developers: current Codex/OpenAI/API behavior and official-doc
  grounding.
- Plugin Eval: plugin/skill quality review and benchmark planning.
- Build Web Apps: dashboard/frontend work only.
- Slack: observer/control-room summaries only. Slack decisions are incomplete
  until written to GitHub or repo artifacts.

## Data And AI Plugins

These can be useful, but only with a named data contract.

- Datadog: use for runtime telemetry, logs, traces, monitors, incidents, p99
  latency, dashboard uptime, broker/session health, and alert design. If
  canompx3 is not instrumented into Datadog, first produce an instrumentation
  plan; do not pretend Datadog has project truth.
- MarcoPolo: use for explicitly named external data sources such as broker
  exports, S3/lakehouse/API/CRM/Jira/log data, or ad hoc cross-system analysis.
  It must not replace `gold.db` or repo MCPs for canonical trading truth.
- Supabase: use only for real Supabase/Postgres/Auth/RLS/Storage/Edge Function
  work. For canompx3 market data, prefer DuckDB and `gold-db`.
- Spreadsheets: use for requested workbook-style exports, operator review
  tables, and spreadsheet-ready reporting. Markdown/CSV remains the default.
- Presentations: use only for explicit slide/deck deliverables.
- Gmail, Google Calendar, and Circleback: personal/admin context only. Never use
  them as project truth unless the user asks for mail, calendar, meetings, or
  prior human coordination.

## Explicit-Only Or Disabled

- CircleCI: do not use for canompx3 unless a task explicitly names CircleCI.
  The repo's CI surface is GitHub Actions.
- Documents: not a normal canompx3 artifact path. Use Markdown docs unless the
  user asks for DOCX/Word.
- PostHog: useful only if canompx3 intentionally instruments product usage,
  dashboard/operator analytics, or LLM trace analytics. It is not a market-edge
  validation tool.
- Hugging Face: useful only for deliberate model/dataset publishing, training,
  or benchmark work. It is not a default research brain for this repo.
- agent-sdk-dev: use only for a standalone Agents SDK app. Prefer OpenAI
  Developers for ordinary OpenAI/Codex/API work.
- skill-creator and plugin-dev: use only when changing durable skills/plugins.
- claude-md-management: use only when explicitly auditing startup docs. Do not
  casually churn `CLAUDE.md` or `.claude/`.

## Prompt Mapping

- "logs", "metrics", "traces", "latency", "monitor", "incident",
  "observability" -> Datadog if configured; otherwise repo runtime artifacts and
  an instrumentation plan.
- "external data", "S3", "warehouse", "lakehouse", "API export", "CRM",
  "Jira", "broker export" -> MarcoPolo only after naming the source and scope.
- "Postgres", "Supabase", "RLS", "auth", "storage", "edge function" ->
  Supabase docs/MCP/skill; not `gold.db`.
- "spreadsheet", "xlsx", "workbook", "table export" -> Spreadsheets.
- "deck", "slides", "presentation" -> Presentations.
- "email", "calendar", "meeting", "what did we discuss" -> Gmail, Google
  Calendar, or Circleback.
- "CI", "workflow", "checks", "PR failed" -> GitHub Actions unless CircleCI is
  explicitly named.

