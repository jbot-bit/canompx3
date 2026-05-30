# Plugin And External Data Routing

Claude should use plugins and external connectors as helpers, not project
authority.

Canonical canompx3 truth remains:

- code and tests
- `gold.db` and repo MCPs (`gold-db`, `repo-state`, `research-catalog`,
  `strategy-lab`)
- committed docs, runtime ledgers, GitHub, generated artifacts, and `HANDOFF.md`

Detailed shared routing policy lives in `.codex/PLUGIN_ROUTING.md` so Claude
and Codex apply the same boundaries.

## Claude Prompt Routing

The hook `.claude/hooks/plugin-router.py` injects a compact cue when prompts
mention plugin/data domains.

- Datadog/logs/metrics/traces/latency/alerts: use Datadog only if canompx3 is
  instrumented there. Otherwise inspect repo runtime artifacts and propose an
  instrumentation plan.
- MarcoPolo/external data/S3/warehouse/API exports/broker exports: use only
  with a named external source and scope. Never replace `gold.db`.
- Supabase/Postgres/RLS/Auth/Storage/Edge Functions: use only for real
  Supabase/Postgres app work. canompx3 market data stays DuckDB unless a
  deliberate migration is requested.
- Spreadsheets/presentations/documents: use only when the user requests those
  artifact formats. Default durable truth remains Markdown/YAML/code/DB.
- Gmail/Calendar/Circleback: personal/admin context only. Never project truth.
- CircleCI: explicit-only. canompx3 CI defaults to GitHub Actions.
- PostHog/Hugging Face/agent SDK/plugin-dev/skill-creator: explicit-only unless
  the user is actually instrumenting analytics, building models, building an
  agent app, or editing durable plugins/skills.

## Conflict Rule

If a plugin output conflicts with repo truth, trust repo truth and mark the
plugin output as `UNSUPPORTED` until verified.

## Claude Plugin Alignment Notes

Claude is the primary tool for this repo, but Codex may have a broader or
different plugin set. When a task would clearly benefit from a Codex-side
plugin that Claude does not currently have, Claude should not silently ignore
the gap and should not install broad plugins by habit.

Claude should do this instead:

1. Infer the capability from the user's intent.
2. Check whether Claude already has an equivalent tool, MCP, skill, or plugin.
3. If not, say the missing capability plainly and propose the smallest useful
   install or setup.
4. Keep canonical truth in repo surfaces even after a plugin is installed.

Current alignment targets:

- GitHub / PR / CI: use Claude's GitHub or `gh` surfaces when available. CI for
  canompx3 defaults to GitHub Actions, not CircleCI.
- Browser / Playwright / Chrome: Claude should use browser automation for local
  dashboard and generated-HTML proof when available; otherwise leave an
  explicit browser-smoke gap.
- Datadog: install/connect only if canompx3 is actually instrumented into
  Datadog or the user asks for telemetry setup. Otherwise inspect repo runtime
  artifacts and propose instrumentation.
- MarcoPolo: connect only for a named external data source. Do not use it as a
  replacement for `gold.db`, `gold-db`, `repo-state`, `research-catalog`, or
  `strategy-lab`.
- Supabase: install/connect only for real Supabase/Postgres/Auth/RLS work. It
  is not a market-data backend for canompx3 unless the user explicitly requests
  a migration.
- semgrep / security plugins: useful for broker/live/auth/webhook/secrets and
  generated-code review. Prefer scoped security scans over broad noisy scans.
- Slack: observer/control-room only. Summaries are non-authoritative until
  written into repo/GitHub artifacts.
- Spreadsheets / Presentations / Documents: install/use only for explicit
  artifact output requests. Markdown/YAML/code/DB remain the default durable
  project records.
- PostHog / Hugging Face / agent SDK: explicit-only unless the task is product
  analytics/LLM trace instrumentation, model/dataset work, or a standalone
  agent app.

If the user asks vaguely for "brains", "data discovery", "improve this",
"check this", or "make this better", do not ask them which plugin to use.
Run the targeted grounding route first, then select the smallest tool/plugin
that fits the evidence source.
