# Agent Control Plane Evaluation - Paperclip, LONA, amux, and reasoning sidecars

Date: 2026-05-12
Owner: Codex
Status: draft implementation plan

## Objective

Add a layer above the existing Claude/Codex/worktree/MCP setup that can organize work, route agents, track state, and prevent lost context without weakening the deterministic trading-research core.

The control plane must organize and utilize the tools we already have:

- Claude Code and Codex launchers/worktrees
- `HANDOFF.md`, `docs/plans/`, runtime ledgers, and action queues
- repo-local MCPs: `gold-db`, `repo-state`, `research-catalog`, `strategy-lab`
- LONA Trading Assistant for sandboxed Backtrader-style experiments
- optional local reasoning proxy for audit/review only

## Non-negotiable boundary

The deterministic core stays sacred.

- No LLM scoring trades.
- No AI writing `orb_outcomes`, `validated_setups`, live profile activation, promotion gates, or canonical validation logic.
- No external agent platform can promote an edge or tune thresholds.
- AI outputs are advisory unless converted into a pre-registered, deterministic repo artifact and verified by existing gates.
- All trading truth remains in project canon and canonical layers.

## Recommended stack shape

```text
Human operator
  |
  v
Control plane (Paperclip first; amux only if terminal control is the missing piece)
  |
  v
Deterministic canompx3 task router and worktree manager
  |
  +-- Claude/Codex agents in isolated worktrees
  +-- Optional reasoning sidecar for audit/review prompts
  +-- LONA sandbox for external Backtrader experiments
  |
  v
Read-only MCPs and canonical repo surfaces
  |
  v
gold.db canonical layers and repo validation gates
```

## External shortlist

| Candidate | Best fit | Strength | Weakness / risk | canompx3 decision |
|---|---|---|---|---|
| Paperclip | top-level AI labor organizer | org chart, tasks, budgets, approvals, heartbeats, agent roles, governance | heavier product/control-plane; must avoid giving it direct trading authority | **Primary candidate** |
| amux | local coding-agent fleet control | tmux sessions, dashboard, watchdog, Kanban, phone/PWA, coordination | appears Claude/tmux-first; npm package metadata is not the install path; must verify source/install separately | **Secondary if terminal/session control is the real pain** |
| Cogpit | Claude Code observability/control | tool-call visibility, token/cost timelines, live diffs, conversation control | Claude-focused; less obviously a cross-tool org/governance layer | **Useful adjunct, not primary** |
| OctoAlly | Claude/Codex dashboard/hive sessions | explicit Claude + Codex dashboard and multi-agent orchestration | hive/consensus/vector-memory claims are high-risk for this repo unless tightly fenced | **Defer** |
| LONA | trading/backtest sandbox worker | strategy creation, OHLCV upload/download, Backtrader reports | not an organizer; not canonical; can create false confidence if misused | **Use under control plane as `ADVISORY_EXTERNAL_SANDBOX` only** |
| agent-reasoning sidecar | optional reasoning wrapper | ReAct/reflection/debate around read-only tools | can create tunnel vision or meta-loop waste if used for validation/discovery | **Defer until routing is stable** |

Research notes:

- Paperclip (`paperclipai` npm package) exists and describes itself as a CLI to orchestrate AI agent teams. Its package depends on `@paperclipai/server`, `embedded-postgres`, `drizzle-orm`, and `postgres`, so the first smoke test must verify local state paths and DB behavior before trust.
- amux public docs describe it as a self-hosted open-source control plane for coding agents with tmux, SQLite, watchdog, Kanban, REST API, channels, scheduler, and PWA. The npm package named `amux` is `0.0.0`, so npm is not the right install source.
- LONA connector is installed and callable. Current account state: no saved strategies and no reports. Global market symbols are available. Treat it as empty sandbox capacity, not existing canompx3 intelligence.

## Tool roles

### Paperclip

Best fit: the top-level organizer.

Use for:

- company/org-chart style ownership of canompx3 work
- tickets/tasks with owners, status, and audit trail
- managing Claude/Codex/Bash/HTTP agents as workers
- worktree-aware task assignment
- heartbeat checks and background follow-up
- approvals, budget controls, and visibility

Do not use Paperclip for:

- direct strategy validation
- direct writes to trading truth tables
- uncontrolled autonomous repo mutation
- deciding whether a strategy is live-ready

Implementation stance:

- local/self-hosted first
- keep Paperclip state outside repo, likely under `~/.paperclip`
- repo contains only a thin integration plan/config, not Paperclip runtime data
- every mutating worker still enters through repo worktree manager or existing launchers

### amux

Best fit: direct terminal/session control if Paperclip is too broad or too product-shaped.

Use if we need:

- a dashboard for many tmux/Codex/Claude sessions
- watchdog/self-healing of stuck terminal agents
- Kanban and session notes tied to running shells
- phone/PWA control of local agents

Current package check:

- `npm view amux version` returned `0.0.0`, which is not enough to install blindly.
- No scoped `@amux/cli` package was found on npm.

Decision:

- keep amux as a secondary candidate until its real install path/source repo is verified.

### Aberon / "Arber" / "Arder" family

Best fit, if this is the remembered tool: governance/audit for custom agents.

Use if we need:

- policy enforcement around bespoke Python/HTTP agents
- audit trails and compliance gates
- production governance for agent actions

Decision:

- not first choice for canompx3 orchestration unless the remembered product is confirmed.
- if found, evaluate as a policy layer below Paperclip, not as the main organizer.

### LONA Trading Assistant

Best fit: sandboxed strategy/backtest worker.

Current observed state:

- `lona_list_strategies`: no saved strategies.
- `lona_list_reports`: no saved reports.
- `lona_list_symbols(is_global=true)`: global symbols are available.
- Tool surface includes strategy creation, strategy update, market data upload/download, async backtest execution, and report inspection.

Use LONA for:

- fast Backtrader prototype checks on non-canonical OHLCV
- external strategy intake smoke tests
- sanity-checking whether an idea is mechanically implementable
- producing advisory reports for audit agents to critique

Do not use LONA for:

- canompx3 edge validation
- Mode A/OOS claims
- promotion decisions
- futures ORB canonical outcome truth
- profile activation

Required adapter discipline:

- any LONA result entering the repo must be labeled `ADVISORY_EXTERNAL_SANDBOX`
- a LONA backtest cannot cite canompx3 canon unless its input data was exported from canonical layers through a read-only, stamped export
- no LONA strategy becomes a canompx3 strategy without a new prereg, K-budget accounting, FDR controls, and deterministic replay in repo code

### agent-reasoning / local reasoning proxy

Best fit: optional reasoning wrapper between the deterministic router and read-only tools.

Use for:

- audit agents
- repo reasoning
- adversarial review
- affidavit/evidence synthesis
- design exploration

Recommended modes:

- `+react`: MCP/repo audits, evidence tracing, legal synthesis
- `+reflection`: code review and document refinement
- `+debate`: bias detection and "why this finding is fake"
- `+tot`: brainstorming/design only

Do not use for:

- auto-router/meta validation of trading results
- unrestricted discovery loops
- OOS tuning
- recursive self-improvement
- autonomous writes to repo state

Implementation stance:

- sidecar proxy only
- Ollama can remain at `11434`
- reasoning proxy can live at `8080`
- existing repo tools stay unchanged
- adapter must handle providers that return reasoning in a `reasoning` field instead of `content`, or explicitly set thinking off for models/tools that cannot consume it

## Proposed Paperclip org model

Company: `canompx3 Research Ops`

Teams:

- `Research Integrity`
  - owns prereg audits, dead-confluence re-audits, literature grounding
- `Runtime Safety`
  - owns live gates, account/profile readiness, kill-switch controls
- `Data Pipeline`
  - owns DB rebuild, drift, ingestion, canonical layer health
- `External Intake`
  - owns TradingView/AI backtester teardown, LONA sandbox checks, third-party strategy intake
- `Operator Desk`
  - owns handoff, queue hygiene, worktree cleanup, daily status

Agent roles:

- `Planner`
  - reads context router, writes scoped plans, no code mutation unless approved
- `Implementer`
  - mutates only assigned worktree files
- `Verifier`
  - runs tests/drift/CI checks, no broad refactor
- `Adversarial Auditor`
  - challenges findings against canon and evidence
- `LONA Sandboxer`
  - runs LONA-only experiments, emits advisory reports only
- `Operator Clerk`
  - keeps worktree/PR/HANDOFF state coherent

## Initial rollout

Stage 0 - inventory only:

- Keep this plan in the repo.
- Do not install Paperclip into project files.
- Keep `main` clean.
- Identify exact install/runtime for Paperclip and any alternative remembered tool.
- Success criterion: `scripts/tools/agent_control_plane_inventory.py` gives a correct, read-only view of active workstreams from both main and linked worktrees.

Stage 1 - local Paperclip smoke test:

- Install/run Paperclip outside repo state.
- Create `canompx3 Research Ops` manually or via Paperclip import if supported.
- Add read-only tasks for current active worktrees:
  - `tv-ai-backtester-teardown`
  - `parallel-20260508-221751`
- Confirm Paperclip can represent worktree path, branch, owner, status, next action.
- Stop condition: if Paperclip cannot safely launch existing repo wrappers or stores opaque mutable state in the repo, stop and evaluate amux instead.

Stage 1B - amux smoke test if Paperclip fails:

- Install only from verified upstream source, not the empty npm placeholder.
- Register canompx3 read-only first.
- Check whether it can launch Codex as well as Claude, not just Claude Code.
- Verify it does not auto-answer prompts, run YOLO mode, or mutate branches without explicit configuration.
- Success criterion: dashboard can monitor isolated worktrees and does not bypass repo preflight.

Stage 2 - adapter contract:

- Add a repo-local script that exports current workstream inventory as JSON.
- Keep it read-only. Implemented draft: `scripts/tools/agent_control_plane_inventory.py`.
- Include:
  - worktree name/path/branch/head
  - dirty status
  - PR link if any
  - purpose
  - recommended next action

Current export command:

```bash
python scripts/tools/agent_control_plane_inventory.py --format json
python scripts/tools/agent_control_plane_inventory.py --format markdown
```

Stage 3 - controlled worker launch:

- Paperclip tasks launch only existing repo launchers:
  - `scripts/infra/codex-worktree.sh`
  - `scripts/infra/codex-project*.sh`
  - `scripts/infra/codex-wsl-sync.sh`
- No direct mutation outside isolated worktrees.
- No bypass of preflight/session guard.

Stage 4 - LONA integration:

- Add a LONA sandbox SOP under `docs/plans/` or `docs/runtime/`.
- Add a standard result label: `ADVISORY_EXTERNAL_SANDBOX`.
- First test should use global symbol data or a tiny synthetic CSV, not canompx3 canonical futures data.
- Do not upload proprietary/canonical DB exports until an explicit export/redaction policy exists.
- Success criterion: one trivial LONA backtest report is produced and recorded as an advisory sandbox artifact with no edge claim.

## First concrete tasks

1. Resolve the stale `tv-ai-backtester-teardown` worktree:
   - rebase on current main
   - run its tool tests
   - decide PR vs delete
   - this belongs under `External Intake`

2. Build a workstream inventory export:
   - probably `scripts/tools/worktree_manager.py list --managed-only --json` is enough
   - wrap only if Paperclip needs a different format

3. Run Paperclip smoke test outside repo:
   - verify CLI help/version
   - verify storage location
   - create one company/project/task manually
   - do not grant broad repo write autonomy

4. LONA smoke test:
   - create or import one trivial Backtrader strategy
   - run against one global symbol
   - inspect report
   - record limitations
   - do not cite as canompx3 edge evidence

## Decision

Use Paperclip as the first control-plane candidate because the primary need is organization, ownership, budgets, approvals, and cross-tool task state. Use LONA as a sandboxed research worker underneath it. Keep amux as the fallback if Paperclip is too broad or cannot manage local terminal agents safely. Keep Cogpit as a possible observability adjunct for Claude Code only. Do not add a reasoning proxy until work routing is stable.

## ROI gate

Proceed only if the next step reduces one of these concrete costs:

- lost branch/worktree context
- duplicate agent work
- unclear task ownership
- unobserved long-running agents
- forgotten PR/CI cleanup
- unclear distinction between advisory sandbox output and canonical trading evidence

Do not proceed if the next step is merely "cool agent tooling." Every integration must either improve workstream visibility, reduce cleanup overhead, or enforce existing canompx3 safety boundaries.
