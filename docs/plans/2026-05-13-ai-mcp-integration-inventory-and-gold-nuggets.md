# AI / MCP / LLM Project Integration — Inventory + Golden Nuggets

**Date:** 2026-05-13
**Owner:** Josh
**Status:** notes — one-screen truth of what's wired, what's installed-but-not-wired, what's missing, and where the highest leverage is.

**Why this exists:** the project has accreted a lot of AI/MCP pieces (5 MCP servers, 11 subagents, 27 skills, OpenRouter wiring, Pinecone, LHP proposer, agent-control-plane plan, LONA, deepseek/opencode wrappers). Hard to tell from inside Claude what's actually load-bearing vs cruft. This is the inventory + the gold-nuggets to act on.

**Authority:** non-canonical notes. Decisions/actions land in `docs/runtime/action-queue.yaml` or follow-up plans. This file is read-only after stabilization.

---

## 1. MCP servers — what's wired in `.mcp.json`

| Server | Type | What it actually does | Wired? | Used in practice? |
|---|---|---|---|---|
| `gold-db` | stdio, repo venv | Read-only DuckDB query templates over `gold.db` (strategy fitness, validated_setups, lane allocation, trade book, canonical context). | ✅ | **HIGH** — every trading question routes through it |
| `repo-state` | stdio | Context views, task routes, startup packet, project pulse, system context. Read-only control plane. | ✅ | **MEDIUM** — under-used; should be the first call every session |
| `research-catalog` | stdio | Literature excerpts, hypothesis status (open/closed via filename-stem heuristic), audit results, K-budget estimator (Bailey 2013). | ✅ | **MEDIUM** — used for `/nogo` + literature grounding |
| `strategy-lab` | stdio | Lane allocation summary, fitness, deployability, promotable candidates. Overlaps `gold-db.get_strategy_fitness` intentionally. | ✅ | **LOW** — overshadowed by direct `gold-db` queries |
| `code-review-graph` | stdio (uvx) | Tree-sitter knowledge graph, impact analysis, semantic search, predicate lineage, dead-code preview. | ✅ | **MEDIUM** — gated behind `/crg-*` skills + auto-routing rule |

**Health check:** all five servers were verified working on 2026-05-12 after `gold.db.wal` orphan cleanup + cryptography pin (`feedback_mcp_venv_drift_cryptography47.md`).

## 2. Subagents — what's in `.claude/agents/`

| Agent | Purpose | Used in practice? |
|---|---|---|
| `blast-radius` | Pre-edit impact map | HIGH (auto-routed for `pipeline/`, `trading_app/` edits) |
| `db-analyst` | Fast `gold.db` queries (formatted, not essays) | HIGH |
| `evidence-auditor` | Independent skepticism pass on claims | HIGH for capital-class decisions |
| `executor` | Scope-locked stage executor | MEDIUM (only when stage-gate is active) |
| `live-risk-auditor` | Broker/session/risk-manager audit | LOW (only pre-deploy) |
| `planner` | Read-only staged plan | MEDIUM |
| `preflight-auditor` | 6-question prereq verifier | LOW (only when stage-gate requires it) |
| `ralph-loop` | Autonomous one-iteration audit→fix→commit loop | LOW |
| `research-methodologist` | Literature-grounded review (Bailey/Harvey/Chordia) | MEDIUM |
| `test-coverage-scout` | Test gap map + pytest targets | MEDIUM |
| `verify-complete` | Post-edit completeness audit | HIGH (auto-routed post-edit) |

## 3. Skills — `.claude/skills/`

27 skills. The high-traffic / load-bearing ones:

- **`orient`, `next`, `verify`, `ship`, `open-pr`** — daily-driver workflow
- **`research`, `discover`, `propose-hypothesis`, `nogo`** — research routing
- **`brain`** — master orchestrator (auto-intent classifier)
- **`capital-review`, `code-review`** — pre-capital scrutiny
- **`quant-debug`, `quant-tdd`** — pipeline bug/feature flow
- **`regime-check`, `trade-book`** — live state
- **`design`, `stage-gate`, `task-splitter`** — implementation discipline

The CRG skills (`crg-context`, `crg-blast`, `crg-search`, `crg-lineage`, `crg-tests`, `crg-deadcode`, `crg-visualize`) are commands not skills, and they're well-documented in `auto-skill-routing.md`.

## 4. Hooks — `.claude/hooks/`

The hooks layer is the **most under-appreciated piece** of project intelligence. They're invisible but enforce safety:

- `session-start.py` — records `branch_at_start` lock, prints origin drift status, loads HANDOFF
- `branch-flip-guard.py` (PostToolUse) — blocks mid-session branch switches
- `bias-grounding-guard.py` — flags ungrounded claims
- `data-first-guard.py` — enforces "query data before reading code" pattern
- `discovery-loop-guard.py` — catches research-loop tunnel vision
- `mcp-git-guard.py` — prevents MCP servers from doing git writes
- `post-edit-pipeline.py`, `post-edit-schema.py` — auto-run drift / tests after canonical edits
- `pre-edit-discovery-marker.py`, `pre-edit-guard.py` — pre-edit truth gating
- `risk-tier-guard.py` — escalates rigor for high-risk work
- `stage-awareness.py`, `stage-gate-guard.py` — enforces stage discipline
- `post-compact-reinject.py` — re-injects critical context after Claude compacts

These are the **brain stem** — without them, none of the other layers would catch their own drift.

## 5. LLM integration scripts — `scripts/tools/`

| Script | Purpose | State |
|---|---|---|
| `render_ai_research_packet.py` | Canonical research-packet renderer for OpenRouter | ✅ Working, used by `gold-db.get_ai_research_packet` MCP |
| `run_ai_research_task.py` | Bounded OpenRouter research-task runner | ✅ Working |
| `eval_openrouter_profiles.py` | Profile evaluation against OpenRouter | ✅ Working |
| `check_or_credits.py` | OpenRouter credit balance | Utility |
| `claude_review_deepseek.py` | Claude-reviews-DeepSeek bridge | Niche |
| `claude_superpower_brief.py` | Bootstrapping brief for Claude | Niche |
| `opencode_resolve_model.py` | OpenCode model resolution | Niche |
| `opencode-agent.ps1`, `deepseek-agent.ps1` | PS1 wrappers for alt agents | Probably stale |
| `agent_control_plane_inventory.py` | Workstream inventory for external organizers (e.g. Paperclip) | ✅ Built, not connected to anything yet |
| `agent_prompt_audit.py` | Agent-prompt auditor | Unknown usage |
| `sync_pinecone.py`, `pinecone_snapshots.py` | Pinecone sync + snapshots | ✅ Wired via `pinecone-assistant` skill |

## 6. Pinecone — project knowledge surface

Wired via `pinecone-assistant` skill. Routes questions like "what did we find about X" / "why did we do Y" / "history of" / "remind me" to Pinecone (project knowledge) vs `gold-db` MCP (live data). The skill description in `MEMORY.md` indicates it handles design decisions, NO-GOs, research findings.

**Gap:** Pinecone sync is manual (`sync_pinecone.py`). No automated upsert hook after PR merge, after audit-result MD commit, or after debt-ledger update. Pinecone freshness depends on you remembering to sync.

## 7. LHP — autonomous self-detecting hypothesis proposer

Landed in PR #279 (commit `87b4eb15`, 2026-05-12). Scans canonical layers for adjacencies, drafts a YAML prereg, refuses to fabricate citations, outputs `.draft.yaml` requiring human review before promotion. The 3 `.rejected.txt` files in `docs/audit/hypotheses/drafts/` are LHP outputs you vetoed.

**Wired into:** `propose-hypothesis` skill (manual invocation).
**Not wired into:** any automated trigger. The proposer doesn't run on a schedule, doesn't fire after canonical data updates, doesn't propose adjacent cells when a strategy fails fitness. Pure pull-mode.

## 8. Plans not yet executed

- **`2026-05-12-agent-control-plane-evaluation.md`** — Paperclip-first / amux-fallback control-plane plan. Stage 0 (inventory) is done (`agent_control_plane_inventory.py`). Stages 1-4 not started. **Open question: do we need a control plane, or is the existing worktree-manager + HANDOFF + MCP triad already sufficient?**
- **`2026-05-12-chordia-audit-queue-v2-plan.md`** — landed in PR #281. ✅
- **`2026-05-12-deployment-throughput-leverage.md`** — open, ties to 78 ROUTABLE_DORMANT decision (HANDOFF step 3).
- **`2026-05-12-next-literature-ingest-baton.md`** — open, ties to Dalton / pending acquisitions.
- **`2026-05-12-harris-full-text-ingestion.md`** — landed in PR #265. ✅

---

# GOLDEN NUGGETS — highest-leverage next moves

Ordered by ROI (effort low, payoff high). Each is one focused action you could take without scope creep.

## NUGGET 1 — Auto-sync Pinecone on canonical doc commit ⭐⭐⭐

**Current waste:** Pinecone is the project-knowledge memory; staleness corrupts every "what did we find about X" lookup. Sync is manual. You forget. The knowledge base drifts behind reality.

**Fix:** post-commit hook (`.githooks/post-commit`) that triggers `scripts/tools/sync_pinecone.py` when the commit touches `docs/audit/results/`, `docs/institutional/literature/`, `docs/runtime/debt-ledger.md`, or `memory/MEMORY.md`. Background process, fail-open (don't block the commit if sync fails).

**Effort:** 30 min. **Payoff:** Pinecone always fresh; `pinecone-assistant` skill stops returning stale verdicts.

**Risk:** sync API failures spam terminal. Mitigation: log to `logs/pinecone-sync.log`, only print on failure rate >X%.

## NUGGET 2 — Wire LHP into a weekly cron / event trigger ⭐⭐⭐

**Current waste:** The hypothesis proposer exists, refuses to fabricate, and produces `.draft.yaml` you can review. But it only runs when you manually invoke `/propose-hypothesis`. So 99% of the time it's dormant.

**Fix:** schedule a weekly LHP run (Sunday morning Brisbane) that scans for: (a) strategies that crossed from FIT→WATCH/DECAY this week (adjacency rerun), (b) any new daily_features columns added in the last week (sweep opportunity), (c) any audit-result MDs with `verdict: UNVERIFIED_INSUFFICIENT_POWER` (waiting for power to mature). Output to `docs/audit/hypotheses/drafts/weekly-YYYY-MM-DD/`. You review Monday morning.

**Effort:** 1-2 hours (cron + dispatcher + dedup against existing 152 preregs). **Payoff:** turns LHP from a pull tool into a discovery flywheel without burning your attention.

**Risk:** generates noise / too many drafts. Mitigation: hard cap at 5 drafts/week, sorted by Chordia K-budget headroom + audit-gap rank.

## NUGGET 3 — Cross-MCP query consolidation ⭐⭐

**Current waste:** `gold-db.get_strategy_fitness` and `strategy-lab.get_recent_fitness` overlap intentionally. But this means two code paths to maintain, and you have to remember which one returns what. The doctrine says "strategy-lab is for deployment-readiness, gold-db is for raw fitness" — but in practice you'd reach for whichever you remember first.

**Fix:** pick ONE as canonical for fitness queries, deprecate the other's overlapping endpoint with a redirect message. Either remove `strategy-lab.get_recent_fitness` (consolidate to gold-db) or move all fitness ops into strategy-lab (gold-db stays pure raw queries).

**Effort:** 1 hour decision + 1 hour migration. **Payoff:** removes one class of "wait, which MCP has X?" cognitive overhead.

**Risk:** breaks existing skill prompts that reference the deprecated endpoint. Mitigation: grep skills/agents for the deprecated name before cutover.

## NUGGET 4 — `repo-state.get_startup_packet` should be the FIRST call every session ⭐⭐

**Current waste:** session-start.py prints HANDOFF + origin drift. But `repo-state.get_startup_packet` returns a structured packet (active work, fresh pulse, system context, task routes) that's strictly more useful — and we never call it automatically.

**Fix:** add to `session-start.py` — invoke `repo-state.get_startup_packet` via MCP and dump a compact summary to the session-start banner. Or: add a `/orient` autorun on session start.

**Effort:** 30 min. **Payoff:** zero-cost context loading on every session start; reduces the "where are we?" tax.

**Risk:** session-start hook hangs if MCP server is slow. Mitigation: 5-second timeout, fail-open.

## NUGGET 5 — Drop OctoAlly / Cogpit / Paperclip evaluation. They're a trap. ⭐⭐

**Current waste:** the agent-control-plane plan considers 6 external tools (Paperclip, amux, Cogpit, OctoAlly, LONA, reasoning sidecar). The honest assessment in the plan itself is "use Paperclip first, fall back to amux." But the cost/benefit math is shaky — the ROI gate at the bottom of the plan says "do not proceed if the next step is merely 'cool agent tooling.'"

**Honest read:** the existing worktree-manager + HANDOFF + 5 MCP servers + 11 subagents + 27 skills + 17 hooks is **already a control plane**. It's just not branded as one. Paperclip adds an org-chart UI on top — but you don't have a coordination problem; you have a context-loading problem (which NUGGET 4 fixes).

**Fix:** mark the agent-control-plane plan as PARKED with reopen trigger = "if I'm spending >2hrs/week on worktree/branch/PR cleanup instead of research." Document in HANDOFF.

**Effort:** 10 min. **Payoff:** kills 6 weeks of "evaluate external orchestrators" research debt.

**Risk:** none. You can reopen anytime.

## NUGGET 6 — `agent_prompt_audit.py` — unknown usage, possible dead code ⭐

**Current waste:** script exists, no caller found, not referenced in any skill or agent or hook. Possibly a stub.

**Fix:** grep its imports and callers. If zero callers → run `/crg-deadcode` and confirm, then delete or pin a reopen trigger.

**Effort:** 5 min. **Payoff:** small. But cleanup compounds.

## NUGGET 7 — OpenCode / DeepSeek wrappers — likely stale ⭐

`deepseek-agent.ps1`, `opencode-agent.ps1`, `opencode_resolve_model.py`, `claude_review_deepseek.py` — these look like alt-LLM experiments from before OpenRouter became the default routing layer. Verify last-used date and either consolidate into a single `scripts/tools/llm_router.py` or delete.

**Effort:** 30 min audit. **Payoff:** consolidates LLM access surface to OpenRouter as the single integration point.

---

## Anti-nuggets — things NOT worth doing

- **LONA integration deeper than `ADVISORY_EXTERNAL_SANDBOX`** — the doctrine boundary in the control-plane plan is correct; LONA can't enter canonical truth without a full prereg + Mode-A replay anyway. Don't bolt it in further.
- **Reasoning sidecar (ReAct / reflection / debate)** — the plan correctly says "defer until routing is stable." Routing is *not* stable yet (see NUGGETS 3, 4). Don't add this layer.
- **New MCP servers** — the 5 you have cover canonical query / state / research / strategy / code-graph. Adding more multiplies confusion.
- **Pinecone schema overhaul** — current schema is fine. The problem is sync cadence (NUGGET 1), not data shape.

---

## What good looks like after these nuggets

- Pinecone is always fresh (NUGGET 1)
- LHP proposes 0-5 drafts/week without you asking (NUGGET 2)
- One canonical MCP for fitness queries (NUGGET 3)
- Session start loads structured context, not just HANDOFF prose (NUGGET 4)
- Agent-control-plane decision is PARKED with a clear reopen trigger (NUGGET 5)
- Dead/stale scripts cleaned (NUGGETS 6, 7)

Total effort: ~6 hours over a weekend. Total payoff: every future session opens with fresh memory, structured context, and a discovery flywheel running in the background.

---

## How this gets used

Read on session-start when the user asks about "AI integrations" / "MCP setup" / "what's wired" / "what's broken." Otherwise this is reference-only. Decisions land in `docs/runtime/action-queue.yaml` with this doc as `decision_refs`.
