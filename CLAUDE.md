# CLAUDE.md

Guidance for Claude Code working with this repository.

@docs/ARCHITECTURE.md

## Project Overview

Multi-instrument futures data pipeline — builds clean, replayable local datasets for ORB breakout trading research and backtesting from Databento DBN files. Active instruments: MGC (Micro Gold), MNQ (Micro Nasdaq), MES (Micro S&P 500), M2K (Micro Russell). Dead for ORB: MCL, SIL, M6E, MBT.

**For instruments, cost models, sessions, entry models, and all trading logic → see `TRADING_RULES.md`.**
**For research methodology, statistical standards, and market structure knowledge → see `RESEARCH_RULES.md`.**

---

## Document Authority

| Document | Scope | Conflict Rule |
|----------|-------|---------------|
| `CLAUDE.md` | Code structure, commands, guardrails, AI behavior | Wins for code decisions |
| `TRADING_RULES.md` | Trading rules, sessions, filters, research findings, NO-GOs | Wins for trading logic |
| `RESEARCH_RULES.md` | Research methodology, statistical standards, trading lens | Wins for research/analysis decisions |
| `ROADMAP.md` | Planned features, phase status | Updated on phase completion |
| `docs/specs/*.md` | Feature specs pending implementation | **Check before building ANY feature** |

Full file inventory → `REPO_MAP.md` (auto-generated, never hand-edit).
Frozen specs (`CANONICAL_*.txt`) → read-only; live code is truth.

**Conflict resolution:** Code behavior → CLAUDE.md. Trading logic → TRADING_RULES.md. Research → RESEARCH_RULES.md.

**Cross-tool state:** Shared decisions live in `HANDOFF.md` and `docs/plans/`, not in Claude-private memory. Read `HANDOFF.md` on session start. See `AGENTS.md` § Cross-Tool Coordination.

---

## Architecture

### Key Design Principles
- **Fail-closed:** Any validation failure aborts immediately
- **Idempotent:** All operations safe to re-run (INSERT OR REPLACE / DELETE+INSERT)
- **Pre-computed outcomes:** 5/15/30m ORB apertures, reused for all discovery
- **One-way dependency:** pipeline/ → trading_app/ (never reversed)

### Time & Calendar Model
- Local timezone: `Australia/Brisbane` (UTC+10, no DST)
- Trading day: 09:00 local → next 09:00 local
- Bars before 09:00 assigned to PREVIOUS trading day
- All DB timestamps are UTC (`TIMESTAMPTZ`)

### DST & DOW (FULLY RESOLVED)
- DST: All sessions dynamic/event-based from `pipeline/dst.py` SESSION_CATALOG. Detail → `docs/DST_CONTAMINATION.md`
- DOW: Brisbane DOW = exchange DOW except NYSE_OPEN (midnight crossing). Guard in `pipeline/dst.py`. Detail → `docs/DOW_ALIGNMENT.md`

---

## Database Location & Workflow (CRITICAL)

**ONE database** (`gold.db`) at `<project>/gold.db` — local disk, no cloud sync.

`C:\db\gold.db` is an auto-synced scratch copy — drift check #37 copies canonical → scratch when stale. All scripts default to canonical `gold.db` via `pipeline.paths.GOLD_DB_PATH`. Drift check #62 blocks any hardcoded scratch DB defaults in active code.

**Rules:**
- NEVER run two write processes against the same DuckDB file simultaneously
- `pipeline/paths.py` reads `DUCKDB_PATH` env var to override default path

---

## MCP Server (gold-db)

`gold-db` MCP server (`trading_app/mcp_server.py`) — 4 read-only tools. **ALWAYS prefer over raw SQL.** Tools: `list_available_queries`, `query_trading_db` (18 templates, row cap 5000), `get_strategy_fitness` (always `summary_only=True` for portfolio-wide), `get_canonical_context`. See `.claude/rules/mcp-usage.md` for intent→tool mapping.

---

## Guardrails

Five layers enforce quality: pre-commit hook (`.githooks/pre-commit`), drift detection (`pipeline/check_drift.py` — count self-reported at runtime), Claude Code hooks (auto-run drift/tests on file edits), GitHub Actions CI, and built-in pipeline validation gates (7 ingestion gates, 4 aggregation gates). Setup: `git config core.hooksPath .githooks`

### Volatile Data Rule
**NEVER cite strategy counts, session counts, drift check counts, cost model numbers, or any other changing stat from memory files or docs.** These go stale after every rebuild. Instead:
- Strategy counts/performance → query `gold-db` MCP tools
- Session list/times → `from pipeline.dst import SESSION_CATALOG`
- Cost models → `from pipeline.cost_model import COST_SPECS`
- Active instruments → `from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS`
- Live portfolio → `from trading_app.live_config import LIVE_PORTFOLIO`

### Research Provenance Rule
Config values derived from research (e.g. `EARLY_EXIT_MINUTES`) must include `@research-source`, `@entry-models`, and `@revalidated-for` annotations. Drift check #45 enforces this. When entry models change, re-validate all research-derived values against the new model before citing them as validated.

### 2-Pass Implementation Method (MANDATORY)
Every non-trivial code change follows two passes:

1. **Discovery:** Read ALL affected files. Understand architecture, patterns, blast radius. Articulate PURPOSE (why it matters, what breaks without it) before writing code. Run guardian prompts if applicable.
2. **Implementation:** Write code → verify (run `check_drift.py`, `audit_behavioral.py`, tests) → fix regressions → code review.

One task at a time. Implement → verify → review → next. Never batch without verification.

### Design Proposal Gate (MANDATORY)
Before writing ANY code on a non-trivial change, present a brief proposal:

1. **What:** One sentence — what you're about to do and why
2. **Files:** Which files you'll touch (create/modify/delete)
3. **Blast radius:** What else could break (callers, tests, drift checks)
4. **Approach:** Key design choice if there are alternatives

Wait for user confirmation ("go", "yes", "do it", "looks good") before writing code. If the user says "no" or redirects, update the proposal. Do NOT silently proceed.

**Exceptions** (no proposal needed):
- Trivial changes: typo fixes, comment updates, single-line bug fixes
- User explicitly said "just do it" or "skip the proposal"
- Git operations (commit, push, merge) — covered by Git Operations rule
- Running queries or read-only exploration

**Why this exists:** 38+ sessions hit "wrong approach" because Claude dove into code without confirming intent. A 10-second checkpoint prevents 30-minute reversals.

---

## Strategy Classification — Behavioral Rules

1. NEVER treat "low trade count" alone as evidence of a bug
2. ALWAYS verify `trade_days <= eligible_days` before investigating
3. If `trade_days > eligible_days` → assume corruption until proven otherwise
4. Do NOT suggest "fixing" filters to increase sample size
5. NEVER recommend REGIME strategies as standalone trading systems
6. For trading logic (filters, entry models, edge zones, Sharpe formulas) → see `TRADING_RULES.md`
7. For classification thresholds and trade day invariant → see `docs/ARCHITECTURE.md`

---

## What's NOT Built Yet

See `ROADMAP.md` for current phase status and planned features.

**Do NOT reference unbuilt features in code or tests. Build guardrails for what exists.**
