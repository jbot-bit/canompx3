# CLAUDE.md

Guidance for Claude Code working with this repository.

@docs/ARCHITECTURE.md

## Project Overview

Multi-instrument futures data pipeline — builds clean, replayable local datasets for ORB breakout trading research and backtesting from Databento DBN files. Active instruments: MGC (Micro Gold), MNQ (Micro Nasdaq), MES (Micro S&P 500). Dead for ORB: MCL, SIL, M6E, MBT, M2K.

**For instruments, cost models, sessions, entry models, and all trading logic → see `TRADING_RULES.md`.**
**For research methodology, statistical standards, and market structure knowledge → see `RESEARCH_RULES.md`.**
**For strategy research routing, test sequences, variable space, and NO-GO registry → see `docs/STRATEGY_BLUEPRINT.md`.**

---

## Document Authority

| Document | Scope | Conflict Rule |
|----------|-------|---------------|
| `CLAUDE.md` | Code structure, commands, guardrails, AI behavior | Wins for code decisions |
| `TRADING_RULES.md` | Trading rules, sessions, filters, research findings, NO-GOs | Wins for trading logic |
| `RESEARCH_RULES.md` | Research methodology, statistical standards, trading lens | Wins for research/analysis decisions |
| `ROADMAP.md` | Planned features, phase status | Updated on phase completion |
| `docs/STRATEGY_BLUEPRINT.md` | Research routing, test gates, variable space, NO-GOs | **Open before ANY research/planning session** |
| `docs/specs/*.md` | Feature specs pending implementation | **Check before building ANY feature** |

Full file inventory → `REPO_MAP.md` (auto-generated, never hand-edit).
Frozen specs (`CANONICAL_*.txt`) → read-only; live code is truth.

**Conflict resolution:** Code behavior → CLAUDE.md. Trading logic → TRADING_RULES.md. Research → RESEARCH_RULES.md.

**Cross-tool state:** Shared decisions live in `HANDOFF.md` and `docs/plans/`, not in Claude-private memory. Read `HANDOFF.md` on session start. For parallel work, prefer `scripts/infra/claude-worktree.sh open <task>` instead of sharing one mutable branch with Codex. See `AGENTS.md` § Cross-Tool Coordination.

**Staleness rule:** `HANDOFF.md`, `docs/plans/`, memory notes, and prior session summaries capture intent and coordination state — but they may be stale. They are NOT authoritative over live code, current DB state, canonical config modules, or current command output. If a doc's claims about state contradict the repo/DB → repo/DB is truth; explicitly report the contradiction.

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

`C:\db\gold.db` scratch copy is **DEPRECATED** (Mar 2026) — caused stale-data bugs. `pipeline.paths.GOLD_DB_PATH` blocks it. Drift check #37 verifies canonical DB exists; drift check #62 blocks hardcoded scratch DB defaults in code.

**Rules:**
- NEVER run two write processes against the same DuckDB file simultaneously
- `pipeline/paths.py` reads `DUCKDB_PATH` env var to override default path

---

## MCP Server (gold-db)

`gold-db` MCP server (`trading_app/mcp_server.py`) — 4 read-only tools. **ALWAYS prefer over raw SQL.** Tools: `list_available_queries`, `query_trading_db` (19 templates, row cap 5000), `get_strategy_fitness` (always `summary_only=True` for portfolio-wide), `get_canonical_context`. See `.claude/rules/mcp-usage.md` for intent→tool mapping.

---

## Guardrails

Five layers enforce quality: pre-commit hook (`.githooks/pre-commit`), drift detection (`pipeline/check_drift.py` — count self-reported at runtime), Claude Code hooks (auto-run drift/tests on file edits), GitHub Actions CI, and built-in pipeline validation gates. Setup: `git config core.hooksPath .githooks`

### Project Truth Protocol
Discovery uses ONLY canonical layers (`bars_1m`, `daily_features`, `orb_outcomes`). Derived layers (`validated_setups`, `edge_families`, `live_config`, docs) are **banned for truth-finding**. Full rules → `RESEARCH_RULES.md` § Discovery Layer Discipline. Enforcement → `.claude/rules/research-truth-protocol.md`.

### Volatile Data Rule
**NEVER cite strategy counts, session counts, drift check counts, cost model numbers, or any other changing stat from memory files or docs.** These go stale after every rebuild. Instead:
- Strategy counts/performance → query `gold-db` MCP tools
- Session list/times → `from pipeline.dst import SESSION_CATALOG`
- Cost models → `from pipeline.cost_model import COST_SPECS`
- Active instruments → `from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS`
- Deployed lanes → `from trading_app.prop_profiles import ACCOUNT_PROFILES`

### Research Provenance Rule
Config values derived from research (e.g. `EARLY_EXIT_MINUTES`) must include `@research-source`, `@entry-models`, and `@revalidated-for` annotations. Drift check #45 enforces this. When entry models change, re-validate all research-derived values against the new model before citing them as validated.

### Source-of-Truth Chain Rule
For any audit, debugging, or research task:
- First identify the canonical source of truth for the layer being examined
- Verify whether each downstream file/module/table derives from that source or silently redefines it
- If the source-of-truth itself may be wrong, audit upstream first before touching downstream
- Never patch downstream behavior to compensate for suspected upstream corruption

### Local Academic / Project-Source Grounding Rule
For methodological, statistical, validation, monitoring, execution-realism, or prop-rule claims:
- Prefer local project sources and local academic PDFs/books in `resources/` over training memory
- For nontrivial methodological claims, name the local source(s) consulted when relevant
- If project canon and an academic source differ, project canon governs implementation; academic source is used as adversarial reference material
- If no local source supports a claim, say UNSUPPORTED rather than improvising

**PDF grounding protocol (MANDATORY — do not silently fake this):**
- When grounding a claim in a `resources/` PDF: **EXTRACT text from the file first.** Do not cite from training memory as if you read it.
- If extraction fails (tool error, reader not installed, garbled output): say **"PDF read failed for [filename]. Cannot ground this claim in local copy."**
- You MAY still use training knowledge, but you MUST label it explicitly: "From training memory — not verified against local PDF."
- **NEVER** silently substitute training memory for a failed PDF read. The user has caught this multiple times. If you can't read the file, say so.

### Audit-First Default for Research Layers
For research pipeline layers (data integrity, outcomes, discovery, validation, noise/significance, portfolio construction, execution realism, monitoring): default workflow is **audit → adversarial audit → fix → rerun → freeze layer → move on**. Do not skip to implementation when truth-state is unverified. Do not advance to the next layer until the current layer is explicitly frozen or blocked upstream.

### 2-Pass Implementation Method (MANDATORY)
Every non-trivial code change follows two passes:

1. **Discovery:** Read ALL affected files. Understand architecture, patterns, blast radius. Articulate PURPOSE (why it matters, what breaks without it) before writing code. Run guardian prompts if applicable.
2. **Implementation:** Write code → verify (run `check_drift.py`, `audit_behavioral.py`, tests) → fix regressions → code review.

One task at a time. Implement → verify → review → next. Never batch without verification.

**Completion evidence ("done" means PROVEN, not CLAIMED):**
- **Tests:** Run affected tests. Show actual output. "Tests should pass" is not evidence.
- **Dead code:** `grep -r` for functions, imports, config entries orphaned by this change. Remove them. Do not leave dead code for future discovery.
- **Drift:** `python pipeline/check_drift.py` must pass.
- Do not close the stage or claim completion without all three.

### Design Proposal Gate (MANDATORY)
Before writing ANY code on a non-trivial change, present a brief proposal:

1. **What:** One sentence — what you're about to do and why
2. **Files:** Which files you'll touch (create/modify/delete)
3. **Blast radius:** What else could break (callers, tests, drift checks)
4. **Approach:** Key design choice if there are alternatives
5. **Self-check (DO NOT SKIP):** Before presenting the proposal, simulate it through 3 scenarios internally:
   - **Happy path:** does the change work as intended end-to-end?
   - **Edge case:** what if inputs are NULL/empty/missing? What about first-time runs, concurrent access, or instruments with sparse data?
   - **Failure mode:** what if a dependency changed, a downstream consumer expects the old interface, or a test mock doesn't cover the new behavior?
   Fix everything that breaks in simulation. Then present. **Do NOT present your first draft — it has flaws you haven't found yet. Your first draft is always wrong; iterate internally until you can't find more problems.** If you present and the user has to push back with obvious flaws, you skipped this step.

   **Anti-performative rule:** Do not just SAY "I simulated three scenarios and they all passed." SHOW what you tested and what you found. State the specific edge case and how your plan handles it. If all three scenarios found zero issues, you didn't look hard enough — go deeper. Performative self-review (claiming you checked without showing evidence) is worse than no self-review because it creates false confidence.

Wait for user confirmation ("go", "yes", "do it", "looks good") before writing code. If the user says "no" or redirects, update the proposal. Do NOT silently proceed.

**Exceptions** (no proposal needed):
- Trivial changes: typo fixes, comment updates, single-line bug fixes
- User explicitly said "just do it" or "skip the proposal"
- Git operations (commit, push, merge) — covered by Git Operations rule
- Running queries or read-only exploration

**Entry-point invariance:** This gate applies regardless of how work is initiated — direct request, `/stage-gate`, `/plan`, `/design`, or any other entry point. The quality bar is the same.

**Why this exists:** 38+ sessions hit "wrong approach" because Claude dove into code without confirming intent, or presented a flawed first-draft plan that the user had to catch and correct. A 30-second self-check prevents 30-minute reversals.

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
