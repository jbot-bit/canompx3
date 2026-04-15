# CLAUDE.md

Guidance for Claude Code working with this repository.

## Project Overview

Multi-instrument futures data pipeline — builds clean, replayable local datasets for ORB breakout trading research and backtesting from Databento DBN files. Active instruments: MGC (Micro Gold), MNQ (Micro Nasdaq), MES (Micro S&P 500). Dead for ORB: MCL, SIL, M6E, MBT, M2K.

**For instruments, cost models, sessions, entry models, and all trading logic → see `TRADING_RULES.md`.**
**For research methodology, statistical standards, and market structure knowledge → see `RESEARCH_RULES.md`.**
**For strategy research routing, test sequences, variable space, and NO-GO registry → see `docs/STRATEGY_BLUEPRINT.md`.**

---

## Document Authority

Canonical registry for document roles: `docs/governance/document_authority.md`.

**Conflict resolution:** Code → CLAUDE.md. Trading logic → `TRADING_RULES.md`. Research → `RESEARCH_RULES.md`. Features → check `docs/specs/*.md` BEFORE building. Research routing → `docs/STRATEGY_BLUEPRINT.md`. **Research methodology / statistical thresholds / institutional standards → `docs/institutional/` (literature-grounded passages + locked criteria).**

**Cross-tool state:** `HANDOFF.md` + `docs/plans/` — read on session start. May be stale: repo/DB is truth over docs. `REPO_MAP.md` = auto-generated file inventory. `docs/institutional/HANDOFF.md` — Phase 0 literature grounding status (2026-04-07).

## Task Routing

For non-trivial repo tasks, do not rely on `CLAUDE.md` alone as the task
router.

Resolve the task first:

`python scripts/tools/context_resolver.py --task "<user request>" --format markdown`

The resolver is the deterministic front door for:

- which doctrine to read
- which code/data files are canonical
- which live views to query
- which surfaces are explicitly not live truth
- which verification profile applies

If the resolver is unavailable or ambiguous, fall back to:

- `AGENTS.md`
- `docs/governance/document_authority.md`
- `docs/governance/system_authority_map.md`
- `scripts/tools/system_context.py`
- `scripts/tools/project_pulse.py`

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

`gold-db` MCP server (`trading_app/mcp_server.py`) — 4 read-only tools. **ALWAYS prefer over raw SQL.** Tools: `list_available_queries`, `query_trading_db` (18 templates, row cap 5000), `get_strategy_fitness` (always `summary_only=True` for portfolio-wide), `get_canonical_context`. See `.claude/rules/mcp-usage.md` for intent→tool mapping.

---

## Guardrails

Five layers enforce quality: pre-commit hook (`.githooks/pre-commit`), drift detection (`pipeline/check_drift.py` — count self-reported at runtime), Claude Code hooks (auto-run drift/tests on file edits), GitHub Actions CI, and built-in pipeline validation gates. Setup: `git config core.hooksPath .githooks`

### Project Truth Protocol
Discovery uses ONLY canonical layers (`bars_1m`, `daily_features`, `orb_outcomes`). Derived layers (`validated_setups`, `edge_families`, `live_config`, docs) are **banned for truth-finding**. Full rules → `RESEARCH_RULES.md` § Discovery Layer Discipline. Enforcement → `.claude/rules/research-truth-protocol.md`.

### Volatile Data Rule
**NEVER cite changing stats from memory/docs.** Query live: strategy counts → `gold-db` MCP, sessions → `pipeline.dst.SESSION_CATALOG`, costs → `pipeline.cost_model.COST_SPECS`, instruments → `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`, lanes → `trading_app.prop_profiles.ACCOUNT_PROFILES`, **ORB window timing → `pipeline.dst.orb_utc_window(trading_day, orb_label, orb_minutes)`** (never derive from `break_delay_min`; never fall back to `break_ts`; see `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`).

### Research Provenance Rule
Config values from research need `@research-source`, `@entry-models`, `@revalidated-for`. Drift check #45 enforces.

### Source-of-Truth Chain Rule
Identify canonical source → verify downstream derives from it → if source may be wrong, audit upstream first → never patch downstream to compensate for upstream corruption.

### Local Academic / Project-Source Grounding Rule
Prefer local sources (`resources/` PDFs, project canon) over training memory. If no local source, say UNSUPPORTED.
**PDF protocol:** EXTRACT text from the file — never cite from training memory as if you read it. If extraction fails, say so explicitly. Label training-memory claims as "From training memory — not verified against local PDF." **Before dismissing a PDF as "bibliography only" / "front matter only" / "nothing relevant" based on a keyword grep, extract the table of contents AND at least 3 sample pages from the middle of the PDF to confirm the actual structure.** A single-keyword grep can miss whole chapters when the terminology is different (e.g., "walk-forward" vs "random walk", "half" vs "half-life"). A 2026-04-07 self-review caught a case where `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf` was incorrectly characterized as "bibliography only" when pp 6-28 are actually Chapter 1 "Introduction" with substantive backtest-overfitting content — see `docs/specs/research_modes_and_lineage.md` § 9.2 Revision history. Commit `aec7730` is the fix.

### Audit-First Default for Research Layers
Research layers: **audit → adversarial audit → fix → rerun → freeze → move on**. Do not skip to implementation when truth-state is unverified.

### Institutional Rigor (MANDATORY — non-negotiable)
**Always take the proper long-term institutional-grounded fix.** No band-aids, no skipping, no dead code, no silent failures, no re-encoding canonical logic. Review own work before claiming done. When review cycles keep finding new bugs, refactor — don't patch. Full rules → `.claude/rules/institutional-rigor.md`.

**Research methodology grounding (Phase 0 established 2026-04-07):** All discovery, validation, and deployment decisions must be literature-grounded. `docs/institutional/literature/` contains verbatim extracts from Bailey-Lopez de Prado (DSR, MinBTL, False Strategy Theorem), Harvey-Liu, Chordia et al, Pepelyshev-Polunchenko, Fitschen (intraday trend-follow — CORE ORB premise), and Carver (volatility targeting + Kelly sizing — Stage 2+ framework). `docs/institutional/pre_registered_criteria.md` locks 12 criteria every validated strategy must meet. `docs/institutional/hypothesis_registry_template.md` is the pre-registration format. **`docs/institutional/mechanism_priors.md` (added 2026-04-15)** is the live trading-logic doc: what we think drives ORB edge, signal-to-role mapping (R1 FILTER → R8 PORTFOLIO allocator), staged deployment roadmap. Read before proposing any new filter to avoid pigeonholing. **No brute-force enumeration of >300 trials allowed** (MinBTL bound). Discovery must cite a committed hypothesis file in `docs/audit/hypotheses/`.

### 2-Pass Implementation Method (MANDATORY)
1. **Discovery:** Read affected files, understand blast radius, articulate PURPOSE before writing code.
2. **Implementation:** Write → verify (drift + tests + behavioral audit) → fix regressions → **self-review** → fix new findings.

One task at a time. Never batch without verification. **"Done" = tests pass (show output) + dead code swept (`grep -r`) + `check_drift.py` passes + self-review passed.** All four required.

### Design Proposal Gate (MANDATORY)
Before writing ANY code on a non-trivial change, present: (1) **What** and why, (2) **Files** to touch, (3) **Blast radius**, (4) **Approach**.

**Self-check (DO NOT SKIP):** Simulate happy path, edge case (NULL/empty/sparse), and failure mode internally. SHOW what you tested and found — don't just claim you checked. Performative self-review (claiming you checked without showing evidence) is worse than no self-review. Fix flaws before presenting. Your first draft is always wrong.

Wait for user confirmation before writing code. Exceptions: trivial changes, "just do it", git ops, read-only exploration.

---

## Strategy Classification — Behavioral Rules
Low trade count ≠ bug (G6/G8 filters expected). Verify `trade_days <= eligible_days` first. If `trade_days > eligible_days` → corruption. Never "fix" filters to increase N. Never recommend REGIME as standalone. Trading logic → `TRADING_RULES.md`. Thresholds → `docs/ARCHITECTURE.md`.


Do NOT reference unbuilt features in code or tests — see `ROADMAP.md` for what's planned.
