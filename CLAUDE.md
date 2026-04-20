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

Six layers enforce quality: pre-commit hook (`.githooks/pre-commit` — lint + format + drift + CRLF guard + staged-file-aware tests), commit-msg hook (`.githooks/commit-msg` — body required for substantial diffs), pre-push hook (`.githooks/pre-push` — branch staleness + scope-doc check via `scripts/tools/verify_branch_scope.py`), drift detection (`pipeline/check_drift.py` — count self-reported at runtime), Claude Code hooks (auto-run drift/tests on file edits), GitHub Actions CI, plus built-in pipeline validation gates.

**One-shot setup (per clone/worktree):** `bash scripts/tools/setup_git_hooks.sh` — wires up `core.hooksPath=.githooks` and verifies executability. Without this, all git hooks are silently dormant (this is how the 2026-04-20 codex/live-book-reaudit CRLF + empty-body + stale-base incident slipped through). See `docs/governance/agent_handoff_protocol.md` for what each hook enforces.

### Project Truth Protocol
Discovery uses ONLY canonical layers (`bars_1m`, `daily_features`, `orb_outcomes`). Derived layers (`validated_setups`, `edge_families`, `live_config`, docs) are **banned for truth-finding**. Full rules → `RESEARCH_RULES.md` § Discovery Layer Discipline. Enforcement → `.claude/rules/research-truth-protocol.md`.

### Volatile Data Rule
**NEVER cite changing stats from memory/docs.** Query live: strategy counts → `gold-db` MCP, sessions → `pipeline.dst.SESSION_CATALOG`, costs → `pipeline.cost_model.COST_SPECS`, instruments → `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`, lanes → `trading_app.prop_profiles.ACCOUNT_PROFILES`, **ORB window timing → `pipeline.dst.orb_utc_window(trading_day, orb_label, orb_minutes)`** (never derive from `break_delay_min`; never fall back to `break_ts`; see `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`).

### Backtesting Methodology (MANDATORY)
Every backtest / discovery scan MUST follow `.claude/rules/backtesting-methodology.md` (auto-loads when editing `research/`, `trading_app/strategy_*`, `docs/audit/`, `docs/institutional/`). 13 rules covering look-ahead gates, comprehensive scope (324 combos), multi-framing BH-FDR, two-pass overlay testing, tautology/fire-rate/arithmetic flags, red-flag stop conditions. Historical failure log embedded — append new failures there.

### Research Provenance Rule
Config values from research need `@research-source`, `@entry-models`, `@revalidated-for`. Drift check #45 enforces.

### Source-of-Truth Chain Rule
Identify canonical source → verify downstream derives from it → if source may be wrong, audit upstream first → never patch downstream to compensate for upstream corruption.

### Local Academic / Project-Source Grounding Rule
Prefer local sources (`resources/` PDFs, project canon) over training memory. If no local source, say UNSUPPORTED. **PDF protocol:** extract text from the file — never cite from training memory as if you read it. Label training-memory claims explicitly. Before dismissing a PDF as "bibliography only" or "nothing relevant" based on keyword grep, extract the TOC + 3 mid-document pages first (terminology differs — "random walk" vs "walk-forward", "half-life" vs "half"). Detail + 2026-04-07 incident → `docs/specs/research_modes_and_lineage.md` § 9.2.

### Audit-First Default for Research Layers
Research layers: **audit → adversarial audit → fix → rerun → freeze → move on**. Do not skip to implementation when truth-state is unverified.

### Institutional Rigor (MANDATORY — non-negotiable)
Take the proper long-term institutional-grounded fix. No band-aids, dead code, silent failures, or re-encoded canonical logic. Self-review before claiming done; refactor (don't patch) when review cycles keep finding new bugs. Full rules → `.claude/rules/institutional-rigor.md` (auto-loads on production-code edits).

**Phase 0 grounding (2026-04-07):** All discovery / validation / deployment decisions must cite `docs/institutional/literature/` extracts (Bailey-Lopez de Prado, Harvey-Liu, Chordia, Pepelyshev-Polunchenko, Fitschen, Carver). `docs/institutional/pre_registered_criteria.md` locks 12 strategy-validation criteria. `docs/institutional/mechanism_priors.md` is the live trading-logic doc — read before proposing any new filter. **No brute-force >300 trials** (MinBTL bound). Discovery requires a committed hypothesis file in `docs/audit/hypotheses/`.

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
