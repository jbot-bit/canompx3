# CLAUDE.md

Guidance for Claude Code working with this repository.

## Project Overview

Multi-instrument futures data pipeline — builds clean, replayable local datasets for ORB breakout trading research and backtesting from Databento DBN files. Active instruments: MGC (Micro Gold), MNQ (Micro Nasdaq), MES (Micro S&P 500). Dead for ORB: MCL, SIL, M6E, MBT, M2K.

**For instruments, cost models, sessions, entry models, and all trading logic → see `TRADING_RULES.md`.**
**For research methodology, statistical standards, and market structure knowledge → see `RESEARCH_RULES.md`.**
**For strategy research routing, test sequences, variable space, and NO-GO registry → see `docs/STRATEGY_BLUEPRINT.md`.**

## Operating Discipline (every task)

Summary of `institutional-rigor.md` + `workflow-preferences.md` — reference, don't restate.

- **THINK FIRST:** state assumptions + success criteria; stop when confused, don't guess.
- **SURGICAL:** smallest diff; touch only what's asked; flag unrelated issues, don't fix them.
- **GOAL-DRIVEN:** define verification first; show evidence before claiming done. A finding/title/status word is a *claim* — falsify vs ground truth first.
- **AUDITS IMPLEMENT FIXES:** when running an audit/code review, ALWAYS implement the fixes in the same session — never just present findings ("GROUND IN TRUTH AND CONTEXT AND YES IMPLEMENT FIXES PHASES"). After fixes, run the full test suite + commit; report pass/fail counts.
- **CHECKPOINT:** at a natural stop → 1-line baton → `/clear` early; save reusable findings to memory the turn they emerge.
- **Multi-terminal/Codex sync:** `add → commit → pull --rebase → push`; never `--force` on reject; one terminal pushes or each owns its branch; `HANDOFF.md` = cross-tool baton. Push has no lock yet — don't race a peer (tracked follow-up).
- **Tooling default:** trade book / HTML reports open in browser by default (don't pass `--no-open` unless asked).

## Quick Commands

```bash
python pipeline/check_drift.py                                         # all guardrails (must pass pre-commit)
python scripts/tools/context_resolver.py --task "<x>" --format markdown  # task front-door
python scripts/tools/generate_trade_sheet.py                           # trade book HTML (opens in browser)
```
One-time setup: `git config core.hooksPath .githooks`

---

## Document Authority

Canonical registry for document roles: `docs/governance/document_authority.md`.

**Conflict resolution:** Code → CLAUDE.md. Trading logic → `TRADING_RULES.md`. Research → `RESEARCH_RULES.md`. Features → check `docs/specs/*.md` BEFORE building. Research routing → `docs/STRATEGY_BLUEPRINT.md`. **Research methodology / statistical thresholds / institutional standards → `docs/institutional/` (literature-grounded passages + locked criteria).**

**Cross-tool state:** `HANDOFF.md` + `docs/plans/` — read on session start. May be stale: repo/DB is truth over docs. `REPO_MAP.md` = auto-generated file inventory. `docs/institutional/HANDOFF.md` — Phase 0 literature grounding status (2026-04-07).

## Task Routing

Non-trivial repo tasks: resolve via the deterministic front door first —
`python scripts/tools/context_resolver.py --task "<user request>" --format markdown`

Fallback if unavailable or ambiguous: `AGENTS.md`, `docs/governance/document_authority.md`, `docs/governance/system_authority_map.md`, `scripts/tools/system_context.py`, `scripts/tools/project_pulse.py`. Path-scoped `.claude/rules/*.md` auto-load on matching edits — reference, don't restate.

## Default Mode

Institutional rigor: cite literature, query live, falsify before validating (detail → `.claude/rules/institutional-rigor.md`). **Autonomy:** act silently on reversible work (config, docs, grounding, trivial refactors, read-only research, git of verified work) + report after; STOP first for capital / schema / destructive / ambiguous-design / outward-facing. Holds only when an action is BOTH cheap and rigorous. Detail → `.claude/rules/autonomy-contract.md`.

## Shell

**Canonical shell: bash** (Bash tool) for git, python/uv/pytest/ruff, file ops, search — bash-native across infra scripts, pre-commit, venv routing, WSL paths. PowerShell = **Windows-only** (GUI launchers, scheduled tasks, `*.ps1`, `Get-Process`/`Get-Item`, `wsl`, registry, `stop_live`). `shell-canon-guard.py` soft-steers; fail-open. Detail → `.claude/rules/shell-canon.md`.

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

`gold-db` MCP server (`trading_app/mcp_server.py`) — read-only. **ALWAYS prefer over raw SQL.** Discover tools via `list_available_queries`; pass `summary_only=True` to `get_strategy_fitness` for portfolio-wide. Tool/template counts are volatile — query live, never cite from memory. Routing → `.claude/rules/mcp-usage.md`.

**Sidecar pip-installs (constraints):** `code-review-graph` is not in `uv.lock`. Install via `pip install -c constraints.txt code-review-graph` so the `cryptography<47` pin holds (Authlib 1.7.0 still imports `cryptography.hazmat.backends`, removed in cryptography 47 — breaks every FastMCP server). See `constraints.txt` and `memory/feedback_mcp_venv_drift_cryptography47.md`.

---

## Guardrails

Five layers enforce quality: pre-commit hook (`.githooks/pre-commit`), drift detection (`pipeline/check_drift.py` — count self-reported at runtime), Claude Code hooks (auto-run drift/tests on file edits), GitHub Actions CI, and built-in pipeline validation gates. Setup: `git config core.hooksPath .githooks`

**Branch-flip protection:** `.claude/hooks/branch-flip-guard.py` (PostToolUse/Bash) + pre-commit step 0c block mid-session branch switches. If BLOCKED: `git checkout <original-branch>` or `rm .git/.claude.pid` + restart. → `.claude/rules/branch-flip-protection.md`

### Project Truth Protocol
Discovery uses ONLY canonical layers (`bars_1m`, `daily_features`, `orb_outcomes`). Derived layers (`validated_setups`, `edge_families`, `live_config`, docs) are **banned for truth-finding**. → `RESEARCH_RULES.md` § Discovery Layer Discipline; enforcement in `.claude/rules/research-truth-protocol.md`.

### Volatile Data Rule
**NEVER cite changing stats from memory/docs.** Query live: strategy counts → `gold-db` MCP, sessions → `pipeline.dst.SESSION_CATALOG`, costs → `pipeline.cost_model.COST_SPECS`, instruments → `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`, lanes → `trading_app.prop_profiles.ACCOUNT_PROFILES`, **ORB window timing → `pipeline.dst.orb_utc_window(trading_day, orb_label, orb_minutes)`** (never derive from `break_delay_min`; never fall back to `break_ts`; see `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`).

### Backtesting Methodology (MANDATORY)
Every backtest / discovery scan MUST follow `.claude/rules/backtesting-methodology.md` (auto-loads on research/ / trading_app/strategy_* / docs/audit/ / docs/institutional/ edits). 14 rules + companion failure log.

### Research Provenance Rule
Config values from research need `@research-source`, `@entry-models`, `@revalidated-for`. Drift check #45 enforces.

### Source-of-Truth Chain Rule
Identify canonical source → verify downstream derives from it → if source may be wrong, audit upstream first → never patch downstream to compensate for upstream corruption.

### Local Academic / Project-Source Grounding Rule
Prefer `resources/` PDFs + project canon over training memory; say **UNSUPPORTED if no local source**. **Entry point: `resources/INDEX.md`** → curated page-cited extracts in `docs/institutional/literature/` (canonical citation source). Extract text — never cite from memory as if read; label training-memory claims explicitly. Detail (index generation, dismissal mechanics, MCP query) → `docs/specs/research_modes_and_lineage.md` § 9.2.

### Audit-First Default for Research Layers
Research layers: **audit → adversarial audit → fix → rerun → freeze → move on**. Do not skip to implementation when truth-state is unverified.

### Institutional Rigor (MANDATORY — non-negotiable)
Proper institutional-grounded fixes only. No band-aids, dead code, silent failures, or re-encoded canonical logic. Self-review before done; refactor (don't patch) when cycles keep finding bugs. → `.claude/rules/institutional-rigor.md` (auto-loads on production-code edits).

**Phase 0 grounding (2026-04-07):** discovery/validation/deployment decisions must cite `docs/institutional/literature/` (Bailey-López de Prado, Harvey-Liu, Chordia, Pepelyshev-Polunchenko, Fitschen, Carver). Pre-registered criteria in `docs/institutional/pre_registered_criteria.md`; mechanism priors in `docs/institutional/mechanism_priors.md`; hypothesis files in `docs/audit/hypotheses/`. **No brute-force >300 trials** (MinBTL bound).

### 2-Pass Implementation Method (MANDATORY)
1. **Discovery:** Read affected files, understand blast radius, articulate PURPOSE before writing code.
2. **Implementation:** Write → verify (drift + tests + behavioral audit) → fix regressions → **self-review** → fix new findings.

One task at a time. Never batch without verification. **"Done" = tests pass (show output) + dead code swept (`grep -r`) + `check_drift.py` passes + self-review passed.** All four required.

### Design Proposal Gate (MANDATORY)
Before writing ANY code on a non-trivial change, present: (1) **What** and why, (2) **Files** to touch, (3) **Blast radius**, (4) **Approach**.

**Self-check (DO NOT SKIP):** Simulate happy path, edge case (NULL/empty/sparse), and failure mode internally. SHOW what you tested and found — don't just claim you checked. Performative self-review (claiming you checked without showing evidence) is worse than no self-review. Fix flaws before presenting. Your first draft is always wrong.

Wait for user confirmation before writing code. Exceptions: trivial changes, "just do it", git ops, read-only exploration.

### Context Budget (automatic — no action required)
A per-turn `[CTX: N%]` gauge is injected by `.claude/hooks/context-gauge.py`; follow its tier directives. Past Tier 4 (~100% of cap), subagent spawns are hard-blocked by `.claude/hooks/subagent-budget-guard.py` — `/clear` is the only path forward. Detail → `.claude/rules/subagent-budget.md`.

---

## Strategy Classification — Behavioral Rules
Low trade count ≠ bug (G6/G8 filters expected). Verify `trade_days <= eligible_days` first. If `trade_days > eligible_days` → corruption. Never "fix" filters to increase N. Never recommend REGIME as standalone. Trading logic → `TRADING_RULES.md`. Thresholds → `docs/ARCHITECTURE.md`.

Sample tiers: REGIME 30-99; CORE >= 100. Caveats: `RESEARCH_RULES.md`.


Do NOT reference unbuilt features in code or tests — see `ROADMAP.md` for what's planned.
