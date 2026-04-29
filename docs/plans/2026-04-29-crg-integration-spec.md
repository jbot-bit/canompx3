# CRG Integration — Spec & Phased Plan (v2 — gap-audited)

**Date:** 2026-04-29
**Status:** APPROVED — implementing phased
**Authority:** `.claude/skills/design`, `superpowers:brainstorming`
**Predecessor:** `docs/runtime/handoff-crg-calibration-2026-04-28.md` (parked at PASS 1)
**Supersedes:** v1 of this doc (gaps closed: G1–G8, S1–S4)

## Purpose

Maximise honest use of `code-review-graph` (CRG) inside this repo. CRG is built (1,034 files / 13,738 nodes / 12,693 embeddings, v2.3.2) and `.mcp.json`-registered, but only 1 of 24 MCP tools is wired into our workflow and zero hooks consume the graph.

This spec is the canonical plan. It is intentionally complete: every capability CRG offers, every place in our system that should consume it, every silence (with reason), and every measurable acceptance criterion.

## Hard non-negotiables (Bloomberg-grade discipline)

1. **No silences.** Every CRG capability is enumerated with a verdict: ADOPT / DEFER / SKIP-with-reason.
2. **No fake numbers.** Token-savings claims must be measured on this repo via `code-review-graph eval`. The published "6.8× / 49×" is theirs, not ours, until we benchmark.
3. **Fail-open advisory; fail-closed structural.** A graph staleness or MCP outage must NEVER block a commit. Drift checks built on CRG queries fail-closed only when the contract is structural-must-hold (canonical-import enforcement, test-coverage of canonical functions).
4. **One source of truth.** No CRG-managed artifact may compete with hand-curated doctrine (CLAUDE.md, TRADING_RULES.md, RESEARCH_RULES.md, STRATEGY_BLUEPRINT.md, `memory/*.md`).
5. **Worktree-aware or worktree-agnostic — never half-pregnant.** The graph DB lives at the main checkout; linked worktrees consume it read-only and accept the staleness contract specified in §Freshness.
6. **Zero risk to live trading.** CRG is meta-tooling. No code path it touches feeds into trading execution. Every drift check fails-open if CRG itself is broken.

## Full capability inventory — every CRG surface and our verdict

Source: `docs/external/code-review-graph/{COMMANDS,FEATURES,USAGE,LLM-OPTIMIZED-REFERENCE}.md` + installed `.venv/Lib/site-packages/code_review_graph/` modules.

### MCP tools (24)

| Tool | Verdict | Where consumed |
|------|---------|----------------|
| `get_minimal_context_tool` | ADOPT | Phase 1 — session-start hook, all agents' first call |
| `list_graph_stats_tool` | ADOPT | Phase 1 — session-start, drift-check freshness gate |
| `build_or_update_graph_tool` | ADOPT | Phase 1 — auto-update hook (PostToolUse) |
| `get_impact_radius_tool` | ADOPT | Phase 1 — `/crg-blast`; Phase 3 — `pre-edit-guard` advisory |
| `query_graph_tool` | ADOPT | Phase 2 — drift checks #126/#127; Phase 3 — `/crg-tests`, `/crg-lineage` |
| `get_review_context_tool` | ADOPT | Phase 3 — `evidence-auditor`, `verify-complete` agents |
| `semantic_search_nodes_tool` | ADOPT | Phase 1 — `/crg-search`; replaces first-call grep across agents |
| `embed_graph_tool` | ADOPT (one-shot) | Phase 1.5 — verify embeddings index, document model |
| `find_large_functions_tool` | ADOPT | Phase 2 — drift check #128 (canonical-path size cap) |
| `get_docs_section_tool` | ADOPT | Phase 1 — slash commands cite via this, not hardcoded |
| `list_flows_tool` | ADOPT | Phase 3 — `evidence-auditor` flow tracing |
| `get_flow_tool` | ADOPT | Phase 3 — `evidence-auditor` |
| `get_affected_flows_tool` | ADOPT | Phase 2 — `/crg-lineage`; Phase 3 — `verify-complete`, `evidence-auditor` |
| `list_communities_tool` | ADOPT | Phase 3 — `planner` agent architecture context |
| `get_community_tool` | ADOPT | Phase 3 — `planner` |
| `get_architecture_overview_tool` | ADOPT | Phase 3 — `planner`, `evidence-auditor` |
| `detect_changes_tool` | ADOPT | Phase 3 — `verify-complete` agent, PR-review skill |
| `get_hub_nodes_tool` | ADOPT | Phase 1 — branch-context hook uses to identify "canonical" files |
| `get_bridge_nodes_tool` | ADOPT | Phase 2 — drift check #129 (chokepoint test-coverage) |
| `get_knowledge_gaps_tool` | ADOPT | Phase 3 — `ralph-loop` audit preamble |
| `get_surprising_connections_tool` | ADOPT | Phase 2 — drift check #125 (cross-layer coupling) |
| `get_suggested_questions_tool` | DEFER | Low-leverage; revisit if `evidence-auditor` shows gaps |
| `refactor_tool` | ADOPT (preview-only) | Phase 3 — `/crg-deadcode` slash command |
| `apply_refactor_tool` | SKIP | Refactors touching production go through stage-gate; never auto-apply |
| `generate_wiki_tool` / `get_wiki_page_tool` | SKIP | Competes with hand-curated doctrine; one-source-of-truth rule |
| `list_repos_tool` / `cross_repo_search_tool` | SKIP | Single-repo project |

**ADOPT count:** 20 of 24 tools wired. **DEFER:** 1. **SKIP-with-reason:** 4 (one tool family + apply_refactor + cross_repo + wiki).

### MCP prompts (5)

| Prompt | Verdict | Where consumed |
|--------|---------|----------------|
| `review_changes` | ADOPT | Phase 3 — `verify-complete` agent uses this as its diff-review preamble |
| `architecture_map` | ADOPT | Phase 3 — `planner` agent, on demand |
| `debug_issue` | ADOPT | Phase 3 — `quant-debug` skill preamble |
| `onboard_developer` | DEFER | New-dev onboarding not a current need |
| `pre_merge_check` | ADOPT | Phase 3 — `/open-pr` / `/ship` skill preamble |

### CLI commands

| Command | Verdict | Trigger |
|---------|---------|---------|
| `code-review-graph build` | ADOPT | Manual / one-shot after canonical refactors |
| `code-review-graph update` | ADOPT | Phase 1 — PostToolUse hook on `Edit\|Write` to canonical paths |
| `code-review-graph status` | ADOPT | Phase 1 — session-start prints stats |
| `code-review-graph watch` | DEFER (use hook instead) | Hook is more deterministic and lives inside our existing `post-edit-pipeline.py` flow |
| `code-review-graph visualize` | ADOPT (on-demand) | Phase 3 — `/crg-visualize` for human auditors |
| `code-review-graph wiki` | SKIP | One-source-of-truth |
| `code-review-graph eval` | ADOPT | Phase 1.5 — baseline measurement |
| `code-review-graph register/repos/daemon` | SKIP | Single-repo |
| `code-review-graph serve` | ALREADY-WIRED | `.mcp.json` invokes via `uvx` |

### Hooks integration patterns (per official guide)

| Pattern | Verdict | Implementation |
|---------|---------|----------------|
| PostToolUse `Edit\|Write` → `update` | ADOPT | Phase 1 — extend `.claude/hooks/post-edit-pipeline.py` with `_crg_update()` |
| SessionStart hook → graph awareness | ADOPT | Phase 1 — extend `.claude/hooks/session-start.py` with `_crg_context_lines()` |
| `code-review-graph watch` daemon | SKIP | Replaced by PostToolUse — deterministic, bounded, in-process |
| Pre-commit hook → `update` | ADOPT | Phase 1 — extend `.githooks/pre-commit` with `update --base HEAD` line |
| Git post-commit hook → `update` | DEFER | Pre-commit + PostToolUse already cover the surface |

### Internal modules (verified by reading source)

| Module | Capability | Verdict |
|--------|-----------|---------|
| `flows.py` | Entry-point detection (HTTP, CLI, tests) + BFS criticality scoring | Used via `list_flows`/`get_flow` tools |
| `communities.py` | Leiden algorithm clustering with weighted edges | Used via community tools |
| `analysis.py` | Hub nodes, bridge nodes (betweenness), knowledge gaps, surprise scoring | Used via tools above |
| `search.py` | FTS5 + vector RRF hybrid, query-aware kind boosting | Used via `semantic_search_nodes` (auto-uses both) |
| `refactor.py` | Preview-then-apply with 10-min expiry, path-traversal prevention | Used via `refactor_tool` (preview only) |
| `incremental.py` | git-diff-driven re-parse, 8-worker pool, default ignore patterns | Wired via `update` hook |
| `embeddings.py` | Local (MiniLM 384), Gemini, MiniMax (1536) | Phase 1.5 verifies which is in use |
| `eval/scorer.py` | MRR, precision/recall, token-efficiency metrics | Phase 1.5 baseline |
| `memory.py` | Saves Q&A markdown to `.code-review-graph/memory/` for re-ingestion | **SKIP** — competes with `memory/*.md` (NO-GO `nogo_claude_mem_2026_04_27.md`) |
| `hints.py` | In-memory session state, next-step suggestions appended as `_hints` | Free — auto-active when MCP tools called |
| `registry.py` | Multi-repo registry at `~/.code-review-graph/registry.json` | SKIP — single-repo |

## Design principles

1. **Token efficiency first.** Every CRG call uses `detail_level="minimal"` per official LLM-OPTIMIZED-REFERENCE. Escalate only when needed. Target ≤5 tool calls / ≤800 tokens per task.
2. **Replace, don't duplicate.** When CRG covers what an existing hook/check does (worse), retire the duplicate. When it complements, layer.
3. **Fail-open advisory; fail-closed structural.** See non-negotiable §3.
4. **Incremental verification.** Each phase ships independently with its own drift-check pass + test run + measurable acceptance.
5. **Worktree-aware staleness contract.** Defined in §Freshness.

## Freshness contract (S2 + S3 closed)

The graph DB lives at `<main_checkout>/.code-review-graph/graph.db`. Linked worktrees read it; they do not maintain their own.

**Staleness states:**
- **FRESH**: graph DB mtime ≥ HEAD-commit timestamp on the main checkout. All MCP tools return current truth.
- **DIRTY-EDIT**: any uncommitted file in `pipeline/`, `trading_app/`, `scripts/` since last `update`. PostToolUse hook auto-runs `update` after each edit, so this state should last <2s in practice.
- **STALE**: graph DB mtime > 24h old AND HEAD has moved. Session-start prints a 1-line warning and runs `update` opportunistically (timeout 10s, fail-silent).
- **MISSING**: graph DB doesn't exist. All CRG drift checks emit `ADVISORY: CRG unavailable — run code-review-graph build` and exit 0 (advisory only).

**Drift-check freshness gate:** every CRG-backed drift check checks `list_graph_stats_tool` first. If FRESH or DIRTY-EDIT → run check. If STALE → emit `ADVISORY: graph stale, refreshing`, run `update`, retry once. If still STALE or MISSING → emit `ADVISORY` and exit 0.

**Worktree consumers:** linked worktrees (`canonaudit`, `cockpit-ledger-20260428`, `gitnexus-eval`, our own `feature/crg-integration-2026-04-29`) read the main checkout's graph DB via absolute path. They do NOT write. This means a worktree-only edit isn't reflected until that branch lands on main and the main checkout's PostToolUse hook fires. Acceptable because:
- worktree branches are short-lived
- drift checks run on CI against the main-checkout graph after merge
- in-worktree advisory tools still get useful (slightly stale) signal

## Curated use-set

Tier ordering = EV ÷ implementation cost. ✅ items ship in this spec.

### Tier 0 — Foundation (Phase 1)

- ✅ **F1. Wire `get_minimal_context` + `list_graph_stats` into `session-start.py`.** One ~80-token line: graph stats + freshness state + risk overview. Replaces nothing; adds always-on graph awareness.
- ✅ **F2. Branch-context pre-edit hook.** PreToolUse on `Edit|Write` to `pipeline/`, `trading_app/`, `scripts/`. Block when current branch starts with `research/` or `session/` AND target is a canonical (= hub-node-identified) file. CRG narrows "canonical" via `get_hub_nodes_tool`; the block decision itself uses git only (so it works even when CRG is down).
- ✅ **F3. PostToolUse auto-update hook.** Extend `.claude/hooks/post-edit-pipeline.py` with `_crg_update()`. Runs `code-review-graph update --base HEAD~1` for any edit under `pipeline/`, `trading_app/`, `scripts/`, `research/`, `tests/`. Timeout 5s. Failures silent. Closes G1.
- ✅ **F4. Pre-commit hook calls `update`.** Add a quick `code-review-graph update --base origin/main` line to `.githooks/pre-commit`. Idempotent on top of F3; defends against PostToolUse misses.
- ✅ **F5. Direct MCP calls in `/crg-*` slash commands.** Retire bash fallbacks. Per-call permission prompts handled by per-prompt approval (settings.json was reverted — DO NOT auto-amend).
- ✅ **F6. SKIP-document.** Add explicit lines to spec marking CRG memory (`.code-review-graph/memory/`), wiki, cross-repo, `apply_refactor` as DO-NOT-USE in this repo (G5 closed).

### Tier 1 — Measurement & freshness (Phase 1.5 — NEW)

- ✅ **M1. One-shot `embed_graph_tool` run.** Verify embeddings are current; document the model in `docs/external/code-review-graph/EMBEDDINGS.md` (NEW). Closes G6. (~30s.)
- ✅ **M2. Run `code-review-graph eval` baseline.** Store output in `docs/external/code-review-graph/EVAL-BASELINE-2026-04-29.json`. Closes S1. Re-run quarterly. Honest numbers, not their published.
- ✅ **M3. Token-savings instrumentation.** Add a tiny logging shim that records, per agent invocation, whether CRG was used and approximate token count (input/output). Append to `.code-review-graph/usage-log.jsonl`. Closes S1. Quarterly review reveals adoption rate.

### Tier 2 — Drift checks built on CRG (Phase 2)

- ✅ **D1. Drift check #125 — cross-layer surprising connections.** Uses `get_surprising_connections_tool`. Rejects new edges between `pipeline/` and `trading_app/` that bypass published canonical surfaces (`pipeline.dst.SESSION_CATALOG`, `pipeline.cost_model.COST_SPECS`, `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`, etc.). Catches Project Truth Protocol violations.
- ✅ **D2. Drift check #126 — canonical-source import enforcement.** **AST-based fallback** (CRG v2.1.0 has no reliable `imports_from` graph pattern at the qualified-name granularity required; pivoted in implementation). Walks `research/` ASTs and flags any module that defines local symbols named after a canonical callable without an `import` from the canonical module. Hard list (verified at implementation):
  - `parse_strategy_id` ← `trading_app.eligibility.builder`
  - `orb_utc_window` ← `pipeline.dst`
  - `detect_break_touch` ← `trading_app.entry_rules`
  - `_load_strategy_outcomes` ← `trading_app.strategy_fitness` (added 2026-04-29 post-audit; 27 callers across repo, 0 redefinitions in research/ today)

  Spec-vs-impl audit-trail (2026-04-29): original spec listed 5 symbols including `reprice_e2_entry`. Excluded because canonical lives in `research/databento_microstructure.py` (not yet promoted to `pipeline.cost_model`); flagging research/ for "re-implementing" itself would be a false positive. Re-add when promoted. Memory: `feedback_aperture_overlay_canonical_parser.md`, `feedback_canonical_value_unit_verification.md`. Currently 3 real findings on landing day.
- ✅ **D3. Drift check #127 — canonical functions must have at least one import-and-call test.** **AST-based fallback** (CRG v2.1.0 `tests_for` graph proven incomplete during implementation: returns 0 edges for canonical functions that have verified tests). **Scope: callables only.** For each canonical callable below, walks `tests/` ASTs looking for `from <canonical_module> import <symbol>` plus a Call site referencing it. Hard list:
  - `pipeline.dst.orb_utc_window`
  - `trading_app.eligibility.builder.parse_strategy_id`
  - `trading_app.entry_rules.detect_break_touch`

  **Constants are out of scope** (audit-trail 2026-04-29). The original spec listed `SESSION_CATALOG`, `COST_SPECS`, `ACTIVE_ORB_INSTRUMENTS`, `HOLDOUT_SACRED_FROM`, and `reprice_e2_entry` too. AST cannot detect `Call` nodes against constants — they are referenced, not called — so a test that imports `SESSION_CATALOG` and asserts on it would not register under this mechanism. **D1 also does NOT cover constants** (D1 explicitly skips edges that pass through canonical surfaces — that's its design, not a gap to lean on). Constant test-coverage is a separate, currently-unbuilt drift check (planned: presence of `from <canonical_module> import <CONSTANT>` in any `tests/` file). Tracked as open follow-up below. `reprice_e2_entry` excluded for the same reason as D2.
- ✅ **D4. Drift check #128 — canonical-path function size cap.** Uses CRG `analysis.find_large_functions` directly (bypasses CRG `analysis_tools._func` wrapper bug — see helpers module). `min_lines=200`, file-path filter `pipeline/` or `trading_app/`. Catches monster-function class. Closes G7. Currently 29 findings.
- ✅ **D5. Drift check #129 — bridge-node test coverage.** Uses CRG `analysis.find_bridge_nodes` + `analysis.query_tests_for`. Filters bridge nodes (top betweenness-centrality) to canonical paths, then asks for tests. Catches the "everything depends on this and nothing tests it" class. Currently 8 findings — **likely shares D3's incomplete-graph false-positive class** (CRG `tests_for` graph still consulted here; AST rebuild is a known follow-up before raising D5 from advisory to blocking).
- ✅ **D6. Predicate-lineage auditor (`/crg-lineage`).** Uses `get_affected_flows_tool` + `query_graph_tool(pattern="callers_of")`. Enumerates every research script that consumes a given `daily_features.<column>`. Replaces hand-grep that built the 29-entry E2 LA registry. Closes the next contamination class before it bites.

### Tier 3 — Agent / workflow upgrades (Phase 3)

- ✅ **A1. Pre-edit blast-radius advisory.** Extends `pre-edit-guard.py`. On `Edit|Write` to `pipeline/` or `trading_app/`: call `get_impact_radius_tool(detail_level="minimal", max_depth=2)` and emit top-5 affected files as a stderr notice. Non-blocking. 2s timeout. Fail-open. (G3 closed.)
- ✅ **A2. `verify-complete` agent gets `detect_changes_tool`.** Adds CRG risk-scored change report after the existing drift+tests pass. Catches stale-test and missing-coverage classes that drift checks #126/#127 don't (those are static lists; `detect_changes` is diff-aware). (G2 closed.)
- ✅ **A3. `evidence-auditor` agent gets CRG structural ground-truth.** Per `adversarial-audit-gate.md`, this agent runs after every CRIT/HIGH commit and requires "independent context." Wire `get_review_context_tool` + `get_affected_flows_tool` + `query_graph_tool(pattern="tests_for")` into its preamble. Iteration 174's kill-switch race would have surfaced via `get_affected_flows`. (G2 closed.)
- ✅ **A4. `blast-radius` agent gets `get_impact_radius_tool` as first call.** Text edit only — instructs the agent to query CRG before grep. Falls back to grep if CRG unavailable. (G3 closed.)
- ✅ **A5. `planner` agent gets `get_architecture_overview_tool` + `list_communities_tool`.** Text edit. Architecture context cheap on first turn. (G3 closed.)
- ✅ **A6. `ralph-loop` agent gets `get_minimal_context_tool` + `get_knowledge_gaps_tool` preamble.** Highest-leverage wiring — this loop runs nightly and audits everything. (G4 closed.)
- ✅ **A7. `/crg-deadcode` slash command.** Wraps `refactor_tool(mode="dead_code")`. Preview only; never apply.
- ✅ **A8. `/crg-visualize` slash command.** Triggers `code-review-graph visualize` (D3.js HTML), opens in browser. On-demand human-auditor surface.
- ✅ **A9. `/open-pr` and `/ship` skills get `pre_merge_check` MCP prompt.** Preamble injection — risk-scored review before PR opens.

### Excluded — explicit DO-NOT-USE list (S4 + G5 closed)

- ❌ **`.code-review-graph/memory/`.** Saves Q&A markdown for re-ingestion. Competes with `memory/*.md` and would create a second source of truth. NO-GO `nogo_claude_mem_2026_04_27.md` documents this exact failure class for `claude-mem`. **Verified disabled** — no `memory/` directory under `.code-review-graph/` and we never call `crg.memory.save_result`.
- ❌ **`generate_wiki` / `get_wiki_page`.** Hand-curated CLAUDE.md, TRADING_RULES.md, RESEARCH_RULES.md, STRATEGY_BLUEPRINT.md are doctrine. Auto-wiki competes; any divergence creates ambiguity over which is authoritative.
- ❌ **`apply_refactor_tool` automation.** Refactors touching production go through stage-gate. CRG previews only.
- ❌ **`code-review-graph watch` daemon.** Replaced by PostToolUse hook (more deterministic, in-process, bounded).
- ❌ **Cross-repo / multi-repo registry.** Single-repo project.
- ❌ **`get_suggested_questions_tool`.** Deferred; low leverage relative to direct flow analysis.
- ❌ **`onboard_developer` MCP prompt.** Not a current need.

## Architecture

### Where CRG lives in our stack

```
.mcp.json                         — MCP server registration (already present)
.claude/settings.json             — MCP perms via per-prompt approval (NOT pre-allowlisted; respects user revert)
.claude/hooks/session-start.py    — Phase 1 adds _crg_context_lines() (F1)
.claude/hooks/branch-context.py   — Phase 1 NEW: PreToolUse hook (F2)
.claude/hooks/post-edit-pipeline.py — Phase 1 extends with _crg_update() (F3)
.claude/hooks/pre-edit-guard.py   — Phase 3 extends with _blast_radius_advisory (A1)
.claude/commands/crg-*.md         — Phase 1 retires bash fallbacks (F5); Phase 2/3 adds /crg-lineage, /crg-deadcode, /crg-visualize
.claude/agents/{verify-complete,evidence-auditor,blast-radius,planner,ralph-loop}.md
                                  — Phase 3 text-only edits adding CRG to first-call ladder (A2-A6)
.githooks/pre-commit              — Phase 1 adds `code-review-graph update` line (F4)
pipeline/check_drift.py           — Phase 2 adds checks #125-#129 (D1-D5)
pipeline/check_drift_crg_helpers.py — Phase 2 NEW (D1-D5 helpers)
docs/external/code-review-graph/  — vendored official docs (already present)
docs/external/code-review-graph/EMBEDDINGS.md         — Phase 1.5 NEW (M1)
docs/external/code-review-graph/EVAL-BASELINE-2026-04-29.json — Phase 1.5 NEW (M2)
.code-review-graph/usage-log.jsonl — Phase 1.5 NEW (M3, gitignored)
```

### One-way dependency check

CRG is meta-tooling. It reads code; it never writes to `pipeline/` or `trading_app/` runtime artifacts. No new runtime dependency on the graph DB. Drift checks fail-closed only on structural contracts; they fail-open if CRG is unavailable.

### Trading-execution isolation

Live trading paths (`trading_app/live/`, `trading_app/risk/`) consume zero CRG output. The graph influences only:
- Pre-commit and PreToolUse hooks (advisory)
- Drift checks (advisory if CRG broken)
- Agent system prompts (read-only)

This isolation is why fail-open is safe.

## Phase 1 — Foundation (~half day)

**Scope-lock files:**
- `.claude/hooks/session-start.py` (add `_crg_context_lines()` + call from `main()`) — F1
- `.claude/hooks/branch-context.py` (NEW) — F2
- `.claude/hooks/post-edit-pipeline.py` (add `_crg_update()`) — F3
- `.githooks/pre-commit` (add update line) — F4
- `.claude/commands/crg-context.md`, `crg-search.md`, `crg-blast.md`, `crg-tests.md` (retire bash fallbacks) — F5
- This spec doc updated with explicit DO-NOT-USE list — F6

**Acceptance:**
- Session-start prints one CRG line: `Graph: 1034 files / 13738 nodes — risk low — fresh` (or `stale`/`unavailable`).
- Edit to `pipeline/cost_model.py` from a `research/*` branch is BLOCKED with a clear message.
- Edit to `pipeline/cost_model.py` from `main` proceeds normally.
- Edit to a `research/` script from a `research/*` branch proceeds normally.
- Edit to `pipeline/cost_model.py` triggers a background `code-review-graph update` (verify via graph mtime).
- `/crg-context <task>` calls MCP tool directly when allowed; bash fallback still present (F5 changes preference, doesn't delete fallback).
- Drift check passes; existing test suite passes (no regressions).

## Phase 1.5 — Measurement & freshness (~1 hour)

**Scope-lock files:**
- `docs/external/code-review-graph/EMBEDDINGS.md` (NEW) — M1
- `docs/external/code-review-graph/EVAL-BASELINE-2026-04-29.json` (NEW) — M2
- `.gitignore` (add `.code-review-graph/usage-log.jsonl`) — M3
- (no Python yet — the usage-log shim ships in Phase 3 with the agent edits)

**Acceptance:**
- `embed_graph_tool` run completes; embeddings model documented.
- `code-review-graph eval` produces baseline JSON; committed to repo.
- `.gitignore` includes the usage-log path.

## Phase 2 — Drift checks built on CRG (~3 days)

**Scope-lock files:**
- `pipeline/check_drift.py` (add 5 new check functions, all fail-open on CRG outage) — D1-D5
- `pipeline/check_drift_crg_helpers.py` (NEW — isolates CRG MCP/CLI calls so check_drift.py stays graph-DB-agnostic)
- `.claude/commands/crg-lineage.md` (NEW) — D6
- `tests/test_pipeline/test_check_drift_crg.py` (NEW — fixture-based, no live graph) — coverage for D1-D5

**Canonical-function lists:** as in §D3.

**Acceptance:**
- Each new check passes on current `main`.
- Pressure tests:
  - Introduce deliberate violation (research script importing a re-implementation of `parse_strategy_id`) → check #126 catches it.
  - Add 250-line function to `pipeline/cost_model.py` → check #128 catches it.
  - Delete a test for `orb_utc_window` → check #127 catches it.
- `/crg-lineage daily_features.rel_vol_NYSE_OPEN` enumerates ≥3 research files (E2 LA registry already documents this).
- Each check fails-open with `ADVISORY: CRG unavailable` when graph DB is moved/locked; never blocks commits when CRG itself is broken.
- `python pipeline/check_drift.py` reports all checks passing, with 5 new advisory entries (D1-D5) under headings 125-129.

### Open follow-ups (Phase 2.x — non-blocking, advisory-only carry-forwards)

Recorded 2026-04-29 from the institutional-rigor audit on f4d3cab5 (HIGH/MEDIUM findings 1-5):

1. **D5 AST rebuild.** D5 still consults CRG `tests_for` (via `query_tests_for` helper). CRG v2.1.0's `tests_for` graph is the same incomplete graph that forced D3 to pivot to AST; D5 likely shares the false-positive class. Recommended: rebuild D5's per-bridge-node test query as an AST scan (parity with D3) before raising D5 from advisory to blocking.
2. **D3 constants extension (currently-unbuilt check).** Cover canonical constants (`SESSION_CATALOG`, `COST_SPECS`, `ACTIVE_ORB_INSTRUMENTS`, `HOLDOUT_SACRED_FROM`) via a new AST mechanism: presence of `from <canonical_module> import <CONSTANT>` in any `tests/` file. Today no check covers this class — D1 explicitly skips canonical surfaces, D3 only counts `Call` nodes.
3. **D3 attribute-access detection.** `import X as Y; Y.parse_strategy_id(...)` is currently flagged as untested even when it exercises the canonical via attribute access. Mechanically detectable but requires module-aliasing tracking. Documented as a known limitation in `tests/test_pipeline/test_check_drift_crg.py`.
4. **`reprice_e2_entry` re-add when promoted.** When `research/databento_microstructure.py::reprice_e2_entry` is promoted to `pipeline.cost_model`, add it back to D2's canonical dict and D3's `canonical_callable` dict.
5. **Live-graph integration test.** All 33 D1-D5 unit tests use mocks. CRG API regression (e.g., `find_bridge_nodes` signature change) would silently degrade D1/D4/D5 from advisory-with-findings to advisory-skip without test failure. Add a smoke test that runs the helpers against the real graph DB if present, skips otherwise.

## Phase 3 — Agent / workflow upgrades (~2 days)

**Scope-lock files:**
- `.claude/hooks/pre-edit-guard.py` (extend with `_blast_radius_advisory()`) — A1
- `.claude/agents/verify-complete.md` (text edit) — A2
- `.claude/agents/evidence-auditor.md` (text edit) — A3
- `.claude/agents/blast-radius.md` (text edit) — A4
- `.claude/agents/planner.md` (text edit) — A5
- `.claude/agents/ralph-loop.md` (text edit) — A6
- `.claude/commands/crg-deadcode.md` (NEW) — A7
- `.claude/commands/crg-visualize.md` (NEW) — A8
- `.claude/skills/open-pr/SKILL.md` and/or `.claude/commands/open-pr.md` (text edit) — A9
- Usage-log shim in `.claude/hooks/_crg_usage_log.py` (NEW) — M3 implementation

**Acceptance:**
- Edit to a `pipeline/` file emits ~5-line stderr notice with top-5 impacted files (A1).
- `verify-complete` agent's protocol now includes a `detect_changes_tool` call before declaring done; prompt verified in agent file (A2).
- `evidence-auditor` agent invokes `get_review_context_tool` + `get_affected_flows_tool` first (A3).
- `blast-radius`, `planner`, `ralph-loop` agents have CRG as first-call instruction (A4-A6).
- `/crg-deadcode` returns dead-code list for current branch's diff (A7).
- `/crg-visualize` produces HTML and opens in browser (A8).
- `/open-pr` preamble shows `pre_merge_check` output (A9).
- Usage log records 1 entry per agent invocation that uses CRG.

## Failure-mode matrix

| Mode | Detection | Recovery | Trading impact |
|------|-----------|----------|----------------|
| Graph DB locked | `list_graph_stats` returns lock error | Retry once after 100ms; if still locked, advisory only | None (advisory) |
| Graph DB missing | File doesn't exist | All CRG checks skip with `ADVISORY: CRG unavailable` | None |
| Graph stale (>24h) | `list_graph_stats` mtime check | Auto `update`, retry once; if still stale, advisory only | None |
| MCP server crashed | First call timeout | Drift check fail-open; advisory hook silent | None |
| Embeddings model missing | `semantic_search_nodes` falls back to FTS5 | Documented behavior, no action | None |
| `update` runs slow on big rebuild | Hook timeout 5s | Skip silently, next call retries | None — edit proceeds |
| Branch-context hook false-positive | Block message + escape hatch | `BRANCH_CONTEXT_OVERRIDE=1` env var | None |
| Pre-edit blast-radius hook timeout | 2s cap | Skip notice silently | None |
| Worktree consumes stale graph | Documented in §Freshness | Accept; update on merge | None |

## Rollback plan per phase

- **Phase 1:** revert the commit. `session-start.py`, slash commands, branch-context hook, post-edit-pipeline addition are isolated; no DB or canonical changes.
- **Phase 1.5:** delete the new docs + eval artifact. No code change to revert.
- **Phase 2:** drift checks are advisory + fail-open by default. Worst case: revert the 5 new check functions + the helper module.
- **Phase 3:** every agent edit is text-only; revert reverts behavior. Pre-edit-guard extension is opt-out via `CRG_ADVISORY_DISABLED=1` env var.

## Verification commands per phase

```bash
# Phase 1
python pipeline/check_drift.py
.venv/Scripts/python.exe -m pytest tests/test_claude_hooks/ -x -q
# Manual smoke test: switch to research/* branch, attempt edit to pipeline/ file → BLOCKED

# Phase 1.5
.venv/Scripts/python.exe -c "from code_review_graph.tools.context import get_minimal_context; print(get_minimal_context(task='smoke', repo_root='.'))"
code-review-graph eval --output docs/external/code-review-graph/EVAL-BASELINE-2026-04-29.json

# Phase 2
python pipeline/check_drift.py    # 129 checks pass on main
.venv/Scripts/python.exe -m pytest tests/test_pipeline/test_check_drift_crg.py -x -q

# Phase 3
python pipeline/check_drift.py
.venv/Scripts/python.exe -m pytest tests/ -x -q --timeout=60
# Manual: invoke verify-complete agent, confirm detect_changes appears in protocol
```

## Measurable success criteria (Bloomberg-grade)

These are committed, not aspirational:

1. **Token reduction baseline:** `eval` output captures pre-CRG token use across 5 sample tasks; post-Phase 3, re-run shows ≥2× reduction on at least 3 of 5 (we set the bar honestly low — their "6.8×/49×" is published data we don't yet replicate).
2. **Drift check coverage:** 5 new structural checks, all green on `main` after Phase 2.
3. **Adversarial audit ground-truth:** `evidence-auditor` agent receives flow-trace input on every CRIT/HIGH commit (verifiable by usage log).
4. **Branch-context block rate:** the F2 hook fires at least once per session in ≥80% of sessions where edits to canonical paths are attempted (verifiable from hook stderr logs over 2 weeks).
5. **No false-positive blocks:** zero verified user reports of incorrect block from F2 over 2-week observation period; if any, override env var documented and escape works.
6. **Zero trading-execution regressions:** no test in `tests/test_trading_app/test_live*` or `tests/test_trading_app/test_risk*` changes status. Verified pre/post each phase.

## Sources

- [code-review-graph (tirth8205)](https://github.com/tirth8205/code-review-graph)
- [code-review-graph CLAUDE.md integration guide](https://github.com/tirth8205/code-review-graph/blob/main/CLAUDE.md)
- `docs/external/code-review-graph/{COMMANDS,FEATURES,USAGE,LLM-OPTIMIZED-REFERENCE,architecture,schema}.md` (vendored)
- `.venv/Lib/site-packages/code_review_graph/{flows,communities,analysis,search,refactor,incremental,embeddings,memory,hints,registry}.py` (verified by reading source)
- `docs/runtime/handoff-crg-calibration-2026-04-28.md` (parked PASS 1 handoff)
- Memory: `code_review_graph_calibration.md`, `feedback_parallel_session_awareness.md`, `feedback_aperture_overlay_canonical_parser.md`, `nogo_claude_mem_2026_04_27.md`

## Decision ledger

| Date | Decision | Reason |
|------|----------|--------|
| 2026-04-29 | Adopt 20 of 24 MCP tools | Only 4 incompatible with one-source-of-truth + single-repo + no-auto-apply rules |
| 2026-04-29 | Skip CRG memory module | Two-source-of-truth class; documented NO-GO precedent |
| 2026-04-29 | Skip wiki generation | Competes with hand-curated doctrine |
| 2026-04-29 | Use PostToolUse hook over `watch` daemon | Deterministic, in-process, bounded timeout |
| 2026-04-29 | Per-prompt MCP perm approval (no settings.json allowlist) | User reverted prior allowlist edit; respect that signal |
| 2026-04-29 | Linked worktrees consume read-only graph from main checkout | Avoids per-worktree DB build; documented staleness contract |
| 2026-04-29 | Branch-context hook block decision uses git only (CRG only narrows scope) | Keeps hook working when CRG is down |

