# Handover — Perf Lazy-Import Sweep (2026-04-20)

**Hand this prompt to a fresh Claude session to resume.**

---

## Active state

| Branch | Where | Status |
|---|---|---|
| `perf/lazy-imports-import-time-cleanup` | PR #24 (open) | Done — 4-module original sweep, ~3× check_drift speedup |
| `perf/lazy-imports-broad-sweep-v2` | `.worktrees/broad-sweep/` (this worktree, pushed) | **WIP — Phase 1 of 3 modules in progress** |
| `audit/mgc-adversarial-reexamination` | local-only | 2 MGC commits saved here from another agent that polluted the wrong branch |

## What's done in this worktree

1. Pre-reg plan: `docs/plans/2026-04-20-broad-lazy-import-sweep.md` (committed)
2. Stage file: `docs/runtime/stages/broad_lazy_sweep_phase1.md` (committed)
3. Phase 1 module 1 — `trading_app/strategy_discovery.py` (committed but **not done**, see KNOWN ISSUE)

## Phase 1 plan (3 low-risk CLIs)

| # | Module | Predicted save | Status |
|---|---|---|---|
| 1.1 | `trading_app/strategy_discovery.py` | ~3s | WIP — module-level imports removed, lazy-loads added inside `_flush_batch_df` and `main()`, **but pandas+duckdb still load transitively** — needs `python -X importtime` trace to find the transitive importer |
| 1.2 | `trading_app/walkforward.py` | ~3s | NOT STARTED |
| 1.3 | `trading_app/strategy_validator.py` | ~2.5s | NOT STARTED |

Phases 2 + 3 (live_config 11.9s, build_daily_features 5.8s) come after.

## KNOWN ISSUE (resume here)

After Stage 1.1 commit, `import trading_app.strategy_discovery` still takes 4.6s warm and pulls pandas + duckdb into `sys.modules`. The lazy imports inside `_flush_batch_df` and `main()` are correct, but something ELSE in the import chain (one of `trading_app.config`, `trading_app.outcome_builder`, `trading_app.db_manager`, `trading_app.hypothesis_loader`, `trading_app.phase_4_discovery_gates`) is loading them at module level.

**Resume command:**
```bash
cd /c/Users/joshd/canompx3/.worktrees/broad-sweep
PYTHONPATH=. python -X importtime -c "import trading_app.strategy_discovery" 2>&1 | head -50 > /tmp/importtime.txt
# Find the line that imports pandas/duckdb at module top
```

Once found, decide:
- If transitive importer is shared by many production paths (e.g., `trading_app.config`) — that's a bigger blast-radius change, may need its own stage
- If it's a single helper (e.g., `db_manager`) — apply same lazy pattern there

## Concurrent agent warning

**The MAIN worktree (`C:/Users/joshd/canompx3/`) is being actively modified by another agent** doing a `try/finally → with` refactor across ~33 files. Some of their refactors have left files broken (caught an `IndentationError` in `pipeline/check_drift.py` mid-session — they fixed it later via commits `cb7b532a`/`2ed166ee` which were MGC commits, NOT the refactor). Stay in the worktree (`.worktrees/broad-sweep`); do NOT switch back to main worktree.

## User preferences applied this session

- Token-optimize own context: keep responses concise, avoid re-reading files
- Best Claude practices for code: PEP 8 lazy imports, `functools.lru_cache(maxsize=1)`, `TYPE_CHECKING` guards
- Pre-register plans grounded in official sources (PEP 8 / PEP 562 / PEP 563)
- Phased iteration with measurement before/after each commit
- One commit per module
- Branch from `origin/main`, never local main

## Outstanding requests not addressed yet

User asked late in session: "ensure we are token optimizing at the same time - with best claude practises and coding. fetch and research". Did not have time to:
1. Fetch official Anthropic docs on prompt caching / token optimization (use `mcp__plugin_context7_context7__query-docs` or WebFetch)
2. Audit project's Claude-API code (`trading_app/ai/query_agent.py`, `scripts/tools/coaching_digest.py`, `scripts/tools/trading_coach.py`) for prompt-caching + model-selection best practices
3. Apply findings as a parallel improvement

## Suggested resume prompt

```
Resume the broad lazy-import sweep at .worktrees/broad-sweep on branch
perf/lazy-imports-broad-sweep-v2. Read HANDOVER_2026-04-20-perf-sweep.md
first. Continue from KNOWN ISSUE: trace the transitive pandas/duckdb load in
trading_app.strategy_discovery using `python -X importtime`, then complete
Stage 1.1, then proceed to Stage 1.2 (walkforward) and Stage 1.3
(strategy_validator). After Phase 1, also research+apply Anthropic
prompt-caching best practices to trading_app/ai/* and scripts/tools/{trading_coach,coaching_digest}.py per pre-reg plan
docs/plans/2026-04-20-broad-lazy-import-sweep.md.
```
