---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# 2026-04-20 — Broad Lazy-Import Sweep

**Status:** PRE-REGISTERED PLAN. Not yet implemented.
**Owner:** claude (this session, autonomous)
**Trigger:** PR #24 fixed check_drift import wall but the underlying anti-pattern (heavy module-level imports) affects many other CLIs. User asked: "what about other tools or calls that take ages and shit".
**Predecessor PR:** #24 (perf/lazy-imports-import-time-cleanup). This plan extends the same pattern to the rest of the codebase.

---

## 1. Hypothesis (pre-registered)

**Claim:** Lazy-loading heavy module-level imports in 7 production modules will reduce their cold-cache import wall from ~38s combined to ~7s combined (>80% reduction), without changing observable behavior in any caller. CLI startup time (`--help` etc.) drops proportionally.

**Falsifiable predictions:**
- P1: Per-module wall <2.0s warm for each Phase-1 module.
- P2: All companion tests pass (`tests/test_pipeline/`, `tests/test_trading_app/`).
- P3: Public API surface unchanged — no caller needs modification.
- P4: `python pipeline/check_drift.py` exits 0 after each phase.
- P5: For each module that's a CLI entry point, `python -m <module> --help` still produces help text and exits 0.

**Rollback rule:** any P2/P3/P4/P5 failure → revert that phase's commit and reassess. Do not stack fixes.

---

## 2. Grounding (official Python best practices)

Same as PR #24:
- **PEP 8 — Style Guide**: "It is sometimes useful to delay imports... for performance reasons."
- **PEP 562 — Module __getattr__**: lazy module-level attribute access (used only if needed for clean public API).
- **PEP 563 + `from __future__ import annotations`**: type annotations become strings, allowing `TYPE_CHECKING`-guarded imports for return types.
- **`functools.lru_cache(maxsize=1)`**: official memoization decorator for singleton getters.
- **scientific Python convention**: heavy submodules lazy-loaded at function call time when not always used.

---

## 3. Discovery results (actual measurements, cold subprocess)

| Module | Cold wall | Status | Phase |
|---|---|---|---|
| `trading_app.live_config` | **11.9s** | ok | 2 (biggest single win, blast radius needs review first) |
| `pipeline.build_daily_features` | 5.8s | ok | 3 |
| `trading_app.strategy_discovery` | 5.1s | ok | 1 (CLI, low risk) |
| `trading_app.walkforward` | 4.0s | ok | 1 (research, low risk) |
| `trading_app.execution_engine` | 3.9s | ok | DEFERRED — live trading core, separate plan |
| `trading_app.paper_trader` | 3.8s | ok | DEFERRED — live trading core, separate plan |
| `trading_app.strategy_validator` | 3.4s | ok | 1 (pre-deploy gate, runs once) |
| `trading_app.live.bot_dashboard` | n/a | OSError 10106 | covered by separate plan (FastAPI restructure) |
| `pipeline.ingest_dbn*` | n/a | OSError 10106 | needs warm measurement |
| `trading_app.lane_allocator` | n/a | IndentationError | PRE-EXISTING BUG, not mine |
| `trading_app.eligibility.builder` | n/a | SyntaxError | PRE-EXISTING BUG, not mine |

The Windows OSError 10106 is a `pandas` import-time network probe that fails on this machine — not related to my changes. Modules that fail with it cannot be cold-profiled here, so they get warm measurements during their phase.

---

## 4. Phased plan

### Phase 1 — Low-risk CLIs (~12s saveable)

Start here. These modules are entry-point CLIs or pre-deploy gates — they run once per invocation, not in hot loops. Lowest blast-radius risk.

| Module | Pattern (TBD per module) | Predicted save |
|---|---|---|
| `trading_app/strategy_discovery.py` | profile + lazy heavy deps | ~3s |
| `trading_app/walkforward.py` | profile + lazy heavy deps | ~3s |
| `trading_app/strategy_validator.py` | profile + lazy heavy deps | ~2.5s |

**Phase-1 acceptance:**
- Each module ≤2s warm import
- Each module's CLI `--help` still works
- Companion tests pass
- One commit per module

### Phase 2 — `trading_app/live_config.py` (~11s saveable, MEDIUM risk)

This is the single biggest target but has wide blast radius (loaded by every live-trading entry, every CLI tool that touches strategies). Approach:

1. Measure: `grep -rl "from trading_app.live_config\|import live_config"` to enumerate importers.
2. Profile: which heavy import inside live_config.py is the cost?
3. Decide: lazy-load that import, OR add a minimal "cheap-imports-only" entry point.
4. Implement + verify.

**Phase-2 acceptance:**
- `trading_app.live_config` ≤3s warm import
- All importers verified via spot-check (smoke import → call → result)
- Companion tests pass

### Phase 3 — `pipeline/build_daily_features.py` (~5s saveable)

Same pattern as Phase 2, narrower blast radius (called by ingest scripts and feature-builder CLI).

**Phase-3 acceptance:**
- `pipeline.build_daily_features` ≤2s warm import
- `python pipeline/build_daily_features.py --help` works
- Tests pass

### DEFERRED — Live trading core (`execution_engine`, `paper_trader`)

These run in the live trading hot path. Lazy-import patterns can introduce first-call latency that matters in live execution. Need their own plan with broker-mock test coverage and explicit benchmarking. NOT in this branch.

---

## 5. Methodology (per phase)

For each module:

1. **Discover root cause:** which specific module-level import is the cost? Run:
   ```
   python -c "import time; t=time.perf_counter(); import <heavy_dep>; print(time.perf_counter()-t)"
   ```
   for each suspected import.
2. **Map blast radius:** `grep -rl "from <module>" .` and inspect each importer.
3. **Apply pattern:** lazy import inside function, OR `@lru_cache` getter, OR `TYPE_CHECKING` guard.
4. **Behavioral verification:** smoke import → call API → assert result unchanged.
5. **Test verification:** companion tests pass.
6. **Commit:** one commit per module, with before/after numbers in the message.

---

## 6. Out of scope (explicitly NOT touching)

- Pre-existing syntax/indentation errors in `trading_app.lane_allocator` and `trading_app.eligibility.builder` — these are bugs that need their own fix.
- `pipeline.ingest_dbn*` modules — fail to cold-profile due to pandas/Windows OSError. Need warm measurements; if turn out heavy, defer to a follow-up.
- `trading_app/live/*` (bot_dashboard, session_orchestrator, multi_runner) — fastapi restructure is its own design proposal.
- Any module under `research/` — research scripts are one-shot and not on a hot path.
- Hook timeout (30s) — already at right level after PR #24.

---

## 7. Acceptance (single decision rule)

After all 3 phases land:

```
Sum of (Phase 1 + Phase 2 + Phase 3) module wall times <= 10s warm
AND P2-P5 all pass for every module
```

If yes → declare done, document in HANDOFF, move on.
If no → document delta, halt, do not stack more fixes.
