# 2026-04-20 — Import-Time Side-Effect Reduction

**Status:** PRE-REGISTERED PLAN. Not yet implemented.
**Owner:** claude (this session)
**Trigger:** Code review of f22052ad surfaced cold-cache spike at 30s hook ceiling.

---

## 1. Hypothesis (pre-registered)

**Claim:** Lazy-importing the heavy module-level dependencies in 4 narrow-blast-radius modules will reduce `check_all_imports_resolve` warm wall from 17-25s to ≤ 10s, and cold wall from ~76s to ≤ 30s, **without changing behavior in any consumer**.

**Falsifiable predictions:**
- P1: Warm `check_all_imports_resolve` wall ≤ 10s (3-run median, isolated subprocess)
- P2: Cold `check_all_imports_resolve` wall ≤ 30s (single first-run-after-system-idle measurement)
- P3: All 241 `tests/test_pipeline/` + `tests/test_trading_app/` tests still pass
- P4: Full `pipeline/check_drift.py` (no `--fast`) exits 0
- P5: Each refactored module's public API surface unchanged (function signatures, return types, observable behavior)

**Rollback rule:** If ANY of P3, P4, or P5 fail at any stage, REVERT that stage's commit and re-investigate. Do not stack fixes.

---

## 2. Grounding (official Python best practices)

No quant-specific institutional literature applies — this is general Python performance. Authoritative sources:

- **PEP 8 — Style Guide for Python Code** (van Rossum, Warsaw, Coghlan):
  > "Imports are always put at the top of the file"

  AND immediately after:

  > "It is sometimes useful to delay imports until after they are needed... where the imports are needed inside conditional code or to avoid circular imports or for performance reasons."

  Lazy imports are explicitly endorsed for performance and circular-import resolution.

- **PEP 562 — Module __getattr__ and __dir__** (Levkivskyi, 2017): allows module-level lazy attribute access. Used in this plan only if needed for a clean public API.

- **Python stdlib `functools.lru_cache`**: official memoization decorator. Pattern: `@lru_cache(maxsize=1)` on a getter function makes the first call do the work, every subsequent call returns cached result. Same data, paid once.

- **scientific Python convention** (numpy, scipy, pandas docs): heavy submodules are lazy-loaded at function call time when not always used. `from scipy import stats` at module level is paid by EVERY importer of the parent module — anti-pattern unless every code path uses it.

---

## 3. Measurement methodology (run-then-report, no narrative)

For each stage:
1. **Baseline:** `python /tmp/run_check_isolated.py` × 3, record warm wall (median).
2. **Refactor:** apply lazy-import pattern to ONE module.
3. **Re-measure:** same script × 3, warm wall.
4. **Behavioral verification:** run the affected module's companion tests (`test_<module>.py`).
5. **Drift verification:** `python pipeline/check_drift.py` exits 0.
6. **Commit:** include before/after wall-time numbers in the commit message.
7. **Stop and reassess:** if delta < 50% of predicted, investigate WHY before proceeding.

Per-module isolated profile via the harness already written in this session at `/tmp/profile_imports.py`.

---

## 4. Per-module plan (ordered by blast-radius ascending)

| # | Module | Predicted saving | Pattern | Blast radius |
|---|---|---|---|---|
| 1 | `pipeline/stats.py` | 4.8s | move `from scipy import stats` inside `jobson_korkie_p()` | 2 importers (`trading_app/live_config.py`, `scripts/tools/select_family_rr.py`) |
| 2 | `pipeline/market_calendar.py` | 1.9s | wrap `_CMES = xcals.get_calendar(...)` in `@lru_cache(maxsize=1)` getter | 2 importers (`trading_app/pre_session_check.py`, `trading_app/live/session_orchestrator.py`) |
| 3 | `trading_app/live/bot_dashboard.py` | 1.3s | **DEFERRED** — see § 3a | 1 importer (`scripts/run_live_session.py`) |
| 4 | `pipeline/audit_bars_coverage.py` | 4.5s | this is a CLI standalone — move heavy imports under `if __name__ == "__main__":` block, or guard inside `main()` | 0 production importers (only `pipeline/check_drift.py` references it for the import-resolve check itself) |

**Total predicted saving: ~12.5s warm.** From ~20s baseline → ~7-8s. Inside the ≤10s P1 budget.

**Module 5 (`pipeline/system_brief.py`, 0.6s warm):** investigated only if P1 not met after stages 1-4. Otherwise skip — Pareto says first 4 are 95%.

### § 3a — Stage 3 deferred (decision recorded 2026-04-20)

`trading_app/live/bot_dashboard.py` cost decomposes:
- `import uvicorn` standalone: **0.37s** (cheap)
- `from fastapi import FastAPI` standalone: **2.25s** (the real cost)

Cannot lazy-load fastapi here: `app = FastAPI(...)` is at module level (line 107) and 30+ `@app.get`/`@app.post` decorators reference it directly. The decorators must register at import time.

Lazy-loading uvicorn alone saves ~0.3s — below the noise floor of the measurement methodology (±5-10s). Not worth a commit.

The proper fix is restructuring to FastAPI's `APIRouter` pattern: `router = APIRouter()` at module level, decorators on `@router.X`, then `app = FastAPI(); app.include_router(router)` inside `main()`. That defers FastAPI() instantiation but still imports fastapi at module level — so it doesn't save the 2.2s anyway. The actual fix would be a separate `bot_dashboard_lazy.py` shim, which is a meaningful refactor with its own design proposal.

**Decision: skip Stage 3.** Adjust total predicted saving:
- Stages 1+2+4 = 4.8 + 1.9 + 4.5 = **11.2s** (was 12.5s).
- Target wall ~9s (from ~20s baseline). Still under P1 ≤10s budget.

---

## 5. Out of scope (explicitly NOT touching)

- `trading_app/ai/claude_client.py` (5.1s) — `import anthropic` is the module's POINT. The module exists as a single canonical wrapper for Claude calls. Lazy-loading anthropic inside its functions defeats the purpose.
- Any module not in §4. Pareto says we don't need to touch them.
- Hook timeout itself (currently 30s in `.claude/hooks/post-edit-pipeline.py`). Don't widen it — the goal is to fit comfortably under it, not raise the ceiling.
- `if module in sys.modules: continue` optimization in `check_all_imports_resolve()`. Already present and effective — don't touch.

---

## 6. Acceptance criteria (single decision rule)

After all 4 stages land:

```
P1 ≤ 10s warm   AND   P2 ≤ 30s cold   AND   P3-P5 all pass
```

If yes → declare done, update HANDOFF.
If no → halt, document delta in this file, do not stack more fixes.
