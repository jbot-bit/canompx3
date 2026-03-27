# Blast Radius Report — `build_live_portfolio()` Deprecation

**Date:** 2026-03-28 (revised — full report replacing earlier stub)
**Agent:** blast-radius (read-only)
**Change type:** Function deprecation / removal
**Blast radius:** SIGNIFICANT

---

## What Is Changing

`trading_app.live_config.build_live_portfolio()` is being deprecated and eventually removed.
The function already emits `DeprecationWarning` and `log.warning()` at runtime (added 2026-03-25).
It resolves to ZERO strategies because `LIVE_PORTFOLIO` specs reference filter types
(`ATR70_VOL`, `X_MGC_ATR70`) that no longer exist in `validated_setups`.
The canonical replacement is `trading_app.prop_profiles.ACCOUNT_PROFILES`.

---

## File Inventory

### Group A — Uncommitted changes (5 files, in-flight deprecation)

These files have been modified in the working tree but not yet committed.
The changes replace direct `build_live_portfolio` calls with DB-direct queries
or injection-first patterns.

| File | Change summary | Risk |
|------|---------------|------|
| `scripts/run_live_session.py` | Import moved inside `else` branch; warning note appended to notes list. `build_live_portfolio` still called when no portfolio is injected — path remains live but warns loudly. | LOW — injection path takes precedence |
| `trading_app/live/session_orchestrator.py` | Added `log.warning()` before the fallback call; no call removed yet. Module-level import at line 35 still fires on every import of the orchestrator. | LOW — warning fires, function not yet removed |
| `trading_app/live/performance_monitor.py` | Docstring updated only. No functional change. | NONE |
| `ui/session_helpers.py` | `build_session_briefings()` replaced `build_live_portfolio` loop with `SELECT * FROM validated_setups WHERE status='active'` query. Behavioural change: now shows ALL active strategies, not only those matching a `LIVE_PORTFOLIO` spec. | MODERATE — test `TestBuildSessionBriefings` hits live `gold.db`; result set may differ |
| `ui_v2/state_machine.py` | Same replacement as `ui/session_helpers.py`, parallel implementation. Same behavioural change. | MODERATE — `tests/test_ui_v2/test_state_machine.py` does NOT test `build_session_briefings()` — no coverage for the new path |

**Critical: module-level import remaining in session_orchestrator.py**

Line 35:
```python
from trading_app.live_config import build_live_portfolio  # DEPRECATED — kept for backward compat
```
This fires on every `import trading_app.live.session_orchestrator`, including inside tests.
Until this line is removed, every orchestrator import drags in `live_config` and emits a
`DeprecationWarning`. The orchestrator test suite always injects a `Portfolio` directly, so the
function itself is never called, but the import warning fires for every test module that imports
`SessionOrchestrator`.

---

### Group B — Remaining callers of the function (3 files, not in the uncommitted batch)

| File | Call site | Disposition |
|------|-----------|-------------|
| `trading_app/live_config.py` | Line ~956, `if __name__ == "__main__"` CLI block | CLI entrypoint only; no other module calls it. Delete when the function is removed. |
| `scripts/tmp_prop_firm_proper_pass.py` | Line 25 import; line 103 call in `collect_active_strategies()` | Explicitly labelled `tmp`. Returns 0 strategies silently now (LIVE_PORTFOLIO resolves to nothing). Safe to delete. |
| `trading_app/prop_portfolio.py` | Line 638, docstring only | No functional call. Stale docstring, no runtime effect. |

---

### Group C — Broader importers of `trading_app.live_config`

These files import other symbols from `live_config` (`LIVE_PORTFOLIO`, `LIVE_MIN_EXPECTANCY_R`,
`PAPER_TRADE_CANDIDATES`, `_load_best_regime_variant`). Removing the *function* does not break
them. Removing or renaming the *constants* or *helpers* they import would break them.

| File | Symbols imported | Break if function removed? | Break if LIVE_PORTFOLIO removed? |
|------|-----------------|---------------------------|----------------------------------|
| `pipeline/check_drift.py` | `LIVE_MIN_EXPECTANCY_R`, `LIVE_PORTFOLIO` | NO | YES — drift checks #43 and #54+ break |
| `scripts/tools/generate_trade_sheet.py` | `LIVE_MIN_EXPECTANCY_DOLLARS_MULT`, `LIVE_MIN_EXPECTANCY_R`, `LIVE_PORTFOLIO`, `_load_best_regime_variant` | NO | YES — trade sheet produces zero rows; dollar gate loses its multiplier constant |
| `scripts/tools/generate_promotion_candidates.py` | `LIVE_MIN_EXPECTANCY_DOLLARS_MULT`, `LIVE_MIN_EXPECTANCY_R` | NO | NO |
| `scripts/tools/sensitivity_analysis.py` | `LIVE_PORTFOLIO` | NO | YES |
| `scripts/tools/pinecone_snapshots.py` | `LIVE_MIN_EXPECTANCY_DOLLARS_MULT`, `LIVE_MIN_EXPECTANCY_R`, `LIVE_PORTFOLIO` | NO | YES |
| `scripts/audits/phase_7_live_trading.py` | `LIVE_PORTFOLIO` | NO | YES |
| `scripts/infra/daily_paper_run.py` | `PAPER_TRADE_CANDIDATES` | NO | NO |
| `ui_v2/server.py` | `LIVE_PORTFOLIO` (2 lazy imports) | NO | YES — first-run detection breaks |
| `trading_app/rolling_portfolio.py` | (detected in file scan — verify before touching constants) | NO | UNKNOWN — verify |
| `tests/test_trading_app/test_live_config.py` | `LIVE_PORTFOLIO`, `build_live_portfolio`, and 4 other symbols | NO (function only) | YES — 7 test call sites break |
| `tests/test_ui/test_copilot_integration.py` | `LIVE_PORTFOLIO` | NO | YES |
| `tests/test_scripts/test_generate_promotion_candidates.py` | `LIVE_PORTFOLIO` | NO | YES |
| `tests/tools/test_pinecone_snapshots.py` | `LIVE_PORTFOLIO` | NO | YES |

**Key distinction:** Removing the function is safe for all Group C files.
Removing `LIVE_PORTFOLIO` (the constant) is a separate, larger change that would require
updating 10+ files — it is NOT part of the current deprecation scope.

---

### Group D — Constants that must be rehomed before `live_config.py` can be deleted

If the goal is eventually to delete `live_config.py` entirely (not just the function), these
constants must be rehomed first:

| Constant | Current location | Used by |
|----------|-----------------|---------|
| `LIVE_MIN_EXPECTANCY_DOLLARS_MULT` | `live_config.py` | `generate_trade_sheet.py`, `generate_promotion_candidates.py`, `pinecone_snapshots.py`, `test_live_config.py` |
| `LIVE_MIN_EXPECTANCY_R` | `live_config.py` | `check_drift.py`, `generate_trade_sheet.py`, `generate_promotion_candidates.py`, `sensitivity_analysis.py` |
| `LIVE_PORTFOLIO` | `live_config.py` | 10 files — see Group C |
| `PAPER_TRADE_CANDIDATES` | `live_config.py` | `daily_paper_run.py` |
| `LiveStrategySpec` | `live_config.py` | `test_live_config.py`, `check_drift.py` (via `LIVE_PORTFOLIO`) |
| `_load_best_regime_variant` | `live_config.py` | `generate_trade_sheet.py` |
| `_load_best_experimental_variant` | `live_config.py` | `test_live_config.py` |

Attempting to delete `live_config.py` without rehoming these would break 10+ files at import time.
Full file deletion is out of scope for the current deprecation.

---

## Canonical Source Impact

| Canonical source | Status |
|-----------------|--------|
| `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` | UNTOUCHED |
| `pipeline.asset_configs.ASSET_CONFIGS` | UNTOUCHED |
| `pipeline.dst.SESSION_CATALOG` | UNTOUCHED |
| `trading_app.config` (entry models, filters) | UNTOUCHED |
| `pipeline.cost_model.COST_SPECS` | UNTOUCHED |
| `pipeline.paths.GOLD_DB_PATH` | UNTOUCHED |
| `trading_app.live_config.LIVE_PORTFOLIO` | NOT BEING REMOVED — stays in place; 10+ importers |
| `trading_app.live_config.build_live_portfolio` | BEING DEPRECATED — 5 calling files addressed in Group A/B |

---

## Database Impact

| Dimension | Assessment |
|-----------|-----------|
| Tables affected | `validated_setups` — new direct query path in `ui/session_helpers.py` and `ui_v2/state_machine.py` |
| Schema risk | Replacement queries use `instrument`, `orb_label`, `expectancy_r`, `sharpe_ratio` — correct column names for `validated_setups`. Verified against schema memory (Mar 9 2026). |
| JOIN risk | No multi-table JOINs. Single-table `SELECT *`. No triple-join trap. |
| Write pattern | Read-only throughout. |
| Concurrent write risk | None — both new paths open with `read_only=True`. |
| Behavioural change | `build_live_portfolio` filtered through `LIVE_PORTFOLIO` specs (0 results in practice). New direct query returns ALL `status='active'` strategies. This is a broader, correct set for briefing cards. |

---

## Test Coverage

| File | Test file | Status |
|------|-----------|--------|
| `live_config.py::build_live_portfolio` | `tests/test_trading_app/test_live_config.py` | COVERED — 7 direct calls across `TestSeasonalGate` (3) and `TestWeightOverrideAndRecovery` (4). All suppress `DeprecationWarning`. Must be deleted or rewritten on function removal. |
| `session_orchestrator.py` | `tests/test_trading_app/test_session_orchestrator.py` | COVERED for injection path. Fallback (un-injected) path has NO TEST. Module-level import fires on test collection. |
| `ui/session_helpers.py::build_session_briefings` | `tests/test_ui/test_session_helpers.py::TestBuildSessionBriefings` | COVERED — 3 tests call against live `gold.db`. New broader query may change assertion results depending on DB state. |
| `ui_v2/state_machine.py::build_session_briefings` | `tests/test_ui_v2/test_state_machine.py` | NO COVERAGE — FLAG. No test exercises the new DB-direct path in the v2 state machine. |
| `scripts/run_live_session.py::_run_preflight` | None | NO TEST — integration-only, no unit test exists. |
| `trading_app/live/performance_monitor.py` | `tests/test_trading_app/test_performance_monitor.py` | Docstring change only — no test impact. |
| `trading_app/prop_portfolio.py` | (docstring only) | No test impact. |

---

## Drift Check Impact

**Check #43 (`check_uncovered_fdr_strategies`):**
Imports `LIVE_MIN_EXPECTANCY_R` and `LIVE_PORTFOLIO`. Iterates `LIVE_PORTFOLIO` to build a
`covered` set. Since `LIVE_PORTFOLIO` is NOT being removed, this check is unaffected by
function removal. Advisory-only (WARNING, never blocks).

**Check #54+ (`check_live_config_spec_validity`):**
Imports `LIVE_PORTFOLIO` and validates its entries against `SESSION_CATALOG`, `ALL_FILTERS`,
`ENTRY_MODELS`. Unaffected by function removal. If `LIVE_PORTFOLIO` is later emptied, this
check trivially passes (no specs to validate) — correct behaviour.

**No drift check references `build_live_portfolio` by name.**
Removing the function will not break or suppress any drift check.

---

## docs/plans References

| File | Reference | Staleness |
|------|-----------|-----------|
| `docs/plans/2026-03-02-live-trading-infrastructure.md` line 22 | Lists `build_live_portfolio` as a verified API | STALE — pre-deprecation design doc. Do not treat as authoritative. |
| `docs/plans/2026-03-02-live-trading-infrastructure.md` line 1196 | "MUST unpack build_live_portfolio() tuple" | STALE — same doc. |
| `docs/plans/2026-03-22-live-config-redesign.md` | Design for rebuilding `LIVE_PORTFOLIO` spec list | ACTIVE DESIGN — not yet implemented. If executed, `generate_trade_sheet.py`, drift checks, and sensitivity analysis will be affected because they iterate the spec list. |
| `session_orchestrator.py` module docstring lines 6 | "VERIFIED API NOTES: build_live_portfolio() returns (Portfolio, notes)" | STALE — in production source file, should be removed with the import. |
| `.claude/skills/trade-book/SKILL.md:88` | References `build_live_portfolio()` | STALE — should be updated to reference `prop_profiles.ACCOUNT_PROFILES`. |

---

## Risk Assessment — Uncommitted Changes

| Risk | Level | Details |
|------|-------|---------|
| Test suite breakage on commit | LOW | Tests either inject portfolios or suppress `DeprecationWarning`. No tests assert on 0-strategy behaviour that would suddenly flip. |
| Briefings behavioural change | MODERATE | `ui/session_helpers.py` and `ui_v2/state_machine.py` now return all active strategies from DB, not spec-filtered ones. In practice this is an improvement (old path returned 0), but `TestBuildSessionBriefings` hits live `gold.db` and may return different counts than before. Verify after commit. |
| Live trading safety | LOW | Orchestrator falls back to `build_live_portfolio` if no portfolio injected → returns 0 strategies → `RuntimeError` at line 110 — fail-closed. Correct behaviour. |
| Module-level import pollution | LOW | Every `import SessionOrchestrator` emits a deprecation warning. Pollutes test output and logs until line 35 is removed. |
| `prop_portfolio.py` docstring | NONE | No runtime effect. |
| `scripts/tmp_prop_firm_proper_pass.py` | NONE | Returns 0 strategies silently. Not in any automated workflow. |

---

## Completion Checklist — Full Removal (beyond the 5 uncommitted files)

These items are NOT done by the 5 uncommitted files. Execute in order:

1. **`session_orchestrator.py` line 35** — Remove module-level import. Replace the `else` fallback at lines 100–107 with `raise RuntimeError("Portfolio must be injected — build_live_portfolio has been removed")`.
2. **`session_orchestrator.py` module docstring** — Remove the `build_live_portfolio()` API note from line 6.
3. **`scripts/run_live_session.py`** — Remove the `else` branch entirely. If no portfolio is injected, raise immediately. Remove the conditional import.
4. **`tests/test_trading_app/test_live_config.py`** — Delete `TestSeasonalGate` (3 tests) and `TestWeightOverrideAndRecovery` (4 tests). Remove `build_live_portfolio` from the import at line 17.
5. **`trading_app/live_config.py`** — Remove the function body (lines 566–end of function) and the `__main__` CLI block (~line 956) that calls it.
6. **`trading_app/prop_portfolio.py`** — Update `build_all_books` docstring at line 638 to remove the reference.
7. **`scripts/tmp_prop_firm_proper_pass.py`** — Delete the file (labelled tmp, functionally broken).
8. **`ui_v2/state_machine.py`** — Add a test for `build_session_briefings()` to cover the new DB-direct path. Currently NO COVERAGE.
9. **`docs/plans/2026-03-02-live-trading-infrastructure.md`** — Mark the `build_live_portfolio` API notes as DEPRECATED or remove to prevent confusion.
10. **`.claude/skills/trade-book/SKILL.md:88`** — Update reference from `build_live_portfolio()` to `prop_profiles.ACCOUNT_PROFILES`.

Do NOT attempt to delete `live_config.py` as part of this work — the constants in Group D
must be rehomed first (separate task).

---

## Summary

```
=== BLAST RADIUS REPORT ===
Target:          trading_app.live_config:build_live_portfolio
Change type:     Function deprecation (removal in progress)
Blast radius:    SIGNIFICANT

DIRECT DEPENDENCIES:
  Callers (runtime, not yet migrated):
    trading_app/live/session_orchestrator.py:35 (module-level import) + :105 (fallback call)
    scripts/run_live_session.py:71 (conditional fallback, in else branch)
    trading_app/live_config.py:956 (__main__ CLI block)
    scripts/tmp_prop_firm_proper_pass.py:103 (tmp research script)
  Callers (docstring only, no runtime effect):
    trading_app/prop_portfolio.py:638
  Importers of live_config (broader): 14 files — none break from function removal alone
  One-way dep:    OK — no pipeline/ module calls build_live_portfolio

CANONICAL SOURCES:
  LIVE_PORTFOLIO:  NOT being removed — still imported by 10+ files
  build_live_portfolio: BEING DEPRECATED — 3 remaining callers after Group A commits

DATABASE IMPACT:
  Tables:         validated_setups (new direct query path in ui/ and ui_v2/)
  JOIN risk:      none — single-table SELECT, no join
  Write pattern:  read-only throughout

TEST COVERAGE:
  build_live_portfolio:      COVERED — 7 tests in test_live_config.py (must delete on removal)
  session_orchestrator fallback: NO COVERAGE
  ui/session_helpers.py::build_session_briefings: COVERED (3 tests, hit live gold.db)
  ui_v2/state_machine.py::build_session_briefings: NO COVERAGE — FLAG

DRIFT CHECKS:
  Check #43 (uncovered FDR): references LIVE_PORTFOLIO not the function — unaffected
  Check #54+ (spec validity): references LIVE_PORTFOLIO not the function — unaffected

GUARDIAN PROMPTS REQUIRED: none

RECOMMENDATION:
  Proceed with committing the 5 uncommitted files. Then execute the 10-item completion
  checklist to finish full removal. Priority items: (1) remove module-level import from
  session_orchestrator.py — it currently emits DeprecationWarning on every import; (2) add
  a test for ui_v2/state_machine.py::build_session_briefings before function is fully deleted.
  Do NOT attempt to delete live_config.py — Group D constants must be rehomed first.
===========================
```
