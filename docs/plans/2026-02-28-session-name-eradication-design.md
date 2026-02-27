# Session Name Eradication — Design (A6 Extended)

**Date:** 2026-02-28
**Audit ref:** A6 from SYSTEM_AUDIT_2026-02-27
**Approach:** Option A — Fix active files + drift guard

## Problem

82 Python files contain old fixed-clock session names ("0900", "1000", etc.).
Only ~10 are active code. The rest are frozen historical research/archive scripts
that were correct at runtime and serve as immutable experimental records.

## Decision: Fix Active, Freeze Historical, Guard New

### Why NOT fix all 82 files
1. **Experimental reproducibility** — archived research scripts document what was
   tested, when, and under what assumptions. Modifying them corrupts the ledger.
2. **Zero value** — dead scripts that will never be re-run add no benefit from renaming.
3. **Risk** — bulk replacement could break non-session uses of these strings.

### Why NOT add aliases (Option C)
Silent auto-resolution layers violate SSOT and allow indefinite use of deprecated
terminology. DRY demands importing from the single source of truth, not translating.

## Scope

### Fix (7-9 files)

| File | Issue | Fix |
|------|-------|-----|
| `scripts/tools/hypothesis_test.py` | 5 locations with hardcoded sessions | Import/replace with new names |
| `scripts/tools/rolling_portfolio_assembly.py` | SESSION_UTC, SESSION_IB dicts | Replace keys with new names |
| `scripts/tools/backtest_atr_regime.py` | Strategy family dicts | Replace session values |
| `scripts/tools/find_pf_strategy.py` | TARGET_SESSIONS list | Replace values |
| `research/lib/query.py` | 3 docstring examples + dead E0 | Update docstrings |
| `research/analyze_double_break.py` | ORB_LABELS list | Replace values |
| `research/cross_validate_strategies.py` | Fallback default | Replace value |
| `tests/.../test_portfolio.py` | ~40 strategy_id strings | Replace for consistency |
| `tests/.../test_strategy_fitness.py` | strategy_id strings | Replace for consistency |

### Freeze (75 files — DO NOT TOUCH)

- `research/archive/*` — frozen experimental record
- `research/research_*.py` — one-off scripts, results captured in MEMORY/docs
- `scripts/walkforward/*` — historical walkforward runs
- `docs/archive/*` — archived documentation

### Guard (new drift check #38)

Add to `pipeline/check_drift.py`: grep all `.py` files for old session name
string literals. Exclude frozen files. Fail if any found in active code.

Exclusion list:
- `scripts/tools/migrate_session_names.py` (migration utility)
- `scripts/tools/volume_session_analysis.py` (legacy reference)
- `research/archive/` (frozen)
- `research/research_*.py` at root level (frozen one-offs)
- `scripts/walkforward/` (frozen)
- `docs/archive/` (frozen)

## Session Name Mapping

| Old | New | Source |
|-----|-----|--------|
| 0900 | CME_REOPEN | pipeline/dst.py SESSION_CATALOG |
| 1000 | TOKYO_OPEN | pipeline/dst.py SESSION_CATALOG |
| 1100 | SINGAPORE_OPEN | pipeline/dst.py SESSION_CATALOG |
| 1800 | LONDON_METALS | pipeline/dst.py SESSION_CATALOG |
| 2300 | US_DATA_830 | pipeline/dst.py SESSION_CATALOG |
| 0030 | NYSE_OPEN | pipeline/dst.py SESSION_CATALOG |

## Verification

1. `python -m pytest tests/ -x -q` — all tests pass
2. `python pipeline/check_drift.py` — all 38 checks pass (including new #38)
3. Grep sweep of active files — zero old names remain
