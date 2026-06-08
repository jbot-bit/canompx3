---
task: Fix live daily_features staleness gate to count trading days, not calendar days (false-fails over holiday weekends)
mode: IMPLEMENTATION
created: 2026-06-09
scope_lock:
  - pipeline/market_calendar.py
  - trading_app/live/session_orchestrator.py
  - scripts/tools/pipeline_status.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_pipeline/test_market_calendar.py
---

## Blast Radius

- `pipeline/market_calendar.py` — ADD one pure function `trading_days_between`
  (delegates to existing canonical `is_cme_holiday` / `_CMES`); no change to
  existing exports, which are imported widely. Zero callers broken.
- `trading_app/live/session_orchestrator.py:~1193` — the daily_features staleness
  gate. WAS calendar-day `.days` + fail>5/warn>3. NOW trading-day count +
  fail>3/warn>1, ImportError fallback to calendar days (fail-closed). Gate only
  raises/warns — NO order/sizing/risk logic touched. RuntimeError still aborts
  the session (re-raised before the broad except at ~1260).
- `scripts/tools/pipeline_status.py` — a near-duplicate weekday-only
  `_trading_days_between` already existed (used by drift check ~line 5935). To
  avoid redundancy it now DELEGATES to the canonical helper, keeping only its
  None-handling. The 5 existing pinned tests still pass (their date ranges cross
  no CME holiday → holiday-aware == weekday-only). Bonus: upgrades the
  drift-check staleness path to be holiday-aware too.
- Tests: `tests/test_pipeline/test_market_calendar.py` (+TestTradingDaysBetween),
  `tests/test_trading_app/test_session_orchestrator.py` (+3 gate tests +fallback).
- Reads: gold.db read-only (unchanged). Writes: none.

## Purpose

The live staleness gate protects against trading on stale daily_features (filters
would silently reject all trades if atr_20/regime columns are absent). It is
correct to fail-closed on genuine staleness — but it currently measures staleness
in calendar days, so it false-fails over normal market-holiday gaps. Switch it to
trading-day counting so it only halts on real staleness.

## Self-check (simulated)
- Happy path: normal Fri→Mon weekend, latest=Fri, today=Mon → 1 trading day → OK.
- Edge (holiday): Thanksgiving Wed→Mon → 5 cal days but 3 trading days → OK (was FAIL).
- Failure mode: genuine 4+ trading-day outage → still fail-closed (RuntimeError).
- Import-fail: market_calendar unavailable → fail-closed (mirror line ~3927), never silent-pass.

## Verification
- TDD: failing holiday-gap test first.
- pytest test_session_orchestrator.py + test_market_calendar.py green.
- check_drift.py passes.
- Adversarial-audit gate (capital path, evidence-auditor) before close.
