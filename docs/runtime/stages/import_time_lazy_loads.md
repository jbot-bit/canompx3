---
task: lazy-load heavy import-time deps in narrow-blast-radius modules per docs/plans/2026-04-20-import-time-side-effects.md
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - pipeline/stats.py
  - pipeline/market_calendar.py
  - pipeline/audit_bars_coverage.py
  - trading_app/live/bot_dashboard.py
  - trading_app/ai/claude_client.py
  - docs/plans/2026-04-20-import-time-side-effects.md
blast_radius: pipeline/stats.py — 2 importers (trading_app/live_config.py, scripts/tools/select_family_rr.py); both call jobson_korkie_p which is the only scipy-using function. pipeline/market_calendar.py — 2 importers (trading_app/pre_session_check.py, trading_app/live/session_orchestrator.py); both call functions that internally use _CMES, lazy-cache via @lru_cache preserves API. trading_app/live/bot_dashboard.py — 1 importer (scripts/run_live_session.py) which only invokes start/run, lazy uvicorn/fastapi inside main() preserves CLI entry. pipeline/audit_bars_coverage.py — 0 production importers (CLI tool), heavy imports under if __name__ block. pipeline/check_drift.py — conflict resolution from cherry-pick (origin reformatted SLOW_CHECK_LABELS to multi-line ruff style) plus carrying the cherry-picked entry; no behavioral change. Companion tests — pipeline.stats has none (pure utility); market_calendar has tests/test_pipeline/test_market_calendar.py; bot_dashboard has tests/test_trading_app/test_bot_dashboard.py; audit_bars_coverage has none (CLI). All companion tests must still pass.
acceptance:
  - check_all_imports_resolve warm wall ≤ 10s (3-run median, isolated)
  - check_all_imports_resolve cold wall ≤ 30s (single first-run measurement)
  - All companion tests pass (test_market_calendar, test_bot_dashboard, test_check_drift, test_audit_behavioral)
  - Full pipeline/check_drift.py exits 0
  - Per-module behavioral verification: caller calls function → same result before vs after refactor
agent: claude
---

# Stage — import-time lazy loads

Pre-reg plan: `docs/plans/2026-04-20-import-time-side-effects.md`.

Five sub-stages, one commit each. After each: measure, run companion tests, verify drift. If any sub-stage falsifies P3/P4/P5, revert that commit and reassess.

| # | File | Pattern | Predicted save |
|---|---|---|---|
| 0 | `pipeline/check_drift.py` | resolve cherry-pick conflict (no logic change) | 0s |
| 1 | `pipeline/stats.py` | scipy import inside `jobson_korkie_p()` | 4.8s |
| 2 | `pipeline/market_calendar.py` | `@lru_cache(maxsize=1)` getter for CMES | 1.9s |
| 3 | `trading_app/live/bot_dashboard.py` | uvicorn/fastapi inside `main()` | 1.3s |
| 4 | `pipeline/audit_bars_coverage.py` | heavy imports under `if __name__ == "__main__":` | 4.5s |

Total predicted: 12.5s warm. From ~20s baseline → ~7-8s.
