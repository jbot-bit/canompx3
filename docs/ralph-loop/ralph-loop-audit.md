# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 145

## RALPH AUDIT — Iterations 141-145 (batch)
## Date: 2026-04-05
## Infrastructure Gates: PASS (all worktrees: drift checks PASS, behavioral audit clean, ruff clean, targeted tests PASS)

---

## Iteration 141 — trading_app/live/rithmic/order_router.py (AUDIT ONLY)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open | Line 310: query_open_orders() returns [] when auth is None — submit()/cancel() raise RuntimeError for same condition. Orphaned brackets survive. | HIGH | UNFIXED (agent exhausted turns) |
| Orphan risk | Line 281: unused loop variable `uid` (ruff B007) | LOW | UNFIXED |
| Canonical violation | Lines 86,96,108: hardcoded "E1","E2" entry model strings | LOW | UNFIXED |

---

## Iteration 142 — trading_app/prop_profiles.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation | Line 857: parse_strategy_id hardcoded ("E1","E2","E3") → replaced with ENTRY_MODELS import | MEDIUM | FIXED (694108d) |
| Canonical violation | _LANE_NAMES dict (lines 887-894): hardcoded session strings as keys | LOW | ACCEPTABLE (pattern 1: explicit continuity constraint, DB migration required on rename) |
| Canonical violation | _PV inlined point values (line 1000): avoids circular import | LOW | ACCEPTABLE (pattern 4: runtime guard at 1009-1016 verifies against COST_SPECS) |

---

## Iteration 143 — trading_app/lane_allocator.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open (path divergence) | Line 616: save_allocation() used CWD-relative path; check_allocation_staleness() used file-relative | MEDIUM | FIXED (694108d) |
| Silent failure | Line 572: except ImportError: pass silently dropped report section | LOW | FIXED (694108d) |
| Canonical violation | Hardcoded 'E2' in _compute_session_regime() | LOW | ACCEPTABLE (pattern 1: intentional architectural design — E2 is canonical unfiltered reference model) |
| Silent failure | _P90_ORB_PTS.get(instrument, 100.0) fallback | LOW | ACCEPTABLE (pattern 4: unreachable via validated pipeline — validated_setups only contains ACTIVE_ORB_INSTRUMENTS) |

---

## Iteration 144 — trading_app/live/multi_runner.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open | Lines 110-117: asyncio.gather(return_exceptions=True) absorbs all crashes; run() returns None on total failure; caller sees exit code 0 | HIGH | FIXED (694108d) |
| Orphan risk | Unused import or dead code paths | LOW | CLEAN |

---

## Iteration 145 — trading_app/live/broker_dispatcher.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open (inconsistent guard) | Lines 87-88: update_market_price() secondary loop unguarded; submit() and cancel_bracket_orders() both have try/except | MEDIUM | FIXED (694108d) |
| All other sins | No hardcoded sessions/costs/DB paths; no orphan imports; no dead code | — | CLEAN |

---

## Summary

- 5 iterations, 13 total findings
- 4 FIXED (1 HIGH fail-open, 2 MEDIUM, 1 LOW)
- 3 UNFIXED from iter 141 (agent exhausted turns — HIGH fail-open in rithmic order_router needs manual fix)
- 4 ACCEPTABLE
- 2 CLEAN

---

## Files Fully Scanned

> Cumulative list — 212 files fully scanned (5 new files added this batch).

- trading_app/ — 44 files (iters 4-61)
- trading_app/ml/features.py — added iter 114
- trading_app/ml/config.py — added iter 129
- trading_app/ml/meta_label.py — added iter 130
- trading_app/ml/predict_live.py — added iter 131
- trading_app/walkforward.py — added iter 132
- trading_app/outcome_builder.py — added iter 115
- trading_app/strategy_discovery.py — added iter 116
- trading_app/strategy_validator.py — added iter 117
- trading_app/portfolio.py — added iter 118
- trading_app/live_config.py — added iter 119
- trading_app/db_manager.py — added iter 120
- trading_app/live/projectx/auth.py — added iter 121
- trading_app/live/projectx/order_router.py — added iter 122
- trading_app/live/projectx/data_feed.py — added iter 123
- trading_app/live/broker_base.py — added iter 123
- trading_app/live/tradovate/data_feed.py — added iter 124
- trading_app/live/session_orchestrator.py — added iter 124
- trading_app/live/bar_aggregator.py — added iter 125
- trading_app/live/tradovate/order_router.py — added iter 126
- trading_app/live/tradovate/auth.py — added iter 126
- trading_app/live/tradovate/contract_resolver.py — added iter 126
- trading_app/live/tradovate/positions.py — added iter 126
- trading_app/live/broker_factory.py — added iter 127
- trading_app/live/tradovate/__init__.py — added iter 127
- trading_app/live/circuit_breaker.py — added iter 128
- trading_app/live/cusum_monitor.py — added iter 128
- trading_app/live/projectx/__init__.py — added iter 128
- pipeline/ — 15 files (iters 1-71)
- pipeline/calendar_filters.py — added iter 133
- pipeline/stats.py — added iter 134
- pipeline/audit_log.py — added iter 136
- pipeline/ingest_dbn_mgc.py — added iter 136
- pipeline/ingest_dbn.py — added iter 137
- pipeline/ingest_dbn_daily.py — added iter 137
- pipeline/build_daily_features.py — added iter 138
- pipeline/build_bars_5m.py — added iter 139
- pipeline/run_pipeline.py — added iter 140
- pipeline/run_full_pipeline.py — added iter 140
- scripts/tools/ — 51 files (iters 18-100)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iters 85, 87)
- scripts/ root — 2 files (iter 88)
- scripts/databento_backfill.py — added iter 135
- research/ — 21 files (iters 101-113)
- docs/plans/ — 2 files (iter 103)
- trading_app/live/rithmic/order_router.py — added iter 141
- trading_app/prop_profiles.py — added iter 142
- trading_app/lane_allocator.py — added iter 143
- trading_app/live/multi_runner.py — added iter 144
- trading_app/live/broker_dispatcher.py — added iter 145
- **Total: 212 files fully scanned**

## Next iteration targets
- trading_app/live/rithmic/order_router.py — **PRIORITY: unfixed HIGH fail-open from iter 141** (query_open_orders returns [] on auth=None)
- trading_app/live/copy_order_router.py — unscanned, same dispatcher-pattern layer
- trading_app/pre_session_check.py — high-centrality deployment-path file
- trading_app/live/bot_dashboard.py — unscanned live trading UI
- trading_app/live/position_tracker.py — unscanned position management
