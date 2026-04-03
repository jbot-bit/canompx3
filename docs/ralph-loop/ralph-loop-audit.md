# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 140

## RALPH AUDIT — Iteration 140
## Date: 2026-04-04
## Infrastructure Gates: PASS (77 drift checks PASS, 0 skipped, 7 advisory; behavioral audit 7/7 clean; ruff clean; 29/29 test_run_pipeline.py + 737 pre-commit suite PASS)

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 77 checks PASS, 0 skipped, 7 advisory |
| `audit_behavioral.py` | PASS | all 7 checks clean |
| `ruff check` | PASS | clean |
| pytest targeted | PASS | 29/29 test_run_pipeline.py |

---

## Scope: pipeline/run_pipeline.py + pipeline/run_full_pipeline.py

---

## Seven Sins Scan

### pipeline/run_pipeline.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Orphan risk / stale metadata | Line 14: docstring listed (MGC, MNQ, NQ) as valid instruments. NQ is a full-size data source symbol with orb_active=False; active ORB instrument is MES | LOW | FIXED (312ec41) |
| Canonical violation (help text) | Line 156: list_instruments() in argparse help includes dead instruments; no choices= guard | LOW | ACCEPTABLE (pattern 4: get_asset_config() in every downstream subprocess fails-closed on unknown instruments) |
| Fail-open (vacuous all()) | Line 243: all() on empty results — vacuously True | LOW | ACCEPTABLE (pattern 1: PIPELINE_STEPS constant has 4 entries; dry_run exits before loop; empty list unreachable) |

### pipeline/run_full_pipeline.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation | Line 123: hardcoded --min-sample=30 string instead of REGIME_MIN_SAMPLES | MEDIUM | ACCEPTABLE (pattern 4: pipeline/ cannot import from trading_app/ per one-way dep rule; canon lock documented inline at line 115) |
| All other sins | No hardcoded sessions/costs/DB paths; no orphan imports; no dead code; step count dynamic | — | CLEAN |

---

## Summary

- pipeline/run_pipeline.py: 1 LOW — FIXED; 2 LOW — ACCEPTABLE
- pipeline/run_full_pipeline.py: 1 MEDIUM — ACCEPTABLE (one-way dep constraint)
- Action: fix ([mechanical] — docstring correction, no behavior change)

---

## Files Fully Scanned

> Cumulative list — 207 files fully scanned (2 new files added this iteration).

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
- **Total: 207 files fully scanned**

## Next iteration targets
- trading_app/ml/evaluate.py — LOW tier, unscanned ML evaluation path (1 importer)
- trading_app/ml/evaluate_validated.py — LOW tier, unscanned ML evaluation validation path (1 importer)
- pipeline/pipeline_status.py — unscanned pipeline status/rebuild orchestrator (referenced in ARCHITECTURE.md commands)
