# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 124

## RALPH AUDIT — Iteration 124
## Date: 2026-03-16
## Infrastructure Gates: PASS (audit_behavioral clean; drift 71 PASS + 14 pre-existing env violations + 6 advisory; 87 tests pass)

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS (pre-existing) | 71 checks PASS, 1 FAILED (Check 16: 14 violations — all missing-package env issues: `databento` and `requests` not installed in this Python environment; pre-existing, unchanged from prior runs) |
| `audit_behavioral.py` | PASS | all 6 checks clean |
| `ruff check` | N/A | ruff not on PATH in this env; behavioral audit confirms no structural issues |
| pytest test_session_orchestrator.py | PASS | 87/87 passed (was 86 pass + 1 fail before fix) |

---

## Scope: trading_app/live/tradovate/data_feed.py + trading_app/live/session_orchestrator.py (from iter 123 Next Targets)

**Diminishing Returns check:**
- `consecutive_low_only: 0` in ledger (last HIGH+ finding iter 123)
- Scope: tradovate/data_feed.py = medium centrality; session_orchestrator.py = low centrality (but highest-impact live trading file)
- Proceed: YES

---

## Seven Sins Scan

### tradovate/data_feed.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | `await self.on_bar(bar)` at line 247 unguarded — but exception propagates up through `_handle_frame` → `_session` async-for loop → `run()` reconnect handler. Correct fail-closed path (different from projectx signalrcore drain task which had no outer recovery). | LOW | ACCEPTABLE (exception correctly propagates to reconnect loop — same behavior as pysignalr path in projectx/data_feed.py) |
| Silent failure | `except Exception as exc: log.warning(...)` in heartbeat (line 208) — sets force_reconnect and returns. Correct. | — | CLEAN |
| Fail-open | None | — | CLEAN |
| Look-ahead bias | N/A — data feed module | — | CLEAN |
| Cost illusion | N/A — data feed module | — | CLEAN |
| Canonical violation | `MD_WS_LIVE` and `MD_WS_DEMO` both set to same URL (lines 28-29). Drift check #71 verifies tradovateapi.com domain (passes). Tradovate distinguishes live/demo by account credentials, not separate WebSocket endpoints. | LOW | ACCEPTABLE (intentional: Tradovate architecture uses credentials not URL to distinguish live/demo; drift #71 PASS) |
| Orphan risk | No unused imports or dead code paths | — | CLEAN |
| Volatile data | `_BACKOFF_INITIAL`, `_BACKOFF_MAX`, `_MAX_RECONNECTS`, `_LIVENESS_TIMEOUT`, `_MAX_STALE_BEFORE_RECONNECT` — infrastructure constants, not trading stats | — | ACCEPTABLE |

### session_orchestrator.py

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Silent failure | ML predictor init `except Exception: self._ml_predictor = None` (line 132-134) — intentional fail-open, documented in comment. ML is supplemental. | — | ACCEPTABLE |
| Silent failure | `_build_daily_features_row` converts all exceptions to `RuntimeError` (fail-closed). | — | CLEAN |
| Fail-open | `except Exception: log.exception("Fill poller iteration error...")` (line 1279) — poller must never crash; correct infinite-loop guard. | — | ACCEPTABLE |
| Look-ahead bias | N/A — live session, no backtesting | — | CLEAN |
| Cost illusion | `_compute_actual_r` uses `self.cost_spec.friction_in_points` (line 611) — correct canonical cost usage. | — | CLEAN |
| Canonical violation | None — imports from canonical sources (`pipeline.cost_model`, `pipeline.paths`, `trading_app.config.ALL_FILTERS`) | — | CLEAN |
| Orphan risk | `import time` inline inside `post_session` loop (line 1400) — non-idiomatic but pre-existing pattern, no correctness impact. | LOW | ACCEPTABLE (style, no correctness impact; pattern 3 in ACCEPTABLE rules) |
| Volatile data | None | — | CLEAN |
| **Test regression** | **5 inline feed stubs in `test_session_orchestrator.py` had `__init__(self, auth, on_bar, demo)` — missing `on_stale=None` added to `BrokerFeed.__init__` in commit d4fe8cb. `TypeError` broke `test_run_starts_watchdog_task`; 4 other reconnect stubs (`MockFeed`, `CountingFeed`, `NeverReachFeed`, `CrashOnceFeed`) were silently stale.** | **MEDIUM** | **FIXED (commit fd37cbb)** |

---

## Summary
- tradovate/data_feed.py: 2 LOW findings → both ACCEPTABLE
- session_orchestrator.py: 1 MEDIUM test regression → FIXED; all other findings ACCEPTABLE or CLEAN
- Verdict: FIXED
- Commit: fd37cbb

---

## Files Fully Scanned

> Cumulative list — 181 files fully scanned (2 new files added this iteration: trading_app/live/tradovate/data_feed.py, trading_app/live/session_orchestrator.py).

- trading_app/ — 44 files (iters 4-61)
- trading_app/ml/features.py — added iter 114
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
- pipeline/ — 15 files (iters 1-71)
- scripts/tools/ — 50 files (iters 18-72, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100): audit_behavioral.py, generate_promotion_candidates.py, select_family_rr.py (iter 89); build_edge_families.py, pipeline_status.py (iter 90); assert_rebuild.py, gen_repo_map.py, sync_pinecone.py (iter 91); pinecone_snapshots.py, rolling_portfolio_assembly.py, generate_trade_sheet.py (iter 92); rr_selection_analysis.py, sensitivity_analysis.py (iter 93); gen_playbook.py, ml_audit.py, audit_integrity.py (iter 94); ml_cross_session_experiment.py, ml_hybrid_experiment.py, ml_instrument_deep_dive.py (iter 95); ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py (iter 96); backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py (iter 97); parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py (iter 98); m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py (iter 99); refresh_data.py, m25_ml_audit.py (iter 100)
- pipeline/asset_configs.py — also reviewed iter 100 (orb_active field addition)
- scripts/infra/ — 1 file (iter 72)
- scripts/migrations/ — 1 file (iter 73)
- scripts/reports/ — 3 files (iter 87): report_wf_diagnostics.py, parameter_stability_heatmap.py, report_edge_portfolio.py (iter 85)
- scripts/ root — 2 files (iter 88): run_live_session.py, operator_status.py
- research/ — 19 files (iters 101-111): research_zt_event_viability.py, research_london_adjacent.py, research_mes_compressed_spring.py (iter 101); research_post_break_pullback.py, research_mgc_asian_fade_mfe.py, research_zt_fomc_unwind.py (iter 102); research_zt_cpi_nfp.py (iter 103); research_mgc_mnq_correlation.py (iter 104); research_atr_velocity_gate.py, research_mgc_regime_shift.py (iter 105); research_zt_event_viability.py (iter 106); research_vol_regime_switching.py (iter 107); research_edge_structure.py, research_1015_vs_1000.py (iters 108-109); research_overlap_analysis.py (iter 110); research_aperture_scan.py (iter 111)
- research/ additional (iters 112-113): research_alt_stops.py, research_direction_asymmetry.py, research_signal_stack.py (iter 112); discover.py, research_wf_stress_keepers.py, research_trend_day_mfe.py scanned but already clean (iter 113)
- docs/plans/ — 2 files (iter 103): 2026-03-15-zt-stage1-cpi-nfp-spec.md, 2026-03-15-zt-stage1-triage-gate.md
- **Total: 181 files fully scanned**
- See previous audit iterations for per-file detail

## Next iteration targets
- `trading_app/live/bar_aggregator.py` — high centrality (5 importers), unscanned, sits between data feed and execution engine
- `trading_app/live/tradovate/order_router.py` — medium centrality, unscanned, parallel sibling of projectx/order_router.py (audited iter 122)
