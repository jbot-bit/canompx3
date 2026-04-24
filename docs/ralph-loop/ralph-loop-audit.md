# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 171

## RALPH AUDIT — Iteration 171
## Date: 2026-04-25
## Infrastructure Gates: drift 107/107 PASS (6 pre-existing advisories); behavioral audit 7/7 PASS; ruff PASS; 39 tests PASS

---

## Iteration 171 — lane_allocator.py duplicates RHO_REJECT_THRESHOLD from lane_correlation.py (FIXED 9809f1b8)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation (integrity-guardian.md § 2) | `lane_allocator.py:506` defined `CORRELATION_REJECT_RHO = 0.70` — a duplicate of `RHO_REJECT_THRESHOLD = 0.70` from `lane_correlation.py`. Comment on line 506 explicitly acknowledged the duplication ("Same as lane_correlation.RHO_REJECT_THRESHOLD") but did not import from the canonical source. Silently diverges if threshold changes. | LOW | FIXED 9809f1b8 |

### Audit Notes

- **Auto-targeting:** Priority 3 (unscanned medium) — `trading_app/lane_correlation.py`. Medium tier (2 importers, 10 imports). File itself was clean; finding was in its importer `lane_allocator.py` which duplicated lane_correlation's threshold constant.
- **Doctrine cited:** integrity-guardian.md § 2 (import from single source of truth, never hardcode); institutional-rigor.md § 4 (delegate to canonical sources, never re-encode).
- **PREMISE:** `CORRELATION_REJECT_RHO = 0.70` in `lane_allocator.py` duplicates `RHO_REJECT_THRESHOLD = 0.70` from `lane_correlation.py`.
- **TRACE:** `lane_correlation.py:24` (`RHO_REJECT_THRESHOLD = 0.70`) → `lane_allocator.py:34` (imports `_load_lane_daily_pnl, _pearson` but NOT `RHO_REJECT_THRESHOLD`) → `lane_allocator.py:506` (`CORRELATION_REJECT_RHO = 0.70`, comment "Same as lane_correlation.RHO_REJECT_THRESHOLD") → `lane_allocator.py:628` (`rho > CORRELATION_REJECT_RHO`).
- **EVIDENCE:** grep confirmed two separate literal `0.70` values in both files. The comment on line 506 proves the author recognized the duplication but hardcoded anyway.
- **Fix:** Added `RHO_REJECT_THRESHOLD` to the existing `from trading_app.lane_correlation import ...` line; deleted the local `CORRELATION_REJECT_RHO = 0.70` definition; updated usage on line 628 to `RHO_REJECT_THRESHOLD`; updated `test_lane_allocator.py` to import from `lane_correlation` instead of `lane_allocator`.
- **Verification:** 39/39 test_lane_allocator.py pass. 107/107 drift checks PASS. 8/8 pre-commit hooks PASS.
- **Blast radius:** 2 files (lane_allocator.py + test_lane_allocator.py). No behavioral change — same threshold value, same comparison logic.

### Full Seven Sins scan — lane_correlation.py

| Sin | Result |
|-----|--------|
| Silent failure | None — try/finally (not try/except) ensures connection cleanup while propagating exceptions. Correct pattern. |
| Fail-open | None — gate_pass=len(reject_reasons)==0; empty deployed_lanes → gate_pass=True is semantically correct (no conflicts possible). |
| Look-ahead bias | N/A — correlation gate reads historical P&L data; no temporal alignment issues. |
| Canonical violation | None in lane_correlation.py itself. GOLD_DB_PATH from pipeline.paths. instrument/entry_model/orb_minutes taken from passed-in lane dict. Finding was in importer lane_allocator.py. |
| Dead code | None — all functions used. _pearson exported and used by lane_allocator. |
| Orphan risk | None — check_candidate_correlation called from lane_allocator.py; _load_lane_daily_pnl and _pearson imported by lane_allocator and portfolio_correlation_audit.py. |
| Async safety | N/A — synchronous module |
| State persistence gap | N/A — stateless computation module |
| Contract drift | None — signatures stable; no dead parameters. |
| Hardcoded thresholds (§ 2) | RHO_REJECT_THRESHOLD=0.70 and SUBSET_REJECT_THRESHOLD=0.80 are module-level constants in lane_correlation.py — these ARE the canonical source; importers must reference these, not duplicate them. FIXED in lane_allocator.py. |

---

## Prior: Iteration 170 — outcome_builder.py dead parameter break_ts in _compute_outcomes_all_rr (FIXED 9b16c4eb)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Dead parameter (institutional-rigor.md § 5) | `_compute_outcomes_all_rr` accepted `break_ts=None` but never read it. 4 callsites passed it unnecessarily. | LOW | FIXED 9b16c4eb |

---

## Prior: Iteration 169 — db_manager.py verify_trading_app_schema silent verifier gap (FIXED 6811640a)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open / Silent failure | `verify_trading_app_schema` missing 12 migration-added columns from expected_cols sets. Returned (True, []) silently. | MEDIUM | FIXED 6811640a |

---

## Prior: Iteration 168 — topstep_scaling_plan.py (audit-only — all findings ACCEPTABLE)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Orphan risk | scripts/tmp/lane_analysis.py imports nonexistent SCALING_LADDER. | LOW | ACCEPTABLE — dormant, guarded |
| Contract drift | lots_for_position() misclassifies dead instruments M2K and M6E. | LOW | ACCEPTABLE — dead instruments only |

---

## Files Fully Scanned

> Cumulative list — 251 files fully scanned (lane_correlation.py added iter 171).

- trading_app/ — 44 files (iters 4-61)
- trading_app/ml/features.py — added iter 114
- trading_app/ml/config.py — added iter 129
- trading_app/ml/meta_label.py — added iter 130
- trading_app/ml/predict_live.py — added iter 131
- trading_app/walkforward.py — added iter 132
- trading_app/outcome_builder.py — added iter 115, re-audited iter 170
- trading_app/strategy_discovery.py — added iter 116
- trading_app/strategy_validator.py — added iter 117
- trading_app/portfolio.py — added iter 118
- trading_app/live_config.py — added iter 119
- trading_app/db_manager.py — added iter 120, re-audited iter 169
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
- trading_app/prop_profiles.py — added iter 142, re-audited iter 167
- trading_app/lane_allocator.py — added iter 143
- trading_app/live/multi_runner.py — added iter 144
- trading_app/live/broker_dispatcher.py — added iter 145
- trading_app/pre_session_check.py — added iter 146, re-audited iter 163
- trading_app/live/copy_order_router.py — added iter 147
- trading_app/live/rithmic/auth.py — added iter 148
- trading_app/live/bot_dashboard.py — added iter 149
- trading_app/live/position_tracker.py — added iter 150
- trading_app/account_hwm_tracker.py — added iter 151
- trading_app/live/trade_journal.py — added iter 152
- trading_app/live/bot_state.py — added iter 153
- trading_app/prop_portfolio.py — added iter 154
- trading_app/live/rithmic/__init__.py — added iter 155
- trading_app/ai/sql_adapter.py — added iter 155
- trading_app/ai/grounding.py — added iter 156
- trading_app/ai/query_agent.py — added iter 157
- trading_app/ai/chat_handler.py — added iter 157
- trading_app/mcp_server.py — added iter 158
- trading_app/ai/__init__.py — added iter 159
- trading_app/ai/corpus.py — added iter 159
- trading_app/ai/cli.py — added iter 159
- trading_app/ai/strategy_matcher.py — added iter 160
- trading_app/live/webhook_server.py — added iter 160
- trading_app/live/instance_lock.py — added iter 160
- trading_app/live/broker_connections.py — added iter 160
- trading_app/live/tradovate/contracts.py — added iter 161
- trading_app/live/tradovate/http.py — added iter 161
- trading_app/live/rithmic/contracts.py — added iter 161
- trading_app/live/rithmic/positions.py — added iter 161
- pipeline/db_config.py — added iter 161
- pipeline/paths.py — re-audited iter 161 (modified 2026-04-04)
- trading_app/execution_engine.py — added iter 164
- trading_app/entry_rules.py — added iter 164
- trading_app/sprt_monitor.py — added iter 165
- trading_app/sr_monitor.py — added iter 165
- trading_app/consistency_tracker.py — added iter 166
- trading_app/risk_manager.py — added iter 166
- trading_app/strategy_fitness.py — re-audited iter 167 (WIP-save only, no substantive change)
- trading_app/topstep_scaling_plan.py — added iter 168
- trading_app/lane_correlation.py — added iter 171
- **Total: 251 files fully scanned**

## Next iteration targets
- Priority 2 (stale re-audit, critical): trading_app/lane_allocator.py — already scanned iter 143, but just modified in iter 171. Git log to confirm if meaningful changes since iter 143 audit date warrant re-audit beyond what iter 171 found.
- Priority 3 (unscanned medium): trading_app/eligibility/ directory files (check centrality index for specific files)
- Priority 3 (unscanned medium): scripts/reports/monitor_lane_correlation_rolling.py (companion to lane_correlation, 0 importers/low tier but scanned this iter by proximity)
- Note: pre-existing drift advisories (checks 59/95) require operational resolution by user — run `python scripts/tools/select_family_rr.py` and re-run validator for Mode A strategies
