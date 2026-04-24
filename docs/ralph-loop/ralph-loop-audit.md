# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 172

## RALPH AUDIT — Iteration 172
## Date: 2026-04-25
## Infrastructure Gates: drift 107/107 PASS (6 pre-existing advisories); 204 tests PASS (orchestrator + risk_manager)
## Scope: B6 verification — F-1 EOD balance never seeded in signal-only mode

---

## Iteration 172 — B6 F-1 EOD balance signal-only seed (VERIFIED CLOSED — ca363e1a)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open / Silent failure (integrity-guardian.md § 3, § 6) | F-1 gate in `risk_manager.py:220-229` fail-closes every entry when `_topstep_xfa_eod_balance is None`. In signal-only, the HWM init block at line 586 is gated `if not signal_only`, so `_apply_broker_reality_check` (which calls `set_topstep_xfa_eod_balance`) never ran. Every entry since 2026-04-15 rejected with "EOD XFA balance unknown — refusing entry". | CRITICAL | VERIFIED CLOSED — fix landed ca363e1a (2026-04-25) |

### Audit Notes — B6 Verification

- **Scope:** HANDOFF.md Next Steps #1 / B6 (F-1 EOD balance never seeded in signal-only — REAL BLOCKER). Ralph iter 172 is a verification audit of fix `ca363e1a` which landed before this iteration ran.
- **Root cause confirmed:** `session_orchestrator.py` — `_apply_broker_reality_check` (contains the only call to `risk_mgr.set_topstep_xfa_eod_balance(initial_equity)`) is only reached inside `if not signal_only` at line 586. In signal-only, `order_router=None`, HWM block is skipped, seed never runs, F-1 fail-closes every entry.
- **Fix verified:** `ca363e1a` adds:
  1. `_apply_signal_only_f1_seed(*, risk_mgr, logger=None) -> bool` — module-level helper (lines 140-169) that seeds `$0.00` (canonical day-1 XFA balance per `topstep_scaling_plan.py:51-53`).
  2. Call site at lines 386-387, gated `if signal_only:`, immediately after `self.risk_mgr` is constructed — before any entry attempt.
- **Canonical grounding verified:** `$0.0` is correct per `topstep_scaling_plan.py:51-53` docstring: "Day-1 of any new XFA: balance = $0 (XFA starts at $0 per topstep_mll_article.md), so max position is the bottom-row tier (2 lots for 50K, 3 lots for 100K/150K)." Conservative bottom-tier cap — NOT a bypass of F-1.
- **Guard confirmed:** `if risk_mgr.limits.topstep_xfa_account_size is None: return False` — non-XFA profiles are no-ops.
- **Rejected options (per commit message):**
  - Option (a): seed with `prof.account_size` ($50,000) → resolves to highest tier (5 lots for 50K) — OVER-PERMITS signal-only. Rejected.
  - Option (b): `disable_f1()` in signal-only → mutes F-1 entirely, violates integrity-guardian § 3. Rejected.
- **Regression check:** `e02c529d` (5 silent-failure fixes) touched `session_orchestrator.py` but only at lines for F8/F2/F5/F6 — does NOT overlap with B6 fix block (lines 381-387). Fix preserved.
- **Doctrine cited:** institutional-rigor.md § 6 (no silent failures); integrity-guardian.md § 3 (fail-closed — chose seed over disable_f1); institutional-rigor.md § 4 (delegate to canonical sources — $0.0 from topstep_scaling_plan docstring).
- **Tests:** `TestF1SignalOnlySeed` (3 tests in `test_session_orchestrator.py`): `signal_only_seed_when_f1_active`, `signal_only_no_seed_when_f1_inactive`, `signal_only_seed_unblocks_can_enter` (real RiskManager integration — same RM that was rejecting now accepts after seed). 22 F-1/signal_only tests pass. 204/204 orchestrator+risk_manager tests pass.
- **Commit:** `ca363e1a` (fix), verified intact through `e02c529d` and `8ca4e1c6`.

---

## Prior: Iteration 171 — lane_allocator.py duplicates RHO_REJECT_THRESHOLD from lane_correlation.py (FIXED 9809f1b8)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Canonical violation (integrity-guardian.md § 2) | `lane_allocator.py:506` defined `CORRELATION_REJECT_RHO = 0.70` duplicating `RHO_REJECT_THRESHOLD = 0.70` from `lane_correlation.py`. | LOW | FIXED 9809f1b8 |

---

## Prior: Iteration 170 — outcome_builder.py dead parameter break_ts (FIXED 9b16c4eb)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Dead parameter (institutional-rigor.md § 5) | `_compute_outcomes_all_rr` accepted `break_ts=None` but never read it. 4 callsites passed it unnecessarily. | LOW | FIXED 9b16c4eb |

---

## Prior: Iteration 169 — db_manager.py verify_trading_app_schema silent verifier gap (FIXED 6811640a)

| Sin | Finding | Severity | Status |
|-----|---------|----------|--------|
| Fail-open / Silent failure | `verify_trading_app_schema` missing 12 migration-added columns from expected_cols. | MEDIUM | FIXED 6811640a |

---

## Files Fully Scanned

> Cumulative list — 252 files fully scanned (session_orchestrator.py re-audited iter 172 for B6).

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
- trading_app/live/session_orchestrator.py — added iter 124, re-audited iter 172 (B6 verification)
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
- trading_app/lane_allocator.py — added iter 143, re-audited iter 171
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
- trading_app/risk_manager.py — added iter 166, re-audited iter 172 (B6 verification)
- trading_app/strategy_fitness.py — re-audited iter 167
- trading_app/topstep_scaling_plan.py — added iter 168
- trading_app/lane_correlation.py — added iter 171
- **Total: 252 files fully scanned**

## Next iteration targets
- Priority 0: check HANDOFF.md for remaining BLOCKER items after B6 closure
- Priority 2 (stale re-audit): trading_app/live/session_orchestrator.py — significantly changed post-iter-124 scan; already covered in iter 172 for B6 only; broader re-audit warranted given e02c529d (5 silent-failure fixes) not fully audited
- Priority 3 (unscanned medium): trading_app/eligibility/ directory files
- Note: pre-existing drift advisories (checks 59/95) — check #59 resolved by ca363e1a MGC backfill; check #95 still requires `select_family_rr.py` run
