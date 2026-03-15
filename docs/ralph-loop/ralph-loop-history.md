# Ralph Loop — Iteration History

> APPEND ONLY. Never delete or overwrite entries.
> Each iteration appends a structured block below.

---

## Iteration 60 — 2026-03-15
- Phase: audit-only
- Classification: audit-only
- Target: trading_app/ai/ (cli.py, corpus.py, grounding.py, query_agent.py, sql_adapter.py, strategy_matcher.py)
- Finding: All 6 files CLEAN — 0 actionable findings. 4 ACCEPTABLE observations: (1) broad except in query_agent.py query tool (error surfaced in result), (2) VALID_ENTRY_MODELS includes E3 (query analysis needs historical E3 data), (3) session names in grounding prompt text (informational, not canonical logic), (4) hardcoded MGC in strategy_matcher.py (one-off research tool).
- Action: audit-only; no code changes
- Blast radius: N/A
- Verification: 81 tests PASS (tests/test_trading_app/test_ai/), drift 72/72 CLEAN
- Commit: NONE

---

## Iteration 58 — 2026-03-15
- Phase: audit-only
- Classification: audit-only
- Target: trading_app/live/projectx/ (5 files: auth, contract_resolver, data_feed, order_router, positions) + trading_app/live/tradovate/ (5 files: auth, contract_resolver, data_feed, order_router, positions)
- Finding: All 10 files CLEAN — 0 actionable findings. 1 ACCEPTABLE observation: projectx/positions.py uses int 0 vs float 0.0 default for avg_price (style only, avg_price used solely for logging). INSTRUMENT_SEARCH_TERMS and PRODUCT_MAP are broker API mappings, not canonical instrument lists. Exception handlers are logged (not silent). Order routers fail-closed on unsupported entry models.
- Action: audit-only; no code changes
- Blast radius: N/A
- Verification: 35 tests PASS (test_projectx_auth + test_projectx_feed + test_projectx_router + test_tradovate_positions), drift 72/72 CLEAN
- Commit: NONE

---

## Iteration 57 — 2026-03-15
- Phase: audit-only
- Classification: audit-only
- Target: trading_app/live/ (8 files: circuit_breaker, cusum_monitor, notifications, live_market_state, multi_runner, broker_factory, broker_base, trade_journal)
- Finding: All 8 files CLEAN — 0 actionable findings. Intentional fail-open patterns (notifications, trade_journal) correctly documented in design docs. Canonical violations: none. Exception handling: all either logged CRITICAL or are intentional best-effort paths.
- Action: audit-only; no code changes
- Blast radius: N/A
- Verification: 56 tests PASS (test_circuit_breaker + test_cusum_monitor + test_live_market_state + test_multi_runner + test_trade_journal), drift 72/72 CLEAN
- Commit: NONE

---

## Iteration 56 — 2026-03-15
- Phase: audit-only
- Classification: audit-only
- Target: trading_app/nested/compare.py + scripts/tools/build_edge_families.py
- Finding: Both files CLEAN — 0 actionable findings. 3 ACCEPTABLE observations in compare.py (hardcoded orb_minutes default, redundant `or 0` patterns, cosmetic None-masking in display).
- Action: audit-only; no code changes
- Blast radius: N/A
- Verification: 85 tests PASS (test_nested + test_edge_families), drift 72/72 CLEAN
- Commit: NONE

---

## Iteration 49 — 2026-03-14
- Phase: audit-only
- Target: trading_app/strategy_fitness.py
- Finding: 0 findings — full Seven Sins scan clean
- Action: audit-only; one low-severity observation (SQL f-string from config values) marked ACCEPTABLE
- Blast radius: N/A
- Verification: 31 tests PASS, drift CLEAN, ruff CLEAN
- Commit: NONE

---

## Iteration 50 — 2026-03-15
- Phase: fix
- Target: trading_app/execution_engine.py:457-465 (_arm_strategies)
- Finding: Fail-open — unknown filter_type silently armed strategy instead of blocking
- Action: Inverted condition to if filt is None: logger.error + continue (fail-closed). Matches portfolio.py, rolling_portfolio.py, strategy_fitness.py pattern.
- Blast radius: 1 file
- Verification: 43 tests PASS, drift 72/72 CLEAN
- Commit: 100e9da

---

## Iteration 1 — 2026-03-09
- Phase: audit
- Target: full codebase (trading_app/live/)
- Finding: 8 findings (3 HIGH, 3 MEDIUM, 2 LOW) — silent failures, fail-open, hardcoded values
- Action: Auditor ran 4 infrastructure gates + Seven Sins scan
- Verification: 3/4 PASS (pytest had 1 failure: tradovate positions BUY/SELL vs long/short)
- Commit: NONE (audit only)

## Iteration 2 — 2026-03-09
- Phase: fix
- Target: tradovate/order_router.py:167, projectx/order_router.py:102
- Finding: cancel() silently returns on no-auth — counted as successful cancel when it isn't, could cause double-closure
- Action: Changed silent return to raise RuntimeError (fail-closed). Caller _cancel_brackets catches exceptions and counts as failed.
- Verification: PASS — all 6 gates (71 drift, behavioral clean, 14/14 router tests, ruff clean, blast radius checked, regression scan clean)
- Commit: 9c40b5e
- Also confirmed: Finding 1 (positions.py) and Finding 2 (slippage) were already fixed in HEAD

## Iteration 3 — 2026-03-09
- Phase: fix
- Target: session_orchestrator.py:524-530
- Finding: Hardcoded risk_pts=10.0 fallback distorts P&L differently per instrument (2-5x for MGC)
- Action: Removed hardcoded 10.0, set actual_r=0.0 (neutral) when risk unknown. CUSUM sees no signal instead of wrong signal.
- Verification: PASS — all 6 gates (71 drift, behavioral clean, 77/77 orchestrator tests, ruff clean, blast radius verified, regression clean)
- Commit: e78d63b

## Iteration 4 — 2026-03-09
- Phase: fix
- Target: webhook_server.py:218 (Finding 6) and webhook_server.py:99,227 (Finding 8)
- Finding: Non-constant-time secret comparison on Cloudflare-exposed endpoint + deprecated asyncio.get_event_loop()
- Action: Added hmac.compare_digest() for timing-safe auth. Replaced get_event_loop() with get_running_loop() (2 occurrences).
- Verification: PASS — all 6 gates (71 drift, 185 tests, ruff clean, behavioral clean)
- Commit: ac90d71

## Iteration 5 — 2026-03-09
- Phase: audit (new targets)
- Target: execution_engine.py, strategy_validator.py, outcome_builder.py, build_daily_features.py
- Finding: 5 findings (0 HIGH, 3 MEDIUM, 1 LOW, 1 SKIPPED) — dormant calendar sizing bug, annotation debt, false-positive calendar signals
- Action: Auditor ran 4 infrastructure gates + Seven Sins scan on core trade logic files. outcome_builder.py CLEAN.
- Verification: 4/4 PASS (2745 passed, 0 failed, 9 skipped)
- Commit: NONE (audit only)

## Iteration 6 — 2026-03-09
- Phase: fix (batch — cross-file MEDIUM, shared blast radius)
- Target: batch: execution_engine.py:688,879,1020 + calendar_overlay.py:93-119
- Finding: batch: F1 (size_multiplier not applied to contracts) + F2 (broken month boundary signals always True)
- Action: F1: Applied size_multiplier at all 3 entry model sizing paths (E2/E1/E3). F2: Removed broken month boundary signal detectors and unused imports.
- Verification: PASS — all 6 gates (71 drift, behavioral clean, 2751 passed, ruff clean, blast radius verified, regression clean)
- Commit: e465403

## Iteration 7 — 2026-03-09
- Phase: fix (batch — MEDIUM same-file + LOW cross-file)
- Target: batch: pipeline/build_daily_features.py (F3) + trading_app/strategy_validator.py (F4)
- Finding: batch: F3 (annotation debt — 10 hardcoded thresholds missing @research-source) + F4 (silent risk fallback to min_risk_floor_points)
- Action: F3: Added @research-source + @revalidated-for annotations at 4 threshold clusters (day_type, RSI lookback, ATR velocity, compression z-score). F4: Added logger.warning when risk fallback fires with strategy_id and values.
- Verification: PASS — all 6 gates (71 drift, behavioral clean, 2751 passed/9 skipped, ruff clean, blast radius verified, regression clean)
- Commit: ea46784

## Iteration 8 — 2026-03-09
- Phase: audit (new targets)
- Target: session_orchestrator.py, cusum_monitor.py, performance_monitor.py, scoring.py, portfolio.py
- Finding: 8 findings (0 CRITICAL, 1 HIGH, 2 MEDIUM, 3 LOW, 2 SKIPPED) — orphan detection fail-open, CUSUM reset gap, dead max_contracts field
- Action: Auditor ran 4 infrastructure gates + Seven Sins scan. scoring.py CLEAN. portfolio.py core sizing CORRECT.
- Verification: 4/4 PASS (2751 passed, 0 failed, 9 skipped)
- Commit: NONE (audit only)

## Iteration 10 — 2026-03-09
- Phase: fix (batch — MEDIUM + 2 LOW, same blast radius: live trading path)
- Target: batch: performance_monitor.py:96-99 (F2) + performance_monitor.py:60 (F3) + session_orchestrator.py:1030 (F4)
- Finding: batch: F2 (CUSUM monitors not reset at daily boundary) + F3 (threshold hardcoded) + F4 (fill poller NotImplementedError silent)
- Action: F2: Added `monitor.clear()` loop to `reset_daily()`. F3: Extracted threshold to class constant `CUSUM_THRESHOLD=4.0` with @research-source annotation. F4: Added `log.debug` for NotImplementedError in fill poller.
- Verification: PASS — all 6 gates (71 drift, behavioral clean, 2751 passed/9 skipped, ruff clean, blast radius verified, regression clean)
- Commit: 2ce0c70

## Iteration 9 — 2026-03-09
- Phase: fix (cross-terminal catch-up)
- Target: batch: session_orchestrator.py:931, webhook_server.py:201
- Finding: 2 remaining unsafe result.order_id patterns missed by iterations 1-8. Kill switch path (line 931) crashes on emergency flatten if broker returns non-dict. Webhook server (line 201) ALWAYS crashes — submit() returns dict, .order_id is never valid on dict.
- Action: Applied safe getattr()/dict.get() pattern consistent with entry/exit paths fixed in iteration 4. Webhook server was a confirmed crash-on-every-call bug (endpoint non-functional since creation).
- Verification: PASS — all 6 gates (71 drift, behavioral clean, 185 fast tests, ruff clean, blast radius clean, no regressions)
- Commit: 7002aad

## Iteration 11 — 2026-03-09
- Phase: audit (post-feature integrity)
- Target: execution_engine.py (multi-aperture), live_config.py (orb_minutes), strategy_discovery.py, walkforward.py, strategy_fitness.py, circuit_breaker.py, + all uncommitted changes (ML predictor, liveness probe, model staleness, RR lock fix)
- Finding: 0 new findings. All targets CLEAN. Multi-aperture ORB correct. Discovery/WF methodologically sound. Uncommitted changes verified.
- Action: Auditor ran 4 infrastructure gates + Seven Sins scan on 10 production files. 3 LOW deferred from iter 9.
- Verification: 4/4 PASS (2757 passed, 0 failed, 9 skipped)
- Commit: NONE (audit only)

## Iteration 13 — 2026-03-10
- Phase: fix (HIGH — test infrastructure)
- Target: tests/test_trading_app/test_ml/test_features.py:389,398,407,417,426
- Finding: TestLoadFeatureMatrixIntegration called load_feature_matrix() without date bounds → OOM (1.1-1.2 GiB) on full MGC dataset, blocking CI
- Action: Added min_date="2024-06-01" / max_date="2024-12-31" to all 5 unbounded calls. Function already supported params. No production code changed.
- Verification: PASS — 71 drift, behavioral clean, 5/5 ML integration tests pass (68s), pre-commit hook 28 passed, ruff clean, blast radius = test file only
- Commit: d93fa92

## Iteration 12 — 2026-03-09
- Phase: audit+fix (Bloomey deep dive — live trading critical path)
- Target: risk_manager.py, portfolio.py, cost_model.py, rolling_portfolio.py, strategy_fitness.py
- Finding: 5 findings (0 CRITICAL, 0 HIGH, 4 MEDIUM, 1 LOW) — hardcoded SINGAPORE_OPEN exclusion (portfolio.py:312,352), fail-open unknown filter (strategy_fitness.py:332), dormant orb_minutes=5 in rolling DOW stats (rolling_portfolio.py:304), unannotated thresholds (7 locations), session slippage no provenance
- Action: F2 fixed (portfolio.py exclusion → config.EXCLUDED_FROM_FITNESS), F5 fixed (fail-open → fail-closed with warning log, both per-strategy and batch paths aligned), F3 partially annotated (strategy_fitness + rolling_portfolio thresholds), F1 annotated TODO for multi-aperture extension
- Verification: PASS — 71 drift, behavioral clean, 135/135 companion tests, ruff clean
- Grade: B+ (Bloomey)
- Commit: PENDING

## Iteration 15 — 2026-03-10
- Phase: fix (batch LOW — annotation debt)
- Target: trading_app/walkforward.py:162,242
- Finding: IS minimum sample guard (15) and window imbalance ratio (5.0x) missing @research-source
- Action: Added @research-source + @revalidated-for to both magic numbers (Lopez de Prado AFML Ch.11 for IS guard; Pardo Ch.7 for imbalance ratio). Comments only, no logic change.
- Verification: PASS — 26/26 walkforward tests, 71 drift, behavioral clean, ruff clean
- Commit: e2ca011

## Iteration 13 — 2026-03-10
- Phase: audit+fix (tradebook/pipeline — outcome_builder, strategy_discovery, strategy_validator, build_edge_families, live_config)
- Finding: 5 findings (0 CRITICAL, 0 HIGH, 1 MEDIUM, 4 LOW) — dollar gate fail-open on NULL median_risk_points (live_config.py:367), unannotated edge family thresholds (build_edge_families.py:31-38), WF gate thresholds missing @research-source (strategy_validator.py:654-656), HOT tier thresholds unannotated (live_config.py:54-57), live portfolio constructor magic numbers inline (live_config.py:354-355,583-584)
- Action: N1 fixed (dollar gate fail-open → fail-closed with logger.warning). Test updated (test_none_guard_passes → test_none_guard_blocks). N2-N5 deferred (LOW annotation work).
- Verification: PASS — 71 drift, behavioral clean, 20/20 live_config tests, ruff clean, blast radius verified (2 callers handle False), regression clean
- Grade: A- (Bloomey)
- Commit: PENDING

## Iteration 16 — 2026-03-10
- Phase: fix
- Target: scripts/tools/generate_trade_sheet.py:134,140
- Finding: Dollar gate `_passes_dollar_gate` fail-open on missing data/exception — diverges from live_config.py fail-closed pattern (fixed in iter 13). Trade sheet could show phantom trades user would never actually trade.
- Action: Changed both `return True` paths to `return False` (fail-closed). Aligns with live_config.py:372-391.
- Verification: PASS — 71 drift, behavioral clean, 20/20 live_config tests, ruff clean, blast radius = 0 external callers, trade sheet still generates 33 trades
- Commit: 29f37d1

## Iteration 17 — 2026-03-10
- Phase: fix (batch — T2 + T3, same query)
- Target: scripts/tools/generate_trade_sheet.py:200-226 (_load_best_by_expr)
- Finding: T2: LEFT JOIN family_rr_locks with IS NULL fallback diverges from live_config's INNER JOIN — could show unlocked RR variants. T3: query missing vs.orb_minutes, aperture parsed from strategy_id string instead.
- Action: Changed LEFT JOIN → INNER JOIN, removed IS NULL fallback. Added vs.orb_minutes to SELECT, replaced _parse_aperture() call with variant["orb_minutes"]. Removed dead _parse_aperture function.
- Verification: PASS — 71 drift, behavioral clean, 20/20 tests, ruff clean, 33 trades unchanged
- Commit: 7caa6fa

## Iteration 18 — 2026-03-10
- Phase: fix (code review finding — IMPORTANT)
- Target: scripts/tools/generate_trade_sheet.py:112,216
- Finding: _exp_dollars_from_row adds spec.total_friction to 1R base (inflating Exp$), diverging from live_config which uses median_risk_pts * point_value only. Also: missing NULLS LAST in ORDER BY.
- Action: Removed + spec.total_friction from 1R calculation. Added NULLS LAST. One marginal trade correctly dropped: MES CME_PRECLOSE VOL_RV12_N20 (old $5.48 inflated → real $4.82, gate $4.86).
- Verification: PASS — 71 drift, behavioral clean, 20/20 tests, ruff clean, 32 trades (1 correctly dropped)
- Commit: f82c408

## Iteration 19 — 2026-03-10
- Phase: audit-only
- Target: trading_app/execution_engine.py (1229 lines)
- Finding: 3 LOW (conditional EXITED prune, E3 silent exit, IB hardcoded 23:00 UTC). All dormant — E3 soft-retired, IB TOKYO_OPEN only, prune harmless.
- Action: Full Seven Sins scan. Engine CLEAN on all critical paths (E2/E1/E3 entry, exit logic, state management, canonical imports, fail-closed unknowns).
- Verification: 4/4 PASS (71 drift, behavioral clean, 41/41 engine tests, ruff clean)
- Commit: NONE (audit only)

## Iteration 20 — 2026-03-10
- Phase: audit+fix (trading logic pipeline — config.py, strategy_discovery.py, outcome_builder.py)
- Target: strategy_discovery.py:630,634 + portfolio.py:965 + backfill_dollar_columns.py:92-95
- Finding: 3 findings (0 CRIT, 0 HIGH, 1 MEDIUM, 2 LOW). SD1: median_risk_dollars and avg_risk_dollars include total_friction, inflating stored values. Same error class as trade sheet T5 (iter 18). SD2: session fallback to ORB_LABELS (LOW). SD3: CORE/REGIME_MIN_SAMPLES missing @research-source (LOW).
- Action: SD1 fixed — removed + total_friction from both lines in compute_metrics(). Updated portfolio.py back-computation (was subtracting friction to undo the inflation). Aligned backfill_dollar_columns.py. All three core files CLEAN on Seven Sins (no look-ahead, fail-closed unknowns, correct cost model, BH FDR/DSR/FST properly computed).
- Verification: PASS — 71 drift, behavioral clean, 113/113 tests (discovery + portfolio), ruff clean, blast radius verified (3 files)
- Commit: 137bf27

## Iteration 21 — 2026-03-10
- Phase: fix (order_router.py — both brokers)
- Target: tradovate/order_router.py:136,140,202,206 + projectx/order_router.py:74,88,171
- Finding: OR1: Fill price `or` pattern uses falsy check — 0.0 fill price treated as None. 7 instances across 2 broker routers. Python antipattern: `x or y` and `if x` on numeric types.
- Action: Replaced `or` with `if x is None: x = fallback`, replaced `if x` with `if x is not None`, added float() cast to ProjectX query_order_status for consistency. Also found OR2 (no fill_price parsing tests) — deferred.
- Verification: PASS — 6/6 gates (71 drift, behavioral clean, 8/8 router tests, ruff clean, blast radius confirmed — all callers already use `is not None`, 83/83 orchestrator regression)
- Commit: 3b10732

## Iteration 25 — 2026-03-11
- Phase: test (OR2 — fill_price parsing coverage)
- Target: tests/test_trading_app/test_order_router.py + test_projectx_router.py
- Finding: OR2: No unit tests for fill_price parsing in submit() / query_order_status() — the is-None guard from iter 21 was untested.
- Action: 14 new tests (7 Tradovate + 7 ProjectX): primary field, fallback field, both absent → None, zero price not falsy. Mock at module level (no HTTP required).
- Verification: PASS — 28/28 tests, 6/6 hooks (ruff auto-formatted, M2.5 skipped test-only change), drift clean
- Commit: 8261a0e

## Iteration 24 — 2026-03-11
- Phase: fix (batch — slate-clear: 5 annotation/logging/warning fixes)
- Target: config.py:970, live_config.py:61-64, execution_engine.py:262, auth.py:42, strategy_discovery.py:1030
- Finding: SD3 (CORE/REGIME_MIN_SAMPLES unannotated), N4 (HOT tier unannotated), EE3 (IB 23:00 UTC hardcode), auth log gap (refresh_if_needed silent trigger), SD2 (ORB_LABELS fallback silent)
- Action: @research-source annotations on 4 constants; log.debug at auth trigger site; logger.warning on SD2 session fallback. N5 closed (no magic numbers at current lines). EE3 annotated inline.
- Verification: PASS — 71 drift, behavioral clean, 61/61 tests, ruff clean
- Commit: 7cf57cb

## Iteration 23 — 2026-03-11
- Phase: fix (batch — EE1 + Bloomey finding, same file)
- Target: execution_engine.py:1152-1154 (EE1) + execution_engine.py:640-641 (_try_entry silent drop)
- Finding: EE1: `if events:` guard prevented pruning of ghost EXITED trades from silent-exit paths (832/965/973). Bloomey new finding: _try_entry:640 zero-risk was a silent drop — no EXITED state, no completed_trades, no event. Inconsistent with all other reject paths.
- Action: EE1: Removed `if events:` — prune now unconditional (no-op when no EXITED trades). _try_entry: Added EXITED state + completed_trades.append + REJECT event before early return. Aligns with E1/E3 ARMED paths.
- Verification: PASS — all 6 gates (71 drift, behavioral clean, 41/41 engine + 83/83 orchestrator, ruff clean, blast radius verified, 185 fast tests)
- Commit: f7bd0c4

## Iteration 22 — 2026-03-10
- Phase: fix (batch — contract_resolver.py, strategy_fitness.py, portfolio.py)
- Target: contract_resolver.py:40, strategy_fitness.py:124, portfolio.py:953
- Finding: CR1: account ID `or` falsy-zero (same class as OR1). F3a: Sharpe decay threshold -0.1 inline → SHARPE_DECAY_THRESHOLD constant. F3b: trade frequency 0.4 annotated. Also closed iter 9 PRODUCT_MAP finding (has fallback, not a gate).
- Action: CR1 fixed (`or` → `is None`). F3a extracted to named constant with @research-source. F3b annotated inline. CR2 closed. F3 now resolved except cost_model.py (self-documenting canonical source).
- Verification: PASS — 6/6 gates (71 drift, behavioral clean, 31 fitness + 68 portfolio + 20 live_config tests, ruff clean, blast radius verified)
- Commit: 684a37c
## Iteration 26 — 2026-03-11
- Phase: fix
- Target: trading_app/live/position_tracker.py:189
- Finding: PT1: best_entry_price() uses `or` chain — fill_entry_price=0.0 silently falls through to engine_entry_price. Same falsy-zero antipattern as OR1 (iter 21) / OR2 (iter 25). Discovered during fresh audit of live/ modules.
- Action: bar_aggregator.py audited (CLEAN). position_tracker.py: replaced `or` chain with explicit `is not None` guards. Added zero-fill guard test to test_position_tracker.py (20 tests total).
- Verification: PASS — 4/4 gates (62 drift checks, behavioral clean, 20/20 position_tracker tests, ruff clean)
- Commit: f713a1c

## Iteration 30 — 2026-03-12
- Phase: fix (LOW — stale comment)
- Target: trading_app/strategy_discovery.py:1082
- Finding: SD1: comment "# E2+E3 (CB1 only)" stale — E3 is in SKIP_ENTRY_MODELS and never runs, but is intentionally still counted in total_combos for conservative n_trials_at_discovery (higher FST hurdle). Comment didn't explain the intentional overcounting.
- Action: Updated comment + added explanatory line. No code or logic change. Fresh full-file audit: sessions fallback already has warning (unlike outcome_builder at iter 29), canonical imports CLEAN, holdout temporal isolation correct, BH FDR annotation informational-only.
- Blast radius: 1 file, comment-only. Callers unaffected. total_combos value unchanged.
- Verification: ACCEPT — all 4 gates (71 drift, behavioral clean, 45/45 strategy_discovery tests, ruff clean). Pre-commit: 185/185 fast tests. M2.5 advisory on pre-existing file patterns.
- Commit: 371bc51

## Iteration 29 — 2026-03-12
- Phase: fix (LOW — observability)
- Target: trading_app/outcome_builder.py:677-678
- Finding: OB1: build_outcomes() silently falls back to ORB_LABELS when get_enabled_sessions() returns empty — misconfigured instruments produce invisible no-ops, no diagnostic log
- Action: Added logger.warning() before the fallback assignment. No logic change; fallback behavior preserved. Fresh audit of full file — no other actionable findings (look-ahead clean, canonical imports correct, idempotent writes correct).
- Blast radius: 1 file changed (log-only). Callers unaffected. logger already defined at line 20.
- Verification: ACCEPT — all 4 gates (71 drift, behavioral clean, 27/27 outcome_builder tests, ruff clean). Pre-commit: 185/185 fast tests. M2.5 advisory on pre-existing file patterns (not the added lines).
- Commit: 07b4ba9

## Iteration 28 — 2026-03-12
- Phase: fix (batch LOW — annotation debt + ledger cleanup)
- Target: trading_app/live_config.py:75,89
- Finding: DF-08: LIVE_MIN_EXPECTANCY_R=0.10 and LIVE_MIN_EXPECTANCY_DOLLARS_MULT=1.3 lacked @research-source annotations. DF-05 and DF-06 were already resolved (stale ledger entries — annotations confirmed present in build_edge_families.py and strategy_validator.py)
- Action: Added @research-source + @revalidated-for annotations to both constants. Closed DF-05 and DF-06 as already-resolved. DF-08 now fully resolved.
- Blast radius: 1 file changed (comment-only). Callers import constants by value — unaffected. Drift check #43 imports LIVE_MIN_EXPECTANCY_R — unaffected.
- Verification: ACCEPT — all 4 gates (71 drift, behavioral clean, 20/20 live_config tests, ruff clean). Pre-commit: 185/185 fast tests pass.
- Commit: 43a86ba

## Iteration 27 — 2026-03-12
- Phase: fix (batch LOW — annotation debt + warning log)
- Target: trading_app/rolling_portfolio.py:48,323,414
- Finding: batch: RP1 (silent filter skip in compute_day_of_week_stats — no log when ALL_FILTERS.get returns None) + RP2 (DEFAULT_LOOKBACK_WINDOWS=24 missing @research-source) + RP3 (min_expectancy_r=0.10 unannotated magic number)
- Action: RP1: Added logger.warning for unknown filter_type. RP2: Added @research-source annotation (Lopez de Prado AFML Ch.7 rolling window convention). RP3: Extracted MIN_EXPECTANCY_R=0.10 constant with @research-source (circular import prevents referencing live_config.LIVE_MIN_EXPECTANCY_R directly). RP4 (hardcoded E1/E2/E3 set in aggregate_rolling_performance:228) deferred — dormant, no E4 yet.
- Blast radius: 1 file checked. DEFAULT_LOOKBACK_WINDOWS imported by live_config.py (value unchanged). Callers always pass min_expectancy_r explicitly. compute_day_of_week_stats has no external callers.
- Verification: PASS — all 6 gates (71 drift, behavioral clean, 36/36 rolling_portfolio tests + 185 fast, ruff clean, blast radius confirmed, regression clean)
- Commit: 0515f15

## Iteration 28-30 — 2026-03-12
- See previous session context (outcome_builder, strategy_validator, strategy_discovery audits)
- Commits: various (iter 28 audit-only, iter 29 OB1 silent fallback fix, iter 30 SD1 stale comment)

## Iteration 31 — 2026-03-12
- Phase: fix (batch MEDIUM — canonical integrity)
- Target: trading_app/paper_trader.py:462,474 + trading_app/rolling_portfolio.py:238
- Finding: PT1+DF-11 — Hardcoded ("E1","E2","E3") in strategy ID parsers, should use config.ENTRY_MODELS
- Action: Added ENTRY_MODELS import to both files. Replaced 3 hardcoded tuples with canonical reference. Zero functional change — same values, different source.
- Blast radius: 2 files changed. All callers internal (private helpers). 15+ modules already import ENTRY_MODELS correctly. Drift check #39 validates config.py, not consumers.
- Verification: ACCEPT — Gate 1 (72 drift), Gate 2 (behavioral clean), Gate 3 (22/22 paper_trader + 36/36 rolling_portfolio), Gate 5 (no sins). Pre-commit: 186/186 fast tests pass.
- Commit: 9158b77

## Iteration 32 — 2026-03-13
- Phase: fix (batch MEDIUM+LOW — volatile data + dead code)
- Target: trading_app/mcp_server.py:213 + lines 54-55
- Finding: MCP1 — hardcoded "735 FDR-validated" and instrument data years in MCP instructions (volatile data violation). MCP2 — unused _CORE_MIN/_REGIME_MIN aliases.
- Action: Replaced hardcoded stats with dynamic values from ACTIVE_ORB_INSTRUMENTS. Removed unused constant aliases and their CORE_MIN_SAMPLES/REGIME_MIN_SAMPLES imports.
- Blast radius: 1 file changed. _build_server called only from __main__. No tests assert on instructions string. 39 files already import ACTIVE_ORB_INSTRUMENTS.
- Verification: ACCEPT — Gate 1 (72 drift), Gate 2 (behavioral clean), Gate 3 (17/17 mcp_server), Gate 5 (no sins). Pre-commit: 186/186 fast tests.
- Commit: da8af67

## Iteration 34 — 2026-03-13
- Phase: fix (batched)
- Target: trading_app/strategy_discovery.py:23,1125
- Finding: SD1 — PROJECT_ROOT dead variable (Orphan Risk); SD2 — hardcoded "2376+" combo count in comment (Volatile Data)
- Action: Deleted PROJECT_ROOT line; removed inline number from comment, kept intent
- Blast radius: 1 file
- Verification: PASS (45/45 tests, 72/72 drift checks)
- Commit: d318da7

## Iteration 35 — 2026-03-13
- Phase: fix
- Target: trading_app/strategy_validator.py:32
- Finding: SV1 — PROJECT_ROOT module-level constant defined but never referenced (Orphan Risk)
- Action: Deleted the single dead line; Path import retained (used at lines 641, 1287)
- Blast radius: 1 file, 0 callers, 0 importers
- Verification: PASS — 49/49 tests, drift clean (72 checks)
- Commit: c0b6cf6

## Iteration 36 — 2026-03-13
- Phase: fix
- Target: pipeline/build_daily_features.py:884,1143
- Finding: BDF1 — ["CME_REOPEN","TOKYO_OPEN","LONDON_METALS"] duplicated verbatim at two for-loop sites with no shared constant (Canonical Violation)
- Action: Extracted to module-level COMPRESSION_SESSIONS constant at line 91 with @research-source/@revalidated-for annotations and schema cross-reference; both loops replaced with constant reference
- Blast radius: 1 file, 0 external callers
- Verification: PASS — 72/72 drift checks, 6/6 behavioral, ruff clean, 60/60 pytest
- Commit: 49b32a9

## Iteration 37 — 2026-03-13
- Phase: fix
- Target: trading_app/cascade_table.py:17
- Finding: CT1 — Dead `PROJECT_ROOT` assignment (orphan risk) + relative path in module docstring usage example
- Action: Removed unused `PROJECT_ROOT = Path(__file__).resolve().parent.parent`. Updated docstring example to use `GOLD_DB_PATH` from `pipeline.paths`.
- Blast radius: 1 file (cascade_table.py); 3 importers checked — none reference PROJECT_ROOT
- Verification: PASS (7/7 test_cascade_table.py, drift 72/72 clean, ruff clean)
- Commit: 00511df

## Iteration 38 — 2026-03-13
- Phase: fix
- Target: trading_app/market_state.py:20 + docstring:10
- Finding: MS1+MS2 — Dead `PROJECT_ROOT` assignment (Orphan Risk, defined but never referenced in file) + relative `Path("gold.db")` in module docstring usage example (Canonical violation). Identical pattern to CT1 fixed in cascade_table.py iter 37.
- Action: Removed unused `PROJECT_ROOT = Path(__file__).resolve().parent.parent`. Updated docstring usage example to `from pipeline.paths import GOLD_DB_PATH` + `GOLD_DB_PATH` usage.
- Blast radius: 1 file changed; 2 callers checked (paper_trader.py, test_market_state.py) — neither references PROJECT_ROOT
- Verification: PASS (19/19 test_market_state.py, drift 72/72 clean)
- Commit: 94dfe8c

## Iteration 39 — 2026-03-13
- Phase: fix
- Target: trading_app/risk_manager.py:15-17
- Finding: RM1 — Dead PROJECT_ROOT assignment + unused `from pathlib import Path` import — neither referenced anywhere in file (Orphan Risk)
- Action: Removed `from pathlib import Path` import and `PROJECT_ROOT = Path(__file__).resolve().parent.parent` assignment (3 lines deleted)
- Blast radius: 1 file (risk_manager.py only — callers use RiskLimits/RiskManager API, no API change)
- Verification: PASS (30/30 test_risk_manager.py, drift 63 OK)
- Commit: adf475f

### scoring.py scan — CLEAN (except SC1 noted below)
- SC1: Hardcoded session names SINGAPORE_OPEN/TOKYO_OPEN in heuristic bonus logic — ACCEPTABLE. These are intentional per-session heuristic adjustments, not a canonical list. Worst case on session rename: bonus silently stops applying. Not a safety/correctness issue.
- No silent failures, no fail-open, no look-ahead bias, no cost illusion, no volatile data.

## Iteration 40 — 2026-03-13
- Phase: fix
- Target: trading_app/execution_engine.py:21,23
- Finding: EE1 — Dead `from pathlib import Path` import and `PROJECT_ROOT = Path(__file__).resolve().parent.parent` assignment; never referenced in the file. Same orphan pattern as CT1/MS1/RM1 (iters 37-39).
- Action: Removed both dead lines; added missing blank line between stdlib and first-party import groups (ruff I001).
- Blast radius: 0 files (pure dead code removal, no callers affected)
- Verification: PASS — 64 tests passed, 72 drift checks passed, ruff clean
- Commit: 1c7a133

## Iteration 41 — 2026-03-13
- Phase: fix
- Target: trading_app/paper_trader.py:23
- Finding: PT1 — Dead PROJECT_ROOT assignment (never referenced; same orphan pattern as EE1/CT1/MS1/RM1)
- Action: Removed 1-line dead assignment. Path import retained (used for db_path type annotation at line 202).
- Blast radius: 1 file (paper_trader.py only; 2 importers unaffected — they import replay_historical, not PROJECT_ROOT)
- Verification: PASS (22/22 tests, 72/72 drift checks)
- Commit: 6a09e64

## Iteration 42 — 2026-03-13
- Phase: fix
- Target: trading_app/live_config.py:21
- Finding: LC1 — Dead PROJECT_ROOT assignment (never referenced; same orphan pattern as EE1/PT1/CT1/MS1/RM1)
- Action: Removed 1-line dead assignment. Path import retained (used for db_path type annotations and Path(args.output) in main()).
- Blast radius: 1 file (live_config.py only; no external callers of PROJECT_ROOT)
- Verification: PASS (36/36 tests, 72/72 drift checks)
- Commit: 27604b9

## Iteration 43 — 2026-03-14
- Phase: fix
- Target: trading_app/rolling_portfolio.py:24
- Finding: RP1 — Dead PROJECT_ROOT assignment (never referenced; same orphan pattern as EE1/PT1/LC1/CT1/MS1/RM1)
- Action: Removed 1-line dead assignment. Path import retained (used for Path(args.output) in main() at line 599).
- Blast radius: 1 file (rolling_portfolio.py only; 3 importers unaffected — none import PROJECT_ROOT)
- Verification: PASS (37/37 tests, 72/72 drift checks)
- Commit: aa818e1

## Iteration 44 — 2026-03-14
- Phase: audit-only
- Target: trading_app/strategy_fitness.py (1153 lines)
- Finding: 0 findings — full Seven Sins scan clean
- Action: No fix. Audit-only.
- Blast radius: N/A
- Verification: PASS (4/4 gates: drift 72 checks, behavioral 6 checks, ruff clean, pytest 31 tests)
- Commit: NONE

## Iteration 45 — 2026-03-14
- Phase: fix
- Target: trading_app/execution_engine.py:410-412
- Finding: DF-02 (LOW) — ARMED/CONFIRMING trades silently discarded at session_end with no log entry; orphan trades invisible to diagnostics
- Action: Added logger.debug() emitting strategy_id and state before the existing state=EXITED + completed_trades.append() path. Zero behavior change.
- Blast radius: 1 file (execution_engine.py only; callers paper_trader.py and session_orchestrator.py unaffected — pure logging addition)
- Verification: PASS (43/43 tests, 72/72 drift checks)
- Commit: 4c6bc4d

## Iteration 46 — 2026-03-14
- Phase: fix
- Target: trading_app/outcome_builder.py:22
- Finding: OB1 (LOW) — Dead `PROJECT_ROOT` assignment at module level, never referenced in file or imported by callers. Same orphan-risk pattern as RP1 (iter 43).
- Action: Removed the single dead assignment line. `Path` import retained (used elsewhere). Also conducted full Seven Sins scan of outcome_builder.py (all clean) and reassessed DF-04 structural blocker in rolling_portfolio.py (confirmed deferred — blast radius >5 files).
- Blast radius: 1 file (outcome_builder.py only)
- Verification: PASS (27/27 tests, 72/72 drift checks)
- Commit: f6b34f6

## Iteration 47 — 2026-03-14
- Phase: fix
- Target: trading_app/strategy_validator.py:7-14
- Finding: SV1 (LOW) — Module docstring Phases list omitted phases 4c (Deflated Sharpe/DSR, informational) and 4d (False Strategy Theorem hurdle, informational), both added after the original docstring was written. Misleads readers about the full validation sequence.
- Action: Added two lines to the docstring Phases list documenting 4c and 4d as informational-only sub-phases. The "7-phase" header count was not changed — it is accurate for the 7 hard-gate phases. Also conducted full Seven Sins scan of paper_trader.py and strategy_discovery.py (both clean, no findings).
- Blast radius: 1 file (docstring only; "7-phase" string has no callers)
- Verification: PASS (49/49 tests, 72/72 drift checks)
- Commit: 7ed02ab

## Iteration 48 — 2026-03-14
- Phase: fix
- Target: trading_app/portfolio.py:22
- Finding: PF1 (LOW) — Dead `PROJECT_ROOT` assignment at module level, never referenced in file or imported by callers. Same orphan-risk pattern as RP1 (iter 43) and OB1 (iter 46).
- Action: Removed the single dead assignment line. `Path` import retained — used in function signatures throughout portfolio.py (load_validated_strategies, build_portfolio, build_strategy_daily_series, correlation_matrix, main). Seven Sins scan of walkforward.py (CLEAN), portfolio.py (1 finding FIXED), strategy_fitness.py (pending next iter).
- Blast radius: 1 file (portfolio.py only; no callers import PROJECT_ROOT)
- Verification: PASS (68/68 tests, 72/72 drift checks)
- Commit: e792bb5

## Iteration 51 — 2026-03-15
- Phase: fix
- Target: trading_app/live_config.py:499
- Finding: Bare `except Exception` in `_check_dollar_gate` — narrowed to `(ValueError, TypeError)`
- Action: Changed `except Exception as exc:` to `except (ValueError, TypeError) as exc:` — aligns with pipeline fortification pattern; behavior identical (fail-closed return preserved)
- Blast radius: 1 file (live_config.py; _check_dollar_gate is private)
- Verification: PASS (72 drift checks, 36 tests)
- Commit: b486e9a

---

## Iteration 52 — 2026-03-15
- Phase: audit-only
- Target: trading_app/entry_rules.py + trading_app/db_manager.py
- Finding: CLEAN — no actionable findings in either file
- Action: Seven Sins scan complete, 0 findings across 2 files
- Blast radius: N/A (no changes)
- Verification: PASS (64 entry_rules tests, drift 72/72)
- Commit: NONE

---

## Iteration 53 — 2026-03-15
- Phase: fix
- Target: trading_app/execution_spec.py:46
- Finding: Hardcoded ["E1", "E3"] in ExecutionSpec.validate() — E2 (active primary entry model) rejected, E3 (soft-retired) accepted. Canonical violation (Sin 5).
- Action: Imported ENTRY_MODELS from trading_app.config; replaced hardcoded list in validate(); updated error message to dynamic format; updated test_execution_spec.py to cover E2 and use dynamic error match
- Blast radius: 2 files (execution_spec.py + test_execution_spec.py)
- Verification: PASS (26 tests, drift 72/72)
- Commit: 41f19b4

---

## Iteration 54 — 2026-03-15
- Phase: audit-only
- Target: trading_app/pbo.py + trading_app/nested/builder.py + trading_app/nested/schema.py
- Finding: CLEAN — no actionable findings across all 3 files
- Action: Seven Sins scan complete. pbo.py iterrows() is build-time documented; missing orb_minutes filter in _get_eligible_days is inefficient but correct (set deduplication). nested/builder.py CB1 guards for E2/E3 are behavioral, not canonical violations. nested/schema.py expected_tables list is self-referential verification, not a canonical list.
- Blast radius: N/A (no changes)
- Verification: PASS (73 tests across test_nested/ + test_pbo.py, drift 72/72)
- Commit: NONE

---

## Iteration 55 — 2026-03-15
- Phase: fix
- Classification: [judgment]
- Target: trading_app/nested/discovery.py:174
- Finding: nested/discovery.py missing SKIP_ENTRY_MODELS guard — E3 (soft-retired) processed in nested grid search, generating stale E3 nested strategies and wasting ~14% compute. Parent strategy_discovery.py applies this guard at line 1090; nested variant was missing it.
- Action: Imported SKIP_ENTRY_MODELS from trading_app.config; added skip guard `if em in SKIP_ENTRY_MODELS: continue` inside the ENTRY_MODELS loop, matching the pattern in strategy_discovery.py
- Blast radius: 1 file (discovery.py), 1 test file (test_discovery.py — no test exercises the skip guard directly)
- Verification: PASS (63 tests across test_nested/, drift 72/72)
- Commit: 52c74c5
