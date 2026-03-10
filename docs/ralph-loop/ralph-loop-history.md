# Ralph Loop — Iteration History

> APPEND ONLY. Never delete or overwrite entries.
> Each iteration appends a structured block below.

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
- Commit: PENDING
