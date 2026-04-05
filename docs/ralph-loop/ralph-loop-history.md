# Ralph Loop — Iteration History

> APPEND ONLY. Never delete or overwrite entries.
> Each iteration appends a structured block below.

---

## Iteration 131 — 2026-03-18
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/ml/predict_live.py:307
- Finding: Aggressive RR mismatch (trade skip, take=False) logged at logger.debug() — invisible at standard INFO log level; aperture mismatch (take=True) inconsistently logged at INFO
- Action: Changed logger.debug → logger.info for aggressive RR skip path only; conservative path left at debug (trade proceeds, informational only)
- Blast radius: 1 file (predict_live.py); log level change only, no callers affected
- Verification: PASS (36/36 test_predict_live.py + 218 fast tests pass)
- Commit: d0fe929

---

## Iteration 130 — 2026-03-17
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/ml/meta_label.py:386
- Finding: CPCV failure logged at DEBUG (invisible at INFO log level), silently bypasses Gate 2 (CPCV AUC check) with no visible warning
- Action: Changed logger.debug to logger.warning with exc_info=True so CPCV failures are visible and the cause is captured
- Blast radius: 1 file (meta_label.py); no callers affected
- Verification: PASS (10/10 test_meta_label.py + 72 drift checks + 218 fast tests pass)
- Commit: c7b0774

---

## Iteration 129 — 2026-03-17
- Phase: fix
- Classification: [judgment]
- Target: trading_app/ml/config.py:267-274
- Finding: compute_config_hash() omits GLOBAL_FEATURES, SESSION_FEATURE_SUFFIXES, ATR_NORMALIZE, CATEGORICAL_FEATURES, and LOOKAHEAD_BLACKLIST — changes to these primary feature-engineering lists would not be detected as "retrain needed" (silent failure in safety mechanism)
- Action: Added missing feature-engineering lists to config_str. Used sorted(LOOKAHEAD_BLACKLIST) for deterministic hashing of the set.
- Blast radius: 1 file (config.py); callers unchanged
- Verification: PASS (46 ML tests pass, 72 drift checks pass, 218 fast tests pass)
- Commit: 3a0e01b

---

## Iteration 128 — 2026-03-17
- Phase: diminishing-returns
- Classification: audit-only
- Target: trading_app/live/circuit_breaker.py, trading_app/live/cusum_monitor.py, trading_app/live/projectx/__init__.py
- Finding: All three scope files clean — zero Seven Sins violations. consecutive_low_only = 3 (≥3 threshold); all targets LOW centrality.
- Action: DIMINISHING_RETURNS triggered — audited all three files, found zero findings. No fix performed. Recommend re-scope to trading_app/ml/config.py (CRITICAL tier).
- Blast radius: 0 files
- Verification: PASS (7/7 test_circuit_breaker.py + drift clean + behavioral audit clean)
- Commit: NONE

---

## Iteration 126 — 2026-03-17
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/live/tradovate/contract_resolver.py:16-17, order_router.py:26-27, positions.py:11-12
- Finding: LIVE_BASE/DEMO_BASE URL constants duplicated across 4 tradovate modules — canonical source is auth.py; 3 files hardcoded them independently (maintenance hazard: URL change requires 4 edits)
- Action: Removed duplicate constants from contract_resolver.py, order_router.py, positions.py; added `from .auth import DEMO_BASE, LIVE_BASE` to each
- Blast radius: 3 production files
- Verification: PASS (29/29 targeted tests + 218/218 pre-commit suite + drift clean)
- Commit: 4650a54

---

## Iteration 125 — 2026-03-17
- Phase: audit-only
- Classification: audit-only
- Target: trading_app/live/bar_aggregator.py
- Finding: No findings — file is entirely clean
- Action: Audit only, no changes
- Blast radius: 0
- Verification: PASS (8/8 tests)
- Commit: NONE

---

## Iteration 124 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: tests/test_trading_app/test_session_orchestrator.py:729,1015,1048,1074,1100
- Finding: 5 inline feed stub __init__ signatures missing on_stale=None — TypeError introduced when d4fe8cb added on_stale kwarg to BrokerFeed; test_run_starts_watchdog_task broken, 4 other reconnect stubs silently stale
- Action: Added on_stale=None to all 5 stubs (InstantFeed, MockFeed, CountingFeed, NeverReachFeed, CrashOnceFeed) — no production code touched
- Blast radius: 1 file (test only)
- Verification: PASS (87/87 tests pass, was 86+1 fail; drift unchanged — 14 pre-existing env violations in Check 16)
- Commit: fd37cbb

---

## Iteration 123 — 2026-03-16
- Phase: fix
- Classification: [judgment]
- Target: trading_app/live/projectx/data_feed.py:298
- Finding: _drain_bar_queue silent crash — unguarded await self.on_bar(bar) in infinite loop; any unhandled exception from on_bar kills the signalrcore drain task silently, halting bar delivery with no log or recovery; pysignalr path propagates to outer reconnect loop but signalrcore drain task has no equivalent recovery
- Action: Wrapped on_bar call in try/except — re-raise CancelledError for clean task shutdown, log+continue on all other exceptions; drain loop now survives individual bar processing failures
- Blast radius: 1 file changed; broker_factory.py imports only; test_projectx_feed.py tests drain queue
- Verification: PASS (20/20 tests pass, 1 deselected pre-existing pysignalr env issue; drift 72/72 PASS + 6 advisory)
- Commit: c6be6d9

---

## Iteration 123 — 2026-03-16
- Phase: fix
- Classification: [judgment]
- Target: trading_app/live/projectx/data_feed.py:298-302
- Finding: _drain_bar_queue silent crash — unguarded await self.on_bar(bar) in infinite drain loop; any unhandled exception from the callback kills the signalrcore bar-delivery task silently with no log or recovery path
- Action: Wrapped on_bar call in try/except; re-raises CancelledError for clean shutdown, logs+continues on all other exceptions
- Blast radius: 1 production file; broker_factory.py imports only; test_projectx_feed.py + test_broker_base.py verified
- Verification: PASS (20/20 tests; drift 72/72 PASS + 6 advisory; ruff clean)
- Commit: c6be6d9

---

## Iteration 122 — 2026-03-16
- Phase: fix
- Classification: [judgment]
- Target: trading_app/live/projectx/order_router.py:110
- Finding: cancel() fail-open — ProjectX /api/Order/cancel returns {"success": false} on HTTP 200 for rejected cancels; cancel() only called raise_for_status() and silently returned success, unlike submit() which already checks data.get("success")
- Action: Added data = resp.json() + success check after raise_for_status(); raises RuntimeError on success=False. Also fixed 3 pre-existing UP038 ruff lint errors (isinstance tuple -> union syntax) in gen_repo_map.py, orb_size_deep_dive.py, strategy_fitness.py.
- Blast radius: 1 production file + 3 lint files; session_orchestrator._cancel_brackets() already wraps cancel() in exception handler; 22 tests verified
- Verification: PASS (22/22 test_projectx_router.py; drift 72/72 checks PASS + 6 advisory)
- Commit: 4db63d2 (UP038 fixes + cancel fix already in d8e0f67)

---

## Iteration 121 — 2026-03-16
- Phase: fix
- Classification: [judgment]
- Target: trading_app/live/projectx/auth.py:84
- Finding: `except Exception:` in `_validate_or_login` broader than intended — swallows programming errors (AttributeError, TypeError) alongside network errors, masking real bugs as token refresh failures
- Action: Narrowed to `except requests.RequestException:` — only HTTP/network errors now trigger fallback to full login
- Blast radius: 1 file; 3 callers unaffected (broker_factory, order_router, fetch_broker_fills); 2 test files verified
- Verification: PASS (9 tests passed; tradovate failure pre-existing missing websockets package)
- Commit: 468ea1e

---

## Iteration 120 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/db_manager.py:verify_trading_app_schema (lines 602-644)
- Finding: verify_trading_app_schema expected_cols for experimental_strategies missing 9 migration-added columns (p_value, sharpe_ann_adj, autocorr_lag1, n_trials_at_discovery, fst_hurdle, median_risk_dollars, avg_risk_dollars, avg_win_dollars, avg_loss_dollars) — schema verification gate silently passes DBs missing production columns
- Action: Added 9 missing columns to experimental_strategies expected_cols verification set (11 lines inserted)
- Blast radius: 3 files (test_db_manager.py, test_app_sync.py, db_manager.py CLI)
- Verification: PASS (11 db_manager tests pass; pre-existing app_sync failure unrelated to change confirmed by stash test)
- Commit: 360ec68

---

## Iteration 119 — 2026-03-16
- Phase: audit-only
- Classification: audit-only
- Target: trading_app/live_config.py
- Finding: No actionable findings — file clean (4 ACCEPTABLE patterns documented)
- Action: Scan only; no fix applied
- Blast radius: N/A
- Verification: PASS (36 tests pass)
- Commit: NONE

---

## Iteration 118 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/portfolio.py:699
- Finding: Hardcoded `_COMPRESSION_SESSIONS = ["CME_REOPEN", "TOKYO_OPEN", "LONDON_METALS"]` inside loop body in `build_strategy_daily_series` duplicates canonical `COMPRESSION_SESSIONS` from `pipeline.build_daily_features` (canonical violation)
- Action: Added `from pipeline.build_daily_features import COMPRESSION_SESSIONS` to imports; removed inline list assignment; replaced `_COMPRESSION_SESSIONS` reference with `COMPRESSION_SESSIONS`
- Blast radius: 1 file
- Verification: PASS (68 tests, ruff clean, drift clean)
- Commit: 603542e

---

## Iteration 117 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/strategy_validator.py:824,1056
- Finding: SV-01 — `rd.get("entry_model", "E1")` hardcoded canonical fallback is unreachable dead code (schema enforces NOT NULL) but would silently assign wrong entry model if triggered; no annotation to document this
- Action: Replaced `.get("entry_model", "E1")` with `.get("entry_model") or "E1"` + inline comment at both sites documenting the NOT NULL schema constraint; same treatment for `filter_type` at line 824
- Blast radius: 1 file
- Verification: PASS — 49 tests, drift 72/72, ruff clean
- Commit: 0b9b466

---

## Iteration 116 — 2026-03-16
- Phase: fix
- Classification: [judgment]
- Target: trading_app/strategy_discovery.py:1082-1088 (high centrality, 8 importers)
- Finding SD-01: Fail-open pattern — `get_enabled_sessions()` returning empty triggered `logger.warning` + silent fallback to `ORB_LABELS` (all sessions). A misconfigured instrument would silently run discovery for all sessions instead of aborting. Same sin as OB-01 (iter 115).
- Action: Replaced `logger.warning` + `sessions = ORB_LABELS` fallback with `raise ValueError`. Removed now-unused `ORB_LABELS` import from `pipeline.init_db`.
- Blast radius: strategy_discovery.py only — change is local to `run_discovery()`. All callers pass valid instruments from config.
- Verification: 71 drift checks pass (5 env-only violations pre-existing), ruff clean, 45/45 strategy_discovery tests pass, 218 fast tests pass
- Commit: 6b9a3f6

---

## Iteration 115 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/outcome_builder.py (critical centrality, 11 importers)
- Finding OB-01: Fail-open pattern — `get_enabled_sessions()` returning empty triggered `logger.warning` + silent fallback to `ORB_LABELS` (all sessions). A misconfigured instrument would silently build outcomes for wrong sessions with no abort.
- Finding OB-02: Dead E3 branch in `cb_options` assignment (line ~833) — `em == "E3"` branch unreachable since E3 is in `SKIP_ENTRY_MODELS`. Missing TODO annotation for future re-enablement.
- Action OB-01: Changed `logger.warning` + `sessions = ORB_LABELS` fallback to `raise ValueError` (fail-closed). Removed unused `ORB_LABELS` import from `pipeline.init_db`.
- Action OB-02: Added `# TODO(E3-retired)` annotation explaining the branch is currently unreachable and when to remove it.
- Blast radius: outcome_builder.py only — both changes are local to `build_outcomes()`. All callers pass valid instruments (guarded by asset_configs).
- Verification: 72 drift checks pass, ruff clean, 27 tests pass
- Commit: d9c2609

---

## Iteration 114 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/ml/features.py:84
- Finding: `_backfill_global_features` used `GLOBAL_FEATURES[0]` ("atr_20") as sole proxy for post-backfill NaN count — if atr_20 was backfilled but overnight_range (#1 ML feature) was not, warning would silently not fire. Pre-backfill check (lines 56-59) correctly iterates ALL features; post-backfill count was asymmetric.
- Action: Replaced `df[GLOBAL_FEATURES[0]].isna().sum()` with `max(df[col].isna().sum() for col in GLOBAL_FEATURES if col in df.columns, default=0)` — consistent with pre-backfill check pattern
- Blast radius: 1 file (private function, 0 external callers)
- Verification: ruff PASS, 6/6 TestBackfillGlobalFeatures PASS; unrelated joblib env failure in other test class
- Commit: 2c7e419

---

## Iteration 113 — 2026-03-16
- Phase: diminishing-returns
- Classification: N/A
- Target: research/discover.py, research/research_wf_stress_keepers.py, research/research_trend_day_mfe.py
- Finding: No violations found — all 3 target files already pass ruff; consecutive_low_only=6 in ledger (threshold: 3)
- Action: DIMINISHING_RETURNS triggered — no fix applied. Re-scope recommended.
- Blast radius: 0 files
- Verification: drift 72/72 PASS, ruff PASS (targets already clean)
- Commit: NONE

---

## Iteration 110 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: research/research_overlap_analysis.py
- Finding: 27 F541 bare f-strings without placeholders + 1 I001 unsorted import block (ruff auto-fixable)
- Action: `ruff check --fix` applied — 27 fixes. Import block sorted. All print strings unchanged.
- Blast radius: 0 files (standalone research script, no callers)
- Verification: ruff clean, pyright 0 errors, drift 72/72 PASS
- Commit: adfa5cd

---

## Iteration 109 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: research/research_edge_structure.py + research/research_1015_vs_1000.py
- Finding: PE-01–PE-07: Pyright errors — (1) utcoffset() returns timedelta|None; .total_seconds() called directly without None guard at 3 sites; (2) pearsonr() return type inferred as tuple|float causing numpy ufunc mismatch at 2 sites; (3) unused `csv` import; (4) 4x `all_days` assigned but never used in function scope (q2/q3/q4/Q1 inner loop)
- Action: Added `offset = dt.utcoffset(); assert offset is not None` pattern at all 3 utcoffset() call sites; replaced `orb_r, orb_p = pearsonr(...)` with `_res = pearsonr(...); orb_r, orb_p = float(_res[0]), float(_res[1])`; removed `csv` import; renamed 4x `all_days` → `_all_days` at unpack sites where value was discarded
- Blast radius: 0 callers, 0 importers (standalone research scripts, no tests)
- Verification: pyright 0 errors 0 warnings, ruff check PASS (both files clean), check_drift.py PASS (72/72, 6 advisory)
- Commit: df4ead3

## Iteration 108 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: research/research_edge_structure.py + research/research_1015_vs_1000.py
- Finding: ES-01–ES-07: combined 65 ruff violations in edge_structure (37) and 28 in 1015_vs_1000 — F541 extraneous f-strings, F841 unused variables (drop, n, window_mins), B007/B023 loop variable binding in closures (band_stats, bar_stats), E702 semicolon statements, I001 import sort
- Action: ruff --fix applied 44 auto-fixes in edge_structure + 37 in 1015_vs_1000; manual fixes: removed unused n/window_mins assignments, split 6 E702 semicolons to two-line form, refactored bar_stats() out of loop with explicit params (volumes/highs/lows/closes/opens/m/start_1000), added regime_days param to band_stats(); batched both files (all LOW, same type, 0 callers)
- Blast radius: 0 callers, 0 importers (standalone research scripts)
- Verification: ruff check PASS (both files clean), check_drift.py PASS (72/72, 6 advisory), audit_behavioral.py PASS (6/6)
- Commit: 150618a

## Iteration 107 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: research/research_vol_regime_switching.py:192,784
- Finding: (1) VS-01–VS-04: 45 ruff violations (import sort, 4x unused loop vars, 40x extraneous f-string prefixes, unused os import) — partially applied in interrupted prior session, completed here; (2) VS-05: second hardcoded `IN ('E1','E2')` in load_data() SQL — not using ENTRY_MODELS variable; (3) VS-04: unused datetime import + hardcoded static date '2026-03-01' in output
- Action: fixed load_data() SQL to use ENTRY_MODELS f-string expansion (matching get_validated_sessions fix); fixed markdown output date to use datetime.date.today().isoformat(); all ruff violations confirmed clean
- Blast radius: 1 file, 0 production callers
- Verification: ruff check PASS, check_drift.py PASS (72/72, 6 advisory)
- Commit: 9c14df0

## Iteration 106 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: research/research_zt_event_viability.py:358
- Finding: Hardcoded variation count "18" in report template — should be computed dynamically from EVENT_FAMILIES structure
- Action: Added `variation_count = sum(len(family.follow_windows) * 2 for family in EVENT_FAMILIES.values())` and used f-string interpolation in report line
- Blast radius: 1 file (standalone research script, zero importers)
- Verification: PASS (72 drift checks, ruff clean)
- Commit: 18de696

---

## Iteration 102 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: research/research_mgc_asian_fade_mfe.py:112-118, research/research_post_break_pullback.py:21-25,286
- Finding: Ruff violations in two new research files — (1) unused `bar_mfe`/`bar_mae` scaffolding assignments (F841) in mgc_asian_fade_mfe.py; (2) unsorted import block (I001) and unused loop vars `t`/`pct` should be `_t`/`_pct` (B007) in post_break_pullback.py. research_zt_fomc_unwind.py clean.
- Action: Removed 4 dead lines from mgc_asian_fade_mfe.py; fixed import sort and renamed `t`→`_t`, `pct`→`_pct` in post_break_pullback.py
- Blast radius: 2 files (standalone untracked research scripts, zero importers)
- Verification: PASS (72 drift checks, 6/6 behavioral, ruff clean)
- Commit: 35c3b46

---

## Iteration 101 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: research/research_mes_compressed_spring.py:42
- Finding: Used os.environ.get("DUCKDB_PATH", ...) instead of canonical GOLD_DB_PATH from pipeline.paths — inconsistent with all other research scripts
- Action: Removed `import os`, added `from pipeline.paths import GOLD_DB_PATH`, replaced DB_PATH assignment with `str(GOLD_DB_PATH)`
- Blast radius: 1 file (standalone research script, zero importers)
- Verification: PASS (72 drift checks, module imports cleanly)
- Commit: f9618c0

---

## Iteration 100 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: scripts/tools/refresh_data.py, pipeline/asset_configs.py, tests/test_pipeline/test_asset_configs.py
- Finding: Uncommitted changes adding orb_active field to ASSET_CONFIGS and 2YY/ZT research-only instrument support in refresh_data.py — correctly gated via cfg.get("orb_active", True)
- Action: Verified clean (7 sins scan: no violations), ran tests (8/8 pass), committed cohesive change set
- Blast radius: 1 production file (refresh_data.py), 1 config (asset_configs.py), 1 test
- Verification: PASS
- Commit: e98dba4

## Iteration 99 — 2026-03-16
- Phase: audit-only
- Classification: N/A (no fix)
- Target: scripts/tools/m25_nightly.py, m25_audit.py, m25_auto_audit.py, m25_preflight.py, m25_run_grounded_system.py
- Finding: All 5 files clean. M25 suite is well-designed advisory tooling. Exception handlers in budget tracker and context-enrichment helpers are advisory-path patterns, not production safety failures. Hardcoded OneDrive path in m25_run_grounded_system.py is a local research script with graceful skip on missing files. Fail-open in m25_preflight.py is intentional (M2.5 is advisory only per .claude/rules/m25-audit.md).
- Action: Audit only — no fixes required.
- Blast radius: N/A
- Verification: PASS (72 drift checks, behavioral audit 6/6, ruff clean)
- Commit: NONE

---

## Iteration 98 — 2026-03-16
- Phase: audit-only
- Classification: N/A (no fix)
- Target: scripts/tools/parity_check.py, build_outcomes_fast.py, build_mes_outcomes_fast.py, prospective_tracker.py, profile_1000_runners.py
- Finding: All 5 files clean. Hardcoded parameters are intentional research-scope configuration (read-only scripts or fast-path MES-specific builders). Connection leaks in write scripts match existing ~22 CLI script pattern (process exit closes). No safety impact.
- Action: Audit only — no fixes required.
- Blast radius: N/A
- Verification: PASS (72 drift checks, behavioral audit 6/6, ruff clean)
- Commit: NONE

---

## Iteration 97 — 2026-03-16
- Phase: audit-only
- Classification: N/A (no fix)
- Target: scripts/tools/backtest_1100_early_exit.py, backtest_atr_regime.py, beginner_tradebook.py, find_pf_strategy.py, rank_slots.py
- Finding: All 5 files clean. Hardcoded parameters in all files are intentional research-scope configuration (read-only scripts, not canonical lists). No safety impact.
- Action: Audit only — no fixes required.
- Blast radius: N/A
- Verification: PASS (72 drift checks, behavioral audit 6/6, ruff clean)
- Commit: NONE

---

## Iteration 96 — 2026-03-16
- Phase: audit-only
- Classification: N/A (no fix)
- Target: scripts/tools/ml_per_session_experiment.py, ml_level_proximity_experiment.py, ml_threshold_sweep.py, ml_session_leakage_audit.py, ml_license_diagnostic.py, audit_15m30m.py
- Finding: All 5 primary targets clean. 1 ACCEPTABLE finding (GP-96a: hardcoded instrument list in read-only diagnostic SQL in audit_15m30m.py — no safety impact).
- Action: Audit only — no fixes required. SESSION_ORDER complete across all ML scripts.
- Blast radius: N/A
- Verification: PASS (72 drift checks, behavioral audit 6/6, ruff clean)
- Commit: NONE

---

## Iteration 95 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: scripts/tools/gen_playbook.py:27 (SESSION_ORDER)
- Finding: GP-94 — EUROPE_FLOW session missing from SESSION_ORDER. SESSION_CATALOG has 12 sessions but SESSION_ORDER only listed 11 (gap between SINGAPORE_OPEN and LONDON_METALS). Any EUROPE_FLOW validated strategies would be silently omitted from MARKET_PLAYBOOK.md.
- Action: Added EUROPE_FLOW tuple ("EUROPE_FLOW", "17:00", "18:00", "London open 7:00/9:00 AM London") to SESSION_ORDER between SINGAPORE_OPEN and LONDON_METALS. Coverage now matches SESSION_CATALOG exactly.
- Blast radius: 1 file (gen_playbook.py only — standalone script, no importers)
- Verification: PASS (72 drift checks, behavioral audit 6/6, ruff clean, SESSION_ORDER == SESSION_CATALOG)
- Commit: 69ac9ac

---

## Iteration 88 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: scripts/run_live_session.py:50
- Finding: RLS-01 — `checks_total = 5` hardcoded in `_run_preflight()`. Controls pass/fail logic (`checks_passed == checks_total`). If a 6th check is added without updating this, preflight returns False even when all checks pass, silently blocking live sessions.
- Action: Added inline comment making the constraint explicit: `# NOTE: must match number of check blocks (1-5) below — update if adding/removing checks`. operator_status.py audited as clean (no findings).
- Blast radius: 1 file (run_live_session.py)
- Verification: PASS (72 drift checks, behavioral audit 6/6, ruff clean)
- Commit: c57130b

---

## Iteration 87 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: scripts/reports/parameter_stability_heatmap.py:43
- Finding: PSH-01 — `APERTURES = [5, 15, 30]` hardcoded. Should use `VALID_ORB_MINUTES` from `pipeline.build_daily_features`. If a 4th aperture were added, neighbor discovery and matrix rendering would silently miss it.
- Action: Imported `VALID_ORB_MINUTES`; replaced `APERTURES = [5, 15, 30]` with `APERTURES = VALID_ORB_MINUTES`.
- Blast radius: 1 file (parameter_stability_heatmap.py)
- Verification: PASS (19/19 tests, 72 drift checks)
- Commit: 55fe2ec

---

## Iteration 86 — 2026-03-15
- Phase: audit-only
- Target: Deep sweep — coaching_digest.py, trading_coach.py, rolling_eval.py, rolling_eval_parallel.py, run_parallel_ingest.py, scratch_ingest.py + pattern sweeps
- Finding: CLEAN — no actionable findings. All major canonical violation patterns eliminated from production code. Remaining items are LOW severity (CLI connection leaks, experiment scripts).
- Action: No code changes. Codebase at steady state for Seven Sins.
- Commit: NONE

---

## Iteration 85 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: report_edge_portfolio.py (FIX) + instrument list sweep
- Finding: REP-01 — `for inst in ["MGC", "MNQ", "MES", "M2K"]` in `--all` loop. Should use ACTIVE_ORB_INSTRUMENTS. If a 5th instrument is added, report would miss it.
- Action: Imported ACTIVE_ORB_INSTRUMENTS; replaced hardcoded list. Ruff auto-fixed import ordering.
- Blast radius: 1 file (report_edge_portfolio.py)
- Verification: PASS (72 drift checks, ruff clean except pre-existing B905)

---

## Iteration 84 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: Last 3 production files with `PROJECT_ROOT / "gold.db"` pattern
- Finding: VSA-01 + OSD-01 + IMN-01 — volume_session_analysis.py, orb_size_deep_dive.py, ingest_mnq.py all had hardcoded DB path + duplicated env var logic.
- Action: Imported GOLD_DB_PATH in all 3; simplified get_db_path() functions; removed unused os imports. Pattern now **fully eliminated** from production code (only scripts/archive/ remains).
- Blast radius: 3 files
- Verification: PASS (72 drift checks, ruff clean, grep confirms only archive/ remains)

---

## Iteration 83 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: stress_test.py (FIX)
- Finding: ST-01 — Same as EX-01/HT-01: `DEFAULT_DB = PROJECT_ROOT / "gold.db"` + duplicated env var logic. Swept remaining: 3 production files left (volume_session_analysis.py, orb_size_deep_dive.py, ingest_mnq.py).
- Action: Imported GOLD_DB_PATH; simplified get_db_path(). Removed unused os import.
- Blast radius: 1 file (stress_test.py)
- Verification: PASS (72 drift checks, ruff clean)

---

## Iteration 82 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: explore.py (FIX) + backfill_garch.py (CLEAN) + stress_test.py (noted)
- Finding: EX-01 — `DEFAULT_DB = PROJECT_ROOT / "gold.db"` + reimplemented env var logic in `get_db_path()`. Should use GOLD_DB_PATH.
- Action: Added sys.path.insert, imported GOLD_DB_PATH, simplified get_db_path() to CLI override → GOLD_DB_PATH fallback.
- Blast radius: 1 file (explore.py)
- Verification: PASS (72 drift checks, ruff clean)

---

## Iteration 81 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: hypothesis_test.py (FIX) + explore.py (noted) + detect_volume_spikes.py (CLEAN)
- Finding: HT-01 — `hypothesis_test.py:48-57` reimplements `pipeline.paths._resolve_db_path()` logic (check CLI, check env var, fallback to `PROJECT_ROOT / "gold.db"`). Should import `GOLD_DB_PATH` which already handles all this.
- Action: Imported GOLD_DB_PATH; simplified `get_db_path()` to CLI override → GOLD_DB_PATH fallback. Removed unused `os` import.
- Blast radius: 1 file (hypothesis_test.py)
- Verification: PASS (72 drift checks, ruff clean)

---

## Iteration 80 — 2026-03-15
- Phase: fix
- Classification: [judgment]
- Target: generate_promotion_candidates.py (CLEAN) + prospective_tracker.py (CLEAN) + rolling_portfolio_assembly.py (CLEAN) + build_mes_outcomes_fast.py (FIX)
- Finding: BMOF-01 — `build_mes_outcomes_fast.py` had 3 canonical violations: (1) `DB_PATH = Path(r"C:\db\mes.db")` pointing to non-existent DB, (2) hardcoded `RR_TARGETS`, (3) hardcoded `CONFIRM_BARS_OPTIONS`. All should import from canonical sources.
- Action: Imported GOLD_DB_PATH, RR_TARGETS, CONFIRM_BARS_OPTIONS from canonical sources. Removed unused Path import.
- Blast radius: 1 file (build_mes_outcomes_fast.py)
- Verification: PASS (72 drift checks, ruff clean)

---

## Iteration 79 — 2026-03-15
- Phase: fix
- Classification: [judgment]
- Target: Drift check #62 regex hardening + downstream fixes
- Finding: DC62-01 — Drift check #62 regex only matched forward-slash paths (`C:/db/gold.db`), missing backslash (`C:\db\gold.db`) and escaped (`C:\\db\\gold.db`). Fixed regex with `[/\\]{1,2}` separator. Improved check immediately caught 2 hidden violations in `ingest_mes.py:37` and `ingest_mnq_fast.py:45`.
- Action: (1) Fixed regex in check_drift.py, (2) Fixed ingest_mes.py DB_PATH, (3) Fixed ingest_mnq_fast.py DB_PATH + removed unused Path import.
- Blast radius: 3 files (check_drift.py, ingest_mes.py, ingest_mnq_fast.py)
- Verification: PASS (72 drift checks — including the now-stronger #62, ruff clean)

---

## Iteration 78 — 2026-03-15
- Phase: fix
- Classification: [judgment]
- Target: audit_integrity.py (CLEAN) + parallel_rebuild.py (CLEAN) + build_outcomes_fast.py (FIX)
- Finding: BOF-01 — `DB_PATH = Path(r"C:\db\gold.db")` hardcoded scratch path. Evaded drift check #62 because regex only matches forward slashes. Fixed to use `GOLD_DB_PATH`.
- Action: Imported GOLD_DB_PATH; replaced hardcoded path. Noted drift check #62 regex gap for future fix.
- Blast radius: 1 file (build_outcomes_fast.py)
- Verification: PASS (72 drift checks, ruff clean)

---

## Iteration 77 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: Codebase sweep for hardcoded `"gold.db"` paths
- Finding: A15-01 — `scripts/tools/audit_15m30m.py:9` uses `duckdb.connect("gold.db")` instead of GOLD_DB_PATH. Last production file with this violation (20 remaining are `scripts/tmp_*` — acceptable).
- Action: Added sys.path setup, imported GOLD_DB_PATH, replaced hardcoded path.
- Blast radius: 1 file (audit_15m30m.py)
- Verification: PASS (72 drift checks, ruff clean)

---

## Iteration 76 — 2026-03-15
- Phase: fix
- Classification: [judgment]
- Target: ML subpackage completion (5 files CLEAN) + scripts/tools/rr_selection_analysis.py (FIX)
- Finding: RR-01 — rr_selection_analysis.py had 3 sins: (1) hardcoded `"gold.db"` path instead of GOLD_DB_PATH, (2) hardcoded `entry_model IN ('E1','E2')` instead of ENTRY_MODELS - SKIP_ENTRY_MODELS, (3) connection leak (no try/finally). Fixed all 3.
- Action: Added sys.path setup, imported GOLD_DB_PATH + ENTRY_MODELS + SKIP_ENTRY_MODELS, wrapped in try/finally.
- Blast radius: 1 file (rr_selection_analysis.py)
- Verification: PASS (72 drift checks, ruff clean)

---

## Iteration 75 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/ml/ subpackage — config.py (CLEAN), meta_label.py (FIX), features.py (CLEAN)
- Finding: ML-01 — `meta_label.py:1103` connection leak. `_con.close()` not in `finally` block during `--sweep-rr` query path. Same pattern as HC-01.
- Action: Wrapped query in try/finally block.
- Blast radius: 1 file (meta_label.py), CLI sweep path only
- Verification: PASS (72 drift checks, ruff clean)

---

## Iteration 74 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: sensitivity_analysis.py (FIX) + report_edge_portfolio.py (CLEAN) + gen_playbook.py (CLEAN, noted gap)
- Finding: SA-01 — `orb_minutes_list = [5, 15, 30]` in sensitivity_analysis.py default param. 6th instance of same canonical violation.
- Action: Imported VALID_ORB_MINUTES; replaced hardcoded default.
- Blast radius: 1 file (sensitivity_analysis.py), function default only
- Verification: PASS (72 drift checks, ruff clean)

---

## Iteration 73 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: gen_repo_map.py (CLEAN) + sync_pinecone.py (CLEAN) + retire_e3_strategies.py (CLEAN) + refresh_data.py (FIX)
- Finding: RD-01 — `ORB_APERTURES = [5, 15, 30]` in refresh_data.py. Same canonical violation (5th instance across codebase). Also swept all remaining `[5, 15, 30]` instances — remaining are research/tests/tmp (acceptable).
- Action: Imported VALID_ORB_MINUTES; replaced hardcoded list.
- Blast radius: 1 file (refresh_data.py)
- Verification: PASS (72 drift checks, ruff clean)

---

## Iteration 72 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: scripts/tools/assert_rebuild.py + scripts/infra/backup_db.py
- Finding: AR-01 — `APERTURES = [5, 15, 30]` hardcoded (same pattern as PS-01/CD-01/RP-01). backup_db.py CLEAN.
- Action: Imported VALID_ORB_MINUTES; replaced hardcoded list.
- Blast radius: 1 file (assert_rebuild.py), assertion A5 only
- Verification: PASS (72 drift checks, ruff clean, import smoke test)

---

## Iteration 71 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: pipeline/paths.py + pipeline/log.py + pipeline/check_db.py + scripts/tools/select_family_rr.py (full scan, 5 files)
- Finding: SFR-01 — `select_family_rr.py:148` hardcodes `entry_model IN ('E1', 'E2')`. Should derive from `ENTRY_MODELS - SKIP_ENTRY_MODELS`. paths.py, log.py, check_db.py all CLEAN.
- Action: Imported ENTRY_MODELS and SKIP_ENTRY_MODELS; built active models list and SQL IN clause dynamically.
- Blast radius: 1 file (select_family_rr.py), query filter only
- Verification: PASS (72 drift checks, ruff clean, import smoke test)

---

## Iteration 70 — 2026-03-15
- Phase: audit-only
- Target: pipeline/dashboard.py + pipeline/db_lock.py + pipeline/audit_log.py
- Finding: CLEAN — no actionable findings across all 3 files
- Action: Seven Sins scan complete, 0 findings. dashboard.py: proper connection management. db_lock.py: atomic lock with O_EXCL. audit_log.py: intentional fail-open (documented).
- Blast radius: N/A (no changes)
- Verification: PASS (72 drift checks)
- Commit: NONE

---

## Iteration 69 — 2026-03-15
- Phase: fix
- Classification: [judgment]
- Target: pipeline/init_db.py (full Seven Sins scan)
- Finding: ID-01 — `--force` mode uses hardcoded 8-table drop list, missing 5 tables (edge_families, regime_strategies, regime_validated, strategy_trade_days, validation_run_log). Users expecting clean slate get orphaned data.
- Action: Replaced hardcoded list with dynamic `information_schema.tables` query — drops ALL user tables, no maintenance burden.
- Blast radius: 1 file (init_db.py), `--force` path only
- Verification: PASS (10 init_db tests, 72 drift checks, ruff clean, smoke test)

---

## Iteration 68 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: pipeline/run_pipeline.py (full Seven Sins scan)
- Finding: RP-01 — `--orb-minutes` CLI arg uses `choices=[5, 15, 30]` instead of `choices=VALID_ORB_MINUTES`. Same pattern in `build_daily_features.py` already uses canonical source correctly.
- Action: Imported `VALID_ORB_MINUTES`; replaced hardcoded list.
- Blast radius: 1 file (run_pipeline.py), CLI arg validation only
- Verification: PASS (72 drift checks, ruff clean, import smoke test)

---

## Iteration 67 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: pipeline/health_check.py (full Seven Sins scan)
- Finding: HC-01 — `check_staleness()` connection leak. `con.close()` not in `finally` block; if `staleness_engine()` raises, connection stays open. Violates project's consistent try/finally pattern.
- Action: Wrapped connection usage in try/finally block, matching `check_database()` pattern.
- Blast radius: 1 file (health_check.py), 1 function
- Verification: PASS (22 health tests, 72 drift checks, ruff clean)

---

## Iteration 66 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: pipeline/check_drift.py (scan completed, no new fix) + scripts/tools/audit_behavioral.py (full scan)
- Finding: AB-01 — `_INST` regex hardcodes 7/8 instrument symbols, missing MBT. Built from `ASSET_CONFIGS` canonical source instead.
- Action: Added `sys.path.insert`, imported `ASSET_CONFIGS`, built `_INST` regex dynamically from all stored symbols. Also completed check_drift.py full scan (orphan/dead code CLEAN, volatile data CLEAN).
- Blast radius: 1 file (audit_behavioral.py), internal regex pattern
- Verification: PASS (23 audit tests, 72 drift checks, 6 behavioral checks, ruff clean)

---

## Iteration 65 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: pipeline/check_drift.py (partial Seven Sins scan — 3539 lines)
- Finding: CD-01 — `check_daily_features_row_integrity` hardcodes `HAVING COUNT(*) != 3` where `3` = number of ORB apertures. Should derive from `len(VALID_ORB_MINUTES)`.
- Action: Imported `VALID_ORB_MINUTES` inside function; replaced hardcoded `3` with `len(VALID_ORB_MINUTES)` in SQL, error message, docstring, and check registry label.
- Blast radius: 1 file (check_drift.py), self-contained check function
- Verification: PASS (63 tests, drift 72/72, ruff clean)

---

## Iteration 64 — 2026-03-15
- Phase: fix
- Classification: [judgment]
- Target: scripts/tools/pipeline_status.py (full Seven Sins scan completion)
- Finding: PS-03 — 4/5 PREFLIGHT_RULES keys unreachable. Keys ("strategy_discovery", "strategy_validator", "build_edge_families", "select_family_rr") don't match step base names ("discovery", "validator", "edge_families", "family_rr_locks") from _parse_step_preflight(). Lookup silently returns (True, "No pre-flight rule..."), skipping actual checks.
- Action: Renamed 4 PREFLIGHT_RULES keys to match step base names. Full Seven Sins scan now complete for pipeline_status.py (3 fixes across iters 62-64).
- Blast radius: 1 file (pipeline_status.py, internal-only key lookup)
- Verification: PASS (31 tests, drift 72/72, ruff clean)

---

## Iteration 61 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/prop_portfolio.py:16, trading_app/prop_profiles.py:13
- Finding: I001 import block un-sorted — extra blank lines after import block causing ruff I001 errors
- Action: ruff --fix applied; removed 1 blank line in each file (2 lines total)
- Blast radius: 2 test files (test_prop_portfolio.py, test_prop_profiles.py), no pipeline callers
- Verification: 35 tests PASS, drift 72/72 CLEAN
- Commit: af5ff0b

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

## Iteration 89 — 2026-03-15
- Phase: audit-only
- Classification: N/A
- Target: scripts/tools/generate_promotion_candidates.py, scripts/tools/select_family_rr.py, scripts/tools/audit_behavioral.py
- Finding: No actionable findings — all three files clean
- Action: Audit only, no changes made
- Blast radius: 0 files
- Verification: PASS (infrastructure gates 3/3)
- Commit: NONE

## Iteration 90 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: scripts/tools/build_edge_families.py:217-220
- Finding: Dead variable `orb_minutes_map` — built but never read; orphan code from refactor
- Action: Removed 2 lines (dict initialization + single assignment). No behavior change.
- Blast radius: 1 file, 0 callers
- Verification: 22/22 tests passed, drift 72/72 PASS
- Commit: e529f42

Also audited: scripts/tools/pipeline_status.py — CLEAN (no findings)

---

## Iteration 92 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: scripts/tools/pinecone_snapshots.py:57-62
- Finding: PS-92 — hardcoded classification thresholds 100/30 in portfolio state SQL query; should reference CORE_MIN_SAMPLES/REGIME_MIN_SAMPLES from trading_app.config
- Action: Added import of CORE_MIN_SAMPLES, REGIME_MIN_SAMPLES from trading_app.config; converted triple-quoted SQL string to f-string; replaced 3 literals with canonical constants. 4 lines changed.
- Blast radius: 2 files (sync_pinecone.py imports generator functions, test_pinecone_snapshots.py)
- Verification: 4/4 tests passed, drift 72/72 PASS
- Commit: d2f582a

Also audited: rolling_portfolio_assembly.py (clean), generate_trade_sheet.py (clean — E2-only HTML note acceptable as display-only)

---

## Iteration 91 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: scripts/tools/build_edge_families.py:218
- Finding: BEF-02 — unused loop variable `orb_min` in first strategies loop, introduced by iter 90 fix (ruff B007)
- Action: Renamed `orb_min` to `_orb_min` on line 218 (1 line). Second loop at line 240 which uses `orb_min` for family_key was untouched. Also audited assert_rebuild.py, gen_repo_map.py, sync_pinecone.py — all clean.
- Blast radius: 1 file
- Verification: PASS (ruff clean, 22/22 test_edge_families tests pass, 72 drift checks pass)
- Commit: 5d576c4

---

## Iteration 92 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: scripts/tools/pinecone_snapshots.py:57-62
- Finding: Canonical violation — hardcoded classification thresholds 100 and 30 in SQL query; should reference CORE_MIN_SAMPLES/REGIME_MIN_SAMPLES from trading_app.config
- Action: Added import of CORE_MIN_SAMPLES, REGIME_MIN_SAMPLES from trading_app.config; replaced 3 literal values with f-string interpolation (4 lines total). Also audited rolling_portfolio_assembly.py and generate_trade_sheet.py — both clean.
- Blast radius: 2 files (sync_pinecone.py imports generators, test_pinecone_snapshots.py)
- Verification: 4/4 test_pinecone_snapshots tests PASS, 72 drift checks PASS
- Commit: d2f582a

## Iteration 93 — 2026-03-15
- Phase: fix
- Classification: [mechanical]
- Target: scripts/tools/sensitivity_analysis.py:40
- Finding: RR_STEPS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0] hardcoded — duplicates canonical RR_TARGETS from trading_app/outcome_builder.py
- Action: Added `from trading_app.outcome_builder import RR_TARGETS` and replaced literal list assignment with `RR_STEPS = RR_TARGETS`
- Blast radius: 1 file (standalone CLI script, no importers)
- Verification: PASS (import verified, ruff clean, 72/72 drift checks pass)
- Commit: b7804ef

---

## Iteration 95 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: scripts/tools/ml_cross_session_experiment.py:26, scripts/tools/ml_hybrid_experiment.py:25, scripts/tools/ml_instrument_deep_dive.py:24
- Finding: GP-95 — EUROPE_FLOW missing from SESSION_ORDER in all 3 ML experiment scripts. SESSION_CATALOG has 12 sessions; all three SESSION_ORDER lists had 11 (missing EUROPE_FLOW between SINGAPORE_OPEN and LONDON_METALS). EUROPE_FLOW trades silently skipped in cross-session feature computation via `if session not in SESSION_ORDER: continue` guard.
- Action: Added "EUROPE_FLOW" entry between "SINGAPORE_OPEN" and "LONDON_METALS" in SESSION_ORDER of all three files (3 lines, one per file). Same fix type as GP-94.
- Blast radius: 3 files (standalone experiment scripts, no production callers)
- Verification: PASS (72 drift checks pass, ruff clean, behavioral audit 6/6)
- Commit: 3057e0c

## Iteration 103 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: research/research_zt_cpi_nfp.py:101
- Finding: zip() without strict= parameter in infer_tick_size consecutive-diff loop (ruff B905)
- Action: Added strict=False to zip(uniq, uniq[1:]) — lists are deliberately unequal length, strict=False makes intent explicit
- Blast radius: 1 file, 0 external callers
- Verification: ruff PASS, check_drift 72/72 PASS
- Commit: 8f2c05b

## Iteration 104 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: research/research_mgc_mnq_correlation.py:158,309,433
- Finding: RC-01 (F541) — extraneous f-prefix on string with no placeholders (line 158). RC-02 (B905) — zip() without strict= in two BH FDR result-merging loops (lines 309, 433).
- Action: Removed f-prefix from line 158 print string; added strict=False to both zip() calls (bh_fdr always returns same-length list as input, so strict=False is correct)
- Blast radius: 1 file, 0 external callers (standalone research script)
- Verification: ruff PASS, check_drift 72/72 PASS
- Commit: 06436e6

## Iteration 105 — 2026-03-16
- Phase: fix
- Classification: [judgment]
- Target: research/research_atr_velocity_gate.py:87-101, research/research_mgc_regime_shift.py:170-210
- Finding: AV-01 (MEDIUM) — Part 0 COUNT query missing `AND o.orb_minutes = 5`, mixing 5m+15m+30m apertures and inflating removal rate stats ~3x. AV-02 (LOW) — `fetchone()` result not guarded for None before tuple destructure (Pyright reportOptionalMemberAccess). RS-01 (MEDIUM) — Parts 4+5 of regime_shift year-by-year and pre/post queries missing `AND o.orb_minutes = 5`, mixing apertures.
- Action: Added `AND o.orb_minutes = 5` to Part 0 query (AV-01); added `if row is None: continue` guard before destructure (AV-02); added `AND o.orb_minutes = 5` to Parts 4 and 5 queries in regime_shift (RS-01)
- Blast radius: 2 files, 0 external callers (standalone research scripts)
- Verification: ruff PASS, check_drift 72/72 PASS
- Commit: 7a09129

## Iteration 111 — 2026-03-16
- Phase: fix
- Classification: [mechanical]
- Target: research/research_aperture_scan.py
- Finding: AS-01/AS-02 (LOW) — 17 F541 bare f-strings without placeholders in print_honest_summary() and main(); 1 I001 unsorted import block. research_session_stats.py does not exist (skip).
- Action: ruff check --fix — removed extraneous f-prefix from 17 print strings; sorted import block. No logic or behaviour change.
- Blast radius: 1 file, 0 external callers (standalone research script)
- Verification: ruff PASS, check_drift 72/72 PASS
- Commit: 16d472f

## Iteration 125 — 2026-03-17
- Phase: audit-only
- Classification: N/A
- Target: trading_app/live/bar_aggregator.py
- Finding: No findings — file entirely clean (0 Seven Sins violations, 8/8 tests PASS)
- Action: Audit only — no code changes
- Blast radius: 5 importers (broker_base.py, live_market_state.py, projectx/data_feed.py, tradovate/data_feed.py, session_orchestrator.py)
- Verification: PASS (8/8 bar_aggregator tests, 72 drift checks, behavioral audit clean)
- Commit: NONE

## Iteration 127 — 2026-03-17
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/live/broker_factory.py:89-90
- Finding: VALID_BROKERS declared as "canonical source for dispatcher" but only used in ValueError message string — actual dispatch guard was if/elif chain, allowing coherence drift
- Action: Added early guard `if broker not in VALID_BROKERS: raise ValueError(...)` before if/elif; removed redundant else clause; kept `raise AssertionError("unreachable")` as defensive guard
- Blast radius: 1 production file, 1 test file
- Verification: PASS (9/9 tests, 72 drift checks, 218 pre-commit suite)
- Commit: 2f039df

## Iteration 132 — 2026-03-18
- Phase: fix
- Classification: [judgment]
- Target: trading_app/walkforward.py:134-136
- Finding: Silent failure — tight stop silently skipped (no warning) when stop_multiplier != 1.0 but cost_spec is None; current caller (strategy_validator) handles this correctly but future callers could silently misconfigure
- Action: Added elif branch with logger.warning() to emit visible warning when stop is requested but cost_spec missing
- Blast radius: 1 production file, 1 test file (test_walkforward.py — all 32 pass unchanged)
- Verification: PASS (32/32 walkforward tests, 72 drift checks, 218 pre-commit suite, ruff clean)
- Commit: 1b2ac93

## Iteration 133 — 2026-03-18
- Phase: audit-only
- Classification: audit-only
- Target: pipeline/calendar_filters.py
- Finding: No actionable findings. Two LOW/ACCEPTABLE: (1) is_month_end/is_month_start/is_quarter_end are dormant exports with documented re-enablement path in calendar_overlay.py; (2) CPI list stale since 2026-03-12 and FOMC list missing 2026-05+ but both guarded by empty CALENDAR_RULES (0 rules loaded — calendar cascade research found zero BH FDR survivors).
- Action: Audit only. All findings ACCEPTABLE. No code changes.
- Blast radius: 3 callers (build_daily_features.py, calendar_overlay.py, session_orchestrator.py), 1 test file
- Verification: PASS (58/58 test_calendar_filters.py, 72 drift checks, behavioral clean, ruff clean)
- Commit: NONE

## Iteration 134 — 2026-03-18
- Phase: fix
- Classification: [mechanical]
- Target: pipeline/stats.py:3-5
- Finding: Stale module docstring listed meta_label.py as an importer of pipeline.stats, but meta_label.py does not import it. Real callers: evaluate.py, evaluate_validated.py, select_family_rr.py.
- Action: Corrected docstring to list all three actual callers by explicit path. No behavior change.
- Blast radius: 0 production callers affected (docstring-only change)
- Verification: PASS (13/13 test_rr_selection.py, 72 drift checks, 733 pre-commit suite, ruff clean)
- Commit: a514218

## Iteration 135 — 2026-04-01
- Phase: fix
- Classification: [judgment]
- Target: scripts/databento_backfill.py:133-139 + scripts/tools/refresh_data.py:248-249
- Finding: Silent failure — (1) load_manifest() had no JSONDecodeError guard; a corrupt manifest (from crash during save_manifest) would propagate an unhandled exception crashing the entire run_download() call; (2) cleanup unlink exception in refresh_data.py was silently swallowed with `except Exception: pass`, leaving orphaned partial download files with no user visibility.
- Action: (1) Added json.JSONDecodeError catch in load_manifest() with warning log, falls back to fresh empty manifest. (2) Changed except Exception: pass to log WARNING with file name and exception text. Fail-closed behavior of refresh_instrument (still returns False) unchanged.
- Blast radius: 0 external callers (both standalone CLI tools); 1 internal caller each within same file
- Verification: PASS (ruff clean, 67 drift checks pass, pre-commit suite pass)
- Commit: 7e70c22

## Iteration 136 — 2026-04-04
- Phase: fix
- Classification: [judgment]
- Target: pipeline/ingest_dbn_mgc.py:126
- Finding: Silent crash — CheckpointManager._load_checkpoints() calls json.loads() with no JSONDecodeError handler. A corrupt JSONL checkpoint line (from process kill during write_checkpoint) propagates an unhandled exception that crashes the entire ingest run with no recovery message, even when prior checkpoints are valid and the run could resume.
- Action: Wrapped json.loads() in try/except json.JSONDecodeError; corrupt lines are skipped with a stderr warning. Valid lines before and after the corrupt entry continue to load correctly. Added regression test test_corrupt_checkpoint_line_is_skipped to test_ingest_daily.py.
- Blast radius: 1 file changed; 3 production callers (ingest_dbn.py, ingest_dbn_daily.py, ingest_dbn_mgc.py main()), 1 test file
- Verification: PASS (18/18 test_ingest_daily.py, 77 drift checks pass, pre-commit suite pass)
- Commit: 4089b29

---

## Iteration 137 — 2026-04-04
- Phase: fix
- Classification: [judgment]
- Target: pipeline/ingest_dbn_daily.py:379
- Finding: Fail-open exception handler in per-file loop — outer try/except caught all file processing errors and called `continue`, allowing the loop to proceed and flush already-buffered data to DB even after a file failed. The end-of-loop `files_failed` check was too late: DB commits for subsequent files could already have been written, leaving silently incomplete data in bars_1m.
- Action: Changed `except` block from `stats["files_failed"] += 1; continue` to `traceback.print_exc(); sys.exit(1)`. Added `import traceback`. Removed now-dead `files_failed` counter (key from stats dict, Files failed log line, and end-of-loop files_failed > 0 check). Added `FATAL:` prefix to error message for grep-ability.
- Blast radius: 1 file changed; DAILY_FILE_PATTERN import in audit_bars_coverage.py unaffected; test behavior unchanged
- Verification: PASS (18/18 test_ingest_daily.py, 77 drift checks pass, 737 fast tests pass, pre-commit suite pass)
- Commit: 4a62a53

---

## Iteration 138 — 2026-04-04
- Phase: fix
- Classification: [mechanical]
- Target: pipeline/build_daily_features.py:1086
- Finding: Wrong comment "~200 5m bars ≈ ~3.5 days" at line 1086 contradicts the correct comment at line 664 ("200 bars ≈ 16.7 hours") in the same file. 200 × 5min = 1000 min = 16.7 hours; the wrong "3.5 days" comment could mislead someone to reduce days=10 to ~4, silently breaking RSI warm-up.
- Action: Corrected comment to "200 5m bars ≈ 16.7 trading hours; 10d is conservative over-fetch". days=10 value unchanged.
- Blast radius: 1 file (build_daily_features.py); comment-only, 0 callers affected
- Verification: PASS (62/62 test_build_daily_features.py + 77 drift checks + 737 pre-commit suite)
- Commit: 74e051a

---

## Iteration 139 — 2026-04-04
- Phase: fix
- Classification: [judgment]
- Target: pipeline/build_bars_5m.py:336
- Finding: Fail-open in main() — verify_5m_integrity was guarded by `row_count > 0`. If source bars_1m exist but the INSERT produces 0 rows (SQL defect), build_5m_bars returns 0, verification is silently skipped, and sys.exit(0) is reached. This masks a broken build as a success.
- Action: Removed `and row_count > 0` from the verify condition. verify_5m_integrity now runs whenever not dry_run. With 0 rows in range, all checks pass cleanly (0 dupes, 0 misaligned, 0 OHLCV, 0 neg vol) — legitimate empty builds still succeed.
- Blast radius: 1 file (build_bars_5m.py main() only); callers invoke as subprocess; verify_5m_integrity signature unchanged
- Verification: PASS (8/8 test_build_bars_5m.py + 77 drift checks + 737 pre-commit suite)
- Commit: b8a5af8

---

## Iteration 140 — 2026-04-04
- Phase: fix
- Classification: [mechanical]
- Target: pipeline/run_pipeline.py:14
- Finding: Stale NQ symbol in module docstring — line 14 listed "(MGC, MNQ, NQ)" as valid instruments. NQ is the full-size Nasdaq data source symbol (orb_active=False); the correct active ORB micro instrument is MES.
- Action: Updated docstring "(MGC, MNQ, NQ)" → "(MGC, MNQ, MES)". No behavior change — docstring only.
- Blast radius: 1 file (run_pipeline.py docstring only); no callers affected
- Verification: PASS (29/29 test_run_pipeline.py + 77 drift checks + 737 pre-commit suite)
- Commit: 312ec41

---

## Iteration 141 — 2026-04-05
- Phase: audit-only (agent exhausted turns)
- Classification: N/A
- Target: trading_app/live/rithmic/order_router.py
- Finding: HIGH fail-open — query_open_orders() returns [] when auth is None instead of raising RuntimeError (inconsistent with submit()/cancel() which both raise). Leaves orphaned bracket orders alive. Also: B007 ruff lint (unused loop var uid), hardcoded entry model strings.
- Action: Finding documented but not fixed — agent ran out of turns during test cycle
- Blast radius: 3 callers (cancel_bracket_orders, internal)
- Verification: N/A
- Commit: NONE

---

## Iteration 142 — 2026-04-05
- Phase: fix
- Classification: [mechanical]
- Target: trading_app/prop_profiles.py:857
- Finding: parse_strategy_id hardcoded ("E1", "E2", "E3") tuple instead of importing ENTRY_MODELS from trading_app.config — canonical violation. If ENTRY_MODELS changes, parse_strategy_id would silently fail to recognize new entry models.
- Action: Added module-level `from trading_app.config import ENTRY_MODELS`; replaced `if p in ("E1", "E2", "E3"):` with `if p in ENTRY_MODELS:`. No behavior change (ENTRY_MODELS == ["E1", "E2", "E3"] currently).
- Blast radius: 2 callers (paper_trade_logger.py:74, prop_profiles.py:922); behavior identical
- Verification: PASS (44/44 test_prop_profiles.py; drift checks pass)
- Commit: 694108d (cherry-picked from 104d499)

---

## Iteration 143 — 2026-04-05
- Phase: fix
- Classification: [judgment]
- Target: trading_app/lane_allocator.py:616,572
- Finding: LA-01 (MEDIUM): save_allocation() used CWD-relative Path("docs/runtime/lane_allocation.json") while check_allocation_staleness() used file-relative anchor. Running from subdirectory writes/reads different locations. LA-02 (LOW): generate_report() had `except ImportError: pass` silently dropping "Changes vs Current Lanes" section.
- Action: LA-01: Changed to file-relative anchor matching check_allocation_staleness(). LA-02: Added warning line to report string on ImportError.
- Blast radius: 1 file, 2 callers (rebalance_lanes.py, tests)
- Verification: PASS (20/20 test_lane_allocator.py; drift checks pass)
- Commit: 694108d (cherry-picked from 0c4e1f7)

---

## Iteration 144 — 2026-04-05
- Phase: fix
- Classification: [judgment]
- Target: trading_app/live/multi_runner.py:110-117
- Finding: Fail-open in run() — asyncio.gather(return_exceptions=True) absorbs all per-orchestrator exceptions. run() returned None even if every instrument crashed; caller in run_live_session.py saw clean exit (code 0) despite total session failure.
- Action: Added failures counter in result-reporting loop. After stop-file cleanup, if failures == len(tasks), raise RuntimeError. Partial failures (degraded mode) preserved — only total failure is now surfaced.
- Blast radius: 1 file (multi_runner.py); 1 caller (run_live_session.py)
- Verification: PASS (11/11 test_multi_runner.py; drift checks pass)
- Commit: 694108d (cherry-picked from e83d272)

---

## Iteration 145 — 2026-04-05
- Phase: fix
- Classification: [judgment]
- Target: trading_app/live/broker_dispatcher.py:87-88
- Finding: update_market_price() secondary loop had no exception guard, unlike submit() and cancel_bracket_orders() which both wrap secondaries in try/except. Exception in secondary propagates to session_orchestrator bar loop with no handler.
- Action: Added try/except around secondary loop in update_market_price(), logging warning with exc_info=True. Consistent with cancel_bracket_orders() pattern. Primary call remains unguarded (correct — primary is authoritative).
- Blast radius: 1 file (broker_dispatcher.py); 1 caller (session_orchestrator.py:1083)
- Verification: PASS (53/53 test_tradovate.py; drift checks pass)
- Commit: 694108d (cherry-picked from 88ad9ab)
