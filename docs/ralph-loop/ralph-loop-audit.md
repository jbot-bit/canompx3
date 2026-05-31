# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 222

## RALPH AUDIT — Iteration 222 (COMPLETED)
## Date: 2026-05-31
## Infrastructure Gates: 152 drift checks PASS (fast + skip-crg-advisory); ruff PASS; 41/41 tests PASS
## Scope: trading_app/live_config.py — stale re-audit (last iter 119, findings=3 stale in ledger)

---

## Full-File Audit Results

### trading_app/live_config.py — SCANNED (stale re-audit, iter 119 confirmed clean)

**Seven Sins Scan — iteration 222:**
- S1 (Silent failure): `return None, None` at line 314 in `_jk_fallback_rr` — legitimate "no match" signal, caller checks it. `return None` at line 465 — same pattern, caller at line 783 checks and appends WARN note. CLEAN.
- S2 (Fail-open): `_check_noise_floor` returns `False` on NULL noise_risk (fail-closed, line 529). `_check_dollar_gate` returns `False` on missing `median_risk_points` (fail-closed, line 549) and on exception (line 563-564). CLEAN.
- S3 (Canonical violation): `"MGC"` in `INSTRUMENT_ATR_GATE` (line 134) — per-instrument regime heuristic, not a canonical list. ACCEPTABLE rule 1. `instrument: str = "MGC"` at line 569 — deprecated function, no live callers. ACCEPTABLE rule 2. `default="MGC"` at line 939 CLI — deprecated tool, guarded by `choices=get_active_instruments()`. ACCEPTABLE rule 2. All other instrument references use `get_active_instruments()` or caller-supplied values. `GOLD_DB_PATH` from `pipeline.paths`. CLEAN.
- S4 (Impact unawareness): 41 tests all PASS. No behavior changes this iteration. CLEAN.
- S5 (Evidence over assertion): All patterns verified by grep + execution. CLEAN.
- S6 (Spec compliance): No spec violations. CLEAN.
- S7 (Metadata trust): `LIVE_MIN_EXPECTANCY_R`, `HOT_MIN_STABILITY`, `HOT_LOOKBACK_WINDOWS`, `LIVE_MIN_EXPECTANCY_DOLLARS_MULT`, `INSTRUMENT_ATR_GATE` all carry `@research-source` annotations. CLEAN.

**Domain-specific checks:**
- ORB window timing: No `break_ts`, `orb_end`, or hardcoded times. CLEAN.
- Session hardcoding: No session literals — specs use string family IDs only (e.g. `"CME_PRECLOSE"`), not used in DST-sensitive resolution. CLEAN.
- E0 fill-on-touch: `close_outside`/`closed_outside` absent. CLEAN.
- Holdout date: No `date(2026` literals. CLEAN.
- DST contamination: No fixed clock times. CLEAN.
- DB path: `GOLD_DB_PATH` from `pipeline.paths` (line 30, 609). CLEAN.
- Cost inline: `get_cost_spec` from `pipeline.cost_model` (lines 551-556). CLEAN.
- Research stat inline: All numeric calibration values carry `@research-source`. CLEAN.

**Ralph-specific extensions scan:**
- Async safety: No async code. CLEAN.
- State persistence gap: No stateful objects. CLEAN.
- Contract drift: Public API unchanged. CLEAN.
- Look-ahead bias: No feature computation; this is a portfolio builder reading pre-validated data. CLEAN.

**ACCEPTABLE findings (4 patterns, same as iter 119):**
1. `"MGC"` key in `INSTRUMENT_ATR_GATE` dict (line 134): per-instrument regime heuristic — ACCEPTABLE rule 1.
2. `instrument: str = "MGC"` default in deprecated `build_live_portfolio()` (line 569): dormant deprecated function, no live callers — ACCEPTABLE rule 2.
3. `default="MGC"` in CLI argparse (line 939): deprecated CLI tool, guarded by `choices=get_active_instruments()` — ACCEPTABLE rule 2.
4. `except Exception as exc` in `_exp_dollars` display helper (line 992): logged, CLI display only, no capital path — ACCEPTABLE rules 1+3.

**Verdict: CLEAN — no actionable findings. Stale ledger `findings=3` entry was pre-iter-119 state.**

---

## RALPH AUDIT — Iteration 221 (COMPLETED)
## Date: 2026-05-31
## Infrastructure Gates: 152 drift checks PASS (fast + skip-crg-advisory); ruff PASS; 36/36 tests PASS
## Scope: trading_app/outcome_builder.py — first full scan (critical centrality, 11 importers)

---

## Full-File Audit Results

### trading_app/outcome_builder.py — SCANNED (first full scan)

**Seven Sins Scan — iteration 221:**
- S1 (Silent failure): `return None` at lines 83 and 101 in `_check_fill_bar_exit` — early exits when fill bar missing or no exit hit. Both are defensive guards, not silent failures. CLEAN.
- S2 (Fail-open): `build_outcomes()` raises `ValueError` when instrument is None (line 736). `get_enabled_sessions` returns empty → `ValueError` raised (line 793). CLEAN.
- S3 (Canonical violation): OB-221-01 FIXED — `default="MGC"` in CLI argparse (line 1098) bypassed the `build_outcomes()` `instrument is None` guard. Fixed: `required=True`. All other instrument references are explicit caller-supplied values. `GOLD_DB_PATH` from `pipeline.paths`. `orb_utc_window` from `pipeline.dst`. `get_cost_spec` from `pipeline.cost_model`. CLEAN.
- S4 (Impact unawareness): `tests/test_trading_app/test_outcome_builder.py` has 36 tests, all PASS. `TestCLI.test_help` uses `--help` only — unaffected by `required=True`. CLEAN.
- S5 (Evidence over assertion): All assertions verified by execution. CLEAN.
- S6 (Spec compliance): No spec violations. CLEAN.
- S7 (Metadata trust): `RR_TARGETS` and `CONFIRM_BARS_OPTIONS` are grid parameters documented in config.py — not research stats embedded as literals. CLEAN.

**Domain-specific checks:**
- ORB window timing: `orb_utc_window` from `pipeline.dst` used exclusively (line 898). No `break_ts` fallback. CLEAN.
- Session hardcoding: `get_enabled_sessions(instrument)` from `pipeline.asset_configs`. CLEAN.
- E0 fill-on-touch: `close_outside`/`closed_outside` absent. CLEAN.
- Holdout date: No `date(2026` literals. CLEAN.
- DST contamination: No fixed clock times. CLEAN.
- DB path: `GOLD_DB_PATH` from `pipeline.paths`. CLEAN.
- Cost inline: `get_cost_spec`, `pnl_points_to_r`, `risk_in_dollars`, `to_r_multiple` all from `pipeline.cost_model`. CLEAN.

**Ralph-specific extensions scan:**
- Async safety: No async code. CLEAN.
- State persistence gap: No stateful objects. CLEAN.
- Contract drift: `compute_single_outcome` public API (break_ts, orb_high, orb_low, break_dir, rr_target, confirm_bars, trading_day_end, cost_spec) unchanged. CLEAN.
- Look-ahead bias: All outcome computation is post-entry (bars after entry_ts). `orb_end_utc` from canonical window — not from `break_ts`. CLEAN.

**Finding fixed:**

- OB-221-01 [LOW] — canonical_violation: `default="MGC"` in CLI argparse at line 1098 bypassed the `build_outcomes()` guard. The file's own docstring (lines 724-727) documented the prior MGC default as a canonical violation per integrity-guardian.md § 2. Fix: `required=True` replacing `default="MGC"`. Commit: aa5963ea (bundled with iter 219).

**ACCEPTABLE findings:**
- `TODO(E3-retired)` at line 964: dormant infrastructure with existing TODO annotation. ACCEPTABLE pattern 2.

---

## RALPH AUDIT — Iteration 220 (COMPLETED)
## Date: 2026-05-31
## Infrastructure Gates: 152 drift checks PASS (fast + skip-crg-advisory); ruff PASS
## Scope: trading_app/live/broker_factory.py — stale re-audit (last iter 127, findings=1)

---

## Full-File Audit Results

### trading_app/live/broker_factory.py — SCANNED (stale re-audit, iter 127 → 220)

**Iter-127 fix verified:**
- VALID_BROKERS coherence guard: `if broker not in VALID_BROKERS: raise ValueError(...)` present at line 55 — VERIFIED correct.
- `raise AssertionError("unreachable")` at line 108 — defensive guard present — VERIFIED correct.
- The iter-127 finding (VALID_BROKERS declared as "canonical source" but only used in error message, with actual dispatch as unguarded if/elif) has been fully resolved.

**Seven Sins Scan — iteration 220:**
- S1 (Silent failure): No `except` blocks in the file. CLEAN.
- S2 (Fail-open): `ValueError` on unknown broker (line 56). `AssertionError("unreachable")` at line 108. CLEAN.
- S3 (Canonical violation): `VALID_BROKERS` is the canonical source (imported by `deployable_shelf_gap.py`). No instrument names, session names, holdout dates, cost specs, DB paths, ORB windows. `get_broker_name()` default `"projectx"` is a dispatcher default, not a canonical list. CLEAN.
- S4 (Impact unawareness): Tests in `test_run_live_session_preflight.py`, `test_rithmic_router.py`, `test_tradovate.py`. CLEAN.
- S5 (Evidence over assertion): No assertions in code. CLEAN.
- S6 (Spec compliance): No spec violations. CLEAN.
- S7 (Metadata trust): No docstring-as-truth issues. CLEAN.

**Domain-specific checks:**
- ORB window timing, session hardcoding, E0 fill-on-touch, holdout date, DST contamination, DB path, cost inline, instrument hardcoding: all CLEAN (pure broker dispatch module — no ORB/trading logic).

**Ralph-specific extensions scan:**
- Async safety: No async code. CLEAN.
- State persistence gap: No stateful objects. CLEAN.
- Contract drift: `BrokerComponents` TypedDict with 5 keys; callers in `session_orchestrator.py` and `webhook_server.py` consume all 5 keys intact. CLEAN.
- Look-ahead bias: Not applicable (broker dispatch module). CLEAN.

**Findings: 0 actionable findings. CLEAN.**

---

## RALPH AUDIT — Iteration 218 (COMPLETED)
## Date: 2026-05-31
## Infrastructure Gates: 152 drift checks PASS (fast + skip-crg-advisory); ruff PASS; 19/19 tests PASS
## Scope: trading_app/ai/provider_registry.py — P1 unscanned high (6 importers)

---

## Full-File Audit Results

### trading_app/ai/provider_registry.py — SCANNED (first full scan)

**Seven Sins Scan — iteration 218:**
- S1 (Silent failure): `except KeyError as exc: raise KeyError(...)` at line 302 — re-raises with improved message. Not silent. CLEAN.
- S2 (Fail-open): `validation_errors()` accumulates all errors; `assert_ready()` raises on non-empty list. `missing_env()` returns sorted list. No health-check-returns-True pattern. CLEAN.
- S3 (Canonical violation): `CLAUDE_REASONING_MODEL`/`CLAUDE_STRUCTURED_MODEL` imported from `trading_app.ai.claude_client` (canonical source). `OPENROUTER_BASE_URL` now a module-level constant (PR-218-01 FIXED). No instrument names, session names, holdout dates, cost specs. CLEAN.
- S4 (Impact unawareness): test_provider_registry.py has 19 tests, all PASS. CLEAN.
- S5 (Evidence over assertion): All assertions verified by execution. CLEAN.
- S6 (Spec compliance): No spec violations. CLEAN.
- S7 (Metadata trust): No docstring-as-truth violations; notes fields are informational only. CLEAN.

**Domain-specific checks:**
- ORB window timing, session hardcoding, E0, holdout date, DST, DB path, cost inline, instrument hardcoding: all CLEAN (AI provider registry module — no ORB/trading logic).

**Ralph-specific extensions scan:**
- Async safety, state persistence gap, contract drift, look-ahead bias: all CLEAN (pure configuration/registry module, no async code, no stateful objects).

**Finding fixed:**

- PR-218-01 [LOW] — annotation_debt/DRY: `"https://openrouter.ai/api/v1"` hardcoded as 4 separate string literals at lines 203, 224, 248, 267 in deepseek_* profile `base_url=` fields. No module-level constant existed. Fix: extracted `OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"` constant at line 25; replaced all 4 literals with constant reference. Zero behavior change — same URL, now a single-point-of-change. Commit: 0989bde3.

---

## RALPH AUDIT — Iteration 217 (COMPLETED)
## Date: 2026-05-31
## Infrastructure Gates: 152 drift checks PASS (fast + skip-crg-advisory); ruff PASS
## Scope: pipeline/system_brief.py (P1 unscanned high) + trading_app/portfolio.py (P3 stale re-audit)

---

## Full-File Audit Results

### pipeline/system_brief.py — SCANNED (first full scan)

**Seven Sins Scan — iteration 217:**
- S1 (Silent failure): `except Exception as exc` at line 49 in `_load_capsule_summary` — intentional fail-soft; work capsule is non-critical infrastructure. Returns `(None, [BriefIssue("warning", ...)])`. All callers handle `None` capsule_summary at line 160. ACCEPTABLE pattern 2.
- S2 (Fail-open): No health check or success-reporting paths. `route_issues` blockers are passed through to the return dict. CLEAN.
- S3 (Canonical violation): `DECISION_LEDGER_PATH` and `DEBT_LEDGER_PATH` are documentation ledger paths, not DB/instrument/session paths. Consistent with `pipeline/system_authority.py:136-137`. CLEAN.
- S4 (Impact unawareness): test_system_brief.py has 2 tests, both PASS. CLEAN.
- S5 (Evidence over assertion): All claims verified by code trace and test execution. CLEAN.
- S6 (Spec compliance): No spec violations. CLEAN.
- S7 (Metadata trust): `expansion_triggers or` fallback at line 165 — intentional design: empty tuple means "use briefing contract defaults" (ACCEPTABLE pattern 1). Not a silent masking issue.

**Domain-specific checks:**
- ORB window timing, session hardcoding, E0, holdout date, DST, DB path, cost inline: all CLEAN (no ORB/session/instrument/cost references in this module).

**Ralph-specific extensions scan:**
- Async safety, state persistence gap, contract drift, look-ahead bias: all CLEAN (no async code, no stateful objects, no ORB computation).

**Findings:** 0 actionable findings. CLEAN.

---

### trading_app/portfolio.py — SCANNED (stale re-audit, iter 118 → 217)

**Seven Sins Scan — iteration 217:**
- S1 (Silent failure): No silent failure paths in this module's logic. CLEAN.
- S2 (Fail-open): No health check paths. CLEAN.
- S3 (Canonical violation):
  - `ML_OVERLAY_SESSIONS = {"NYSE_OPEN", "US_DATA_1000", "US_DATA_830"}` at line 966: bootstrap-verified research-derived subset, NOT a canonical index. Comment documents BH FDR evidence. ACCEPTABLE pattern 1.
  - `orb_minutes=5` (line 998) and `orb_minutes=30` (line 1012): architectural layer constants for the documented O5/O30 multi-RR design. ACCEPTABLE pattern 1.
  - `instrument: str = "MGC"` (line 468) and `instrument: str = "MNQ"` (lines 614, 971): CLI-facing default convenience values, not canonical enumeration. ACCEPTABLE pattern 1.
  - `GOLD_DB_PATH` imported from `pipeline.paths` (line 32). CLEAN.
- S4 (Impact unawareness): Not checked (file unchanged since iter 118, hash verified stable). CLEAN.
- S5 (Evidence over assertion): All assertions traced. CLEAN.
- S6 (Spec compliance): No spec violations. CLEAN.
- S7 (Metadata trust): `FITNESS_WEIGHTS` at lines 1443-1448 — operational policy weights implementing the fitness system contract, not unannotated research stats. `0.4` trades/day at line 1400 correctly has `@research-source` + `@revalidated-for`. CLEAN.

**Domain-specific checks:**
- ORB window timing: No `break_ts` fallback, no re-derivation of ORB windows. CLEAN.
- E0 fill-on-touch: No `close_outside`/`closed_outside` references. CLEAN.
- Holdout date: No `date(2026` literals. CLEAN.
- DST contamination: Session strings are labels only. CLEAN.
- DB path: `GOLD_DB_PATH` from `pipeline.paths`. CLEAN.
- Cost inline: `get_cost_spec` imported from `pipeline.cost_model`. CLEAN.

**Findings:** 0 actionable findings. CLEAN (iter-118 findings were resolved upstream).

---

## RALPH AUDIT — Iteration 216 (COMPLETED)
## Date: 2026-05-31
## Infrastructure Gates: 152 drift checks PASS (fast + skip-crg-advisory); ruff PASS
## Scope: pipeline/asset_configs.py — P1 unscanned critical (82 importers)

---

## Full-File Audit Results

### pipeline/asset_configs.py — SCANNED (first full scan)

**Seven Sins Scan — iteration 216:**
- S1 (Silent failure): `get_enabled_sessions` returns `[]` for unknown instruments — both callers (outcome_builder.py:792-795, strategy_discovery.py:1299-1302) raise ValueError on empty. No silent path. CLEAN.
- S2 (Fail-open): `require_dbn_available` raises ValueError on all failure paths (missing dbn_path, missing minimum_start_date, FileNotFoundError on disk check). CLEAN.
- S3 (Canonical violation): `ACTIVE_ORB_INSTRUMENTS` derives from `ASSET_CONFIGS` dict + `DEAD_ORB_INSTRUMENTS` exclusion — canonical source, not hardcoded list. All session names in `enabled_sessions` validated against `SESSION_CATALOG` (0 unknown). `PROJECT_ROOT / "DB"` paths are raw DBN data stores (not deprecated gold.db scratch path). CLEAN.
- S4 (Impact unawareness): test_asset_configs.py has 40 tests, all pass. CLEAN.
- S5 (Evidence over assertion): All claims verified by execution. CLEAN.
- S6 (Spec compliance): No spec violations. CLEAN.
- S7 (Metadata trust): AC-216-01 FIXED — M2K `orb_active: True` was misleading metadata. Changed to `orb_active: False` (commit 0e77cda4). FIXED.

**Domain-specific checks:**
- ORB window timing: No `break_ts`, no hardcoded orb_minutes. CLEAN.
- Session hardcoding: All session strings validated against SESSION_CATALOG. CLEAN.
- E0 fill-on-touch: No `close_outside`/`closed_outside` references. CLEAN.
- Holdout date: No `date(2026` literals in instrument configuration. CLEAN.
- DST contamination: No fixed timing constants; sessions are labels only. CLEAN.
- DB path: No `gold.db` references; `PROJECT_ROOT / "DB"` paths are raw DBN stores (correct). CLEAN.

**Ralph-specific extensions scan:**
- Async safety: No async code in this module. CLEAN.
- State persistence gap: No stateful objects. CLEAN.
- Contract drift: `get_asset_config`/`require_dbn_available` API split from 2026-04-19 intact. CLEAN.
- Look-ahead bias: Not applicable (configuration module). CLEAN.

**Finding fixed:**

- AC-216-01 [LOW] — annotation_debt: M2K `"orb_active": True` at line 234 was misleading — M2K has been in `DEAD_ORB_INSTRUMENTS` since Mar 2026 (0/18 families survive noise screening). The drift check `check_no_raw_orb_active_reads` (check_drift.py:6126) explicitly documented M2K as a "trap" with `orb_active=True`. Fixed: changed to `"orb_active": False` with clarifying comment. Zero behavior change — `ACTIVE_ORB_INSTRUMENTS` derivation still correctly excludes M2K via `DEAD_ORB_INSTRUMENTS`. Commit: 0e77cda4.

**Known acceptable patterns from prior iter (182 scan):** none recorded in ledger for this file.

---

## RALPH AUDIT — Iteration 215 (COMPLETED)
## Date: 2026-05-31
## Infrastructure Gates: 152 drift checks PASS (fast + skip-crg-advisory); ruff PASS
## Scope: pipeline/build_daily_features.py — stale re-audit (findings=4 since iter 189, 3 commits since)

---

## Full-File Audit Results

### pipeline/build_daily_features.py — SCANNED (stale re-audit)

Stale re-audit covering 3 commits since iter 189 (last audited):
- `62c51a14`: NYSE_PREOPEN session added (comment-only change to build_daily_features.py)
- `296c54f2`: NYSE_PREOPEN holiday-contamination guard added to compute_orb_range
- `19a344c6`: verify_daily_features moved before COMMIT (fail-before-commit fix)

**Iter-189 fix verified:**
- GARCH-SILENT-N1: logger.debug → logger.warning at except Exception block (line 860-862) — VERIFIED correct.

**New changes verified:**

**296c54f2 (holiday guard):** `compute_orb_range` at line 231 correctly gates NYSE_PREOPEN on NYSE holidays using `is_nyse_holiday()` (canonical `pipeline.market_calendar` source). Returns all-None routing into the existing empty-ORB path. Uses `_orb_utc_window` for non-holiday days. Fail-closed. CLEAN.

**19a344c6 (verify-before-commit):** `verify_daily_features` now called BEFORE `con.execute("COMMIT")`. On integrity failure, raises `RuntimeError` before committing — outer `except Exception as e` catches it, executes ROLLBACK, re-raises. Correct pattern. CLEAN.

**ACCEPTABLE findings (not fixed):**

- `SESSION_WINDOWS = {"asia": (9, 0, 17, 0), "london": (18, 0, 23, 0), "ny": (23, 0, 2, 0)}` at line 95: Fixed Brisbane-time windows for session_asia/london/ny range features. Comment at line 93-94 explicitly documents "These do NOT track actual market opens which shift with DST." Features not used in any strategy filter (config.py has zero references). Used only for informational market profile data. ACCEPTABLE per pattern 1 (intentional per-session heuristic, documented non-DST approximation).

- `insert_count = len(rows) - existing_count` at line 1743: minor logging inaccuracy (counts rows-before-update, not true new inserts). No capital path impact, no correctness impact. ACCEPTABLE per pattern 3 (style, no correctness impact).

**Seven Sins Scan — iteration 215:**
- S1 (Silent failure): `except Exception as exc` at line 860 → `logger.warning` + `return None` (logging elevated from debug in iter 189). `except Exception as e` at line 1779 → ROLLBACK + logger.error + re-raise. `except Exception` at line 1958 (main) → `sys.exit(1)` (already logged by inner handler). CLEAN.
- S2 (Fail-open): `verify_daily_features` called before COMMIT; RuntimeError on failure triggers ROLLBACK. CLEAN.
- S3 (Canonical violation): `_orb_utc_window` used for all ORB windows. `SESSION_WINDOWS` documented as intentional approximation (no strategy filter use). `GOLD_DB_PATH` from `pipeline.paths`. `COMPRESSION_SESSIONS` defined here and imported by portfolio.py (not a duplication — single definition). CLEAN.
- S4 (Impact unawareness): test_build_daily_features.py updated in all 3 commits. CLEAN.
- S5 (Evidence over assertion): All assertions verified by execution in commit messages. CLEAN.
- S6 (Spec compliance): No spec violations identified. CLEAN.
- S7 (Metadata trust): No docstring-as-truth violations. CLEAN.

**Domain-specific checks:**
- ORB window timing: `_orb_utc_window` (aliased from `pipeline.dst.orb_utc_window`) used exclusively. No `break_ts` fallback, no re-derivation. CLEAN.
- Session hardcoding: `SESSION_WINDOWS` is a documented approximation for informational features only. ORB sessions all use `ORB_LABELS` from init_db. CLEAN.
- E0 fill-on-touch: No `close_outside`/`closed_outside` references. CLEAN.
- Holdout date: No `date(2026` literals in feature computation. CLEAN.
- DST contamination: `SESSION_WINDOWS` documented as non-DST-aware approximation. All ORB windows use `_orb_utc_window` resolver. CLEAN.

**Ralph-specific extensions scan:**
- Look-ahead bias: All feature computation is retrospective (daily features builder runs post-session). post-pass rolling features use `post_rows[0..i-1]` indexing (prior-only). CLEAN.
- State persistence gap: No stateful objects; all data passed through `con` transactions. CLEAN.
- Contract drift: `build_daily_features(con, symbol, start_date, end_date, orb_minutes, dry_run)` signature unchanged. CLEAN.

---

## RALPH AUDIT — Iteration 214 (COMPLETED)
## Date: 2026-05-31
## Infrastructure Gates: 152 drift checks PASS (fast + skip-crg-advisory); 247 tests PASS; ruff PASS
## Scope: trading_app/live/session_orchestrator.py — stale re-audit (hash changed since iter 182, 3 commits since ea0d4fec)

---

## Full-File Audit Results

### trading_app/live/session_orchestrator.py — SCANNED (stale re-audit)

Stale re-audit covering 3 commits since iter 182 (last audited):
- `ea0d4fec`: NQ-mini symbol substitution, dormant-only wiring
- `1cc7f4a1`: lifecycle-block silent-fail fix + readiness effective-copies
- `a877fc89` (iter 213): broker-factory bypass in reconnect + TradovateContracts AttributeError

**All iter-213 fixes verified:**
- `_contracts_cls` stored at __init__ line 326; used at reconnect line 3835 — VERIFIED correct.
- All three contracts classes accept `**kwargs` so `demo=self.demo` is absorbed safely — VERIFIED via grep.
- Lifecycle block fail-open → `log.critical` + `_notify` at line 1132-1143 — VERIFIED correct behavior.
- `LIVE_SIGNALS_DIR` from `pipeline.paths` (was `Path(__file__).parent.parent.parent`) at line 297 — canonical path migration VERIFIED.

**ACCEPTABLE finding (not fixed):**
- `_close_min_et or 0` at lines 1362 and 3701: falsy-zero pattern, but produces correct results in all cases. `_close_min_et` is `int | None`; when `None` → `0` (correct); when `0` (e.g., 16:00 close) → `0` (correct); when positive int → itself (correct). ACCEPTABLE per pattern 3 (style/preference difference with no correctness impact; no capital path impacted).

**Seven Sins Scan — iteration 214:**
- S1 (Silent failure): `_publish_state` `except Exception: pass` is intentional (dashboard state is best-effort, documented). `_minutes_to_close_et` `except Exception: return None` is defensive-safe. CLEAN.
- S2 (Fail-open): Lifecycle block failure now emits `log.critical` + `_notify` — CLEAN (fixed iter 213).
- S3 (Canonical violation): No hardcoded instruments, sessions, DB paths, costs, chordia verdicts. `LIVE_SIGNALS_DIR` from `pipeline.paths`. CLEAN.
- S4-S7: No spec violations, no look-ahead bias, no research stats inline, no holdout date hardcoding. CLEAN.
- Async safety: `time.sleep` only in `post_session()` (post-asyncio.run(), no event loop — documented intentional). CLEAN.
- State persistence gap: `_kill_switch_fired` → `_safety_state.save()` on `_fire_kill_switch()`. CLEAN.
- Contract drift: `_contracts_cls` contract preserved, `**kwargs` absorbs broker-specific params. CLEAN.

**Domain-specific checks:**
- ORB window timing: No `break_ts` fallback, no hardcoded orb_minutes. CLEAN.
- E0 fill-on-touch: No `close_outside`/`closed_outside` references. CLEAN.
- Holdout date: No `date(2026` literals in trading code. CLEAN.

---

## Files Fully Scanned

- pipeline/system_context.py (iter 208)
- tests/test_pipeline/test_system_context.py (iter 208)
- .claude/hooks/session-start.py (iter 208)
- trading_app/derived_state.py (iter 209)
- pipeline/check_drift.py (iter 210 — Pyright audit; 0 errors confirmed)
- pipeline/check_drift_crg_helpers.py (iter 210 — Pyright audit; 0 errors confirmed)
- trading_app/eligibility/builder.py (iter 211 — full scan; 1 annotation finding fixed)
- trading_app/opportunity_awareness.py (iter 212 — full scan; 1 canonical violation fixed)
- trading_app/allocation_promotion.py (iter 212 — batch fix; 1 canonical violation fixed)
- trading_app/live/session_orchestrator.py (iter 214 — stale re-audit; 0 findings, ACCEPTABLE falsy-zero LOW)
- pipeline/build_daily_features.py (iter 215 — stale re-audit; 0 findings, 2 ACCEPTABLE LOW)
- pipeline/asset_configs.py (iter 216 — full scan; 1 LOW fixed annotation_debt AC-216-01)
- pipeline/system_brief.py (iter 217 — first full scan; 0 findings, 2 ACCEPTABLE)
- trading_app/portfolio.py (iter 217 — stale re-audit iter 118; 0 findings, 5 ACCEPTABLE)
- trading_app/ai/provider_registry.py (iter 218 — first full scan; 1 LOW fixed PR-218-01)
- trading_app/live/broker_factory.py (iter 220 — stale re-audit iter 127; 0 findings, iter-127 fix verified present)
- trading_app/outcome_builder.py (iter 221 — first full scan; 1 LOW fixed OB-221-01, ACCEPTABLE TODO(E3-retired))

## Next Iteration Targets

Priority 0 — Open deferred HIGH/CRITICAL: NONE (SHADOW-MLL is MEDIUM, intentional design, dormant).
Priority 1 — Unscanned high files: `pipeline/ingest_dbn_mgc.py` (high, 9 importers, findings=1 from iter 136 — stale re-audit needed, hash changed since iter 219 fix).
Priority 2 — Stale re-audits: `trading_app/live_config.py` (high, last iter 119, findings=3, hash unchanged — re-audit to verify fixes). `trading_app/conditional_overlays.py` (medium, 2 importers, never scanned).
Priority 3 — Stale medium files: `trading_app/db_manager.py` (last iter 180, findings=2 — check hash).
