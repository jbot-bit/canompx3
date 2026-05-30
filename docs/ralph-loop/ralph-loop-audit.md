# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 215

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

## Next Iteration Targets

Priority 0 — Open deferred HIGH/CRITICAL: NONE (SHADOW-MLL is MEDIUM, intentional design, dormant).
Priority 1 — Unscanned critical/high files: `pipeline/asset_configs.py` (82 importers, critical centrality — unscanned in ledger, highest centrality unscanned file).
Priority 2 — Stale re-audits: `trading_app/conditional_overlays.py` (new module referenced in session_orchestrator, unscanned). `trading_app/execution_engine.py` (critical usage, not in centrality JSON).
Priority 3 — Unscanned medium files: `trading_app/portfolio.py` (imports COMPRESSION_SESSIONS from build_daily_features — verified in this iter, but portfolio.py itself unscanned).
