# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 214

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

## RALPH AUDIT — Iteration 212 (COMPLETED)
## Date: 2026-05-30
## Infrastructure Gates: 151 drift checks PASS (fast); 10 tests PASS; ruff PASS; backfill dry-run drifted=0
## Scope: trading_app/opportunity_awareness.py — first full scan (high centrality, 5 importers, unscanned); trading_app/allocation_promotion.py — batch same fix type

---

## Full-File Audit Results

### trading_app/opportunity_awareness.py — SCANNED

Shadow-only opportunity awareness module (display/pre-session, no trade authorization).
Well-structured: fail-closed exception handling with proper `available=False` returns,
canonical delegation to `resolve_profile_id`, `resolve_allocation_json`, `build_state_envelope`.

**Finding FIXED (canonical_violation, MEDIUM):**
`PASSING_CHORDIA_VERDICTS = frozenset({"PASS_CHORDIA", "PASS_PROTOCOL_A"})` at line 38
duplicates the policy encoded in `chordia.chordia_verdict_allows_deploy()` (chordia.py:468).
Two call sites used this local frozenset instead of the canonical function. If the verdict
taxonomy gains a new passing label (e.g., `PASS_PROTOCOL_B`), the local copy would silently
miss it — wrong tier assignment in PRIME_SHADOW detection, wrong "Chordia gate:" warnings.

Also found same pattern in `trading_app/allocation_promotion.py:18` (`PASS_CHORDIA_VERDICTS = {"PASS_CHORDIA", "PASS_PROTOCOL_A"}`) — batched in same fix.

Fix: Added `from trading_app.chordia import chordia_verdict_allows_deploy`; replaced all 3
call sites with the canonical function call.

**ACCEPTABLE findings (not fixed):**
- `date.today()` at lines 329/383/446: shadow-only display module; trading_day is always
  passed explicitly by session_orchestrator callers via `today=` param; staleness is bounded
  by `OPPORTUNITY_MAX_AGE_DAYS=1`. ACCEPTABLE per pattern 1 (intentional per-session heuristic).
- `trailing_expr >= 0.20` at line 265: PRIME_SHADOW threshold — operational heuristic for
  display tier, not a deployed trading parameter. No `@research-source` required per project
  standards (operator-visible only, no capital path). ACCEPTABLE per pattern 3 (style/preference
  with no correctness impact).
- Broad `except Exception` at lines 456/481: both handlers return `available=False/valid=False`
  explicitly — fail-closed, not swallowed. ACCEPTABLE per pattern 4 (guarded by fail-closed return).

---

## Seven Sins Scan — iteration 212

- Sin 1 (Silent failure): Both except blocks return structured error envelope with `available=False`. CLEAN.
- Sin 2 (Canonical violation): FIXED — local chordia verdict set replaced with canonical function. allocation_promotion.py also fixed.
- Sin 3 (Fail-open): No path returns success after failure; `_validated_opportunity_state` returns (False, reason, None) on any validity failure. CLEAN.
- Sin 4-7: Shadow-only module; no capital gate, no inline research stats, no spec violations, no holdout contamination. CLEAN.

**Ralph-specific extensions scan:**
- Async safety: No async code in scope. CLEAN.
- State persistence gap: `write_state=True` path uses `state_path.write_text()` atomically. CLEAN.
- Contract drift: `validate_state_envelope` receives full inputs; `_validated_opportunity_state` return tuple used consistently. CLEAN.

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

## Next Iteration Targets

Priority 0 — Open deferred HIGH/CRITICAL: NONE (SHADOW-MLL is MEDIUM, intentional design, dormant).
Priority 1 — Unscanned critical/high files: `pipeline/build_daily_features.py` (critical centrality, 82 importers, last audited iter 189 — 5 audits but may have changed). Or `trading_app/execution_engine.py` (critical, not listed in centrality JSON but high usage).
Priority 2 — Stale re-audits: `pipeline/asset_configs.py` (82 importers, critical centrality — check last scan date vs recent instrument changes). `trading_app/live/session_orchestrator.py` freshly audited iter 214.
Priority 3 — Unscanned medium files: `trading_app/conditional_overlays.py` (new module referenced in session_orchestrator, unscanned).
Priority 1 — Unscanned high files: `pipeline/paths.py` already audited iter 207 (3 audits). `trading_app/live/session_orchestrator.py` is Priority 2 stale re-audit but ranks highest.
