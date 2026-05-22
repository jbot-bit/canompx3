# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 193

## RALPH AUDIT — Iteration 193 (COMPLETED)
## Date: 2026-05-23
## Infrastructure Gates: 160 drift checks PASS; 44 deployability tests PASS
## Scope: trading_app/deployability.py — hardcoded threshold literals on capital-decision path

---

## Iteration 193 — trading_app/deployability.py (full scan)

### Auto-Targeting
- Scope provided: `trading_app/deployability.py` — medium centrality, never scanned by Ralph.

---

## Finding CANON-193 — LOW — FIXED

**PREMISE:** `trading_app/deployability.py` hardcoded `0.50` (WFE floor), `100` (CORE_MIN_SAMPLES), and `30` (REGIME_MIN_SAMPLES) as inline magic numbers on the capital-decision path, instead of importing from `trading_app.config`.

**TRACE:**
- `_singleton_clears_binding_criteria` (deployability.py:299,303) → `float(wfe) < 0.50` / `int(n) < 100`
- `_trade_context` (deployability.py:629,631) → `int(sample_size) < 30` / `int(sample_size) < 100`
- `_classify_strategy` (deployability.py:845,850) → `float(row["wfe"]) < 0.50` / `int(row["sample_size"]) < 100`
- `_build_instrument_summary` (deployability.py:958) → `int(report.metrics["sample_size"]) < 100`

**EVIDENCE:** `trading_app.config` exports `MIN_WFE = 0.50`, `CORE_MIN_SAMPLES = 100`, `REGIME_MIN_SAMPLES = 30` — already imported by `deployability.py` (import line 24), just not referenced.

**FIX:** `trading_app/deployability.py:24` — added `CORE_MIN_SAMPLES, MIN_WFE, REGIME_MIN_SAMPLES` to existing config import; replaced 7 inline literal uses with the canonical constants. 8 lines changed.

**DOCTRINE:** `integrity-guardian.md § 2` / `institutional-rigor.md § 10` — import from canonical module, never inline magic numbers on capital-class decision paths.

**VERDICT:** FIXED — commit `8dfb78e5`

---

## Other Findings (No Fix Required)

### `except Exception` at line 322 — ACCEPTABLE
`except Exception: failed.append("C10_micro_check_error")` — explicitly fail-closed (appends to `failed` list which gates deployment). Annotated `# noqa: BLE001 - fail closed`. Matches ACCEPTABLE pattern 1 (intentional fail-closed exception swallow).

### `except Exception as exc` at line 461 — ACCEPTABLE
`_replay_strategy` returns `{"ok": False, "error": str(exc)}` — fail-closed (replay error → `replay_mismatch` hard issue → `BLOCKED_REPLAY_MISMATCH` verdict). Exception is recorded in the `error` field. Correct pattern.

### `< 0.05` FDR alpha (lines 274, 437) — ACCEPTABLE
No named canonical constant for BH FDR alpha exists in this codebase (`strategy_validator.py` uses `alpha=0.05` as a function default, not an exported constant). Style difference with no correctness impact; would need a new constant added to config.py which is out of scope.

### `< 0.95` DSR threshold (line 861) — ACCEPTABLE
No canonical constant for the DSR cross-check threshold exists. `dsr_below_cross_check` is a warning only (not a hard gate), so no capital risk.

### `< 7` years tested (line 853) — ACCEPTABLE
`pipeline/check_drift.py` has a local `MIN_YEARS = 7` but it is not exported from any canonical module. Short-history issue is a warning only.

---

## Seven Sins Scan — trading_app/deployability.py

- **Silent failure:** No unguarded silent failures. Both `except Exception` blocks are fail-closed.
- **Fail-open:** No fail-open patterns detected. All hard-blocker paths route to BLOCKED verdicts.
- **Canonical violation:** FIXED (WFE/sample thresholds now import from config).
- **Impact awareness:** `_classify_strategy` correctly uses `HARD_BLOCKER_TO_VERDICT` to prevent verdict bypass.
- **Chordia gate:** `chordia_verdict_allows_deploy` + `chordia_verdict_label` imported from canonical `trading_app.chordia`. No bypass detected.
- **Theory_grant gate:** `has_theory=False` hardcoded for SINGLETON path — correct by design (docstring explicitly documents this and adversarial-audit 2026-05-11 verified it).
- **Instrument lists:** No hardcoded instrument list. Uses `ACTIVE_ORB_INSTRUMENTS` from `pipeline.asset_configs` via `_active_in_sql()`.
- **Entry model lists:** No hardcoded entry model list.
- **Hardcoded status strings:** verdict strings (`DEPLOYABLE_CANDIDATE` etc.) are module-level constants, not duplicated inline.

---

## Files Fully Scanned

- trading_app/deployability.py (iter 193)
- trading_app/chordia.py (iter 192)
- trading_app/live/session_orchestrator.py (iter A.6)
- trading_app/pre_session_check.py (iter A.6)
- trading_app/derived_state.py (iter A.6)
- trading_app/live/alert_engine.py (iter ALERT-CONTAM-N2)
- trading_app/live/tradovate/order_router.py (iter PR301)
- trading_app/live/tradovate/http.py (iter PR301)
- pipeline/check_drift.py (iter 192 — partial, check_am33 function)
- scripts/run_live_session.py (iter A.6)

## Next Iteration Targets

**Priority 1 — Unscanned medium-centrality files:**
- `trading_app/strategy_fitness.py` — used by deployability and chordia; never scanned
- `trading_app/lifecycle_state.py` — called by deployability on every profile audit; never scanned
- `trading_app/strategy_validator.py` — critical path for C3/C8 gates; partial scan only
