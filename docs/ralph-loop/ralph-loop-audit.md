# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## Last iteration: 198

## RALPH AUDIT — Iteration 198 (COMPLETED)
## Date: 2026-05-23
## Infrastructure Gates: 160 drift checks PASS; 26 chordia tests PASS
## Scope: trading_app/chordia.py

---

## chordia.py Audit Summary

`chordia.py` implements the Chordia t-statistic gate (Criterion 4), the
audit-log reader, and the verdict taxonomy (PASS_CHORDIA / PASS_PROTOCOL_A /
FAIL_CHORDIA / FAIL_BOTH / MISSING). Capital-class: verdicts determine live
deployment eligibility via `apply_chordia_gate` in `lane_allocator.py`.

The file is well-structured with one finding:

---

## Finding CHORDIA-IO-198 — MEDIUM — FIXED

**PREMISE:** `load_chordia_audit_log` documents a fail-closed contract
(missing OR malformed YAML → empty log with default_has_theory=False). But
the try/except at line 350 only catches `yaml.YAMLError`. A call to
`p.read_text()` on an existing file can raise `OSError` (PermissionError,
disk error) or `UnicodeDecodeError` (corrupted encoding) — neither is
`yaml.YAMLError`, so these bypass the fail-closed path and propagate to
the capital-class allocator gate.

**TRACE:**
- `chordia.py:351` — `p.read_text()` inside `try/except yaml.YAMLError`
- `lane_allocator.py:315` — `audit_log = load_chordia_audit_log()` inside
  `try/finally:con.close()` (no except — OSError propagates out)
- `lane_allocator.py:735` — same call in `apply_chordia_gate`, no guard

**EVIDENCE:** `chordia.py:350-362` shows only `yaml.YAMLError` caught.
`compute_lane_scores` uses `try/finally` (no except) so any uncaught
exception from `load_chordia_audit_log` exits the function with no scores.

**Fix:** Widened except to `(yaml.YAMLError, OSError, UnicodeDecodeError)`.
Also added `type(exc).__name__` to the warning message for grep-ability.
1 new test `TestLoadChordiaAuditLogIOError` added. Commit 4257c32e.

**Doctrine cited:** integrity-guardian.md § 3 (fail-closed mindset) + § 6
(no silent failures — every except must log the exception)

---

## Seven Sins Scan — chordia.py

- Sin 1 (Silent failure): `load_chordia_audit_log` now correctly catches all
  IO errors and logs at WARNING. FIXED (this iteration).
- Sin 2 (Fail-open): File-missing and YAML-parse error paths both return
  `default_has_theory=False` and empty entries — strictly fail-closed.
  ACCEPTABLE.
- Sin 3 (Canonical violation): Thresholds `CHORDIA_T_WITH_THEORY=3.00` and
  `CHORDIA_T_WITHOUT_THEORY=3.79` are module-level constants with literature
  annotations. `chordia_verdict_label` body uses the constants (not literals).
  Docstring at lines 447-450 contains the literal values `3.00` / `3.79` for
  human readability — documentation-only, no correctness impact. ACCEPTABLE
  per pattern 3 (style difference with no correctness impact).
- Sin 4 (Impact awareness): No hardcoded instrument or entry-model lists found.
  CLEAN.
- Sin 5 (Evidence over assertion): N/A (audit mode).
- Sin 6 (Spec compliance): No `docs/specs/chordia.md`; module docstring is
  the spec and matches behavior exactly. CLEAN.
- Sin 7 (Metadata trust): `chordia_gate` is marked DEPRECATED in its docstring
  and warns correctly. `chordia_verdict_label` is the production interface.
  No stale metadata found.
- Theory_grant vs audit_log independence: VERIFIED. The allocator reads
  `audit_log.verdict(sid)` (the pre-recorded YAML field) at line 340, NOT a
  recomputed value. The `has_theory` field in `theory_map` affects the
  threshold only inside `chordia_verdict_label`, which the live allocator does
  NOT call (it reads the YAML verdict directly). Two independent trust surfaces
  confirmed per AM3.3 doctrine.

---

## Files Fully Scanned

- pipeline/check_drift.py (iter 153)
- pipeline/build_daily_features.py (iter 158)
- pipeline/dst.py (no-touch, iter 160)
- trading_app/strategy_discovery.py (iter 162)
- trading_app/outcome_builder.py (iter 165)
- trading_app/entry_rules.py (iter 168)
- trading_app/strategy_validator.py (iter 171)
- trading_app/live/session_orchestrator.py (iter 174)
- trading_app/live/execution_engine.py (iter 177)
- trading_app/live/alert_engine.py (iter 180)
- trading_app/derived_state.py (iter 183)
- trading_app/deployability.py (iter 193)
- trading_app/strategy_fitness.py (iter 194)
- trading_app/live_config.py (iter 195)
- trading_app/prop_portfolio.py (iter 195, partial — fitness gate path)
- trading_app/lane_correlation.py (iter 196)
- trading_app/lane_allocator.py (iter 197, partial — _classify_status + greedy path)
- trading_app/chordia.py (iter 198, full)

---

## Next Iteration Targets

**Priority 1 — Unscanned high/medium centrality files:**
- `trading_app/prop_portfolio.py` — partially scanned (iter 195); remainder not audited
- `trading_app/lane_allocator.py` — continue remainder of file (apply_c8_gate,
  build_allocation, silent-failure surface in correlation path)

**Priority 2 — Stale re-audits:**
- `trading_app/lane_allocator.py` was modified (iter 197 fix) — re-audit the
  `apply_chordia_gate` path to confirm no regression from the signature change
