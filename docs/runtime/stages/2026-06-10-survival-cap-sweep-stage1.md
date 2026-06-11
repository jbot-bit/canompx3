---
task: "Survival-cap sweep Stage 1 — sweep_survival_cap() + C11 envelope persist + blocking AST drift guard, non-binding at DEPLOYED_MAX_CONTRACTS_CLAMP=1"
mode: CLOSED
scope_lock:
  - trading_app/account_survival.py
  - pipeline/check_drift.py
  - tests/test_trading_app/test_account_survival_sweep.py
  - tests/test_pipeline/test_check_drift_survival_cap_sweep_guard.py
blast_radius: |
  - trading_app/account_survival.py — ADD sweep_survival_cap() (additive; reuses _scenarios_for_context + simulate_survival, NO new sim math). ADD survival_safe_ceiling + per-n probs to the C11 envelope payload via a sweep-persist helper. evaluate_profile_survival / check_survival_report_gate behavior UNCHANGED at clamp=1 (sweep is a separate on-demand entry; payload key additive, readers use .get()).
  - pipeline/check_drift.py — ADD one BLOCKING check: any active-profile live max_contracts > 1 must trace to a persisted swept ceiling with live_cap <= swept_ceiling; else BLOCK. Fail-direction: false-BLOCK safe, false-PASS forbidden (feedback_capital_guard_fail_direction_matters). No-op PASS at clamp=1 (no non-1 live cap exists).
  - Reads: gold.db read-only (sim already does). Writes: only the existing C11 JSON state file (same writer path). Capital impact at clamp=1: NONE — DEPLOYED_MAX_CONTRACTS_CLAMP=1 gates the live order path; this adds evidence only.
  - CRIT/HIGH truth-layer + capital-path → adversarial-audit gate MANDATORY before commit.
---

## Grounded seams (verified against ebb122ed, 2026-06-10)

- Cap lever: `SizingContext.max_contracts_for(strategy_id)` — account_survival.py:213 (default 1, fail-closed).
- Sweep building block: `_scenarios_for_context(con, ..., size_model=SizingContext(...))` — account_survival.py:763, returns `(scenarios, metadata)`. Vary ONLY size_model per n.
- Sim: `simulate_survival(scenarios, rules, ...)` — account_survival.py:892. Reused unchanged.
- MAE fix CONFIRMED on main: `risk_points = t.risk_points` (:595), not mae-derived. Honest foundation.
- Clamp: `DEPLOYED_MAX_CONTRACTS_CLAMP = 1` — portfolio.py:94.
- C11 envelope: `build_state_envelope(...)` payload written by `evaluate_profile_survival` (:1172-1195); read by `read_survival_report_state` (:302) / `check_survival_report_gate` (:1200); drift consumer check_drift.py:7882.
- No existing `sweep_survival_cap` anywhere (grep clean) — net-new.

## Decisions (operator GO, 2026-06-10)

1. Persistence: EXTEND the existing C11 envelope (one fingerprint-guarded file; .get() backward-compatible).
2. Invocation: ON-DEMAND only — NOT folded into evaluate_profile_survival (zero added cost to existing path).

## Done criteria

- tests pass (show output) — cap=1 byte-parity + synthetic ceiling-flip + drift known-violation injection.
- dead code swept (grep -r).
- python pipeline/check_drift.py passes.
- self-review + adversarial-audit gate (CRIT/HIGH capital path).

## CLOSED (2026-06-11)

Prod code (`sweep_survival_cap` / `_evaluate_gate` / `_contiguous_safe_ceiling` /
`_persist_sweep_into_c11_envelope` + blocking drift guard) shipped on origin/main
in an earlier commit (`89350b44`). This close-out landed the OWED test layer:

- Guard test was ALREADY done at `tests/test_pipeline/test_check_drift_survival_cap_sweep_guard.py`
  (10/10) — only the scope_lock path above was stale.
- NEW behavior test `tests/test_trading_app/test_account_survival_sweep.py` (11/11):
  cap=1 byte-parity vs single `_evaluate_gate`, contiguous-ceiling flip, fail-closed
  at cap=1 fail, non-contiguous-pass-not-honored, ceiling<1 raises, and 3
  `_persist_sweep_into_c11_envelope` round-trip/raise cases. `_evaluate_gate` +
  `_contiguous_safe_ceiling` run REAL (never re-encoded); flip lever is the
  `simulate_survival` result dict's `operational_pass_probability`.
- Gates: drift 187/0; adversarial-audit gate (evidence-auditor) = CLEAN on all
  three criteria (false-PASS direction, mock fidelity, no-vacuous-pass).

Still gated: any clamp lift past `DEPLOYED_MAX_CONTRACTS_CLAMP=1` must run the sweep
at production `n_paths=10_000` (the tests use 16 for speed — proves routing, not
convergence) and is a separate operator decision.
