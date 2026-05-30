# Powered-OOS Stage 2 — hypothesis_loader trade_fraction wiring

task: Teach trading_app/hypothesis_loader.py to parse + validate an optional
  oos_power_floor block declaring holdout_method: trade_fraction, with a schema-level
  guardrail that a fraction split is NOT a deployment gate (Criterion 8 stays required).
mode: IMPLEMENTATION

## Scope Lock
- trading_app/hypothesis_loader.py
- tests/test_trading_app/test_hypothesis_loader.py
- docs/prompts/prereg-writer-prompt.md

## Blast Radius
- trading_app/hypothesis_loader.py — ADDITIVE validator `_validate_oos_power_floor(meta, path)`
  called from load_hypothesis_metadata before the return dict (insert near line 361).
  Mirrors the existing conditional_role validation pattern (lines 340-359). OPT-IN:
  fires ONLY when oos_power_floor.holdout_method key is present, so the ~6 existing
  preregs that declare oos_power_floor WITHOUT holdout_method load UNCHANGED (zero
  regression). No new required key. No schema/DB. No capital path.
- tests/test_trading_app/test_hypothesis_loader.py — new validation tests: accept valid
  trade_fraction block; reject unknown holdout_method; reject trade_fraction without
  target_tier; reject trade_fraction missing the not-a-deploy-gate ack; legacy block
  (no holdout_method) still loads.
- docs/prompts/prereg-writer-prompt.md — document the optional oos_power_floor block +
  the trade_fraction guardrail so generated preregs conform.
- Reads: existing prereg YAMLs (read-only at test time); Writes: none.

## Approved Design (operator-confirmed 2026-05-31)
- Validator strictness: OPT-IN — only validate when oos_power_floor.holdout_method present.
- holdout_method allowed set: {calendar, trade_fraction}.
- trade_fraction REQUIRES: target_tier (in oos_power POWER_TIERS names) AND an explicit
  not-a-deployment-gate acknowledgment (e.g. deployment_gate: false, or a field citing
  Criterion 8 as still-required). Loader REJECTS a fraction split that implies deploy
  readiness — encodes AFML 2018 §12.2 (single-path pitfall) + pre_registered_criteria.md
  Criterion 8 (forward-OOS NOT demoted) at the schema level.

## Literature grounding (Stage 1 carried this in the helper docstring; Stage 2 carries it into the loader)
- Harvey-Liu 2015 (literature/harvey_liu_2015_backtesting.md p.17): OOS != binary veto.
- AFML 2018 §12.2 (literature/lopez_de_prado_2018_afml_ch_3_7_8.md): single train-then-
  trailing-OOS split is WF pitfall #1 (overfit-prone). trade_fraction is a BETTER
  single-path OOS, NOT a CPCV (§12.4) substitute, NOT a Criterion 8 clearance.
- docs/institutional/pre_registered_criteria.md Criterion 8 / Amendment 3.5: forward-OOS
  REQUIRED for deployment, explicitly NOT demoted.

## Done = (all four)
1. tests pass (show output) — new loader validation tests + full test_hypothesis_loader.
2. dead code swept (grep -r).
3. python pipeline/check_drift.py passes (--fast --skip-crg-advisory acceptable; note
   step-2b gold.db lock-race is a known false blocker — prove canonical via dry-run).
4. self-review passed (validator can't reject legacy blocks; can't pass a deploy-claiming
   fraction split).

## State: DONE. Stage 1 committed 0bd9fcf9. Stage 2 implemented + verified:
## tests 80/80 (10 new + parity guard), ruff clean, drift 152/0, dead code swept,
## self-review passed (null/string/0/identity edge cases verified vs yaml.safe_load),
## zero regression on 122 legacy preregs.
