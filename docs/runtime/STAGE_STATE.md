---
mode: IMPLEMENTATION
task: Fix noise floor methodology — per-strategy null + K override for null seeds
scope_lock:
  - trading_app/config.py
  - trading_app/strategy_validator.py
  - pipeline/check_drift.py
  - scripts/infra/revalidate_null_seeds.py
acceptance: |
  Validator accepts fdr_k_overrides (per-session K dict).
  Revalidate script injects real production K.
  Config + drift check updated with methodology notes.
  Tests pass.
---
