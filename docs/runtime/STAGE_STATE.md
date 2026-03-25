---
mode: IMPLEMENTATION
task: FDR K correction — count stop multiplier in K, freeze K at validation time
scope_lock:
  - trading_app/strategy_discovery.py
  - trading_app/strategy_validator.py
  - trading_app/db_manager.py
  - pipeline/init_db.py
acceptance:
  - stop multiplier counted in total_combos (K *= len(STOP_MULTIPLIERS))
  - validated_setups has discovery_k and discovery_date columns
  - 4 Apex lanes still pass BH FDR at corrected K
  - 75/75 drift, 0 regressions
---
