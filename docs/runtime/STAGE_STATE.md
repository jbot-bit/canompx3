---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Post-rebuild deployment — update prop_profiles + paper_trade_logger + commit
updated: 2026-04-02T05:30:00Z
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/paper_trade_logger.py
blast_radius:
  - prop_profiles.py: new honest lane strategy_ids (COST_LT, OVNRNG, ORB_VOL)
  - paper_trade_logger.py: LANES must match prop_profiles
acceptance:
  - All deployed lanes use pre-entry-only filters
  - All strategy_ids exist in validated_setups
  - Tests pass, drift clean
---
