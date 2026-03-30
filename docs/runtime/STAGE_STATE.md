---
mode: IMPLEMENTATION
task: Post-re-discovery lane upgrades + timestamp bug fix
phase: 1/2
scope_lock:
  - trading_app/strategy_discovery.py
  - trading_app/prop_profiles.py
  - trading_app/paper_trade_logger.py
blast_radius: strategy_discovery.py _flush_batch_df created_at only; prop_profiles.py DailyLaneSpec L1+L6 swap; paper_trade_logger.py lane ID sync; no schema changes
acceptance:
  - Drift clean
  - timestamp fix verified
  - lane strategy_ids exist in validated_setups
---
