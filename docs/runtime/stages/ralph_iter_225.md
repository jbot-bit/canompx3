---
task: Ralph Loop iter 225 — db_manager schema parity + hardcoded instrument DEFAULT (DM-225-01, DM-225-02)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/db_manager.py
updated: 2026-06-04T00:00:00
agent: ralph
---

## Blast Radius
- trading_app/db_manager.py — remove instrument DEFAULT 'MNQ' from paper_trades DDL; add 7 EHR columns to verify_trading_app_schema expected_cols sets for validated_setups and experimental_strategies; zero logic change
