---
stage: IMPLEMENTATION
task: Holdout-clean re-discovery + validation (all 3 instruments)
pass: 2
scope_lock:
  - trading_app/strategy_discovery.py
  - trading_app/strategy_validator.py
  - scripts/tools/build_edge_families.py
  - scripts/tools/select_family_rr.py
blast_radius: >
  experimental_strategies and validated_setups tables will be rebuilt.
  Edge families will be rebuilt. No production code changes — only DB writes.
  New confluence filters (22 base) will increase K, potentially changing
  FDR significance for existing strategies. Net validated count may decrease.
acceptance:
  - Discovery completes for MNQ, MES, MGC with --holdout-date 2026-01-01
  - Validation promotes strategies from clean discovery
  - Edge families rebuilt
  - Drift checks 59 + 82 pass
  - All drift checks pass
updated: 2026-03-30T20:00:00+10:00
---
