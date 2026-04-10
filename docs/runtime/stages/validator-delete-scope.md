---
task: Fix validator Phase C DELETE to scope by strategy_id, not instrument+orb_minutes
mode: IMPLEMENTATION
scope_lock: [trading_app/strategy_validator.py, tests/test_strategy_validator.py]
blast_radius: validated_setups table integrity — affects all future validation runs
acceptance:
  - Phase C DELETE uses strategy_id IN (...) not instrument+orb_minutes
  - Test proves pre-existing validated_setups survive validation of new strategies
  - Existing tests pass
  - Drift checks pass
updated: 2026-04-11T00:00:00Z
---

## Blast Radius
- `strategy_validator.py` Phase C batch write (lines 1616-1644)
- edge_families FK cleanup must also be scoped to processed strategy_ids
- No downstream callers affected — DELETE scope is internal to batch write
