---
mode: IMPLEMENTATION
task: Fix Phase 3 yearly gate bias + SINGLETON threshold for REGIME visibility
scope_lock:
  - trading_app/strategy_validator.py
  - scripts/tools/build_edge_families.py
  - tests/test_trading_app/test_strategy_validator.py
  - tests/test_trading_app/test_edge_families.py
  - .claude/hooks/stage-gate-guard.py
blast_radius:
  - trading_app/regime/validator.py (inherits min_trades_per_year default)
  - trading_app/nested/validator.py (inherits min_trades_per_year default)
  - validated_setups table (future runs produce different population)
  - edge_families table (rebuild changes SINGLETON classification)
acceptance:
  - CORE 747 count unchanged in validated_setups
  - Existing tests pass
  - New boundary tests added
  - Drift checks pass
agent: claude-code-main
---

Design: docs/plans/2026-03-31-regime-validation-design.md
