---
mode: IMPLEMENTATION
task: Fix Phase 3 yearly gate bias + SINGLETON threshold for REGIME visibility
scope_lock:
  - trading_app/strategy_validator.py
  - trading_app/config.py
  - scripts/tools/build_edge_families.py
  - tests/test_trading_app/test_strategy_validator.py
  - tests/test_trading_app/test_edge_families.py
  - .claude/hooks/stage-gate-guard.py
blast_radius:
  - trading_app/regime/validator.py (inherits min_trades_per_year default)
  - trading_app/nested/validator.py (inherits min_trades_per_year default)
  - validated_setups table (future runs produce different population)
  - edge_families table (rebuild changes SINGLETON classification)
  - trading_app/walkforward.py (receives different params, no code change)
acceptance:
  - CORE strategies get CORE WF params (unchanged behavior)
  - REGIME strategies get smaller windows + 2 required (both positive)
  - Existing tests pass
  - Drift checks pass
agent: claude-code-main
---

Design: docs/plans/2026-03-31-regime-validation-design.md
