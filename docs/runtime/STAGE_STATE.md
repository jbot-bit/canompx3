---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: FDR gaps — 5 remaining fixes (test, filter, guard, J clustering, docs)
pass: 1
scope_lock:
  - scripts/tools/audit_fdr_integrity.py
  - tests/test_trading_app/test_strategy_validator.py
  - docs/specs/fdr_methodology.md
blast_radius:
  - audit script: check_6 active filter, empty-guard, check_2/3 docs
  - test file: new test class for discovery_k freeze
  - methodology spec: update J>0.7 honest count after empirical verification
acceptance:
  - GAP-1: test_discovery_k_freeze_on_second_write passes
  - GAP-2: check_6 filters on active instruments
  - GAP-4: empty ACTIVE_ORB_INSTRUMENTS raises at module load
  - GAP-5: J>0.7 cluster count verified empirically
  - GAP-6: check_2/3 output explains post-freeze WARN
  - All tests pass, drift clean
updated: 2026-03-30T19:00:00Z
---
