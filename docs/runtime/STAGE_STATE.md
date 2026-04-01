---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Code review fixes — stale notes, CopyOrderRouter auth, tier test coverage
updated: 2026-04-01T17:00:00Z
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/live/copy_order_router.py
  - tests/test_trading_app/test_prop_profiles.py
blast_radius:
  - prop_profiles.py: notes string only (no logic change)
  - copy_order_router.py: add self.auth (Liskov compliance)
  - test_prop_profiles.py: add tier DD assertions
acceptance:
  - Notes say $3K not $4K for Tradeify 100K
  - CopyOrderRouter has self.auth set
  - Test asserts Tradeify 100K=$3K, 150K=$4.5K, Apex consistency=0.50
---
