---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Allocator UX — report diff, paused collapse, mismatch warning
updated: 2026-04-02T18:00:00Z
scope_lock:
  - trading_app/lane_allocator.py
  - trading_app/pre_session_check.py
  - scripts/tools/rebalance_lanes.py
blast_radius:
  - lane_allocator.py: generate_report gets diff vs current + collapsed paused list
  - pre_session_check.py: warn when deployed lane differs from allocator recommendation
acceptance:
  - Rebalance report shows KEEP/NEW/DROP diff vs prop_profiles
  - Paused list collapsed by reason category (not 115 individual lines)
  - Pre-session check flags deployed vs recommended mismatch
  - Drift clean
---
