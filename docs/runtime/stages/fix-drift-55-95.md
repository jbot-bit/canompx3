---
task: Fix pre-existing drift check failures 55 and 95
mode: IMPLEMENTATION
stage: 1/1
scope_lock:
  - pipeline/check_drift.py
  - trading_app/prop_profiles.py
blast_radius: Drift check 55 (cost model ranges), drift check 95 (profile alignment). No runtime behavior change — check_drift is validation-only, prop_profiles changes lane references.
updated: 2026-04-10T18:00:00+10:00
---

## What
Fix two pre-existing drift check failures:
1. Check 55: GC cost model outside micro-futures ranges (GC is proxy-only, not traded)
2. Check 95: topstep_50k_mnq_auto references 5 stale G8 lanes not in validated_setups

## Approach
1. Check 55: Skip non-ACTIVE_ORB_INSTRUMENTS in cost model range check
2. Check 95: Replace 5 stale G8 lanes with 5 currently validated multi-RR strategies
