"""Pipeline-level test harnesses (null harness + canary contamination suite).

These are importable modules (not pytest files): ``canary_suite`` exposes the
Tier-1 guard-efficacy canaries consumed by ``tests/test_canary/`` and by the
``check_canary_suite_green`` drift gate.
"""
