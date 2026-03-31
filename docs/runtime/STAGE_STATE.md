---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Build auto-scaling profiles — TYPE-A/TYPE-B for TopStep + Tradeify at 50K and 100K tiers
updated: 2026-04-01T10:00:00Z
scope_lock:
  - trading_app/prop_profiles.py
  - scripts/tmp_tier_analysis.py
blast_radius:
  - prop_profiles.py: consumed by prop_portfolio.py, live_config.py, bot dashboard, pre_session_check
  - DD validation runs at import time — will warn if overbudget
acceptance:
  - 4 new profiles: topstep_50k_type_a, topstep_100k_type_a, tradeify_50k_type_b, tradeify_100k_type_b
  - All lanes from DB-validated best per session x instrument
  - P90 ORB caps set per session x instrument from actual data
  - DD budget computed and documented per profile
  - Import-time DD validation passes or warns (expected for AGGRO profiles)
  - Existing profiles unchanged
  - Drift checks pass
---
