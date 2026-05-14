---
task: A.6 Fix max_orb_size_pts conflict — key ORB caps by (orb_label, instrument, orb_minutes)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/live/session_orchestrator.py
  - scripts/tools/forward_monitor.py
  - scripts/tools/slippage_scenario.py
  - tests/test_trading_app/test_prop_profiles.py
  - tests/test_trading_app/test_session_orchestrator.py
  - docs/ralph-loop/deferred-findings.md
---

## Blast Radius

- `trading_app/prop_profiles.py` — `get_lane_registry()` key changes from 2-tuple to 3-tuple; capital-path validator; consumed by live + weekly review. Conflict message updated.
- `trading_app/live/session_orchestrator.py` — `_orb_caps` dict re-keyed (line 354) and per-event lookup at line 2153 now passes `strategy.orb_minutes`. This is the live ORB_CAP gate that prevents oversized trades.
- `trading_app/weekly_review.py` — unpacks the new 3-tuple key when iterating registry (line ~320).
- `trading_app/derived_state.py` — verified read-only of `lane.max_orb_size_pts` per-lane; NOT touched.
- `trading_app/pre_session_check.py` — verified iterates per-lane dict not registry; NOT touched.
- Reads: `docs/runtime/lane_allocation.json` (read-only); Writes: none. No DB schema change.
- Tests: add `test_get_lane_registry_per_aperture_caps_coexist` (same session+instrument, different orb_minutes, different caps → both keep their cap), `test_get_lane_registry_same_aperture_conflict_fails` (same session+instrument+orb_minutes, different caps → ValueError), and orchestrator test verifying the right cap is selected by `strategy.orb_minutes`.
- Companion: 214/214 session_orchestrator tests; 27/27 bot_dashboard tests must still pass.
- Adversarial-audit gate REQUIRED (live capital path, judgment-class fix).
