---
task: "Bootstrap noise floor calibration — replace Gaussian null with sign-randomization"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Build and run sign-randomization bootstrap for noise floor. Update config if stable."
updated: 2026-03-24T02:30+10:00
terminal: main
scope_lock:
  - scripts/tools/noise_floor_bootstrap.py
  - trading_app/config.py
  - HANDOFF.md
acceptance:
  - "Bootstrap script runs for all 3 instruments"
  - "Floors stable across 5 seed runs (spread < 0.02)"
  - "Config updated only if stable"
  - "No threshold or gate logic changes"
proven:
  - "PASS 1 verified: MGC sigma 2.50x overshoot, MNQ 1.04x, MES 1.06x"
  - "PASS 1 verified: MGC lag-1 = -0.058 (sign-randomization OK)"
unproven: []
blockers: []
---
