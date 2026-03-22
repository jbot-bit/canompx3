---
task: "Add 2026 holdout enforcement drift check"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Single drift check in check_drift.py to detect holdout contamination"
updated: 2026-03-22T19:00+10:00
terminal: main
scope_lock:
  - pipeline/check_drift.py
acceptance:
  - "New drift check detects re-discovery with 2026 data after pre-registration"
  - "Current state passes (created_at 2026-03-19 < declaration 2026-03-20)"
  - "No false positives on existing data"
proven:
  - "MNQ experimental_strategies created_at = 2026-03-19 (pre-declaration)"
  - "Pre-registration date = 2026-03-20"
unproven: []
blockers: []
---
