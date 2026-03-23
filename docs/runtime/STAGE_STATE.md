---
task: "null_envelope.py MAX->P95 tooling consistency fix"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Fix null_envelope.py to compute P95 floor (matching production), not MAX. Tooling-only, no threshold change."
updated: 2026-03-24T01:30+10:00
terminal: review
scope_lock:
  - scripts/tools/null_envelope.py
acceptance:
  - "compute_envelope floor uses P95 not MAX"
  - "update_config_file comment says P95 not noise max"
  - "No production threshold changes"
  - "No changes to NOISE_FLOOR_BY_INSTRUMENT or strategy_validator"
proven: []
unproven: []
blockers: []
---
