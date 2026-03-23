---
task: "Post-audit cleanup: stale comments, p_value propagation, doc consistency"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Fix 4 outstanding items from session audits. No logic changes. No threshold changes."
updated: 2026-03-24T01:00+10:00
terminal: main
scope_lock:
  - trading_app/live_config.py
  - HANDOFF.md
  - chatgpt-project-kit/PROJECT_REFERENCE.md
acceptance:
  - "live_config.py stale rolling comment updated"
  - "HANDOFF section 10 removed (duplicate of corrected section 8)"
  - "PROJECT_REFERENCE.md classification matches RESEARCH_RULES.md"
  - "p_values propagated to validated_setups (already done via SQL)"
  - "No logic changes, no threshold changes"
proven:
  - "p_value propagation: 772/772 done, verified matching"
unproven: []
blockers: []
---
