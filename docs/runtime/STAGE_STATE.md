---
task: "Master guardian audit: fix fail-open defects, stale narrative, canonical violations, test staleness"
mode: IMPLEMENTATION
stage: 2
stage_of: 4
stage_purpose: "Phase 2: Fix stale narrative across docs/memory/agents (instrument counts, session counts, dead advice)"
updated: 2026-03-24T21:30+10:00
terminal: main
scope_lock:
  - trading_app/config.py
  - trading_app/ml/predict_live.py
  - trading_app/ml/config.py
  - trading_app/live/multi_runner.py
  - TRADING_RULES.md
  - .claude/agents/verify-complete.md
  - .claude/agents/ralph-loop.md
  - .claude/rules/validation-workflow.md
acceptance:
  - "All '4 active instruments' references corrected to 3 or canonical"
  - "config.py stale live strategy counts removed"
  - "TRADING_RULES.md session count corrected to 12"
  - "Memory files marked stale or corrected"
  - "No new drift introduced"
blockers: []
---
