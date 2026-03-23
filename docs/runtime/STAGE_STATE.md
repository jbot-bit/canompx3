---
task: "Guardian audit phase 2: stale docs cleanup"
mode: IMPLEMENTATION
stage: 4
stage_of: 4
stage_purpose: "Class C: Fix stale docs — PLAN_codex, external AI context docs, HANDOFF"
updated: 2026-03-25T01:15+10:00
terminal: main
scope_lock:
  - PLAN_codex.md
  - docs/ai-context/GEMINI.md
  - chatgpt-project-kit/PROJECT_REFERENCE.md
  - chatgpt-project-kit/PROJECT_INSTRUCTIONS.md
acceptance:
  - "All stale volatile counts removed or marked as snapshots with dates"
  - "No new stale counts introduced"
blockers: []
---
