---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Stress test — fix hardcoded values, None crashes, canonical violations across codebase
updated: 2026-03-31T12:00:00Z
scope_lock:
  - scripts/tools/score_lanes.py
  - .claude/hooks/*.py
  - .claude/settings.json
  - .claude/agents/*.md
  - .claude/rules/*.md
  - trading_app/prop_profiles.py
  - trading_app/weekly_review.py
  - scripts/tools/*.py
  - trading_app/live/*.py
  - scripts/run_live_session.py
blast_radius:
  - score_lanes.py: CLI signature change, no external callers
  - hooks: Claude Code infra only
  - prop_profiles: config layer, no DB writes
acceptance:
  - python scripts/tools/score_lanes.py runs without crash (default args)
  - All hooks exit cleanly on valid JSON input
  - python pipeline/check_drift.py passes
  - Comprehensive grep for hardcoded patterns returns clean
---
