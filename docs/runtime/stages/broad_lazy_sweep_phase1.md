---
task: Phase 1 of broad lazy-import sweep — low-risk CLI modules per docs/plans/2026-04-20-broad-lazy-import-sweep.md
mode: IMPLEMENTATION
scope_lock:
  - trading_app/strategy_discovery.py
  - trading_app/walkforward.py
  - trading_app/strategy_validator.py
  - docs/plans/2026-04-20-broad-lazy-import-sweep.md
blast_radius: trading_app/strategy_discovery.py — CLI entry-point invoked manually for discovery scans; called via subprocess from research scripts. trading_app/walkforward.py — research/backtest framework, called by Phase 2.x research and discovery flows. trading_app/strategy_validator.py — pre-deploy gate, runs once per strategy promotion. All three are CLI / one-shot tools — no hot-loop callers. Common pattern likely: pandas + scipy + duckdb + sklearn imported at module top. Fix pattern: lazy-load heavy deps inside the functions that use them; keep light constants + light imports at module level. No public API changes; existing callers unaffected.
acceptance:
  - each module ≤2s warm import (3-run median, isolated subprocess)
  - python -m <module> --help still works (where applicable)
  - companion tests pass (test_strategy_discovery, test_walkforward, test_strategy_validator if present)
  - python pipeline/check_drift.py exits 0
  - one commit per module with before/after measurements
agent: claude
---

# Stage — broad lazy-sweep Phase 1

Pre-reg plan: `docs/plans/2026-04-20-broad-lazy-import-sweep.md` § Phase 1.

Three modules, one commit each. After each: measure, run companion tests, verify drift. If any falsifies P3-P5, revert and reassess.
