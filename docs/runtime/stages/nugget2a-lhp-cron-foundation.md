---
task: LHP cron foundation — package marker, weekly runner with --dry-run, dedup index init
mode: IMPLEMENTATION
slug: nugget2a-lhp-cron-foundation
scope_lock:
  - scripts/cron/__init__.py
  - scripts/cron/lhp_weekly.py
  - docs/runtime/lhp_dedup_index.json
---

## Blast Radius

- scripts/cron/__init__.py — NEW (empty, package marker)
- scripts/cron/lhp_weekly.py — NEW (~120 lines). Scans docs/audit/hypotheses/ for .yaml files, populates dedup index keyed by file path (full content-hash deferred per plan § Deferred item 4). `--dry-run` mode proves invokability without LLM call.
- docs/runtime/lhp_dedup_index.json — NEW, written by first dry-run as `{}` initially then populated.
- Reads: filesystem (docs/audit/hypotheses/*.yaml). Writes: docs/runtime/lhp_dedup_index.json only (no gold.db writes, no production code mutation).
- Zero changes to llm_hypothesis_proposer.py, lhp/llm_client.py, lhp/static_checks.py, or anything in pipeline/ or trading_app/. Stage 4 will add fitness-transition trigger + GHA schedule.
- 202 hypothesis files currently in docs/audit/hypotheses/ (verified via `ls | wc -l`); plan stated 181 — count has grown since plan write (4-week churn).
