---
task: LHP fitness-transition trigger + GHA schedule (Nugget 2b)
mode: IMPLEMENTATION
slug: nugget2b-lhp-fitness-transition-trigger
scope_lock:
  - scripts/cron/lhp_weekly.py
  - docs/runtime/lhp_fitness_snapshot.json
  - .github/workflows/ci.yml
---

## Blast Radius

- scripts/cron/lhp_weekly.py: MODIFY. Add `--check-fitness-transition` flag. Calls canonical `trading_app.strategy_fitness.compute_portfolio_fitness` per `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`. Diffs aggregated `decay+watch+stale` count vs cached snapshot. First-run guard: no prior snapshot → no transition fires.
- docs/runtime/lhp_fitness_snapshot.json: NEW. Schema `{"as_of": "YYYY-MM-DD", "decay_watch_count": N, "by_instrument": {INSTR: {"fit": N, "watch": N, "decay": N, "stale": N}}}`. Written every fitness-check run.
- .github/workflows/ci.yml: MODIFY. Add `schedule:` trigger cron `0 20 * * 0` (Sun 20:00 UTC = Mon 06:00 Brisbane). Job runs `python scripts/cron/lhp_weekly.py --check-fitness-transition`, gated `if: github.ref == 'refs/heads/main'`.
- Reuse canonical: `compute_portfolio_fitness` (read-only on gold.db); `ACTIVE_ORB_INSTRUMENTS` (instrument iteration); `FitnessReport.summary` (decay+watch+stale counts).
- Reads: gold.db read-only. Writes: docs/runtime/lhp_fitness_snapshot.json only (no DB writes).
- DuckDB concurrency safe: `compute_portfolio_fitness` opens `read_only=True`.
- Risks (explicit): (1) OPENROUTER_API_KEY missing on GHA = LHP propose() exits 3; logged not spammed. Manual post-merge: add secret. (2) Snapshot bootstrap: first run has no prior → treat as `first_run` not `fitness_transition`. (3) DuckDB read concurrency: MVCC-safe; log path for audit.
