---
task: Ralph Loop iter 210 — Fix all Pyright type errors in check_drift.py, bot_dashboard.py, derived_state.py
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - trading_app/live/bot_dashboard.py
blast_radius:
  - pipeline/check_drift.py (fetchone None-guards + CRG list assertions — no behavior change)
  - trading_app/live/bot_dashboard.py (_bg_processes dict type widened to Any; casts added at dict.get call sites — no behavior change)
updated: 2026-05-24T00:00:00
agent: ralph
---
