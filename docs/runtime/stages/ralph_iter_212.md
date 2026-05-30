---
task: Ralph Loop iter 212 — replace local PASSING_CHORDIA_VERDICTS copies with canonical chordia_verdict_allows_deploy()
mode: IMPLEMENTATION
scope_lock:
  - trading_app/opportunity_awareness.py
  - trading_app/allocation_promotion.py
updated: 2026-05-30T00:00:00
agent: ralph
---

## Blast Radius
- trading_app/opportunity_awareness.py — removes local frozenset, calls chordia_verdict_allows_deploy() at 2 sites
- trading_app/allocation_promotion.py — removes local set, calls chordia_verdict_allows_deploy() at 1 site
- Reads: trading_app/chordia.py (canonical source, read-only); Writes: neither file changes any test or DB state
