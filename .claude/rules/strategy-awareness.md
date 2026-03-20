# Strategy Awareness — Blueprint Routing

Before any strategy, research, or trading work:

1. **Route via `docs/STRATEGY_BLUEPRINT.md`** — §3 test sequence, §5 NO-GO registry, §10 assumptions
2. **Variable coverage:** Before declaring ANY dimension dead, test ≥3 values (Blueprint §3 Gate 2)
3. **System is not proven right — just not-yet-wrong.** Every finding is provisional.

Key state (re-verify if stale):
- MNQ E2 = only positive unfiltered baseline. MGC/MES need size filters.
- ML works at RR2.0 O30 (bootstrap p≤0.02). Portfolio-level negative ML = DEAD (p=0.35).
- `filter_type` must match `ALL_FILTERS` exactly. Unknown strings = silent trade drops.
- 2026 holdout is SACRED. 3 pre-registered strategies only.

Canonical sources and hard lessons → see `integrity-guardian.md` (don't duplicate here).
