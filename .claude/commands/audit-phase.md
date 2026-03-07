Run a single audit phase by number. Pass the phase number as argument.

Use when: "run phase 3", "audit phase 5", "just run phase 2"

Usage: /audit-phase <phase_number>

```bash
.venv/bin/python scripts/audits/run_all.py --phase $ARGUMENTS
```

Phases:
- 0: Triage (what changed recently)
- 1: Automated checks (drift, tests, integrity, behavioral)
- 2: Infrastructure config (asset_configs, cost_model, DST, paths)
- 3: Documentation vs reality
- 4: Config sync (filters, entry models, sessions, grids, thresholds)
- 5: Database integrity (schema, row ratios, temporal coverage, orphans)
- 6: Build chain staleness
- 7: Live trading readiness
- 8: Test suite (coverage gaps, stale references)
- 9: Research & script hygiene
- 10: Git & CI
