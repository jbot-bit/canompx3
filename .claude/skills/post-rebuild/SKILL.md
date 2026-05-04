---
name: post-rebuild
description: Run post-rebuild audit chain after outcome/discovery/validation completes
disable-model-invocation: true
---
Run post-rebuild audit chain after outcome/discovery/validation completes: $ARGUMENTS

Use when: after any rebuild chain completes, "post-rebuild", "rebuild done", "sync up", "finish the rebuild", "audit after rebuild"

## Post-Rebuild Chain

Run AFTER outcome_builder + strategy_discovery + strategy_validator have completed successfully.
This skill handles everything that comes AFTER the core rebuild.

Parse $ARGUMENTS for instrument (default: all active instruments).

### Step 1: Retire E3 Strategies

Validator promotes E3 strategies -- this script retires them:
```bash
python scripts/migrations/retire_e3_strategies.py
```

### Step 2: Build Edge Families

If instrument specified:
```bash
python scripts/tools/build_edge_families.py --instrument $INSTRUMENT
```

If no instrument specified, run for all active:
```bash
python -c "
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
for inst in ACTIVE_ORB_INSTRUMENTS:
    print(inst)
" | while read INST; do
    python scripts/tools/build_edge_families.py --instrument $INST
done
```

### Step 3: Regenerate Repo Map

```bash
python scripts/tools/gen_repo_map.py
```

### Step 4: Audit Gates (ALL must pass)

```bash
python scripts/tools/audit_integrity.py
python pipeline/check_drift.py
python scripts/tools/audit_behavioral.py
python -m pytest tests/ -x -q
```

**ANY failure = STOP.** Fix the issue before proceeding to Step 5.

### Step 5: Surface Promotion Candidates

```bash
python scripts/tools/generate_promotion_candidates.py
```

Opens scorecard HTML in browser. Review candidates and decide which to add to `live_config.py`.
If no candidates found, report "Portfolio fully covered" and continue.

### Step 6: Sync Pinecone Knowledge Base

```bash
python scripts/tools/sync_pinecone.py
```

### Step 7: Report

```
=== POST-REBUILD COMPLETE ===
Instrument:     $INSTRUMENT (or ALL)
E3 retire:      PASS/FAIL
Edge families:  PASS/FAIL
Repo map:       PASS/FAIL
Integrity:      PASS/FAIL
Drift:          PASS/FAIL
Behavioral:     PASS/FAIL
Tests:          PASS/FAIL (N passed)
Candidates:     N found (or "fully covered")
Pinecone sync:  PASS/FAIL
=============================
```

### Rules

- Each step depends on the previous one succeeding
- NEVER skip the audit gates -- they catch regressions from the rebuild
- NEVER skip Pinecone sync -- generated snapshots go stale without it
- Stop on first failure, report which step failed
