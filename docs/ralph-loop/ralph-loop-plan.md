## Iteration: 216
## Target: pipeline/asset_configs.py:234
## Cluster: 1 finding (annotation_debt), severity=[LOW]
## Classification: [mechanical]
## Blast Radius: 0 callers affected (drift check enforces no raw orb_active reads outside asset_configs.py)
## Invariants: ACTIVE_ORB_INSTRUMENTS = ['MES', 'MGC', 'MNQ'] must not change; DEAD_ORB_INSTRUMENTS must not change; 40 tests must pass
## Diff estimate: 2 lines (comment update + flag value)
## Doctrine cited: integrity-guardian.md § 7 (Never Trust Metadata — orb_active=True on a dead instrument is misleading metadata)
## Findings deferred: NONE (file otherwise clean)
