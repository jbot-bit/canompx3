## Iteration: 100
## Target: scripts/tools/refresh_data.py (uncommitted changes)
## Finding: Clean uncommitted changes adding 2YY/ZT research-only instrument support — correct use of ASSET_CONFIGS.orb_active gate, no canonical violations.
## Classification: [mechanical]
## Blast Radius: 1 file (no external callers)
## Invariants: ACTIVE_ORB_INSTRUMENTS unchanged, existing active instruments unaffected, orb_active gate used correctly
## Diff estimate: ~28 lines (all in scripts/tools/refresh_data.py — below 20-line cap for production code... wait: this is a tools script in scripts/tools/ not in pipeline/ or trading_app/)
