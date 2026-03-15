## Iteration: 95
## Target: scripts/tools/ml_cross_session_experiment.py:26, scripts/tools/ml_hybrid_experiment.py:25, scripts/tools/ml_instrument_deep_dive.py:24
## Finding: SESSION_ORDER missing EUROPE_FLOW in all three ML experiment scripts — EUROPE_FLOW trades silently skipped in cross-session feature computation
## Classification: [mechanical]
## Blast Radius: 3 files changed (standalone experiment scripts, no production callers)
## Invariants: SESSION_ORDER must remain in chronological Brisbane-time order; EUROPE_FLOW inserts between SINGAPORE_OPEN and LONDON_METALS; no logic changes
## Diff estimate: 3 lines added (one per file)
