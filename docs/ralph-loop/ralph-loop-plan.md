## Iteration: 92
## Target: scripts/tools/pinecone_snapshots.py:57-62
## Finding: Hardcoded classification thresholds 100/30/99 — should reference CORE_MIN_SAMPLES/REGIME_MIN_SAMPLES from trading_app.config
## Classification: [mechanical]
## Blast Radius: 2 callers (sync_pinecone.py imports generators, test_pinecone_snapshots.py), no behavior change (same values)
## Invariants: SQL query logic unchanged; thresholds remain 100 and 30; output format identical
## Diff estimate: 4 lines (1 import + 3 value replacements)
