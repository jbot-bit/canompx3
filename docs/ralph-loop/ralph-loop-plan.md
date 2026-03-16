## Iteration: 101
## Target: research/research_mes_compressed_spring.py:42
## Finding: Uses os.environ.get("DUCKDB_PATH", ...) instead of canonical GOLD_DB_PATH from pipeline.paths
## Classification: [mechanical]
## Blast Radius: 1 file (standalone research script, zero importers)
## Invariants: DB_PATH must still resolve to gold.db; sys.path.insert kept; os import removed only (unused after fix)
## Diff estimate: 3 lines (remove import os, add from pipeline.paths import GOLD_DB_PATH, change DB_PATH assignment)
