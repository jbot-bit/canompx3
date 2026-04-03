## Iteration: 137
## Target: pipeline/ingest_dbn_daily.py:379
## Finding: Outer try/except in the per-file loop catches all exceptions and `continue`s — fail-open: DB commits may proceed with data missing from failed files before the end-of-loop files_failed check can abort.
## Classification: [judgment]
## Blast Radius: 1 file changed; 0 importers change behavior (DAILY_FILE_PATTERN import unaffected); 1 test file (test_ingest_daily.py)
## Invariants:
##   1. Successfully processed files must be committed exactly as before
##   2. Schema validation, PK safety, integrity checks must remain unchanged
##   3. stats counters must remain accurate for the successful-only case
## Diff estimate: 4 lines production code
