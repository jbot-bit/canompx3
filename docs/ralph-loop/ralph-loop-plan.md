## Iteration: 135
## Target: scripts/databento_backfill.py:133-139 + scripts/tools/refresh_data.py:248-249
## Finding: Silent failure — (1) load_manifest() crashes on corrupt JSON with unhandled JSONDecodeError; (2) cleanup unlink exception is silently swallowed with no log
## Classification: [judgment]
## Blast Radius: 0 external callers (both are standalone CLI tools); load_manifest called from 1 place in same file (run_download:373); unlink except called from 1 place in refresh_instrument
## Invariants:
##   1. load_manifest must return a valid dict (same interface); corrupt manifest -> fresh empty dict
##   2. refresh_instrument must still return False after ingest failure (fail-closed not changed)
##   3. save_manifest path and manifest schema are unchanged
## Diff estimate: 6 lines production code
