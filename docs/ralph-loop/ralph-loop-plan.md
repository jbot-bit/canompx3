## Iteration: 139
## Target: pipeline/build_bars_5m.py:336
## Finding: Fail-open — verify_5m_integrity skipped when row_count == 0, allowing a SQL build defect (source rows exist but INSERT produces 0 rows) to exit with code 0
## Classification: [judgment]
## Blast Radius: 1 file (main() only; build_5m_bars and verify_5m_integrity signatures unchanged; subprocess callers unchanged)
## Invariants:
##   1. dry_run must still skip integrity check
##   2. verify_5m_integrity() signature unchanged
##   3. A genuine no-source-data run (source_count == 0) must still return 0 cleanly (verify will run but find nothing — that is correct behavior)
## Diff estimate: 1 line (remove `and row_count > 0` from the condition)
