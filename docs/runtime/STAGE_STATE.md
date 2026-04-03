# STAGE_STATE

stage: IMPLEMENTATION
task: Ralph Loop iter 137 — fix fail-open exception handler in ingest_dbn_daily.py per-file loop
classification: judgment
scope_lock:
  - pipeline/ingest_dbn_daily.py
blast_radius: 1 file; DAILY_FILE_PATTERN import in audit_bars_coverage.py unaffected; 1 test file
acceptance_criteria:
  - File processing exception causes immediate sys.exit(1) not continue
  - Successfully processed files committed exactly as before
  - pytest tests/test_pipeline/test_ingest_daily.py passes
  - python -m pipeline.check_drift passes
