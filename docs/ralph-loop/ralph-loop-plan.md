## Iteration: 63
## Target: scripts/tools/pipeline_status.py:605
## Finding: Silent failure — bare `except Exception` swallows unexpected errors when querying family_rr_locks without instrument filter; silently returns None with no log
## Classification: [judgment]
## Blast Radius: 0 external callers (staleness_engine used only in main()); family_rr_locks is display-only (not in stale_steps logic); 1 file changed
## Invariants: result["family_rr_locks"] must remain in returned dict; when table has no instrument column AND table doesn't exist, result must be None (not exception); duckdb.BinderException outer catch must not change
## Diff estimate: 1 line (except Exception -> except duckdb.CatalogException)
