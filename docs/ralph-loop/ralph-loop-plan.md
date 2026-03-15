## Iteration: 62
## Target: scripts/tools/pipeline_status.py:515
## Finding: APERTURES = [5, 15, 30] is a hardcoded local re-definition of VALID_ORB_MINUTES from pipeline/build_daily_features.py — canonical violation (Volatile data / Canonical violation)
## Classification: [mechanical]
## Blast Radius: 0 external callers of APERTURES, 3 internal usages (lines 552, 626, 668), 0 test references to APERTURES
## Invariants: APERTURES values must remain [5, 15, 30] (sourced from canonical); staleness_engine, format_status, and preflight_check logic must not change
## Diff estimate: 2 lines (add 1 import, replace 1 definition)
