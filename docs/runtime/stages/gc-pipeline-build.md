mode: IMPLEMENTATION
task: "Phase 1: Add GC cost spec and build GC orb_outcomes + daily_features for proxy discovery"
updated: 2026-04-10T17:00:00+10:00
scope_lock:
  - pipeline/cost_model.py
  - trading_app/outcome_builder.py
blast_radius: "Adds GC to COST_SPECS (new entry, no existing code affected). Builds GC orb_outcomes + daily_features in gold.db (new rows, no existing data modified). No production code changes — research branch only."
acceptance:
  - "GC in COST_SPECS with point_value=100"
  - "GC orb_outcomes built (row count > 0)"
  - "GC daily_features built (row count > 0)"
  - "Drift checks pass"
  - "MGC orb_outcomes unchanged (row count same as before)"
