mode: IMPLEMENTATION
task: "Fix discovery DELETE to preserve multi-hypothesis-file strategies + unlock GC PDR/GAP filters"
updated: 2026-04-10T20:00:00+10:00
scope_lock:
  - trading_app/strategy_discovery.py
  - trading_app/config.py
blast_radius: "Discovery write path only. No schema change. No validator change. Legacy mode unchanged."
acceptance:
  - "Discovery with hypothesis_sha skips DELETE (INSERT OR REPLACE only)"
  - "Discovery without hypothesis file keeps current DELETE behavior"
  - "GC sessions in _pdr_validated and GAP gate"
  - "Run file 1 then file 3 for GC — both files' strategies persist"
  - "Drift check no new violations from this change"
