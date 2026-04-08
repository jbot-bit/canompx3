---
mode: IMPLEMENTATION
task: FK-safe DELETE in strategy_discovery.py for Mode A coexistence
updated: 2026-04-09T20:00:00+10:00
scope_lock:
  - trading_app/strategy_discovery.py
  - pipeline/check_drift.py
blast_radius: Single SQL statement change + drift check 18 keyword list fix (pre-existing false positive exposed by multiline SQL).
acceptance:
  - MES discovery runs without FK error (0 protected rows)
  - MNQ discovery runs without FK error (101 protected rows preserved)
  - MGC discovery runs without FK error (0 protected rows)
  - Old grandfathered rows survive in experimental_strategies with hypothesis_file_sha=NULL
  - New Mode A rows have hypothesis_file_sha set
  - Drift checks pass
---

## Design

Change the DELETE statement in strategy_discovery.py batch write section from blanket
`DELETE FROM experimental_strategies WHERE instrument = ? AND orb_minutes = ?`
to FK-safe scoped delete that skips rows referenced by validated_setups.promoted_from.

Per Amendment 2.4, grandfathered research-provisional strategies must survive alongside
new Mode A pre-registered discoveries. Distinguishable by hypothesis_file_sha (NULL vs SHA).
