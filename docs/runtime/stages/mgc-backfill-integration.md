---
mode: IMPLEMENTATION
task: MGC backfill integration — config fix + ingest + rebuild + rediscovery
updated: 2026-04-09T22:00:00+10:00
scope_lock:
  - pipeline/asset_configs.py
  - pipeline/data_era.py
blast_radius: MGC data only. MNQ/MES untouched. See docs/plans/2026-04-09-mgc-backfill-integration.md
acceptance:
  - bars_1m MIN date for MGC = 2022-06-13 (was 2023-09-11)
  - daily_features row count for MGC increases (~3500+ from ~2229)
  - orb_outcomes row count for MGC increases (~900K+ from ~617K)
  - Drift checks pass (87/0/7 or better)
  - MGC discovery runs with updated hypothesis file
---

## Design

See `docs/plans/2026-04-09-mgc-backfill-integration.md` for full hardened plan.
