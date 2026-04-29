# CRG Phase 2 — Drift Checks

mode: IMPLEMENTATION
**Date:** 2026-04-29

## Task

Phase 2 of CRG integration spec (docs/plans/2026-04-29-crg-integration-spec.md):
- D1-D5: 5 new advisory drift checks backed by CRG graph queries
- D6: /crg-lineage slash command
- Helper module isolating all CRG subprocess calls
- Test coverage (fixture-based, no live graph required)

## Scope Lock

- pipeline/check_drift_crg_helpers.py
- pipeline/check_drift.py
- .claude/commands/crg-lineage.md
- tests/test_pipeline/test_check_drift_crg.py

## Blast Radius

`check_drift.py`: grows by ~120 lines (new check functions + CHECKS registrations). No existing check modified. All 5 new checks are ADVISORY — zero commit-blocking risk.

`check_drift_crg_helpers.py`: new file, imported only by check_drift.py. No production path touches it.

Tests: fixture-based, no live graph DB. Isolated to test_check_drift_crg.py.
