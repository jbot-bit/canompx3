---
mode: IMPLEMENTATION
task: Three hardening fixes — pit_range_atr backfill, routed-filter column drift check, pooled-finding annotation schema
design_doc: docs/plans/2026-04-20-hardening-three-fixes-design.md
stage: 1/3
started: 2026-04-20
---

## Scope Lock

- pipeline/backfill_pit_range_atr.py
- pipeline/build_daily_features.py
- pipeline/ingest_statistics.py
- pipeline/check_drift.py
- tests/test_pipeline/test_check_drift_db.py
- .claude/rules/pooled-finding-rule.md
- docs/audit/results/TEMPLATE-pooled-finding.md
- docs/plans/2026-04-20-hardening-three-fixes-design.md

## Blast Radius

- daily_features.pit_range_atr UPDATE path only
- no schema changes, no trading_app changes, no filter logic changes
- zero live-trade impact (no deployed lane routes PIT_MIN)
- two new drift checks added to pipeline/check_drift.py
- one new rule file, one new template file

## Acceptance

- pipeline/backfill_pit_range_atr.py --all exits 0, reports ≥90% coverage for MES/MGC/MNQ
- daily_features.pit_range_atr population ≥ 0.90 for active instruments
- python pipeline/check_drift.py exits 0 (excluding pre-existing HTF MGC failure)
- pytest tests/test_pipeline/test_check_drift_db.py passes
- memory/exchange_range_signal.md reflects RESOLVED status
