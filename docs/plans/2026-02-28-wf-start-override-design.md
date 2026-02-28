# WF Start-Date Override for Regime-Shifted Instruments

**Date:** 2026-02-28
**Status:** Approved
**Approach:** Config-level `WF_START_OVERRIDE` dict (Approach A)

## Problem

MGC's anchored expanding walk-forward starts from 2016 (earliest outcome data). Gold was $1,300-$1,800 from 2016-2022, producing tiny 5-minute ORBs. Under G4+ filters (minimum 4+ points), most 6-month WF windows before 2025 have <15 eligible trades, making them "invalid" per the WF minimum.

**Empirical evidence (MGC CME_REOPEN E2 RR1.0 CB1 G6):**
- 2016-2022: 1-9 trades per half-year (all INVALID, need >=15)
- 2023: 4-6 trades per half-year (INVALID)
- 2024: 4-7 trades per half-year (INVALID)
- 2025-2026: 20-40 trades per half-year (VALID)

Result: Only 3-4 valid windows out of ~20. Most MGC strategies fail WF despite strong recent performance.

**Root cause:** Gold price tripled ($1,300 → $3,500+), fundamentally changing ORB characteristics. The anchored design penalizes MGC for having more data from a different regime.

## Design

### Principle

Full-sample validation (Phase A: sample size, cost, yearly, stress) uses ALL data. Only walk-forward window generation starts from a configurable per-instrument date.

### Data Flow

```
Current:  outcomes(earliest..latest) → skip 12mo → generate windows from month 13+
New:      outcomes(earliest..latest) → max(earliest, override) → skip 12mo → generate windows
```

For MGC with `wf_start_date=2022-01-01`:
- Phase A: Uses 2016-2026 (full 10 years) — unchanged
- Phase B (WF): Anchors from 2022-01-01. First test window: 2023-01-01. Expected ~7 windows, all in high-vol gold era with meaningful trade counts.

### Code Changes (4 files)

**1. `trading_app/config.py`** — New constant
```python
from datetime import date

WF_START_OVERRIDE: dict[str, date] = {
    "MGC": date(2022, 1, 1),  # Gold <$1800 pre-2022 = tiny ORBs, G4+ windows invalid
}
```

**2. `trading_app/walkforward.py`** — New parameter in `run_walkforward()`
- Add `wf_start_date: date | None = None` parameter
- Change window anchor: `anchor = max(earliest, wf_start_date) if wf_start_date else earliest`
- Window generation proceeds from `_add_months(anchor, min_train_months)`

**3. `trading_app/strategy_validator.py`** — Thread override through
- Import `WF_START_OVERRIDE` from config
- `_walkforward_worker()` receives `wf_start_date` and passes to `run_walkforward()`
- `run_validation()` looks up `WF_START_OVERRIDE.get(instrument)` and includes in worker args

**4. Tests** — Verify behavior
- Unit: mock outcomes, confirm window generation with/without override
- Integration: MGC gets override, MNQ/MES/M2K unaffected
- Edge case: override date > latest outcome = no valid windows (fail-closed)

### Post-Implementation Rebuild

Only MGC needs re-validation (outcomes unchanged from tonight's rebuild):
```bash
python trading_app/strategy_validator.py --instrument MGC \
  --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75
python scripts/tools/build_edge_families.py --instrument MGC
```

### Expected Impact

- Phase A survivors: ~286 (unchanged — full sample validation uses all data)
- WF pass rate: Significantly higher — ~7 valid windows instead of 3-4
- FDR significance: More strategies with enough WF windows to show true signal

### Not In Scope

- Rolling window mode (separate future feature if needed)
- Auto-detection of regime shifts
- CLI `--wf-start-date` flag (config is the canonical source)
- Changes to MNQ/MES/M2K (no regime shift detected)
