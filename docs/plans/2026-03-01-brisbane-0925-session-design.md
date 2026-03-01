# Design: BRISBANE_0925 Session

**Date:** 2026-03-01
**Status:** Draft
**Scope:** Add fixed-Brisbane-time 09:25 session for MNQ

---

## Background

Session discovery scan (`research_session_discovery.py`) tested 288 candidate times
across 4 instruments with 5 RR targets and 3 G-filters (14,440 combos).
BH FDR correction at q=0.05 across all combos.

**Finding:** MNQ at 09:25 Brisbane is the only non-existing-session time with FDR survivors.

| Metric | 09:00 (CME_REOPEN) | 09:25 (candidate) |
|--------|--------------------|--------------------|
| Raw avgR (RR2.5 G4) | -0.0149 | +0.3169 |
| Sharpe (annualized) | -0.14 | 2.95 |
| FDR survivors | 0/15 | 15/15 |
| N trades | 1,338 | 1,275 |
| Head-to-head | 0/15 | 15/15 wins |

**Verified:** Imported actual `scan_time()` and ran against raw bars_1m.
Reproduces exact CSV numbers. Zero lookahead (close-based break, entry at break bar close).

---

## DST Analysis (Critical)

CME reopens at 5:00 PM CT. In Brisbane:
- **US Winter (CST):** 5 PM CT = 09:00 Brisbane → 09:25 is 25 min after reopen
- **US Summer (CDT):** 5 PM CT = 08:00 Brisbane → 09:25 is 85 min after reopen

**Test:** Is the edge event-relative (post-reopen consolidation) or fixed-time?

| Time | DST Season | avgR | N |
|------|-----------|------|---|
| 09:25 Brisbane | Winter | +0.24R | — |
| 09:25 Brisbane | Summer | +0.36R | — |
| 08:25 Brisbane (event-relative summer) | Summer | +0.005R | — |

**Conclusion:** Edge is at the **fixed Brisbane clock time**, not relative to CME reopen.
Summer is BETTER despite being 85 min after reopen. 08:25 (event-relative) is noise.

**Resolver:** `return (9, 25)` — no DST logic needed. Simplest possible resolver.

---

## Design Decisions

### Session Name: `BRISBANE_0925`

No market event anchors this time. "POST_CME_REOPEN" would be misleading (85 min in summer).
Naming follows the pattern of being honest about what the session IS.

### Break Group: `"cme"` (MANDATORY)

Break groups prevent adjacent sessions from truncating each other's break detection windows.
Sessions in the **same** group share a boundary; sessions in **different** groups create boundaries.

If BRISBANE_0925 were in its own group:
- CME_REOPEN break window would be truncated from 55 min → 20 min (winter). **Devastating.**

In `"cme"` group:
- CME_REOPEN break window: UNCHANGED (still extends to TOKYO_OPEN "asia" at 10:00)
- BRISBANE_0925 break window: 9:30 → 10:00 (TOKYO_OPEN) = 30 min. Adequate.

### ORB Duration: 5 minutes

The scan tested 5-minute ORBs. Matches all sessions except TOKYO_OPEN (15m).

### Instrument Scope: MNQ only

The scan only found FDR survivors for MNQ at this time. Other instruments can be added
later if independent evidence emerges.

### Entry Models & Filters

No special configuration. Pipeline will test standard E1/E2 entry models with
`BASE_GRID_FILTERS` (G4/G5/G6/G8 size filters, volume filters).
The scan approximated E1 CB1 — pipeline E2 may perform differently.

### Early Exit: None initially

T80 time-stop data requires outcome data that doesn't exist yet.
Set `EARLY_EXIT_MINUTES = None`. Research winner speed after outcomes are built.

### DOW Alignment: Aligned

09:25 Brisbane falls at the start of the Brisbane trading day.
Same calendar day as CME trading session. Brisbane DOW = Exchange DOW.

---

## Files to Change

### 1. `pipeline/dst.py`

**Add resolver function** (before SESSION_CATALOG):
```python
def brisbane_0925_brisbane(trading_day: date) -> tuple[int, int]:
    """Fixed 09:25 AM Brisbane session. No market event anchor.

    Session discovery scan (2026-03-01) found this fixed clock time
    has positive raw expectancy for MNQ across both DST seasons.
    Not event-relative to CME reopen.
    """
    return (9, 25)
```

**Add to SESSION_CATALOG:**
```python
"BRISBANE_0925": {
    "type": "dynamic",
    "resolver": brisbane_0925_brisbane,
    "break_group": "cme",
    "event": "Fixed 9:25 AM Brisbane (not event-relative)",
},
```

**Add to DST_CLEAN_SESSIONS** (fixed time = inherently DST-clean)

**Add to DOW_ALIGNED_SESSIONS** (same calendar day)

### 2. `pipeline/init_db.py`

**Add to ORB_LABELS_DYNAMIC:**
```python
ORB_LABELS_DYNAMIC = [
    "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
    "US_DATA_830", "NYSE_OPEN", "US_DATA_1000", "COMEX_SETTLE",
    "CME_PRECLOSE", "NYSE_CLOSE",
    "BRISBANE_0925",
]
```

This auto-generates 14 daily_features columns. Migration adds columns if table exists.

### 3. `pipeline/asset_configs.py`

**Add to MNQ's `enabled_sessions`:**
```python
"enabled_sessions": [
    "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
    "US_DATA_830", "NYSE_OPEN", "US_DATA_1000",
    "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE",
    "BRISBANE_0925",
],
```

MNQ only. No changes to MGC/MES/M2K.

### 4. `trading_app/config.py`

**Add to three dicts:**
```python
ORB_DURATION_MINUTES["BRISBANE_0925"] = 5
EARLY_EXIT_MINUTES["BRISBANE_0925"] = None
SESSION_EXIT_MODE["BRISBANE_0925"] = "fixed_target"
```

No special filter logic in `get_filters_for_grid()` — returns `BASE_GRID_FILTERS` by default.

### 5. Tests

Update any tests that assert session counts or enumerate session names.

---

## Automatic Validation

These checks fire automatically and catch mistakes:

| Check | What It Validates |
|-------|-------------------|
| Drift #32 | ORB_LABELS ↔ SESSION_CATALOG sync |
| `build_daily_features.py` import guard (L81-89) | Every ORB label classified in DST sets |
| `validate_catalog()` | No permanent time collisions between sessions |
| `validate_dow_filter_alignment()` | DOW filter safety for misaligned sessions |

---

## Rebuild Chain (After Code Changes)

```bash
# 1. Add columns to database
python pipeline/init_db.py

# 2. Rebuild daily features for MNQ (populates BRISBANE_0925 columns)
python pipeline/build_daily_features.py --instrument MNQ --start 2023-01-01 --end 2026-03-01

# 3. Rebuild outcomes (requires orb_outcomes table — may need full rebuild)
python trading_app/outcome_builder.py --instrument MNQ --force --start 2023-01-01 --end 2026-03-01

# 4. Discovery
python trading_app/strategy_discovery.py --instrument MNQ

# 5. Validation
python trading_app/strategy_validator.py --instrument MNQ --min-sample 50 \
  --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward

# 6. Edge families
python scripts/tools/build_edge_families.py --instrument MNQ
```

Note: `--no-walkforward` is mandatory for MNQ (only 2 years of data).

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Break window too short (30 min) | Most ORB breaks occur within 15 min. Monitor N vs scan's 1,275 |
| CME_REOPEN results change | break_group="cme" prevents this. Verify post-rebuild |
| Pipeline finds fewer strategies than scan | Expected — pipeline is stricter (E1/E2 models, more filters) |
| orb_outcomes table missing | Must rebuild via outcome_builder before discovery |
| Collision with CME_REOPEN | validate_catalog() catches permanent collisions. 9:25 ≠ 9:00/8:00 |

---

## Success Criteria

1. Drift checks pass (especially #32)
2. `build_daily_features.py` populates BRISBANE_0925 columns without error
3. outcome_builder produces outcomes for BRISBANE_0925
4. At least some strategies survive BH FDR in discovery
5. CME_REOPEN validated strategy counts unchanged
