---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Calendar Cascade Scanner — Design Document

**Date:** 2026-03-08
**Status:** Approved
**Bloomey Grade:** B+ (6 action items incorporated below)
**Research basis:** `research/research_calendar_effects.py` (Mar 2026, 416 BH FDR tests)

---

## 1. Problem Statement

### What's broken
The production execution engine (`trading_app/execution_engine.py:199`) defaults to
`CALENDAR_SKIP_NFP_OPEX` — a blanket filter that skips ALL NFP and OPEX days for
ALL instruments across ALL sessions. This was built on a stale E0-era assumption
that NFP/OPEX days are "universally toxic."

Mar 2026 comprehensive analysis (416 tests, BH FDR q=0.10) proved this wrong:

**Skipping NFP costs edge on:**
- MES × US_DATA_1000 (EXPLOIT — better on NFP days)
- MNQ × NYSE_OPEN (EXPLOIT)
- MGC × US_DATA_1000 (EXPLOIT)

**Skipping OPEX costs edge on:**
- MNQ × NYSE_OPEN (+0.201R diff, 100% year consistency)
- MES × NYSE_CLOSE (+0.464R diff, 83% consistency)
- MES × LONDON_METALS, MES × NYSE_OPEN (EXPLOIT)

**Legitimate AVOID signals exist but are session-specific:**
- NFP × MNQ × CME_PRECLOSE (-0.365R, 100% consistency)
- OPEX × MGC × NYSE_OPEN (-0.164R, 82% consistency)
- NFP × MGC × TOKYO_OPEN, NFP × M2K × NYSE_OPEN

The blanket skip throws away good edge alongside bad. The correct approach is
per-instrument×session calendar rules.

### Two production bugs (independent of cascade design)
1. **Wiring bug:** `paper_trader.py:258-266` constructs `ExecutionEngine()` WITHOUT
   passing `calendar_overlay`. The engine falls back to its default (`CALENDAR_SKIP_NFP_OPEX`),
   so the CLI `--calendar-filter` flag has NO effect on runtime skipping.
2. **Harmful default:** `execution_engine.py:199` defaults to `CALENDAR_SKIP_NFP_OPEX`.
   Any code that creates an `ExecutionEngine()` without explicitly passing `calendar_overlay=None`
   silently applies the wrong blanket skip.

---

## 2. Design Goals

1. Replace blanket NFP/OPEX skip with per-instrument×session calendar rules
2. Support three actions: SKIP (don't trade), HALF_SIZE (reduced weight), NEUTRAL (normal)
3. Keep hypothesis budget under 30 total tests (vs 416 in flat approach)
4. Use per-day aggregated PnL to eliminate correlation inflation from multiple strategies
5. Fail-open-to-trade: missing rules = trade normally, never skip on error
6. Fix the two production bugs regardless of cascade implementation

---

## 3. Architecture

### 3.1 Two-Phase Design

**Phase 1: Research — Cascade Scanner** (`research/research_calendar_cascade.py`)
Offline analysis script. Runs the portfolio-level cascade test, produces a frozen
JSON rules file. Re-run when validated_setups changes significantly (quarterly).

**Phase 2: Production — Calendar Overlay** (`trading_app/calendar_overlay.py`)
Lightweight module that loads the frozen JSON and exposes `get_calendar_action()`.
No statistics at runtime — just a lookup.

### 3.2 Cascade Testing Method (Harvey, Liu & Zhu)

The key insight: don't test 416 hypotheses. Test at portfolio level first, decompose
only where rejected.

```
LEVEL 1: PORTFOLIO-WIDE (10 tests)
  For each calendar signal (NFP, OPEX, FOMC, CPI, ...):
    Pool ALL instrument×session daily PnLs
    t-test: mean(signal_days) vs mean(non_signal_days)
    BH FDR at q=0.10 across the 10 tests

  NOT REJECTED → classify as NEUTRAL, stop decomposing
  REJECTED → proceed to Level 2

LEVEL 2: PER-INSTRUMENT (up to 4 tests per rejected signal)
  For each instrument (MGC, MNQ, MES, M2K):
    Pool ALL sessions for that instrument
    t-test: signal days vs non-signal days
    BH FDR at q=0.10 across this level's tests

  NOT REJECTED → instrument-wide NEUTRAL
  REJECTED → proceed to Level 3

LEVEL 3: PER-SESSION (up to 11 tests per rejected instrument)
  For each session:
    t-test: signal days vs non-signal days
    BH FDR at q=0.10 across this level's tests
    Year-by-year consistency check

  Classify based on thresholds (see 3.4)
```

**Total hypothesis budget:** 10 + (rejected × 4) + (rejected × ~11) ≈ 25 typical,
never exceeds ~60 even if everything rejects. Compare to 416 flat.

### 3.3 Per-Day Aggregation (Bloomey Critical Fix)

The existing `research_calendar_effects.py` tests at the strategy-outcome level.
If 20 MNQ NYSE_OPEN strategies fire on the same NFP day, that's 20 correlated
observations counted as independent. This inflates N and deflates p-values.

**The cascade scanner MUST aggregate first:**

```python
# WRONG (existing script): strategy-level rows
# 20 strategies × 50 NFP days = 1000 "independent" observations

# RIGHT (cascade scanner): one number per day
daily_agg = df.groupby(["instrument", "session", "trading_day"])["pnl_r"].mean()
# 1 × 50 NFP days = 50 actual independent observations
```

This means the cascade scanner's p-values will be LESS significant than the flat
script's. That's correct — the flat script's p-values are overconfident. Honesty
over outcome.

### 3.4 Signal Classification Thresholds

| Action | Criteria | Numeric threshold |
|--------|----------|-------------------|
| **SKIP** | BH q<0.10 + ≥75% year consistency + mean diff ≤ -0.15R | Don't trade this day |
| **HALF_SIZE** | BH q<0.10 + ≥60% year consistency + mean diff < 0 | Trade at 0.5x weight |
| **NEUTRAL** | Everything else (including EXPLOIT signals) | Trade normally |

**Why no EXPLOIT/BOOST action?** Adding a "trade more on good days" action is
data-snooping the positive tail. We only act on the negative tail (reduce/skip).
Positive signals just mean "don't skip" — which NEUTRAL already handles.

**Year-by-year consistency:** For each year with sufficient data (≥5 on-days,
≥20 off-days), check if the direction matches. 6/8 years = 75% = CONSISTENT.
5/8 = 62.5% = WEAK. 4/8 = 50% = NOISE.

### 3.5 HALF_SIZE Mechanism

**Current engine limitation:** `CalendarSkipFilter` is binary — match (skip) or
don't (trade). There's no concept of partial sizing.

**Solution:** Add `size_multiplier: float` to `ActiveTrade` dataclass.

```python
class CalendarAction(Enum):
    SKIP = 0.0       # Don't trade
    HALF_SIZE = 0.5  # Trade at half weight
    NEUTRAL = 1.0    # Normal

# In ExecutionEngine._arm_strategies():
action = get_calendar_action(instrument, session, trading_day)
if action == CalendarAction.SKIP:
    continue  # skip
# Otherwise proceed, store multiplier
active_trade.size_multiplier = action.value  # 0.5 or 1.0
```

**Paper trader impact:** `pnl_r *= size_multiplier` when recording journal entry.
This correctly reduces the portfolio weight of half-size trades in aggregate stats.

**Live trading reality:** Micro futures = 1 contract minimum. Can't trade 0.5 contracts.
For live trading, HALF_SIZE is logged for tracking but the trade is taken at 1 contract.
The portfolio-level expectation adjusts for this. Future enhancement: could skip if
only 1 contract and HALF_SIZE, or apply to position count when trading 2+ contracts.

### 3.6 Error Handling

| Scenario | Behavior | Rationale |
|----------|----------|-----------|
| `calendar_cascade_rules.json` missing | `CALENDAR_RULES = {}` → all NEUTRAL | Fail-open-to-trade |
| Unknown instrument×session in lookup | Not in dict → NEUTRAL | New sessions trade normally |
| Multiple signals fire same day (NFP + Friday) | Most restrictive wins: SKIP > HALF_SIZE > NEUTRAL | Conservative |
| `get_calendar_action()` raises exception | Catch at engine level, log warning, return NEUTRAL | Never skip on error |
| Cascade scanner finds zero BH survivors | Empty JSON → all NEUTRAL | Blanket skip removed, nothing needed |
| Year-by-year data too short (<4 years) | Skip consistency check, classify NOISE | Insufficient evidence |
| Calendar date lists (FOMC/CPI) run out | Signal not computed for future dates → NEUTRAL | Degrade gracefully |

### 3.7 `get_calendar_action()` Interface

```python
def get_calendar_action(
    instrument: str,
    session: str,
    trading_day: date,
) -> CalendarAction:
    """
    Look up the calendar action for this instrument×session×day.

    Checks all calendar signals that fire on trading_day (NFP, OPEX, FOMC, etc.),
    looks up each in CALENDAR_RULES, returns the most restrictive action.

    Returns NEUTRAL if no rules match or if CALENDAR_RULES is empty.
    """
```

This replaces the entire `CalendarSkipFilter` mechanism in the engine. The engine
no longer needs to know about NFP/OPEX/Friday — it just asks "should I trade this?"
and gets back SKIP/HALF_SIZE/NEUTRAL.

---

## 4. Data Flow

```
gold.db
  │
  ├─ orb_outcomes (pnl_r per strategy per day)
  ├─ daily_features (is_nfp_day, is_opex_day, is_friday, day_of_week)
  └─ validated_setups (which strategies are validated E1/E2)
  │
  ▼
research/research_calendar_cascade.py
  │
  ├─ JOIN validated_setups → orb_outcomes → daily_features
  ├─ AGGREGATE: mean(pnl_r) per (instrument, session, trading_day)
  ├─ COMPUTE calendar signals (NFP, OPEX, FOMC, CPI, month-end, DOW, etc.)
  ├─ CASCADE: Level 1 → Level 2 → Level 3 with BH FDR at each
  ├─ CONSISTENCY: year-by-year check for survivors
  └─ CLASSIFY: SKIP / HALF_SIZE / NEUTRAL per thresholds
  │
  ▼
research/output/calendar_cascade_rules.json
  │  Format: {"rules": [
  │    {"instrument": "MGC", "session": "NYSE_OPEN", "signal": "OPEX",
  │     "action": "SKIP", "diff": -0.164, "p_bh": 0.023,
  │     "yr_consistent": 9, "yr_total": 11},
  │    ...
  │  ]}
  │
  ▼
trading_app/calendar_overlay.py
  │
  ├─ Load JSON at import time → CALENDAR_RULES dict
  ├─ Key: (instrument, session, signal_name)
  ├─ Val: CalendarAction enum
  └─ Expose: get_calendar_action(instrument, session, trading_day)
  │
  ▼
trading_app/execution_engine.py
  │
  ├─ _arm_strategies() calls get_calendar_action()
  ├─ SKIP → continue (don't arm)
  ├─ HALF_SIZE → arm with size_multiplier=0.5
  └─ NEUTRAL → arm with size_multiplier=1.0
  │
  ▼
ActiveTrade.size_multiplier → JournalEntry.pnl_r scaling
```

---

## 5. File Manifest

| File | Action | Lines est. | Purpose |
|------|--------|-----------|---------|
| `trading_app/paper_trader.py` | **EDIT** | ~5 | Fix wiring bug: pass `calendar_overlay` to engine |
| `trading_app/execution_engine.py` | **EDIT** | ~30 | Default→None, replace CalendarSkipFilter with `get_calendar_action()`, add `size_multiplier` |
| `research/research_calendar_cascade.py` | **NEW** | ~400 | Cascade scanner with per-day aggregation |
| `trading_app/calendar_overlay.py` | **NEW** | ~120 | `CalendarAction` enum, `CALENDAR_RULES` loader, `get_calendar_action()` |
| `tests/test_calendar_overlay.py` | **NEW** | ~150 | Action resolution, most-restrictive, empty-rules |
| `tests/test_paper_trader.py` | **EDIT** | ~30 | Wiring test + size_multiplier test |
| `trading_app/config.py` | **EDIT** | ~5 | Add deprecation note to `CALENDAR_SKIP_NFP_OPEX` |
| `research/output/calendar_cascade_rules.json` | **NEW** (gen) | — | Frozen cascade output |

**Files NOT touched:**
- `pipeline/calendar_filters.py` — already updated (docstring fixed this session)
- `trading_app/portfolio.py` — calendar_overlay in portfolio is for backtest PnL, separate concern
- `pipeline/build_daily_features.py` — no changes needed, already computes is_nfp/is_opex

---

## 6. Implementation Steps

### Step 1: Fix Production Bugs (no new features)

**paper_trader.py:258-266** — add `calendar_overlay=calendar_overlay` to `ExecutionEngine()` call.
**execution_engine.py:199** — change default from `CALENDAR_SKIP_NFP_OPEX` to `None`.
**config.py:428-441** — add deprecation comment to `CALENDAR_SKIP_NFP_OPEX`.

Run existing tests → must all pass. This is a safe change because:
- Paper trader CLI default is `--calendar-filter NONE` which sets `calendar_overlay=None`
- Changing engine default to `None` + passing `None` from paper_trader = no calendar skip
- This is the CORRECT behavior — blanket skip is wrong

### Step 2: Build Cascade Scanner

New file `research/research_calendar_cascade.py`:
- Import calendar date builders from `research/research_calendar_effects.py` (reuse, don't duplicate)
- Load validated outcomes (same query as existing script)
- Aggregate to per-day level
- Implement 3-level cascade with BH FDR at each level
- Year-by-year consistency for survivors
- Classify per thresholds
- Output JSON to `research/output/calendar_cascade_rules.json`
- Print human-readable summary

### Step 3: Build Calendar Overlay Module

New file `trading_app/calendar_overlay.py`:
- `CalendarAction` enum with float values (SKIP=0.0, HALF_SIZE=0.5, NEUTRAL=1.0)
- Load JSON from `research/output/calendar_cascade_rules.json`
- `CALENDAR_RULES` dict keyed by `(instrument, session, signal)`
- `get_calendar_action(instrument, session, trading_day)` function
- Calendar signal detection (is_nfp, is_opex, etc.) — reuse from `pipeline/calendar_filters.py`

New file `tests/test_calendar_overlay.py`:
- Test action resolution for known rules
- Test most-restrictive logic (SKIP > HALF_SIZE > NEUTRAL)
- Test empty rules → all NEUTRAL
- Test unknown instrument/session → NEUTRAL

### Step 4: Wire Into Engine

**execution_engine.py:**
- Replace `calendar_overlay: CalendarSkipFilter | None` parameter with import of `get_calendar_action`
- In `_arm_strategies()`, replace the CalendarSkipFilter block (lines 475-494) with:
  ```python
  action = get_calendar_action(strategy.instrument, orb.label, self.trading_day)
  if action == CalendarAction.SKIP:
      logger.info(f"Calendar SKIP: {strategy.strategy_id}")
      continue
  size_multiplier = action.value
  ```
- Add `size_multiplier` to `ActiveTrade` (default 1.0)

**paper_trader.py:**
- Remove `calendar_overlay` parameter from `replay_historical()` (no longer needed — overlay is automatic)
- In journal entry creation, apply `pnl_r *= active_trade.size_multiplier`
- Remove CLI `--calendar-filter` flag (replaced by automatic per-strategy lookup)

### Step 5: Validate

- Run paper_trader for all 4 instruments
- Compare total PnL with old blanket skip vs new cascade overlay
- New should be BETTER (stops skipping EXPLOIT days)
- Run `python pipeline/check_drift.py`
- Run `python scripts/tools/audit_behavioral.py`
- Run full test suite

---

## 7. Calendar Signals Tested

| # | Signal | Column / Detection | Frequency |
|---|--------|-------------------|-----------|
| 1 | NFP | `is_nfp_day` (daily_features) — 1st Friday of month | ~12/year |
| 2 | OPEX | `is_opex_day` (daily_features) — 3rd Friday of month | ~12/year |
| 3 | FOMC | Hardcoded dates from federalreserve.gov + day after | ~16/year |
| 4 | CPI | Hardcoded dates from bls.gov | ~12/year |
| 5 | Month-end | Last 2 trading days of month | ~24/year |
| 6 | Month-start | First 2 trading days of month | ~24/year |
| 7 | Quarter-end | Last 2 trading days of Q (Mar/Jun/Sep/Dec) | ~8/year |
| 8 | OPEX week | Mon-Fri of OPEX week | ~60/year |
| 9 | Monday | day_of_week == 0 | ~52/year |
| 10 | Tuesday | day_of_week == 1 | ~52/year |
| 11 | Wednesday | day_of_week == 2 | ~52/year |
| 12 | Thursday | day_of_week == 3 | ~52/year |
| 13 | Friday | day_of_week == 4 | ~52/year |

Total: 13 signals × cascade decomposition ≈ 25 tests typical.

---

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Per-day aggregation kills all significance | MEDIUM | Expected — fewer but more honest signals. If zero survive, that's the truth. |
| Cascade finds nothing at Level 1 | MEDIUM | All signals classified NEUTRAL. Blanket skip removed. This is a GOOD outcome — we stop skipping for no reason. |
| FOMC/CPI date lists go stale | LOW | Lists cover through 2026-03. Add header with source URL + last-updated date. Annual maintenance. |
| HALF_SIZE adds complexity for negligible benefit | LOW | If no signals classify HALF_SIZE, the code path exists but never fires. No harm. |
| New overlay breaks live trading | LOW | `get_calendar_action()` defaults NEUTRAL on any error. Fail-open-to-trade. |

---

## 9. What This Does NOT Do

- Does NOT add "boost" or "trade more" for positive calendar days (data snooping risk)
- Does NOT change the discovery grid or validated_setups
- Does NOT affect pipeline/ (one-way dependency preserved)
- Does NOT change portfolio.py backtest calendar logic (separate concern, future work)
- Does NOT auto-update FOMC/CPI dates (manual maintenance)
- Does NOT support fractional contracts for live trading (micros = 1 contract minimum)
