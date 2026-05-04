# Full Trading Logic & Data Pipeline Audit

**Date:** 2026-02-15
**Branch:** feature/market-state
**Methodology:** 6 parallel deep reads of every critical module. Wrong until proven right. Each line verified against CANONICAL_LOGIC.txt and TRADING_RULES.md.

---

## VERDICT: 2 REAL BUGS, 1 DESIGN FLAW, REST IS CLEAN

---

## BUG 1: `sharpe_ann` Uses Calendar Year Count, Not Actual Time Span

**File:** `trading_app/strategy_discovery.py:135-142`
**Severity:** HIGH — systematically inflates Sharpe for short-span instruments

```python
years_span = len(yearly) if yearly else 0  # <-- counts DISTINCT calendar years
trades_per_year = (n_traded / years_span) if years_span > 0 else 0
sharpe_ann = sharpe_ratio * (trades_per_year ** 0.5)
```

**Problem:** `len(yearly)` counts how many distinct calendar years have trades, not the actual time span. Two failure modes:

| Scenario | Actual Span | `len(yearly)` | trades_per_year | Effect |
|----------|-------------|---------------|-----------------|--------|
| 2024-12-01 to 2025-01-31 | 2 months | 2 | N/2 | **Deflated** (2 months treated as 2 years) |
| 2024-01-01 to 2024-06-30 | 6 months | 1 | N/1 | **Inflated** (6 months treated as 1 full year) |

**Impact on your data:** MNQ/MES have ~2 years of data (2024-02 to 2026-02). If a strategy only trades in 2024 and 2025, `len(yearly)=2`, trades_per_year = N/2. This is approximately correct for 2-year spans but becomes wrong at the edges. MGC with 2024-01 to 2026-02 gets `len(yearly)=3` for 2.1 actual years — minor error.

**Fix:**
```python
if yearly:
    min_year = min(int(y) for y in yearly)
    max_year = max(int(y) for y in yearly)
    years_span = max(max_year - min_year + 1, 1)
```
Or better: use actual date range from outcome data.

---

## BUG 2: E3 Stop Validation Incomplete in Execution Engine

**File:** `trading_app/execution_engine.py:641-656`
**Severity:** MEDIUM — can allow fills that should be rejected

**Problem:** When E3 checks for retrace fill, it only validates stop breach on the **retrace bar itself**, not on intermediate bars between confirm and retrace.

```python
# Engine only checks THIS bar:
stop_hit = bar["low"] <= trade.stop_price  # long case
```

Meanwhile `entry_rules.py:226-230` correctly scans ALL bars up to the retrace:
```python
if stop_hit_mask[:retrace_idx + 1].any():  # Scans ALL bars
    return no_fill
```

**Scenario:**
- Bar N: confirm detected, `armed_at_bar = N`
- Bar N+1: stop breached (low < stop) — but `armed_at_bar` guard skips fill check
- Bar N+2: price retraces to ORB level — E3 fill triggers, stop NOT breached on this bar

**Result:** Engine fills the trade. Outcome builder would NOT fill it (it scans all intermediate bars).

**Impact:** Affects execution_engine (live/paper trading) only. Pre-computed `orb_outcomes` use `entry_rules.py` which is correct. So backtested results are accurate — but live execution could diverge.

---

## DESIGN FLAW: INSERT OR REPLACE Preserves Stale `validation_status`

**File:** `trading_app/strategy_discovery.py:470-479`
**Severity:** MEDIUM — silent data corruption on re-discovery

The INSERT OR REPLACE in discovery writes all metric columns but does NOT touch `validation_status`. If a strategy was previously validated (`PASSED`), re-running discovery updates its metrics but leaves the stale `PASSED` status. The validator then skips it (thinks it's already validated).

**Current workaround:** Manual DB cleanup. Should be codified by adding explicit `validation_status=NULL, validation_notes=NULL` to the INSERT OR REPLACE columns.

---

## VERIFIED CLEAN (No Issues Found)

| Module | File | Lines Audited | Status |
|--------|------|--------------|--------|
| **Cost model** | `pipeline/cost_model.py` | All 278 | CORRECT — $10/pt, $8.40 RT, no double-deduction |
| **R-multiples** | `cost_model.py:229-261` | to_r_multiple + pnl_points_to_r | CORRECT — friction in both num+denom for PnL, denom-only for MAE/MFE |
| **ORB construction** | `build_daily_features.py:221-333` | ORB window + break detection | CORRECT — [start, end) window, first close outside, no lookahead |
| **Break window grouping** | `build_daily_features.py:256-295` | Break groups | CORRECT — 1100/1130 share boundary, extends to next group |
| **DST sessions** | `pipeline/dst.py:28-104` | 4 resolvers | CORRECT — CT/ET/LT to Brisbane via zoneinfo |
| **E1 entry** | `entry_rules.py:149-176` | Market-on-confirm | CORRECT — next bar OPEN after confirm |
| **E3 entry** | `entry_rules.py:179-238` | Limit-at-ORB | CORRECT — stop-breach guard scans all intermediate bars |
| **Outcome builder** | `outcome_builder.py:56-319` | Fill-bar + post-entry | CORRECT — ambiguous bars = loss, no lookahead |
| **RSI** | `build_daily_features.py:465-513` | Wilder's 14-period | CORRECT — uses bars AT/BEFORE 09:00 only |
| **ATR** | `build_daily_features.py:751-784` | 20-day SMA of TR | CORRECT — prior days only, no lookahead |
| **Gap** | `build_daily_features.py:759-766` | gap_open_points | CORRECT — today open - prev close |
| **Filter logic** | `config.py:100-116` | OrbSizeFilter.matches_row | CORRECT — G4 = size >= 4.0, fail-closed on None |
| **Expectancy** | `strategy_discovery.py:81-92` | Formula | CORRECT — (WR * avg_win_r) - (LR * avg_loss_r) on post-cost R |
| **Validation 6 phases** | `strategy_validator.py:37-146` | All phases | CORRECT — sample, ExpR>0, yearly, stress, sharpe, drawdown |
| **Stress test** | `cost_model.py:264-278` | 1.5x multiplier | CORRECT — extra friction / risk deducted from ExpR |
| **Contract selection** | `ingest_dbn_mgc.py:296-336` | Front contract | CORRECT — highest volume, earliest expiry tiebreak |
| **OHLCV validation** | `ingest_dbn_mgc.py:191-243` | 7 gates | CORRECT — NaN, Inf, non-positive, H>=L, volume>=0 |
| **5m aggregation** | `build_bars_5m.py:80-137` | time_bucket | CORRECT — first open, max high, min low, last close, sum vol |
| **Checkpoint/resume** | `ingest_dbn_mgc.py:85-185` | Append-only JSONL | CORRECT — idempotent via INSERT OR REPLACE |
| **State machine** | `execution_engine.py:438-798` | CONFIRMING->ARMED->ENTERED->EXITED | CORRECT — armed_at_bar guard, session-end scratch |
| **Paper trader** | `paper_trader.py:104-133` | Chronological replay | CORRECT — ORDER BY ts_utc, 24h window |

---

## NO LOOKAHEAD DETECTED

Every module verified:
- **ORB:** uses only bars within [start, end) window
- **Break:** first close outside, timestamp is the breaking bar
- **Outcome:** scans bars strictly AFTER entry (`ts_utc > entry_ts`)
- **RSI:** bars AT/BEFORE 09:00 only
- **ATR/Gap:** prior day data only
- **Filters:** use pre-computed daily_features (no future data)
- **Volume filter:** fail-closed lookback window, no forward peeking

---

## COST MODEL DETAIL

All formulas match CANONICAL_LOGIC.txt:

| Formula | Implementation | Verified |
|---------|---------------|----------|
| Risk = \|entry - stop\| * point_value + friction | `cost_model.py:194-201` | CORRECT |
| Reward = \|target - entry\| * point_value - friction | `cost_model.py:204-211` | CORRECT |
| Realized RR = Reward / Risk | `cost_model.py:214-226` | CORRECT |
| to_r_multiple = (pnl_pts * pt_val - friction) / risk_$ | `cost_model.py:229-244` | CORRECT |
| pnl_points_to_r = (pnl_pts * pt_val) / risk_$ | `cost_model.py:247-261` | CORRECT (MAE/MFE) |
| Expectancy = (WR * avg_win_r) - (LR * avg_loss_r) | `strategy_discovery.py:81-92` | CORRECT |
| Stress = base ExpR - (extra_friction / risk) | `strategy_validator.py:109-129` | CORRECT |

No double-deduction of costs found. Points vs dollars properly distinguished throughout.

---

## RECOMMENDED FIXES (Priority Order)

### 1. E3 engine stop scan (live-trading correctness)

Add intermediate bar stop check to `execution_engine.py:652` to match `entry_rules.py` behavior.

### 2. sharpe_ann formula (ranking precision)

Replace `len(yearly)` with actual date-range span. Affects strategy ranking and CORE/REGIME classification thresholds.

### 3. Discovery validation_status (data hygiene)

Add explicit `validation_status=NULL, validation_notes=NULL` to the INSERT OR REPLACE columns. Eliminates manual cleanup step.

---

## BOTTOM LINE

The pipeline is **structurally sound**. No lookahead. No double-counting costs. No survivorship bias in discovery. The two bugs are real but bounded: Bug 1 affects ranking precision (not direction), Bug 2 affects live execution only (backtested outcomes are correct). The core money math -- cost model, R-multiples, expectancy -- is right.
