# E2 Stop-Market Entry Model Design

**Date:** 2026-02-26
**Status:** Approved
**Scope:** Purge E0 (dishonest), build E2 (stop-market at ORB level + slippage)

---

## Problem Statement

E0 (Limit-On-Confirm) wins 33/33 grid combos at G4+ RR2.0 — a structural artifact from 3 compounding optimistic biases:

1. **Fill-on-touch:** Assumes fill at exact ORB level. Professional standard is fill-through-by-1-tick minimum.
2. **Fakeout exclusion:** Only fills on bars that close outside ORB. Real stop orders fill on ANY touch, including fakeouts.
3. **Fill-bar wins (RR1.0):** Entry + target hit on same 1m bar. Contamination 2-16% depending on session.

Meanwhile, the industry-standard entry for ORB breakout backtesting — **stop-market at the ORB level** (Crabel, Davey, QuantConnect) — is not implemented.

## Decision Record

| Question | Decision | Rationale |
|----------|----------|-----------|
| Slippage model | Option C: N ticks, configurable, default 1 | Enables stress testing (N=2 filter for brittle strategies). Lopez de Prado: scenario simulation. |
| Fakeout handling | Option A: True CB0, no confirmation | Only honest option. Fakeouts eaten as -1R losses. Stop orders are blind. |
| E0 purge scope | Option A: Full purge (code + data) | Bad data is toxic. No value comparing honest vs dishonest models. |
| Discovery grid | Option A: E2 at CB1 slot (reuse infrastructure) | Model simplicity. CB parameter is semantically irrelevant for E2 but avoids grid/schema changes. |
| Overall approach | Approach 1: Swap-in-place | One pass, one rebuild, clean result. |

## E2 Entry Model Specification

### Mechanics

A stop-market order sits at the ORB boundary before any break occurs. When the first 1m bar's range crosses the ORB level, the stop triggers.

- **Long entry:** `orb_high + (N * tick_size)` when any bar's high > orb_high
- **Short entry:** `orb_low - (N * tick_size)` when any bar's low < orb_low
- **No confirmation required** — the break IS the entry
- **Fakeouts included** — bar may close back inside ORB; you're still filled

### Detection Path (New)

E2 cannot use `detect_confirm()` because that requires bar CLOSE outside ORB, which filters fakeouts (the same bias E0 has).

New function `detect_break_touch()`:
- Input: bars_df, orb_high, orb_low, break_dir, detection_window_end
- Logic: find first bar where high > orb_high (long) or low < orb_low (short)
- Output: BreakTouchResult(bar_idx, bar_ts, orb_high, orb_low, break_dir)
- No close requirement — range crossing is sufficient

### Slippage Configuration

```python
# config.py
E2_SLIPPAGE_TICKS = 1   # Default: 1 tick fill-through (industry standard)
E2_STRESS_TICKS = 2     # For stress testing robustness
```

Tick sizes from CostSpec (e.g., MGC = 0.10pt, MNQ = 0.25pt).

### Risk Calculation

Identical to E1/E3 — risk = |entry_price - stop_price|. The N-tick slippage widens risk slightly beyond ORB width (e.g., 10.10pt vs 10.00pt for MGC). This is correct — the slippage is a real cost.

## E0 Purge Specification

### Data Purge (execute first)

```sql
DELETE FROM orb_outcomes WHERE entry_model = 'E0';
DELETE FROM experimental_strategies WHERE entry_model = 'E0';
DELETE FROM validated_setups WHERE entry_model = 'E0';
DELETE FROM edge_families WHERE family_id LIKE '%_E0_%';
```

Expected impact: ~817 validated strategies removed, associated outcomes/experimental rows.

### Code Removal

- Delete `_resolve_e0()` from entry_rules.py
- Remove E0 dispatch in `resolve_entry()`
- Remove E0 branch in execution_engine.py
- Remove E0 from `ENTRY_MODELS` constant
- Remove E0-specific tests

### New Drift Check

Add check: no rows with entry_model = 'E0' exist in orb_outcomes, experimental_strategies, validated_setups.

## Data Flow

### Current (E0/E1/E3)

```
orb_break_ts (daily_features)
  -> detect_confirm() [requires bar CLOSE outside ORB]
  -> resolve_entry() [dispatches to _resolve_e0/e1/e3]
  -> outcome computation
```

### New (E2/E1/E3)

```
orb_break_ts (daily_features)
  -> [E2] detect_break_touch() [requires bar RANGE to cross ORB]
  -> _resolve_e2() [entry at ORB level + N ticks]
  -> outcome computation (identical downstream)

  -> [E1/E3] detect_confirm() [unchanged]
  -> resolve_entry() [dispatches to _resolve_e1/e3]
  -> outcome computation
```

## Files Changed

| File | Change | Est. LOC |
|------|--------|----------|
| config.py | Replace E0 with E2 in ENTRY_MODELS, add E2_SLIPPAGE_TICKS | ~15 |
| entry_rules.py | Delete _resolve_e0(), add detect_break_touch() + _resolve_e2() | ~60 |
| outcome_builder.py | E2 path uses detect_break_touch() | ~20 |
| execution_engine.py | Replace E0 branch with E2 stop-market logic | ~40 |
| nested/builder.py | Replace E0 refs with E2, use break-touch for nested | ~15 |
| strategy_discovery.py | Update grid loop (E2 replaces E0 at CB1) | ~10 |
| paper_trader.py | Update entry model parsing | ~5 |
| check_drift.py | Add no-E0-rows drift check | ~10 |
| Tests (multiple) | Remove E0 tests, add E2 tests, update counts | ~80 |
| TRADING_RULES.md | Update entry model documentation | ~20 |

**Total: ~275 LOC changed/added, ~100 LOC deleted**

## Rebuild Sequence

After code changes, for each instrument (MGC, MES, MNQ, M2K):

1. `outcome_builder.py --instrument X --force --start <earliest> --end <latest>`
2. `strategy_discovery.py --instrument X`
3. `strategy_validator.py --instrument X --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward`
4. `build_edge_families.py --instrument X`

## Expected Outcomes

- E2 will have **more trades** than old E0 (fakeouts now included)
- E2 will have **lower win rate** than old E0 (fakeouts are -1R losses)
- E2 will have **slightly worse entry** than old E0 (N-tick slippage)
- E2 will have **better entry** than E1 (at the level, not chasing)
- Strategies that survive E2 are **mechanically honest** and tradeable live
- N=2 stress test filters brittle strategies that can't survive hostile execution

## References

- Toby Crabel (1990): ORB with stop entries
- Kevin Davey (KJ Trading): stop entries superior for breakouts
- QuantConnect futures fill model: max(stop_price, close) + slippage
- Lopez de Prado: scenario simulation, stress testing
- arXiv 2024 "Negative Drift of Limit Order Fill": fill-on-touch is optimistic
- TradingView docs: "in real trading, orders don't fill easily when prices just touch the limit price"
