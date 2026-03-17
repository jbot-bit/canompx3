# Trade-Count-Based Walk-Forward Windows — Design

**Date:** 2026-03-17
**Status:** Design complete, ready for implementation
**Reference:** Lopez de Prado, AFML Ch.2 — information-driven sampling

## Problem

MGC ATR varies 9.2x (11.5pt 2018 → 105.3pt 2026). Calendar-based WF windows
(6-month test, 15-trade minimum) produce windows that are ALL invalid (N<15)
in low-ATR years and ALL valid in high-ATR years. Result: WF validates ONLY the
2024-2026 high-vol regime. Zero OOS evidence for moderate-vol environments.

Compare: MNQ 1.9x, MES 2.8x, M2K 1.5x — MGC is the outlier.

## Solution

Replace calendar boundaries with trade-count boundaries for affected instruments.
A window of "next 30 trades" spans 2 months at high ATR but 18 months at low ATR —
automatically regime-spanning.

Calendar mode remains default for MNQ/MES/M2K where it works fine.

## Architecture

### New Config (config.py)

```python
WF_TRADE_COUNT_OVERRIDE: dict[str, int] = {
    "MGC": 30,  # 30 trades per OOS window
}
WF_MIN_TRAIN_TRADES: dict[str, int] = {
    "MGC": 45,  # min IS trades before first OOS window
}
```

### Window Generation (walkforward.py)

Calendar mode (existing, unchanged):
```
window_start = anchor + 12 months
while window_start < latest:
    window_end = window_start + 6 months
    OOS = trades in [window_start, window_end)
    IS  = all trades before window_start
    advance window_start = window_end
```

Trade-count mode (new):
```
idx = min_train_trades  (e.g., 45)
while idx + window_size <= len(outcomes):
    IS  = outcomes[:idx]
    OOS = outcomes[idx : idx + window_size]
    idx += window_size
```

### Param Threading (strategy_validator.py)

```python
wf_test_trades = WF_TRADE_COUNT_OVERRIDE.get(instrument)
wf_min_train = WF_MIN_TRAIN_TRADES.get(instrument)
# passed through _walkforward_worker → run_walkforward
```

## Pass Rule

Unchanged. Same 4 conditions:
1. n_valid >= min_valid_windows (3)
2. pct_positive >= min_pct_positive (60%)
3. agg_oos_exp_r > 0
4. total_oos_trades >= min_trades_per_window × min_valid_windows

In trade-count mode every window IS valid (guaranteed N = window_size),
so min_valid_windows = min_windows = total_windows.

## Blast Radius

- `trading_app/walkforward.py` — new window path (~30 lines)
- `trading_app/config.py` — 2 new dicts (~10 lines)
- `trading_app/strategy_validator.py` — thread 2 params (~5 lines)
- `tests/test_trading_app/test_walkforward.py` — new tests (~60 lines)

4 files. No schema changes. No entry model changes. No pipeline changes.

## Test Strategy (TDD)

1. test_trade_count_basic — 90 trades, window=30, min_train=30 → 2 OOS windows
2. test_trade_count_insufficient — 20 trades < min_train → 0 windows, fails
3. test_trade_count_exact_minimum — min_train + window trades → 1 window
4. test_trade_count_pass_rule — 4-condition pass rule unchanged
5. test_calendar_unchanged — existing calendar tests pass (regression)

## Rollback

Delete trade-count code path + config dicts. Calendar mode untouched — zero risk.
