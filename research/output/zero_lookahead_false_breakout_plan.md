# Zero-Lookahead False Breakout Filter (ORB, 1m OHLCV)

Status: drafted for implementation/backtest
Date: 2026-02-22

## Objective
Filter likely false ORB breakouts **at or before entry** using only data known at breakout-bar close (no lookahead).

## Core approach: Breakout Quality Score (BQS)

Compute 5 real-time components on breakout bar close:

1. **Outside Close Confirmed (OCC)**
   - Long breakout: `close > OR_high`
   - Short breakout: `close < OR_low`
   - Score: +1 if true else 0

2. **Wick Rejection Penalty (WRP)**
   - Long: upper wick ratio = `(high - close) / max(high-low, eps)`
   - Short: lower wick ratio = `(close - low) / max(high-low, eps)`
   - Score: +1 if wick ratio <= 0.30 else 0

3. **Break Speed (BSP)**
   - `break_delay_min = minutes from OR end to breakout`
   - Score: +1 if `break_delay_min <= 10` else 0

4. **Range-Energy (RES)**
   - `size_atr = OR_size / ATR20`
   - Threshold: use rolling 70th percentile (or fixed calibrated value)
   - Score: +1 if `size_atr >= q70` else 0

5. **Volume Impulse (VIS)**
   - `vol_impulse = breakout_bar_volume / SMA(volume, 20)` (or robust local baseline)
   - Score: +1 if `vol_impulse >= 1.0` (or tuned quantile) else 0

### Composite
`BQS = OCC + WRP + BSP + RES + VIS` (0..5)

Use as filter:
- Trade only if `BQS >= 4` (initial)
- Aggressive mode: `BQS >= 3`

## 3-5 Testable Rule Sets

### Rule A (Strict quality)
- OCC == true
- WRP == true
- BSP == true (`delay <= 10`)
- RES == true (`OR_size/ATR20 >= q70`)
- VIS == true (`vol_impulse >= 1.0`)

### Rule B (Strong but less restrictive)
- OCC true
- WRP true
- `BQS >= 4`

### Rule C (Momentum quality only)
- OCC true
- BSP true
- RES true
- VIS true

### Rule D (Anti-false breakout veto)
- Veto trade if any of:
  - OCC false (closed back inside OR)
  - wick rejection ratio > 0.30
  - break_delay_min > 30 and vol_impulse < 1.0

### Rule E (Asymmetric side logic)
- Test long and short separately (same thresholds initially)
- Keep side-specific thresholds if one side materially outperforms

## Why this is zero-lookahead
All inputs are known at breakout bar close or earlier:
- OR_high/OR_low, OR_size
- ATR20 from prior bars
- breakout bar OHLCV
- elapsed minutes to breakout

No future bars, no post-break outcomes, no double-break future labels.

## Implementation notes (Pine v6)

- Trigger breakout candidate when price crosses OR boundary.
- Evaluate BQS at `barstate.isconfirmed` on breakout bar.
- Enter next bar only if BQS threshold passes.

## Backtest plan

1. Baseline ORB (no filter)
2. Apply Rules A-E independently
3. Compare:
   - win rate
   - expR / avgR
   - trade count (frequency)
   - max drawdown
   - yearly/OOS stability

Promotion gate:
- uplift > 0
- DD not materially worse
- sufficient frequency (prefer common-ground, not sparse niches)
