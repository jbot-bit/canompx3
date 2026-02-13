# Early Exit Rules — Research Results

Tested 2026-02-12 against gold.db (4,272 trades, 2021-2026, G4+ filter, E1 CB2 + E3 CB1).
Script: `scripts/analyze_early_exits.py`

---

## What Works

### 0900: Kill losers at 15 minutes

| Metric | Without rule | With 15-min kill | Change |
|--------|-------------|-----------------|--------|
| **E1** Sharpe | 0.164 | **0.207** | +26% |
| **E1** MaxDD | -19.0R | **-11.8R** | 38% tighter |
| **E1** ExpR | +0.225 | +0.239 | +6% |
| **E3** Sharpe | 0.123 | **0.157** | +28% |
| **E3** MaxDD | -15.0R | **-16.3R** | similar |
| **E3** ExpR | +0.164 | +0.181 | +10% |

- 40% of 0900 E1 trades are losing at 15 min. Of those, only **24% eventually win**.
- 36% of 0900 E3 trades are losing at 15 min. Of those, only **24% eventually win**.
- Action: At 15 minutes after fill, if mark-to-market is negative, **close the trade**.

### 1000: Kill losers at 30 minutes

| Metric | Without rule | With 30-min kill | Change |
|--------|-------------|-----------------|--------|
| **E1** Sharpe | 0.014 | **0.052** | 3.7x |
| **E1** MaxDD | -32.1R | **-20.8R** | 35% tighter |
| **E1** ExpR | +0.019 | +0.064 | 3.4x |
| **E3** Sharpe | 0.003 | **0.020** | 6.7x |
| **E3** MaxDD | -30.6R | **-22.7R** | 26% tighter |
| **E3** ExpR | +0.004 | +0.025 | 6.3x |

- 24% of 1000 E1 trades are losing at 30 min. Of those, only **12% eventually win**.
- 17% of 1000 E3 trades are losing at 30 min. Of those, only **18% eventually win**.
- Action: At 30 minutes after fill, if mark-to-market is negative, **close the trade**.

### Why different timing?

0900 is faster (Asia open momentum dissipates quickly — 15 min is the cutoff).
1000 is slower (needs 30 min for the signal to clarify). At 10-20 min on 1000, the loser check actually *hurts* because it's too early.

---

## What Doesn't Work

### Breakeven trail — DESTRUCTIVE

| Session | Trigger | Sharpe delta | Notes |
|---------|---------|-------------|-------|
| 1800 E1 | +0.5R | **-0.167** | Worst single result |
| 1800 E3 | +0.5R | **-0.165** | |
| 0900 E1 | +0.5R | **-0.126** | |
| 1000 E3 | +0.5R | **-0.117** | |

Gold is too volatile. Moving stop to breakeven after +0.5R kills 45-62% of trades that would have eventually won. Even the +1.0R trigger hurts everywhere except 2300.

**Never use breakeven trails on Gold ORB strategies.**

### Retrace dwell — Mostly neutral

Dwell time (sitting at entry after going green) showed weak results across sessions:
- 0900: small positive at M15 (Sharpe +0.01), not worth the complexity
- 1000 E3: actually **negative** — dwell trades had 67-75% original WR, meaning it was killing winners
- 1800: slightly negative everywhere
- Only validated for 1800 E3 in isolation (per TRADE_MANAGEMENT_RULES.md), but pooled across all RR targets the signal dilutes

**Not recommended as a systematic rule.** May work as a discretionary "watch closely" signal for 1800 E3 specifically.

### N-bar momentum — Too rare to matter

- Rule 4 (K=3, K=5): triggers on 0-13% of trades
- When it does trigger, the original WR of triggered trades is low (17-29%), confirming the signal is real
- But sample sizes are tiny (1-50 trades affected per group)
- K=5 almost never triggers (0-3% of trades)

**Signal exists but fires too rarely to improve portfolio metrics.**

### Loser check at wrong timing — Harmful

- 0900 at 30 min: **hurts** (Sharpe 0.164 -> 0.058). Too late — by 30 min, 41% of triggered trades still win.
- 1000 at 10 min: **hurts** (Sharpe 0.014 -> -0.032). Too early — noise, not signal.
- 1800 at any timing: all negative or flat. 1800 is a slower session, losers can still recover.

**Timing matters. Wrong timing is worse than no rule at all.**

---

## Rules Summary (for live trading)

```
TRADE FILLS
  |
  +-- 0900 session?
  |     At 15 min: Am I in profit?
  |       YES -> HOLD
  |       NO  -> EXIT (only 24% of these win)
  |
  +-- 1000 session?
  |     At 30 min: Am I in profit?
  |       YES -> HOLD
  |       NO  -> EXIT (only 12-18% of these win)
  |
  +-- 1800 session?
  |     No systematic early exit improves results.
  |     (The 30-min check from TRADE_MANAGEMENT_RULES.md is
  |      specific to 1800 E3 RR1.5/2.0 only — still valid
  |      as discretionary signal, but not systematic edge here.)
  |
  +-- 2300 session?
        No early exit helps. Already negative ExpR baseline.
        (2300 has structural problems — early exits can't fix bad edge.)
```

---

## Validation

| Known fact | Script result | Match? |
|-----------|--------------|--------|
| 1800 E3 losers at 30 min: 24% WR (TRADE_MANAGEMENT_RULES.md) | 21% orig WR of triggered | Yes (small diff from pooling RR 1.5/2.0/2.5 vs 1.5/2.0) |
| Retrace dwell 10+ min: 33% WR (TRADE_MANAGEMENT_RULES.md) | 1800 E3 M10: 34% orig WR | Yes |
| Breakeven trail hurts in volatile instruments (literature) | Worst degradation across all sessions | Confirmed |

---

## Methodology

- Database: gold.db, read-only
- Period: 2021-01-01 to 2026-02-04
- Filter: G4+ ORB size (>= 4 points)
- Entry models: E1 (CB2), E3 (CB1)
- RR targets: 1.5, 2.0, 2.5 (pooled)
- Sessions: 0900, 1000, 1800, 2300
- Scratches excluded (NULL pnl_r)
- Each rule tested independently (not stacked)
- Early exit PnL uses `to_r_multiple()` (deducts friction)
