# Garch Mechanism Hypotheses

**Date:** 2026-04-16  
**Status:** ACTIVE HYPOTHESIS NOTE  
**Purpose:** record realistic market-structure hypotheses for `garch` and nearby
vol-state filters so the next tests stay theory-shaped instead of drifting into
filter stacking.

**Authority chain:**
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/mechanism_priors.md`
- `docs/plans/2026-04-16-garch-deployment-map-proof-plan-reset.md`

---

## Principle

`garch_forecast_vol_pct` is currently best treated as a **state variable**,
not a standalone directional edge. The mechanism question is therefore:

**What kind of market state does this variable represent, and how should that
state alter participation or allocation?**

---

## M1 — Latent Expansion After Quiet Overnight Range

### Hypothesis

When:
- `overnight_range_pct` is low to moderate, and
- `garch_forecast_vol_pct` is elevated,

the market may be carrying latent conditional volatility that has **not yet**
been expressed through overnight realized range. The next event-anchored
session is more likely to exhibit expansion with cleaner directional follow-through.

### Economic use if true

- rank those setups above neutral ones
- prefer allocation rather than hard gating

### Why this is plausible

- conditional volatility can rise before realized intraday expansion shows up
- event sessions convert latent uncertainty into realized movement

### What would falsify it

- confluence underperforms solo `garch`
- low-overnight / high-garch cells do not show better realized trade quality

---

## M2 — High Garch + High ATR = Active Trend-Vol State

### Hypothesis

When both:
- `garch_forecast_vol_pct` is high, and
- `atr_20_pct` is high,

the market is in a broad active volatility state with persistent participation.
For the right session families, this should help continuation-style breakout
quality more than either proxy alone.

### Economic use if true

- bounded upweighting on already-eligible trades
- lane ranking on crowded days

### Why this is plausible

- GARCH captures conditional-vol expectation
- ATR captures realized recent movement
- agreement may indicate a more mature, active regime rather than a noisy spike

### What would falsify it

- `GARCH_ATR` confluence fails to beat the better solo map
- improvement disappears once survival and DD are included

---

## M3 — Composite Vol-State Is Better For Allocation Than For Entry Gating

### Hypothesis

Signals like `garch_pct`, `overnight_range_pct`, and `atr_20_pct` may not be
good at answering "trade or do not trade?" but may be useful for answering:

- which eligible lane deserves scarce risk budget today?
- which trade gets extra size?
- which lane gets deprioritized?

### Economic use if true

- portfolio-ranking allocator
- relative lane priority under finite account risk

### Why this is plausible

- continuous state variables often lose value when forced into isolated binary
  lane decisions
- the same signal can still add value in **cross-opportunity selection**

### What would falsify it

- portfolio ranking adds nothing beyond existing lane selection
- same-day top-ranked trades are not materially better than lower-ranked ones

---

## M4 — Different Profiles Need Different Translation, Not Different Truth

### Hypothesis

The same state variable can be economically useful under one account geometry
and only mildly useful under another, without changing the underlying research
truth.

### Economic use if true

- profile-specific doctrine after allocator proof

### Why this is plausible

- prop-style trailing DD and copier arithmetic reward different tradeoffs than
  self-funded single-account capital

### What would falsify it

- all profiles rank the same map the same way once accounting is correct

---

## Immediate Research Implication

The next theory-shaped test should not be:

- "try another random filter with garch"

It should be one of:

1. **allocator translation test**
   - does a simple confluence map beat solo maps economically?
2. **portfolio-ranking test**
   - does composite state improve allocation among same-day opportunities?
3. **specific mechanism slice**
   - e.g. `high garch + low overnight range` versus base on event sessions

Only after one of those survives should a narrower, more specific filter-style
claim be entertained.
