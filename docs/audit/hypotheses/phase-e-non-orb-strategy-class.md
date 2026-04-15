# Phase E — Non-ORB Strategy Class (SC2) (PRE-REG STUB)

**Status:** NOT EXECUTED. Pre-registration stub for future session.
**Date authored:** 2026-04-15
**Origin:** User question 2026-04-15 — "use these indicators in OTHER ways, not just ORB"

## Premise

The current research framework pigeonholes all signals into ORB-breakout
context. Level / volatility / volume / timing signals all exist
independently of ORB. Applying them in different strategy frameworks
could open a strategy class orthogonal to ORB (SC2), diversifying the
portfolio beyond correlated ORB lanes.

## Candidate SC2 strategies (each requires separate pre-reg when executed)

### SC2.1 — Direct Level Fade

- **Setup:** Limit orders AT PDH, PDL, session levels (after window closes) with stop just beyond
- **Entry:** Passive limit. Fill means level hit + immediate reject
- **Target:** Midpoint of prior-day range OR next structural level
- **Stop:** ~0.3-0.5x ATR beyond the level (tight)
- **Mechanism:** Classical S/R mean-reversion. Dalton / Murphy framework.
- **Literature gap:** Dalton (Mind Over Markets) not in `resources/` — acquisition required
- **Session candidates:** COMEX_SETTLE, CME_PRECLOSE, NYSE_CLOSE (post-NY-close, clean)

### SC2.2 — Level-Break Momentum

- **Setup:** Breakout AFTER clean close through level WITH volume confirmation
- **Entry:** Stop-market AT level + 1-2 ticks, only when break_bar_volume > 1.5x session avg
- **Target:** Dynamic — trail to next structural level
- **Stop:** Below the level
- **Mechanism:** Momentum continuation through institutional trigger
- **Links to:** validated volume signals (rel_vol_HIGH_Q3, bb_volume_ratio_HIGH)

### SC2.3 — Mean-Reversion Within Range

- **Setup:** When INSIDE_PDR fires (middle of prior-day range) + low vol regime
- **Entry:** Fade both edges of the prior-day range with limits
- **Target:** Opposite edge
- **Stop:** Outside range
- **Mechanism:** Range-bound day, auction balance (market profile)
- **Contrast:** Opposite of ORB-breakout — must be DISTINCT regime (gate on day_type)

### SC2.4 — Cross-Asset Pair (MES vs MNQ divergence)

- **Setup:** When MES and MNQ positions in respective PDRs diverge > 2 sigma
- **Entry:** Long underperformer, short outperformer, size-matched by ATR
- **Exit:** Convergence or time-stop
- **Mechanism:** Statistical arb on typically correlated equity index futures
- **Infrastructure:** Requires pairs-trade execution + relative-value backtest

### SC2.5 — Volatility Regime Overlay

- **Setup:** Use GARCH_VOL_PCT as REGIME GATE across ALL strategies (SC1 + SC2.x)
- **Entry:** Reduce size / disable trades on GARCH_VOL_PCT > 90 (extreme vol)
- **Mechanism:** Known volatility-regime edge degradation
- **Light-touch:** Can be applied as portfolio-level overlay, not new strategy

## Execution roadmap (if chosen in future)

1. **Sub-Phase E0 — Literature acquisition:** Dalton (Mind Over Markets), Murphy (Technical Analysis), Steidlmayer (Market Profile) PDFs. Extract passages. Write `docs/institutional/literature/` files.

2. **Sub-Phase E1 — SC2.1 Direct Level Fade pre-reg + research:**
   - Scope: COMEX_SETTLE / CME_PRECLOSE / NYSE_CLOSE × {MNQ, MES} × theta {0.15, 0.25}
   - K budget: ~24 cells. Within Bailey MinBTL.
   - Backtest infra: limit-order simulator in outcome_builder
   - Timeline: 2-3 weeks

3. **Sub-Phase E2 — SC2.2 Level-Break Momentum:** Separate pre-reg, similar scope

4. **Sub-Phase E3 — SC2.5 Volatility Regime Overlay:** Cheapest, existing GARCH signal, apply as day-level gate

## Pre-registered criteria (all sub-phases)

Per `docs/institutional/pre_registered_criteria.md` — all 12 criteria,
PLUS:

- Correlation with existing ORB portfolio < 0.3 (diversification test — must not duplicate existing edge)
- Positive ExpR on pre-committed threshold set (no post-hoc tuning)
- T0-T8 battery pass
- Signal-only shadow mode 2 weeks before live

## Kill criteria

- Any SC2 strategy correlates > 0.5 with deployed ORB lanes → strategy duplication, not expansion
- Any SC2 strategy fails baseline ExpR ≥ 0.10 → below cost threshold
- Literature gap cannot be filled within 2 weeks → defer until sources available

## Relationship to existing ORB work

SC2 is ADDITIVE not REPLACEMENT. ORB (SC1) continues. SC2 provides
orthogonal signal surface. Portfolio construction (via allocator) decides
which lanes of which class to deploy based on correlation + Sharpe.

## Why deferred

Sub-Phase E0 literature acquisition blocks SC2.1-2.3. SC2.5 (vol regime overlay)
is executable now but narrower scope; would not justify full Phase E
dedication.
