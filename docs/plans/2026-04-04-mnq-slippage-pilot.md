# Design Plan 5: MNQ E2 Slippage Pilot

**Date:** 2026-04-04
**Priority:** HIGH — MNQ is the ONLY unfiltered-positive instrument; edge dies at 5 ticks/side
**Status:** DESIGN (awaiting implementation)
**Prerequisite:** `gold.db` must exist with bars_1m + daily_features populated

---

## What & Why

MNQ baseline edge is +0.085R = ~$4.40/trade at $2.00/point.
At $0.50/tick (MNQ tick), each extra tick of slippage costs $1 and kills ~11% of the edge.
**The edge dies at 5 ticks/side of extra slippage.**

The MGC tbbo pilot showed mean=6.75 ticks actual (vs 1 modeled), std=41.57.
MNQ is a different market — NQ is one of the most liquid futures in the world,
but the MICRO (MNQ) has different book depth than full-size NQ.

The pilot script already exists: `research/research_mnq_e2_slippage_pilot.py`.
This plan is about running it and integrating the results.

## Existing Infrastructure

| Component | File | Status |
|-----------|------|--------|
| MNQ pilot script | `research/research_mnq_e2_slippage_pilot.py` | EXISTS — not yet run |
| MGC pilot results | Referenced in `cost_model.py:81` | DONE — mean=6.75 ticks |
| Break-even analysis | `scripts/tools/slippage_scenario.py` | EXISTS — MNQ lanes analyzed |
| Cost model | `pipeline/cost_model.py:90-98` | MNQ: 1 tick slippage modeled |
| Session multipliers | `pipeline/cost_model.py:180-222` | MNQ: all 1.0x or 0.8-0.9x |

### Break-Even Context (from cost_model.py:84-88)

| Session | Extra Ticks to Zero ExpR | Risk Level |
|---------|--------------------------|------------|
| COMEX_SETTLE | 4.9 | **FRAGILE** |
| SINGAPORE_OPEN | 6.0 | FRAGILE |
| NYSE_CLOSE | 15.4 | Robust |
| NYSE_OPEN | 17.7 | Robust |

If actual slippage is even 3 ticks/side, COMEX_SETTLE (a pre-registered 2026 strategy)
is at risk. This is existential for the portfolio.

---

## Pilot Design (Already in Script)

The script (`research/research_mnq_e2_slippage_pilot.py`) has 3 phases:

### Phase 1: `--estimate-cost`
- Estimate Databento API cost for tbbo (top-of-book) data
- Sample: 40 days × 6 sessions = 240 events, ~32 min of tbbo data each
- Cost estimate needed before purchasing data

### Phase 2: `--pull`
- Download tbbo snapshots from Databento around each ORB break moment
- Window: 2 min before → 30 min after break
- Cache to `research/data/tbbo_mnq_pilot/`
- Stratified by: session (6) × ATR regime (high/low) = 12 buckets, 5 days each

### Phase 3: `--reprice`
- For each break event, find the BBO (best bid/offer) at the exact moment price first crosses ORB boundary
- Measure: how many ticks from ORB level to the actually achievable fill?
- Compare to modeled 1-tick assumption
- Report by session and ATR regime

---

## Implementation Plan

### Stage 1: Pre-Flight Verification

Before running the pilot, verify:

1. **gold.db exists** with MNQ data populated
   ```sql
   SELECT COUNT(*), MIN(trading_day), MAX(trading_day)
   FROM daily_features WHERE symbol = 'MNQ' AND orb_minutes = 5;
   ```

2. **Databento API key** is configured (`DATABENTO_API_KEY` env var)

3. **Cost estimate** — run `--estimate-cost` first to check API budget

### Stage 2: Run the Pilot

```bash
# 1. Estimate cost
python -m research.research_mnq_e2_slippage_pilot --estimate-cost

# 2. Pull tbbo data (requires Databento API credits)
python -m research.research_mnq_e2_slippage_pilot --pull

# 3. Reprice against actual book depth
python -m research.research_mnq_e2_slippage_pilot --reprice
```

### Stage 3: Analyze Results

Key questions the pilot must answer:

| Question | How to Measure | Kill Threshold |
|----------|---------------|----------------|
| Mean ticks at break moment? | `(best_offer - orb_high) / tick_size` for longs | > 3 ticks → FRAGILE sessions dead |
| Median ticks? | Same, median | > 2 ticks → cost model update needed |
| Session variation? | Group by session, compare means | Asian > 2x US → session multipliers wrong |
| ATR regime effect? | Group by high/low ATR | High ATR > 2x low → vol-conditional cost needed |
| MNQ vs NQ depth? | Compare MNQ vs NQ.FUT bbos | If NQ has 0 ticks and MNQ has 3+ → micro-specific cost |

### Stage 4: Cost Model Update (If Needed)

Based on pilot results, update `pipeline/cost_model.py`:

**Scenario A: Median ≤ 1 tick (current model is correct)**
- No changes needed
- Document result in cost_model.py comment
- Update `SESSION_SLIPPAGE_MULT` if session variation found

**Scenario B: Median 2-3 ticks (model needs update)**
- Update `slippage` field: `$1.00 → $1.50-2.00`
- Rerun `slippage_scenario.py` to recalculate break-even ticks
- COMEX_SETTLE and SINGAPORE_OPEN may need to be flagged FRAGILE
- Rerun discovery+validation to see which strategies survive higher costs

**Scenario C: Median > 3 ticks (edge is at risk)**
- **HALT live trading on FRAGILE sessions** (COMEX_SETTLE, SINGAPORE_OPEN)
- Update cost model significantly
- Rerun full pipeline with corrected costs
- Strategies that don't survive are DEAD
- Only robust sessions (NYSE_OPEN, NYSE_CLOSE) continue

### Stage 5: Document & Archive

Write results to `research/output/mnq_e2_slippage_pilot_results.md`:
- Raw statistics (mean, median, std, max, min by session × regime)
- Comparison to MGC pilot
- Cost model recommendation
- Sessions at risk

Update `cost_model.py` TODO comment (line 80-89) with actual results.

---

## Files to Touch

| File | Change | Type |
|------|--------|------|
| `research/research_mnq_e2_slippage_pilot.py` | RUN (verify it works, fix any gold.db path issues) | RUN |
| `pipeline/cost_model.py` | Update slippage if results warrant | MODIFY (conditional) |
| `scripts/tools/slippage_scenario.py` | Rerun with updated costs | RUN (conditional) |
| `research/output/mnq_e2_slippage_pilot_results.md` | NEW — results documentation | CREATE |

## Blast Radius

- Pilot script is read-only (queries gold.db + Databento API)
- Cost model change would cascade through ALL discovery + validation
- If cost model changes → full pipeline rebuild required
- This is why we run the pilot FIRST (read-only) before touching production

## Prerequisites

1. `gold.db` must exist with MNQ bars_1m + daily_features
2. Databento API key must be set (`DATABENTO_API_KEY`)
3. Sufficient Databento credits (estimate via `--estimate-cost`)

## Acceptance Criteria

1. Pilot runs to completion on all 6 sessions × 2 ATR regimes
2. Per-session mean/median/std ticks reported
3. Comparison to MGC pilot results documented
4. Cost model recommendation made with supporting evidence
5. If slippage > 2 ticks median: updated cost model + break-even analysis rerun
6. Results archived in `research/output/`

## Decision Tree (Define Before Running)

```
Median ≤ 1 tick → KEEP current model, document, update SESSION_SLIPPAGE_MULT
Median 1-2 ticks → UPDATE cost model, rerun scenario, flag fragile sessions
Median 2-3 ticks → UPDATE cost model, HALT fragile sessions, rerun discovery
Median > 3 ticks → HALT all MNQ live trading, full pipeline rebuild with new costs
```
