---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# 1s Break Speed Research — Plan for Next Session

**Status:** COMPLETE. Step 1 was already done (FAST5/FAST10 in grid, 3 validated). Step 2: KILL — 1s adds no discovery value. Step 3 cancelled.

## Context

### What exists (from `break_speed_signal_retest.md`, Apr 1 2026):
- break_delay_min computed at 1m resolution in daily_features (14 sessions)
- 3 sessions VALIDATED (DSR-deflated, BH FDR at K=23):
  - NYSE_CLOSE MNQ: +13.1% WR spread, DSR p<0.001, WFE 106%
  - CME_REOPEN MGC: +9.6% WR spread, DSR p=0.002, WFE 87%
  - NYSE_OPEN MNQ: +8.2% WR spread, DSR p=0.013, WFE 92%
- 3 active validated strategies already use FAST filters
- Integration into discovery grid = NOT YET DONE (ACTION QUEUE #3)

### What's available:
- 1s bars: 3.9 GB, 16 years, all 3 instruments, all sessions, all UTC hours
- FREE data (no subscription needed)

## Step 1: DEPLOY the 1m break speed filter (FIRST)

Before testing 1s refinement, deploy what's already validated:
- Add FAST_MEDIAN composite filter to the discovery grid for the 3 validated sessions
- Re-run pipeline discovery with the new filter
- Validate the discovered strategies (same T1-T8 battery)
- Deploy to prop profiles if validated

This is the highest-ROI next step. No new research needed — the signal is proven.

## Step 2: Test 1s break speed (AFTER deployment)

### Question:
Does measuring break delay at 1-second resolution (instead of 1-minute) change the optimal threshold or WR spread for break speed filtering?

### Method:
1. Use ORB boundaries from daily_features (already computed, verified)
2. Load 1s bars from raw files (NOT from gold.db — these are research files)
3. For each break event: find first 1s bar closing beyond ORB boundary after ORB end
4. Compute break_delay_sec = that timestamp - ORB end timestamp
5. Compare break_delay_sec vs break_delay_min:
   - Correlation (should be high, r > 0.9)
   - Does 1s reveal sub-minute structure invisible at 1m?
   - Does optimal threshold change? (e.g., FAST120s vs FAST5m)
6. Test ALL sessions and instruments (honest K), not just the 3 validated
7. BH FDR at honest K
8. ATR confound control
9. Era check: pre-2019 (parent symbols) vs post-2019 (native micros)

### Contract alignment:
- 1s bars use parent symbol (.FUT) — contains multiple contract months + spreads
- MUST filter to outright contracts only (no `-` in symbol)
- MUST select front-month by volume (same as 1m pipeline)
- VERIFY prices match daily_features ORB boundaries (spot-check 5 random dates)

### Look-ahead check:
- break_delay_sec = time from ORB end to stop trigger = known at entry moment
- No look-ahead — same logic as 1m, just finer resolution
- Do NOT test post-break features (velocity, follow-through) — those are look-ahead

### Kill criterion:
- If 1s WR spreads are within 1% of 1m WR spreads for all sessions → 1m is sufficient
- If 1s reveals NO new sessions beyond the 3 already validated → 1s adds no discovery value
- Either kill outcome: deploy 1m version (already proven), archive 1s data

### Pass criterion:
- If 1s WR spread is >2% better than 1m for any validated session → worth integrating
- If 1s reveals new sessions with p < 0.05 (BH FDR) → novel discovery from 1s
- Either pass outcome: build 1s break delay computation into pipeline

## Step 3 (only if Step 2 passes): Novel 1s features

If 1s resolution proves useful, explore additional features only possible at 1s:
- ORB formation speed (when did the range boundaries get set?)
- Volume profile within ORB (front-loaded vs back-loaded)
- Pre-break momentum (grinding vs spiking approach to boundary)

These are NOVEL hypotheses — need full Gate 1-5 research sequence.
Do NOT explore these before Steps 1-2 are complete.

## Dependencies
- Step 1 requires: trading_app/config.py changes, discovery grid re-run
- Step 2 requires: research script reading raw 1s files + gold.db daily_features
- Step 3 requires: Step 2 to pass

## Estimated effort
- Step 1: ~1 day (filter integration + discovery + validation)
- Step 2: ~half day (research script + analysis)
- Step 3: ~1 day (if reached)
