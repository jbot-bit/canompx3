# Prompt: Full 24-Hour ORB Time Scan

## Context

We trade ORB (Opening Range Breakout) strategies on CME micro futures (MGC, MNQ, MES, MCL). Currently we compute 5-minute ORBs at 11 specific times (7 fixed + 4 dynamic). These times were chosen based on named market events (Tokyo open, CME open, London open, etc.).

A recent DST edge audit revealed that fixed-clock-time ORBs sometimes outperform event-aligned dynamic ORBs. This suggests the edge may not come from the market event itself but from positioning flow at specific clock times. **We may be missing profitable ORB times that don't correspond to any named event.**

## Task

Build `research/research_orb_time_scan.py` that scans EVERY possible ORB start time across the 23-hour CME trading day, computes breakout quality metrics at each time, and ranks the best times per instrument.

## Database

- Path: `C:/db/gold.db` (DuckDB)
- Table: `bars_1m` — columns: `ts_event` (TIMESTAMPTZ, UTC), `symbol` (VARCHAR), `open`, `high`, `low`, `close`, `volume`
- Instruments: MGC, MNQ, MES, MCL
- MGC data: 2021-02-05 to 2026-02-04 (~5 years)
- MNQ/MES data: 2024-02-04 to 2026-02-04 (~2 years)
- All timestamps are UTC in the database
- Brisbane = UTC+10 (no DST, constant year-round)

## Method

### Step 1: Generate candidate times
- Every 15-minute increment across the CME trading day
- CME daily maintenance break: 4:00 PM - 5:00 PM CT (varies in UTC by DST — winter: 22:00-23:00 UTC, summer: 21:00-22:00 UTC)
- In Brisbane terms, scan from 00:00 to 23:45 in 15-minute steps = 96 candidate start times
- EXCLUDE times that fall within the CME maintenance window on any day (just skip those bars gracefully — if fewer than 5 bars exist in the window, mark that day as NO_ORB for that time)

### Step 2: For each candidate time, each instrument, each trading day
- Compute 5-minute ORB: high = max(high) of first 5 bars at/after start time, low = min(low) of first 5 bars
- ORB size = high - low
- Apply G4+ filter (ORB >= 4 points) — skip days where ORB < 4
- Look for first break in the 4 hours after ORB closes:
  - Break LONG: a 1m bar closes above orb_high
  - Break SHORT: a 1m bar closes below orb_low
  - If both sides break, use FIRST break direction
  - If no break in 4 hours, mark as NO_BREAK
- For broken days, compute simple outcome: did price reach 2R target (RR2.0) before hitting stop (other side of ORB)?
  - Scan bar-by-bar after break bar
  - Target LONG = entry + 2 * orb_size, stop LONG = orb_low
  - Target SHORT = entry - 2 * orb_size, stop SHORT = orb_high
  - Entry price = close of break bar (E1 model)
  - Max horizon: 8 hours after break
  - Win = +2.0R, Loss = -1.0R, Timeout = mark-to-market R at end of window

### Step 3: Aggregate per (instrument, candidate_time)
For each combo, compute:
- `n_days`: total trading days with valid 5m bars at that time
- `n_g4`: days passing G4+ filter
- `n_breaks`: days with a break (long or short)
- `break_rate`: n_breaks / n_g4
- `n_trades`: breaks with outcome computed
- `avg_r`: mean R-multiple across all trades
- `total_r`: sum of R-multiples
- `win_rate`: % of trades hitting +2.0R target
- `avg_orb_size`: mean ORB size in points on G4+ days
- `direction_bias`: (count_long - count_short) / n_breaks — how directionally skewed

### Step 4: Output

**Console output:**
1. Top 20 times per instrument ranked by `total_r` (minimum 30 trades)
2. Highlight which of these match our current 11 sessions (mark with *)
3. Show bottom 10 times (worst avgR) as contrast
4. Summary: how many candidate times have positive avgR vs negative

**Save to CSV:** `research/output/orb_time_scan_full.csv` with all 96 x 4 instrument rows

**Key analysis questions to answer in the console summary:**
- Are our current 11 sessions actually the best times, or are we missing better ones?
- Are there clusters of good times (suggesting a flow pattern) or are good times scattered?
- Do the DST-shifted event times (e.g., 0800 Brisbane = summer CME open) show up as separate opportunities?
- Is there a "dead zone" in the trading day where NO time has edge?
- Do different instruments peak at different times, or do they share the same time structure?

## Important Notes

- Use DuckDB Python API (`import duckdb`)
- Brisbane = UTC+10 always. To convert: `Brisbane hour H:MM` = `UTC (H-10):MM` (wrapping at midnight)
- This is a RESEARCH script — read-only, no writes to gold.db
- Print progress (e.g., "Scanning MGC... 48/96 times done") because this will be slow
- Don't use the pipeline modules — this is standalone research. Query bars_1m directly.
- Cost model: don't apply friction for this scan. We want raw edge signal. Friction analysis comes later for the winners.
- The 4-hour break window and 8-hour outcome window are deliberately generous — we want to find ANY time with breakout energy, then we can tune windows later.

## Expected Runtime

~96 times × 4 instruments × ~1000 trading days × bar-by-bar scanning. This will be slow. Optimize by:
- Bulk-loading all bars_1m for an instrument into a pandas DataFrame once
- Vectorize where possible
- Print per-instrument progress

## Validation

After the scan completes, verify:
1. Our known good sessions (MGC 1000, MGC 1800, MES 1000) should appear near the top — if they don't, something is wrong with the methodology
2. Times during the CME maintenance break should have 0 trades
3. MCL should show no/weak edge at most times (confirmed NO-GO instrument)
4. Total trade counts should be plausible given the date ranges
