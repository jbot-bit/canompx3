# Prompt: Volume DST Analysis + New Session Candidates

## Context

We run ORB breakout strategies on CME micro futures (MGC, MNQ, MES). We discovered that four fixed sessions (0900, 1800, 0030, 2300) drift ±1 hour relative to their named market events during DST. A 24-hour ORB time scan with winter/summer split revealed:

- Some times have STABLE edge (same in both regimes)
- Some times have edge that DIES in one regime (e.g., MES 0900 winter = negative)
- "Secret" times at 09:30 and 19:00 Brisbane have stable edge but aren't current sessions

Read `CLAUDE.md` section "CRITICAL: DST Contamination" and `docs/RESEARCH_ARCHIVE.md` sections "24-Hour ORB Time Scan" and "DST Strategy Revalidation" for full context.

## Task 1: Volume by DST Regime

**Question:** Does trading volume differ systematically between winter and summer at our session times? This tells us whether edge changes are liquidity-driven or structural.

### Method

For each instrument (MGC, MNQ, MES) and each of the following Brisbane times:
- Current sessions: 0900, 1000, 1100, 1800, 0030, 2300
- New candidates: 0830, 0930, 1045, 1700, 1900
- Dynamic session equivalents: CME_OPEN time (0800 summer / 0900 winter), LONDON_OPEN time (1700 summer / 1800 winter)

Compute for each time slot:
1. Mean 5-minute volume (sum of volume in the 5 bars starting at that time)
2. Median 5-minute volume
3. Mean volume in the 60 minutes AFTER the ORB window (break detection period)
4. Split all of the above by US DST regime (winter/summer)
   - EXCEPT for 1800/1700 where the split should use UK DST (Europe/London)

### DST classification
- US DST: Use `America/New_York`. Winter (EST) = UTC offset -5h. Summer (EDT) = UTC offset -4h.
- UK DST: Use `Europe/London`. Winter (GMT) = UTC offset 0. Summer (BST) = UTC offset +1h.
- Use `zoneinfo` (stdlib) — same approach as `pipeline/dst.py`.

### Output for Task 1

Console table per instrument:
```
Time | Winter Vol (mean/med) | Summer Vol (mean/med) | Ratio (W/S) | Post-ORB W | Post-ORB S | Ratio
```

Key analysis questions:
- Does volume drop significantly at 0900 in summer (when it's no longer CME open)?
- Does volume SPIKE at 0800 in summer (actual CME open) compared to winter?
- Is 19:00 volume comparable to 18:00? (validates it as tradeable)
- Is the "MES 0900 winter toxic" finding explainable by volume alone?
- Do Asia times (1000, 1100) show identical volume in both regimes? (They should — no DST in Asia)

## Task 2: New Session Candidates

Based on the time scan results, evaluate these candidate sessions for potential addition to the pipeline:

| Candidate | Instrument | Time Scan avgR (W/S) | Stability |
|-----------|-----------|---------------------|-----------|
| 09:30 | MGC | +0.14W / +0.18S | STABLE |
| 19:00 | MGC | +0.43W / +0.48S | STABLE |
| 10:45 | MNQ | +0.33W / +0.11S | WINTER-DOM |
| 10:45 | MES | +0.35W / +0.10S | WINTER-DOM |

For each candidate, compute:
1. How many G4+ days per year on average?
2. Average ORB size vs the nearest current session
3. Break rate (what % of G4+ days produce a break?)
4. Overlap with existing sessions: what % of trade days at the candidate are ALSO trade days at the nearest current session? (If >90% overlap, the candidate adds no new trades)
5. Volume comparison with nearest current session

### Output for Task 2

Summary table:
```
Candidate | G4+ days/yr | Avg ORB | Break Rate | Overlap with [nearest] | Volume ratio vs [nearest] | Verdict
```

Verdicts:
- **ADD**: unique trades, sufficient volume, stable edge
- **SKIP**: too much overlap with existing session (>80%)
- **INVESTIGATE**: promising but needs more data

## Database

- Path: `C:/db/gold.db` (DuckDB)
- Table: `bars_1m` — columns: `ts_event` (TIMESTAMPTZ, UTC), `symbol` (VARCHAR), `open`, `high`, `low`, `close`, `volume`
- Instruments: MGC, MNQ, MES
- Brisbane = UTC+10 (no DST, constant year-round)

## Important Notes

- Use DuckDB Python API (`import duckdb`)
- This is READ-ONLY research — no writes to gold.db
- For the dynamic session times (CME_OPEN, LONDON_OPEN), you need to compute the actual Brisbane time per-day using the same logic as `pipeline/dst.py`:
  - CME_OPEN: 5PM CT → winter (CST) = 09:00 Bris, summer (CDT) = 08:00 Bris
  - LONDON_OPEN: 8AM UK → winter (GMT) = 18:00 Bris, summer (BST) = 17:00 Bris
- Print progress — multiple instruments × multiple times
- Save full results to `research/output/volume_dst_analysis.csv`

## Validation

1. Asia times (1000, 1100) should show ~identical volume in both regimes (no DST in Japan/Singapore)
2. CME_OPEN time should show the highest volume of any time in the 0800-0900 range
3. LONDON_OPEN time should show higher volume than non-event times in the 1700-1900 range
4. Volume at maintenance break times (roughly 20:00-21:00 Bris winter / 21:00-22:00 Bris summer) should be near zero
