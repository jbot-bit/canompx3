# Volume DST Analysis — Findings (Feb 2026)

## Key Finding: Volume Follows the Market Event, Not the Clock

The volume data conclusively shows that trading volume at DST-affected sessions shifts with the underlying market event. This means:

1. **Any edge at these sessions is event-driven, not time-driven**
2. **WINTER-DOM / SUMMER-DOM verdicts from the DST split are real signal, not noise**
3. **Dynamic sessions (CME_OPEN, LONDON_OPEN, etc.) are the correct approach** for capturing the event consistently year-round

## Session-by-Session Volume Evidence

### 0900 (CME Open in Winter)

| Instrument | Winter Vol | Summer Vol | W/S Ratio | Volume Drop |
|------------|-----------|-----------|-----------|-------------|
| MGC | 585 | 212 | 2.76x | -63.7% |
| MNQ | 4,105 | 1,598 | 2.57x | -61.1% |
| MES | 2,878 | 1,022 | 2.82x | -64.5% |

**Interpretation:** In summer, CME opens at 0800 Brisbane. By 0900, the opening surge is gone. The 63% volume drop proves 0900's edge is the CME open event. Any strategy at fixed 0900 should probably migrate to dynamic CME_OPEN.

### 2300 (US Data / Pre-Equity in Winter)

| Instrument | Winter Vol | Summer Vol | W/S Ratio | Direction |
|------------|-----------|-----------|-----------|-----------|
| MGC | 1,436 | 2,527 | 0.57x | Summer +76% |
| MNQ | 3,344 | 5,506 | 0.61x | Summer +65% |
| MES | 1,922 | 3,645 | 0.53x | Summer +90% |

**Interpretation:** Reverse of 0900. In summer, 2300 catches more US equity activity (market opened earlier). The edge here is also event-tied.

### 1800 (London Open in Winter)

| Instrument | Winter Vol | Summer Vol | W/S Ratio |
|------------|-----------|-----------|-----------|
| MGC | 1,038 | 793 | 1.31x |
| MNQ | 3,422 | 3,188 | 1.07x |
| MES | 2,064 | 1,774 | 1.16x |

**Interpretation:** Moderate shift. Gold (MGC) shows the strongest London-open sensitivity, which makes sense — London is the primary gold market. Equities less affected.

### 1700 (London Open in Summer — confirmation)

| Instrument | Winter Vol | Summer Vol | W/S Ratio |
|------------|-----------|-----------|-----------|
| MGC | 689 | 1,088 | 0.63x |
| MNQ | 2,124 | 3,090 | 0.69x |
| MES | 1,000 | 1,951 | 0.51x |

**Interpretation:** Summer volume at 1700 is 1.5-2x winter, confirming the London open event literally moves from 1800 to 1700. The 1800 winter volume and 1700 summer volume are comparable — same event, different clock time.

### 0030 (US Equity Open in Winter)

| Instrument | Winter Vol | Summer Vol | W/S Ratio |
|------------|-----------|-----------|-----------|
| MGC | 2,647 | 1,997 | 1.33x |
| MNQ | 41,620 | 20,701 | 2.01x |
| MES | 25,193 | 14,980 | 1.68x |

**Interpretation:** Massive volume at this time regardless of DST (it's near the US equity open). MNQ particularly affected — halves in summer because the equity open has already happened 1 hour ago.

### 1000 / 1100 (Asia — Clean)

No DST anywhere. Tokyo open (1000) and Singapore/HK (1100) are the same event year-round. Volume is stable. These sessions are the gold standard for DST-free analysis.

## New Session Candidates

| Candidate | G4+/yr | Overlap with Nearest | Vol Ratio | Verdict |
|-----------|--------|---------------------|-----------|---------|
| MGC 09:30 | 10 | 73% with 0900 | 0.60x | NOT VIABLE — too few G4+ days |
| MGC 19:00 | 13 | 53% with 1800 | 0.75x | NOT VIABLE — too few G4+ days |
| MNQ 10:45 | 240 | 100% with 1000 | 0.63x | SKIP — identical trade days |
| MES 10:45 | 45 | 77% with 1000 | 0.54x | NOT VIABLE — high overlap, low vol |

**Bottom line:** None of the new session candidates add independent value. The existing session grid is adequate. The real fix is dynamic sessions (already built), not new fixed times.

## Actionable Conclusions

1. **For validated strategies at 0900/1800/0030/2300:** The 4 WINTER-DOM strategies found in validation are likely real event-driven effects, not artifacts. They perform better in winter because they're catching the actual market event in winter.

2. **Recommendation for 0900 strategies:** Consider migrating to CME_OPEN dynamic session, which always captures the actual CME open regardless of DST. The 63% volume drop in summer means 0900 summer trades are in a fundamentally different (lower liquidity) environment.

3. **No new sessions needed:** All candidates either overlap too much or don't have enough G4+ frequency.

4. **Asia sessions remain the strongest foundation:** 1000/1100 are DST-immune, high-frequency, and volume-stable. Any portfolio should weight these heavily.

5. **Dynamic sessions are the long-term fix:** CME_OPEN, LONDON_OPEN, US_EQUITY_OPEN already exist in the pipeline. The volume data proves they should be the primary sessions for DST-affected times.
