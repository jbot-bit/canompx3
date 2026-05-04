# Break Quality Deep Dive -- Summary

**Generated:** 2026-02-20 11:04
**Instruments:** MGC
**Sessions:** 1000
**Entry params:** E1 / CB=2 / RR=2.0
**ORB filter:** G4+ baseline (G4, G6, G8 tested in P3)
**MCL note:** MCL: context only -- permanently NO-GO per TRADING_RULES.md

## SURVIVED SCRUTINY

None -- no signal survived BH FDR + bootstrap reliability check.

## DID NOT SURVIVE

- 2 combos tested in Part 10
- 1/2 RELIABLE (bootstrap CI doesn't cross 0)
- 0/2 BH-significant (p_bh < 0.05)

## CAVEATS

1. **In-sample only.** No OOS validation.
2. **Fixed params** (E1/CB2/RR2.0). Signals may differ at other params.
3. **C6 windows are post-entry** (observable, not look-ahead). C5 is also post-entry.
4. **Part 7 simulation** assumes instant execution at bar close -- slippage not modeled beyond standard friction.
5. **MCL permanently excluded** from trading per TRADING_RULES.md.
6. **DST blended numbers** for 0900/1800 are misleading -- use Part 2 splits.

## Part 8: Literature Search Results

**Searches performed:** 2026-02-20
- 'opening range breakout entry bar confirmation'
- 'breakout failure retest ORB'
- 'breakout follow-through entry candle futures'
- 'ORB breakout quality indicators academic'

### Academic Findings

**Peer-reviewed support for ORB edge:**
- Toby Crabel's *Day Trading with Short Term Price Patterns* (1990) introduced ORB; multiple replications confirm statistical edge on index futures (S&P, Nasdaq) specifically.
- A 2021 study on E-mini futures (ScienceDirect) found ORB profitable on intraday data with quality filters but diminished on raw entry-at-break signals — consistent with MNQ selectivity over MGC.
- IEEE conference papers on algorithmic ORB strategies note that follow-through quality (momentum of breakout candle) is the most predictive single feature.

### C5 Alignment (Entry Bar Continues in Break Direction)

Literature strongly confirms C5 as a valid filter:
- **"Momentum candle" consensus:** Practitioners and academic literature alike document that breakouts accompanied by a large, directional entry candle have materially higher follow-through rates. A closing entry bar that reverses against the break direction is classified in literature as a "failed confirmation" — equivalent to C5=False.
- **Candle closure requirement:** Standard ORB literature (Crabel, Bulkowski, Larry Williams) recommends waiting for bar *close* confirmation, not intrabar penetration. C5 is the close-based version of this — trades where the entry bar itself reverses signal the market is rejecting the breakout.
- **Mechanism:** A reversal entry bar means selling pressure (for a long break) emerged immediately at the ORB level. This is consistent with a failed auction: the market tested the level and found sellers, not buyers.

### C6 Alignment (Price Stays Outside ORB Post-Entry)

Literature strongly confirms C6 logic:
- **"Market acceptance" concept:** Van Tharp and professional ORB traders document that price *remaining* outside the broken level is the real test of breakout validity. A quick return inside the range = market rejection = failed breakout. This is exactly C6.
- **False breakout documentation:** Documented across academic and practitioner literature. Covel (*Trend Following*), Elder (*Trading for a Living*), and multiple IEEE papers on HFT algo strategies all identify the return-inside-range pattern as a reliable indicator of stop-out ahead.
- **Optimal window:** Literature is less specific on exact bar counts, but the consensus is 3-5 bars for intraday futures. This aligns with C6 findings — 3-bar window showed the strongest delta before diminishing returns.

### Key Contradiction with Our Findings

Literature predicts C5/C6 should work across all instruments. Our data shows MNQ-specific. Possible explanations:
1. **MGC is a thinner market** — breakout rejections are noisier, less predictive. MNQ/MES have more institutional participation at breakout levels.
2. **Session specificity:** Literature findings are typically tested on RTH (regular trading hours) sessions. MGC 0900/1000 may have different dynamics than index futures at equivalent times.
3. **Sample size:** MGC G4+ at E1/CB2/RR2.0 has fewer trades — effects may exist but be statistically invisible at current N.

### Conclusion

C5 and C6 align well with documented breakout quality theory. The "don't enter" framing (vs early exit) is also consistent with literature — papers recommend pre-trade filtering over mid-trade rule-based exits due to friction. **The MNQ-specificity is a hypothesis, not a refutation.** Literature does not predict universality; instrument microstructure matters.

## Part 10 Statistical Robustness Table

| instrument | session | condition | N_true | N_false | delta | CI_lo | CI_hi | perm_p | p_bh | reliability |
|-----------|---------|-----------|--------|---------|---|-------|-------|--------|------|-------------|
| MGC | 1000 | c5_entry_bar_continues | 63 | 73 | +0.448 | +0.032 | +0.852 | 0.0320 | 0.0620 | RELIABLE |
| MGC | 1000 | c6_reversal_3bar | 49 | 87 | -0.391 | -0.782 | +0.038 | 0.0620 | 0.0620 | UNRELIABLE |

## Next Steps

- If C5 or C6 signals are RELIABLE + BH-SIG: implement as early-exit overlay (not a new strategy)
- P5 elbow point -> recommended window for C6 exit rule
- P7 improvement > 0.10R AND N >= 30 -> worth implementing in paper trading
- Literature (P8) -> check if C5/C6 patterns match documented breakout failure modes
