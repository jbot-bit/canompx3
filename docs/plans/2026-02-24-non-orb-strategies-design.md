# Non-ORB Strategy Research Design

**Date:** 2026-02-24
**Goal:** Discover genuinely uncorrelated alpha sources to diversify the existing ORB breakout portfolio.
**Approach:** Hybrid — mechanism-first archetypes with targeted feature exploration.
**Instruments:** All 7 (MGC, MES, MNQ, M2K, MCL, M6E, SIL)
**Timeframes:** Intraday (prop-compatible) + multi-day (own capital)

---

## Self-Audit Notes

- **Gap Fade KILLED:** 23-hour futures have median gaps of 3% ATR. Not tradeable.
- **Replaced with Failed Breakout Fade** — profits when ORB fails (negative ORB correlation).
- **Vol Regime split:** Contraction variant is an ORB overlay (correlated). Only expansion-fade is different.
- **Cross-Instrument:** Direction doesn't predict (RESEARCH_RULES L168). Reframed as vol-signal.
- **Gold trends intraday** (RESEARCH_RULES L181) — mean reversion archetypes may fail for MGC. Flagged.

---

## Six Strategy Archetypes

### 1. Failed Breakout Fade (Mean Reversion, Intraday)

**Mechanism:** Trapped breakout traders create reverse flow when they stop out after a failed ORB break. Double-break days show this pattern.

**Entry:** After ORB breaks direction D, if price returns inside range within N bars, enter opposite to D.
**Exit:** Target opposite ORB boundary. Stop if original breakout resumes past entry.
**Time stop:** Session end.

**Parameters:** N_return_bars ∈ {5, 10, 15, 20}, RR ∈ {0.5, 1.0, 1.5}
**Grid:** 12 combos × sessions × instruments
**Expected ORB correlation:** Negative (profits when ORB fails)

### 2. Late-Session Reversal (Mean Reversion, Intraday)

**Mechanism:** End-of-session profit-taking and position squaring causes reversion from intraday extremes.

**Entry:** At T-90min or T-60min before session close, if price is extended > N×ATR from session midpoint, trade opposite direction.
**Exit:** Return to session midpoint. Stop: further extension by 0.5×ATR.
**Time stop:** Session close (forced flat).

**Parameters:** Extension threshold ∈ {0.5, 0.75, 1.0, 1.5} ATR, Entry time ∈ {T-60, T-90}
**Grid:** 8 combos × sessions × instruments
**Expected ORB correlation:** ~0 (different timing and logic)

### 3. Multi-Day Momentum (Trend Following, Multi-Day)

**Mechanism:** Macro trends persist over multiple days due to institutional rebalancing (Moskowitz et al., 2012).

**Entry:** If N-day close-to-close return is in top/bottom quintile, enter at next session open.
**Exit:** Fixed hold ∈ {1, 2, 3, 5} days, or trailing stop 1.5×ATR.

**Parameters:** Lookback ∈ {3, 5, 10, 20} days, Exit type ∈ {1d, 2d, 3d, trailing}
**Grid:** 16 combos × instruments
**Expected ORB correlation:** ~0.1 (different timeframe)

### 4. Vol Expansion Fade (Vol Regime, Intraday)

**Mechanism:** After volatility explosion (ATR expansion), vol mean-reverts. Range-bound behavior follows directional bursts.

**Entry:** If ATR(5)/ATR(20) > 1.3, trade mean-reversion next session (fade prior day's direction).
**Exit:** Tight target (0.5×ATR). Stop: 1.0×ATR.
**Time stop:** Session end.

**Parameters:** ATR ratio threshold ∈ {1.2, 1.3, 1.5, 1.8}, Entry direction ∈ {fade_prior, fade_extreme}
**Grid:** 8 combos × instruments
**Expected ORB correlation:** ~0 (trades vol regime, not price direction)

### 5. Cross-Instrument Vol Signal (Cross-Asset, Intraday)

**Mechanism:** Volatility transmits across asset classes with time lag. Large overnight moves in one market signal higher activity in the next session's market.

**Signal:** If leader instrument's overnight_range > 1.5× its ATR(20), trade follower's next session breakout with wider targets.
**Leader-follower pairs:**
- MGC overnight → MES/MNQ US equity open
- M6E (EUR/USD) → MGC London open
- MCL (oil) → MES US equity open

**Parameters:** Vol threshold ∈ {1.0, 1.5, 2.0}, Target multiplier ∈ {1.0, 1.5, 2.0}
**Grid:** ~30 total tests
**Expected ORB correlation:** ~0.2 (enhances ORB on specific days)

### 6. VWAP Reversion (Mean Reversion, Intraday)

**Mechanism:** Institutional orders benchmark to VWAP. Extended deviation triggers institutional flow back to VWAP. Persists because it's driven by execution mandates.

**Entry:** When price deviates > N×σ from running VWAP (computed from session start), trade back toward VWAP.
**Exit:** Return to VWAP (full) or VWAP ± 0.5σ (partial). Stop: deviation extends by 1σ.
**Time stop:** Session close.

**Parameters:** σ multiplier ∈ {1.5, 2.0, 2.5, 3.0}, Target type ∈ {full, partial}
**Grid:** 8 combos × sessions × instruments
**Expected ORB correlation:** ~0 (continuous signal, mean reversion)

---

## Validation Framework

### Phase 1: Raw Signal Test
- Compute signal and backtest per instrument
- Apply friction costs per instrument cost model
- Output: N, WR, ExpR, Sharpe, MaxDD

### Phase 2: Statistical Validation
- Min N: 30 REGIME, 100 PRELIMINARY (RESEARCH_RULES.md)
- t-test on mean PnL_R ≠ 0 (exact p-values)
- Year-by-year breakdown
- BH FDR across all tests within each archetype
- Sensitivity: ±20% on key parameter

### Phase 3: ORB Correlation
- Compute daily PnL correlation with representative ORB strategy per instrument
- Target: |correlation| < 0.3 for "genuinely uncorrelated"

### Phase 4: Honest Report
SURVIVED SCRUTINY / DID NOT SURVIVE / CAVEATS / NEXT STEPS format.

---

## Implementation Plan

**Phase 1 script:** `research/research_non_orb_daily.py`
- Archetypes 3 (Multi-Day Momentum), 4 (Vol Expansion Fade), 5 (Cross-Instrument Vol)
- Uses daily_features only. Fast (~2 min).

**Phase 2 script:** `research/research_non_orb_intraday.py`
- Archetypes 1 (Failed Breakout Fade), 2 (Late-Session Reversal), 6 (VWAP Reversion)
- Scans 1m bars. Slower (~15-30 min).

**Aggregation:** Final section in each script computes FDR correction and ORB correlation.

---

## Total Test Count
- Archetype 1: ~12 × 14 sessions × 7 instruments = ~1,176 (but many session/instrument combos won't have breaks)
- Archetype 2: ~8 × 14 × 7 = ~784
- Archetype 3: 16 × 7 = 112
- Archetype 4: 8 × 7 = 56
- Archetype 5: ~30
- Archetype 6: ~8 × 14 × 7 = ~784

**Conservative estimate:** ~500-800 valid tests after filtering sessions with insufficient data.
**BH FDR at 0.05:** With 700 tests, expect ~35 false positives. Need p < 0.007 to survive FDR.

---

## Results (2026-02-24)

### Phase 1: Daily-Level Strategies
- **Script:** `research/research_non_orb_daily.py`
- **Tests:** 157 valid (N>=30)
- **FDR survivors:** 0
- Multi-Day Momentum: uniformly negative across all instruments
- Vol Expansion Fade: best signal MES ATR>1.8 (N=35, p=0.009) but REGIME-only and not FDR-significant
- Cross-Instrument Vol: promising direction (MGC→M2K) but not FDR-significant

### Phase 2: Intraday Strategies (1m bar scanning)
- **Script:** `research/research_non_orb_intraday.py`
- **Tests:** 383 valid (N>=30), 756 total
- **FDR survivors:** 0
- **Runtime:** 17.8 minutes across 7 instruments
- Failed Breakout Fade: uniformly negative (avg -0.28R). Confirms ORB momentum is real.
- Late-Session Reversal: barely triggers. Only 14 tests reached N>=30.
- VWAP Reversion: confirmed ANTI-EDGE. p=1.0 everywhere (168 tests). Fading VWAP systematically loses.

### Combined Results
- **540 valid tests across 6 archetypes × 7 instruments**
- **0 FDR survivors**
- **Verdict: DID NOT SURVIVE SCRUTINY**

### Key Findings
1. ORB breakout momentum is REAL — fading failed breaks is systematically negative (-0.28R avg)
2. Micro futures TREND during sessions — VWAP reversion is an anti-edge (p=1.0)
3. Gold is the worst for mean reversion (strongest trending behavior)
4. Multi-day momentum doesn't work at 1-5 day holds with ATR stops
5. Only SIL 0030 Failed Breakout Fade showed weak positive signal (+0.22R, N=34, p=0.14) — needs 3+ years

### Conclusion
The ORB breakout system is the correct strategy class for these instruments. Diversification should come from ORB session/configuration expansion, portfolio construction, and regime overlays — not from mean-reversion alternatives.
