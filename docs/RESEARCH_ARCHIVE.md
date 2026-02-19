# Research Archive

Research findings, NO-GO archive, and alternative strategy results.
Supplements `TRADING_RULES.md`. Raw numbers preserved for reproducibility.

**Last Updated:** 2026-02-17

---

## Table of Contents

1. [Cross-Instrument Lead-Lag (MGC x MES x MNQ)](#cross-instrument-lead-lag)
2. [Hypothesis Test: Multi-Instrument Filter Optimization (Feb 2026)](#hypothesis-test-multi-instrument-filter-optimization)
3. [Cross-Instrument Portfolio Analysis (P1) — NO-GO for 1000 LONG Stacking](#cross-instrument-portfolio-analysis-p1)
4. [DST Edge Audit — Preliminary Findings](#dst-edge-audit)
5. [DST Contamination Audit — Full Scope](#dst-contamination-audit)
6. [24-Hour ORB Time Scan with DST Split](#24-hour-orb-time-scan)
7. [DST Strategy Revalidation — All Affected Sessions](#dst-strategy-revalidation)
8. [Alternative Strategy NO-GOs](#alternative-strategy-no-gos)
9. [Edge Structure Analysis (Feb 2026)](#edge-structure-analysis-feb-2026)

---

## Cross-Instrument Lead-Lag

**Date:** 2026-02-17
**Scripts:** `research/research_cross_instrument.py`, `research/research_concordance_stacking.py`
**Data:** 579 overlapping trading days, Feb 2024 -- Feb 2026 (all 3 instruments)
**Status:** DOCUMENTED -- awaiting out-of-sample validation

### Question

Does watching MGC, MES, and MNQ simultaneously give predictive signal for ORB breakouts?

### Baseline: MES/MNQ Concordance

MES and MNQ are both US equity indices. Their break-direction concordance is ~87-89% across all shared sessions. This is a baseline fact, not a discovery.

| Session | Concordance | N |
|---------|------------|---|
| 0900 | 89% | 417/466 |
| 1000 | 88% | 434/496 |
| 1100 | 87% | 444/511 |
| 1800 | 87% | 442/510 |

### Finding 1: Concordance Filter (0900, 1100 ONLY)

When all 3 instruments break the same direction at a session, ORB strategy outcomes improve at **0900 and 1100**. Effect degrades gradually across tiers (3/3 > 2/3 > rest), suggesting a real mechanism.

**0900 E1/CB2/RR2.5/G5+ (representative):**

| Tier | WR | ExpR | Sharpe | N |
|------|-----|------|--------|---|
| 3/3 Concordant | 28.0% | +0.127 | 1.498 | 282 |
| 2/3 Majority | 20.5% | -0.063 | -0.838 | 346 |
| Remaining | 15.8% | -0.109 | -1.620 | 19 |
| Baseline (all) | 23.6% | +0.018 | 0.229 | 647 |

**1100 E1/CB4/RR2.5/G5+ (representative):**

| Tier | WR | ExpR | Sharpe | N |
|------|-----|------|--------|---|
| 3/3 Concordant | 36.5% | +0.185 | 1.872 | 323 |
| 2/3 Majority | 32.1% | +0.036 | 0.382 | 377 |
| Remaining | N/A | N/A | N/A | 0 |
| Baseline (all) | 34.1% | +0.105 | 1.082 | 700 |

Monotonic degradation observed across **every** 0900 and 1100 parameter combo tested (6 combos x 3 size filters each = 18 tests, all monotonic).

**Concordance does NOT help at 1000 or 1800.** At both sessions, the majority-2 tier outperforms concordant-3. Do not apply concordance filter to these sessions.

### Finding 2: Independence from ORB Size

Concordance is genuinely independent from ORB size filters. Overlap between concordant-3 days and MGC G5+ eligible days:

| Session | Concordant-3 & G5+ | Concordant-3 total | Overlap |
|---------|--------------------|--------------------|---------|
| 0900 | 42 | 209 | 20% |
| 1000 | 40 | 223 | 18% |
| 1100 | 73 | 233 | 31% |
| 1800 | 40 | 244 | 16% |

Concordance is NOT a proxy for large ORBs. It captures orthogonal information about cross-asset conviction.

### Finding 3: MGC 2300 ORB Size Gates MES/MNQ 0030

The size of MGC's 2300 ORB (relative to ATR) is a strong predictor of MES/MNQ 0030 quality. MGC 2300 break **direction** does not matter -- only **size**.

**Median split:** MGC 2300 ORB/ATR ratio = 0.094

**MES 0030 E3/CB1/RR1.5 (strongest result):**

| MGC 2300 ORB | WR | ExpR | Sharpe | N |
|--------------|-----|------|--------|---|
| Large (>= median) | 52.4% | +0.173 | 2.445 | 246 |
| Small (< median) | 38.8% | -0.126 | -1.816 | 240 |
| Baseline (all) | 45.6% | +0.023 | 0.328 | 487 |

**MNQ 0030 E3/CB1/RR1.5:**

| MGC 2300 ORB | WR | ExpR | Sharpe | N |
|--------------|-----|------|--------|---|
| Large (>= median) | 45.1% | +0.084 | 1.109 | 237 |
| Small (< median) | 40.1% | -0.035 | -0.468 | 232 |
| Baseline (all) | 42.6% | +0.023 | 0.305 | 470 |

Consistent across all 8 parameter combos tested for both MES and MNQ. Structural interpretation: a large gold ORB at 2300 (~8AM ET) signals an active overnight session, which predicts stronger equity breakouts at 0030 (~9:30AM ET NYSE open).

### Finding 4: Contrarian Signal -- NOT Real

MGC 2300 long -> MES 0030 short showed 69.4% WR (N=124) in the initial scan. However, the concordance stacking analysis confirmed that MGC direction has negligible predictive value for MES/MNQ 0030 outcomes. The contrarian finding was noise from a broad scan with no multiple comparison correction.

### Negative Findings

| Hypothesis | Result |
|-----------|--------|
| 1000 concordance filter | Majority-2 outperforms concordant-3. No benefit. |
| 1800 concordance filter | Majority-2 outperforms concordant-3. No benefit. |
| MGC 2300 direction predicts MES/MNQ 0030 | Direction splits are within noise. Only size matters. |
| Lead-lag at adjacent sessions (e.g. 0900->1000) | Lifts of 1.03-1.09 but most are trivially small with no multiple comparison correction. |
| MES/MNQ disagreement predicts next session | Rare events (49-68 days) with no consistent pattern. |

### Caveats

1. **579 overlapping days is thin.** Many conditional slices have N < 50.
2. **No multiple comparison correction.** ~50+ comparisons tested. P-values shown are raw.
3. **Feb 2024 -- Feb 2026 is one regime.** Gold and equities had a specific relationship during this period (both rallying, gold outperforming).
4. **Concordance reduces trade count.** Filtering to concordant-3 days cuts ~60% of trades. Wider confidence intervals.
5. **Outcomes use orb_outcomes pnl_r** (real strategy parameters with costs) for concordance stacking; daily_features outcome (RR=1.0, no costs) for the initial lead-lag scan.

### Re-Validation Trigger

**Do NOT implement concordance as a production filter until re-validated.**

Trigger: When MES + MNQ overlap days exceed **800** (currently 579). At that point, re-run both scripts. Every day from 2026-02-17 onward is genuine out-of-sample data for these findings.

```bash
# Check current overlap count:
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
r = con.execute(\"\"\"
    SELECT COUNT(*) FROM (
        SELECT trading_day FROM daily_features
        WHERE symbol IN ('MGC','MES','MNQ') AND orb_minutes=5
        GROUP BY trading_day HAVING COUNT(DISTINCT symbol) = 3
    )
\"\"\").fetchone()
print(f'Overlap days: {r[0]} (trigger: 800)')
con.close()
"

# Re-run when ready:
python research/research_cross_instrument.py
python research/research_concordance_stacking.py
```

If 0900/1100 concordance still shows monotonic degradation on the new 220+ days, proceed to option 2 (build ConcordanceFilter in config.py).

---

## Hypothesis Test: Multi-Instrument Filter Optimization

**Date:** 2026-02-17
**Script:** `scripts/tools/hypothesis_test.py`
**Data:** orb_outcomes + daily_features, all instruments, full history
**Status:** COMPLETED — 2 of 5 hypotheses passed, actionable changes identified

### Background

The ORB Size Deep Dive revealed that fixed-point gates (G4, G5, G6, G8) mean completely different things across instruments. A 5pt ORB on MNQ (~21,000 index) is 0.024% of price. A 5pt ORB on MGC (~$2,700 gold) is 0.19% — eight times bigger in relative terms. This prompted testing 5 alternative filter approaches.

### H1: G3 (>=3pt) filter for MNQ — NO-GO

**Question:** Does lowering the MNQ gate from G4 to G3 add real new edge?

The BAND 3-4pt rows show trades that G3 would add that G4 doesn't already capture:

| Session | N (3-4pt band) | avgR | WR |
|---------|---------------|------|----|
| 0900 | 9 | +0.59 | 66.7% |
| 1000 | 3 | -1.00 | 0.0% |
| 1100 | 12 | -0.59 | 16.7% |
| 1800 | 1 | -1.00 | 0.0% |

N=9 on 0900 is statistical noise. 1000/1100/1800 bands are actively toxic. MNQ ORBs rarely fall in the 3-4pt range (the instrument lives at 12pt+), so G3 is cosmetic for MNQ too — just for different reasons than MGC. **Stick with G4+ for MNQ.**

### H2: Band filter for MES (cap at 12pt) — PASS (1000 session only)

**Question:** Does capping MES ORBs at 12pt rescue edges by excluding toxic large-ORB days?

**MES 0900 — Cap HURTS:**

| Filter | N | avgR | totR |
|--------|---|------|------|
| G4+ no cap | 93 | +0.34 | +32.0 |
| G4-L12 capped | 86 | +0.25 | +21.7 |
| 12pt+ only | 7 | +1.47 | +10.3 |

The 12pt+ zone on 0900 is the BEST zone (WR=85.7%). Do NOT cap 0900.

**MES 1000 — Cap HELPS:**

| Filter | N | avgR | totR |
|--------|---|------|------|
| G4+ no cap | 121 | +0.28 | +34.2 |
| G4-L12 capped | 112 | +0.36 | +40.3 |
| G5-L12 capped | 67 | +0.52 | +34.7 |
| 12pt+ only | 9 | -0.68 | -6.1 |

Toxic zone confirmed on 1000 (WR=11.1%, avgR=-0.68). Capping at 12pt improves both avgR AND totR at every gate level. **Add band filters (G4-L12, G5-L12) to MES 1000 discovery grid.**

### H3: Percentage-based ORB filter — NO-GO (as primary)

**Question:** Does a universal % threshold (ORB size / price) work across instruments?

At 0.10%, all instruments show positive avgR, but comparison to fixed-point baselines:

| Instrument | Fixed gate | N | avgR | % filter (0.10%) | N | avgR |
|-----------|-----------|---|------|------------------|---|------|
| MGC 0900 | G5+ | 56 | +0.77 | >= 0.10% | 70 | +0.52 |
| MNQ 0900 | G4+ | 278 | +0.08 | >= 0.10% | 66 | +0.27 |
| MES 0900 | G3+ | 131 | +0.17 | >= 0.10% | 37 | +0.52 |

Fixed gates are already better (MGC) or have far more trades (MNQ, MES). At 0.15%, MES 1000 goes negative. **Not better than instrument-specific fixed gates. Do not add to discovery grid.**

### H4: ORB/ATR(20) ratio filter — HARD NO-GO

**Question:** Does normalizing ORB size to ATR(20) adapt to volatility regime?

| Instrument | 0.20x ATR | N | Baseline (fixed) | N |
|-----------|-----------|---|------------------|---|
| MGC 0900 | +0.30 | 9 | +0.77 (G5+) | 56 |
| MNQ 0900 | -0.41 | 5 | +0.10 (G4+) | 278 |
| MES 0900 | +0.73 | 5 | +0.17 (G3+) | 131 |

ORBs are inherently ~5-15% of daily ATR (one hour vs full day). Even the lowest threshold (0.20x) yields single-digit samples. **Scale mismatch is structural. Kill this hypothesis entirely.**

### H5: Direction filter by time-of-day — PASS (1000 session)

**Question:** Does filtering LONG-ONLY on Asia sessions and SHORT-ONLY on US sessions improve edge?

**1000 session (the clear winner):**

| Instrument | LONG avgR | LONG N | SHORT avgR | SHORT N | BOTH avgR | BOTH N |
|-----------|----------|--------|-----------|---------|----------|--------|
| MGC | +0.66 | 38 | -0.09 | 27 | +0.35 | 65 |
| MNQ | +0.26 | 178 | +0.03 | 203 | +0.14 | 381 |
| MES | +0.19 | 81 | +0.06 | 106 | +0.12 | 194 |

MGC 1000 shorts are NEGATIVE (-0.09 avgR). LONG-ONLY doubles avgR AND improves total R (+25.1 vs +22.5). MNQ 1000 LONG-ONLY nearly doubles avgR with N=178 (statistically robust).

**0900 session — NOT worth it.** Differences exist (MGC LONG +0.87 vs SHORT +0.63) but too small to justify halving trade count. Both directions work at 0900.

**1800 session — Dead regardless.** MES 1800: LONG=-0.23, SHORT=-0.15. No direction filter rescues a dead session.

**Action:** Add LONG-ONLY as a direction filter for 1000 session across MGC, MNQ, MES in discovery grid.

### Summary Scorecard

| Hypothesis | Verdict | Action |
|-----------|---------|--------|
| H1: G3 for MNQ | NO-GO | Stick with G4+. Band 3-4pt adds 9 trades. |
| H2: MES 12pt cap | PASS (1000 only) | Add band G4-L12, G5-L12 to MES 1000 grid. Do NOT cap 0900. |
| H3: % of price | NO-GO (primary) | Not better than tuned fixed gates. |
| H4: ORB/ATR ratio | HARD NO-GO | Scale mismatch is structural. |
| H5: Direction | PASS (1000) | Add LONG-ONLY for 1000 across all instruments. |

### Caveats

1. All tests use E1 (breakout) entry model only. E3 (retrace) may show different patterns for band/direction filters.
2. MES 0900 12pt+ zone (N=7, WR=85.7%) is suspiciously perfect — likely regime-specific. Monitor.
3. Direction filter halves N. MGC 1000 LONG N=38 is below REGIME threshold (50). Needs more data.
4. MNQ samples are post-Feb 2024 only (shorter history than MGC).

---

## Cross-Instrument Portfolio Analysis (P1)

**Date:** 2026-02-17
**Script:** `research/research_cross_instrument_portfolio.py`
**Status:** COMPLETED — NO-GO for 1000 LONG stacking

### Question

Does combining MGC + MNQ + MES at the 1000 session (LONG-ONLY direction filter) improve portfolio risk-adjusted returns via diversification?

### Key Results

**Daily R Correlation at 1000 LONG-ONLY:**

| Pair | Correlation | N (overlapping days) | Interpretation |
|------|------------|---------------------|----------------|
| MNQ / MES | +0.83 | ~200+ | Effectively the SAME trade |
| MGC / MNQ | +0.40 to +0.44 | ~200+ | Moderate — NOT the diversification freebie hoped for |
| MGC / MES | +0.40 to +0.44 | ~200+ | Same as above |

**Portfolio Sharpe Impact:**
Adding MNQ and/or MES to MGC at 1000 LONG-ONLY WORSENS portfolio Sharpe. The equity micros add correlated drawdown without enough independent edge to compensate.

### Verdict: NO-GO

Do NOT run both MNQ and MES at 1000 LONG-ONLY. Pick ONE equity micro per session if needed, but stacking all three instruments on the same session-direction signal provides no diversification benefit.

### Implications for P6 (Regime Hedging)

The original P6 hypothesis assumed MNQ/MES vol is uncorrelated with gold vol, making multi-instrument allocation a natural hedge. P1 partially disproved this: at the 1000 session where all three have confirmed LONG-ONLY edges, correlation is too high to diversify. True portfolio diversification requires a genuinely uncorrelated asset class (bonds, FX, ags).

### Next Steps

User plans to identify a truly uncorrelated micro futures product for proper portfolio diversification. Candidates to investigate: micro bonds (2YY, 10Y), micro FX (M6E, M6A), micro ags (MYM soybean?).

---

## DST Edge Audit

**Date:** 2026-02-17
**Script:** `research/research_dst_edge_audit.py`
**Status:** COMPLETED — preliminary findings, full 24-hour time scan pending

### Question

Fixed sessions (0900, 1800, 0030) drift ±1 hour vs actual market opens during DST. Does this misalignment degrade edge quality? Should we switch to dynamic (DST-aware) sessions?

### Section 2: Winter vs Summer Edge Split (0900 session)

| Instrument | Winter avgR | Summer avgR | Delta | Verdict |
|-----------|------------|------------|-------|---------|
| MGC | Higher | Lower | +0.18R | Winter >> Summer |
| MES | Higher | Lower | +0.35R | Winter >> Summer |
| MNQ | Higher | Lower | +0.09R | ~Equal, KEEP FIXED |

Winter = when fixed 0900 perfectly aligns with actual Asia open. Summer = 1 hour off. All instruments show winter advantage, but MNQ is robust to the shift.

### Section 3: Fixed vs Dynamic Head-to-Head (matched days)

| Instrument | Fixed avgR advantage | Verdict |
|-----------|---------------------|---------|
| MGC 0900 | +0.13R | Fixed beats dynamic |
| MES 0900 | +0.14R | Fixed beats dynamic |
| MGC 1800 | No overlapping days | Inconclusive |

### The Apparent Paradox

Winter edges are stronger (suggesting the actual market event timing matters), BUT on matched days, fixed sessions slightly outperform dynamic. Multiple possible explanations — DO NOT over-conclude:

1. **Pre-positioning theory:** Fixed-time ORB captures calm buildup before the event. Plausible but unproven.
2. **Two independent opportunities:** In summer, both the event time (0800) AND the fixed time (0900) may have separate edge. Winter conflates them because they're the same time.
3. **Sample size caution:** Head-to-head only has summer days (winter = identical times), so N is halved.
4. **We haven't tested the full clock:** Only 11 of 96 possible 15-min slots have been evaluated. There may be better times we've never looked at.

### Preliminary Verdicts (subject to revision after full time scan)

1. **Do NOT rule out dynamic sessions.** The +0.13R advantage is small and summer-only.
2. **Winter is genuinely better** — feed into P2 (calendar effect scan) as a seasonality signal.
3. **MNQ is most DST-robust** — only +0.09R gap, consistent edge year-round.
4. **MGC 1800 inconclusive** — no E3 data, no overlapping days.
5. **NEXT STEP: Full 24-hour ORB time scan** (`research/research_orb_time_scan.py`) to test ALL 96 possible start times across the trading day. This will reveal whether our 11 sessions are actually optimal or if we're missing hidden opportunities.

---

## DST Contamination Audit

**Date:** 2026-02-17
**Status:** AUDIT COMPLETE — remediation in progress

### Discovery

The DST edge audit (above) revealed a deeper systemic issue: four of our seven fixed sessions align with specific market events in one DST regime but miss by 1 hour in the other. Every metric ever computed on these sessions is a blended average of two different market contexts.

### Affected Sessions

| Session | Winter (std time) alignment | Summer (DST) alignment | Shift source |
|---------|---------------------------|----------------------|-------------|
| 0900 | = CME Globex open (5PM CST = 23:00 UTC = 09:00 Bris) | CME opened 1hr earlier at 0800 Bris (5PM CDT = 22:00 UTC) | US DST |
| 1800 | = London metals open (8AM GMT = 08:00 UTC = 18:00 Bris) | London opened 1hr earlier at 1700 Bris (8AM BST = 07:00 UTC) | UK DST |
| 0030 | = US equity open (9:30AM EST = 14:30 UTC = 00:30 Bris) | US equity opened 1hr earlier at 2330 Bris (9:30AM EDT = 13:30 UTC) | US DST |
| 2300 | ≈ US data release (8:30AM EST = 13:30 UTC ≈ 23:30 Bris) | US data released 1hr earlier at 2230 Bris (8:30AM EDT = 12:30 UTC) | US DST |

### Clean Sessions (no contamination)

| Session | Why clean |
|---------|----------|
| 1000 | = Tokyo Stock Exchange open (9:00 AM JST). Japan has NO DST. Permanently aligned. |
| 1100 | = Singapore open. Singapore has NO DST. Permanently aligned. |
| 1130 | = HK/Shanghai equity open. Neither has DST. Permanently aligned. |
| CME_OPEN | Dynamic session — `dst.py` resolver adjusts per-day via `zoneinfo`. |
| LONDON_OPEN | Dynamic session — resolver adjusts per-day. |
| US_EQUITY_OPEN | Dynamic session — resolver adjusts per-day. |
| US_DATA_OPEN | Dynamic session — resolver adjusts per-day. |

### Scope of Contamination

Everything downstream of `daily_features` for sessions 0900/1800/0030/2300 is contaminated:
- `daily_features` ORB columns for those sessions
- `orb_outcomes` rows for those sessions
- `experimental_strategies` discovered at those sessions
- `validated_setups` for those sessions
- `edge_families` containing those sessions
- All TRADING_RULES.md stats for 0900/1800/2300 session playbooks
- Hypothesis tests H1-H5 (used 0900 data for H1, H2)
- Cross-instrument portfolio analysis (partially — 1000 results are clean)

**NOT contaminated:** Raw bar data (bars_1m, bars_5m) is UTC and correct. The ORB computation itself is correct — it computes the ORB at the stated clock time. The problem is interpretation: the stated clock time maps to different market events depending on DST.

### Remediation Plan

1. ✅ 24-hour ORB time scan with winter/summer split — DONE (`research/research_orb_time_scan.py`)
2. Add winter/summer split to `strategy_validator.py` for sessions 0900/1800/0030/2300
3. Re-validate all strategies at affected sessions with split visible
4. Flag strategies where edge is >80% driven by one DST regime
5. For regime-dependent edges: determine if clock time or market event has edge, decide fixed vs dynamic
6. Update TRADING_RULES.md session playbooks with clean split numbers
7. Rule: ALL future research touching 0900/1800/0030/2300 MUST split by DST regime

### Key Implication

If a strategy at 0900 shows +0.30 avgR blended but is actually +0.48R winter / +0.12R summer, the real question is: is the winter edge because 0900 = CME open in winter? If so, you should trade CME_OPEN year-round (dynamic), not 0900 (fixed). The blended number hides this completely.

Conversely, if 0900 shows +0.30 blended and is +0.33W / +0.27S (stable), then the clock time itself has edge regardless of what market event it aligns with. In that case, keep the fixed session.

The time scan showed both patterns exist across different instruments and times.

---

## 24-Hour ORB Time Scan

**Date:** 2026-02-17
**Script:** `research/research_orb_time_scan.py`
**Status:** COMPLETED with DST winter/summer split on all 96 candidate times

### Method

Scanned every 15-minute increment (96 slots) across the 23-hour CME trading day. For each slot, computed 5-minute ORB, applied G4+ filter, measured break rate and RR2.0 outcomes using E1 entry. Split all results by US DST regime (winter = EST, summer = EDT).

### DST Stability Verdicts

| Verdict | Meaning |
|---------|---------|
| STBL | \|winter - summer\| <= 0.10R, both N >= 10. Reliable edge. |
| W>> | Winter avgR much better than summer. Edge may be event-aligned (winter = aligned). |
| S>> | Summer avgR much better than winter. Edge may be event-avoidance (summer = misaligned). |
| LOW-N | One regime has fewer than 10 trades. Cannot assess stability. |

### MGC Findings

- **09:30 STABLE** (+0.14W / +0.18S) — NOT our current 0900 session. 30 min later.
- **19:00 STABLE** (+0.43W / +0.48S) — NOT our current 1800 session. 1 hour later. Strongest stable edge. May be the actual edge we're capturing at 1800 but offset.
- **10:00 STABLE** (+0.15W / +0.08S) — Tokyo open, confirmed clean.
- **22:45 WINTER-DOMINANT** (+0.69W / +0.04S) — nearly all winter edge.
- **18:00 LOW-N for winter** — caveat: scan splits by US DST, but 1800 should use UK DST. UK and US DST overlap mostly but not perfectly (2-4 week gaps at transitions).
- 9 of top 20 are STABLE.

### MNQ Findings

- **10:00 perfectly STABLE** (+0.21W / +0.21S) — most balanced edge in entire scan.
- **10:45 WINTER-DOMINANT** (+0.33W / +0.11S) — still positive in summer but weaker.
- Most DST-robust instrument overall: 10/20 stable.

### MES Findings — MAJOR RED FLAGS

- **10:00 STABLE** (+0.23W / +0.22S) — the ONE reliable time for MES.
- **WARNING: 4 top-20 times have edge that DIES in winter** (09:45, 10:15, 12:15, 12:30 — all summer-only edges).
- **MES 0900 winter is ACTIVELY HARMFUL** (-0.21W avgR) — winter is when 0900 = CME open exactly. The alignment with the market event makes MES WORSE. Summer (misaligned) is better (-0.01S, nearly flat).
- 10:45 winter-dominant (+0.35W / +0.10S).
- MES top-ranked times in aggregate (like 10:30) are summer-dominated and may be fragile.

### Key Insights

1. **1000 is the anchor across all instruments.** DST-stable, clean (Japan no DST), confirmed edge. Build the portfolio around this time.

2. **Our current session times may not be optimal.** MGC 19:00 (STBL, +0.43/+0.48) looks better than our 18:00. MGC 09:30 (STBL) looks better than 09:00. These are different times that we've never explicitly tested as sessions.

3. **Market event alignment can HURT, not help.** MES 0900 in winter (= CME open exactly) is negative avgR. The noise at the event kills breakouts. This contradicts the assumption that "aligning with the event = better."

4. **Winter-dominant edges may be event-driven.** MGC 22:45 at +0.69W / +0.04S is likely US-data-release-driven. When the fixed time aligns with the event (winter), it fires. When misaligned (summer), it's noise.

5. **The 1800 LOW-N issue:** The scan uses US DST as the split variable, but 1800's relevant DST is UK-based. US and UK DST overlap ~90% but diverge by 2-4 weeks at spring/fall transitions. A UK-DST-specific split would give cleaner results for 1800. This is a known limitation, not a bug.

6. **"Secret ORBs" confirmed:** Times like 19:00, 09:30, 10:45 don't correspond to any named market event but have real, stable edge. The market has positioning flow patterns at clock times that aren't driven by exchange opens.

### Next Steps

1. Strategy revalidation with DST split (`research/PROMPT_dst_strategy_revalidation.md`) — quantify how many existing validated strategies are regime-dependent.
2. Consider adding 09:30 and/or 19:00 as new sessions based on stable edge evidence.
3. For 1800: run a UK-DST-specific split to get clean numbers.
4. For MES: seriously question whether 0900 should be traded at all given winter = negative avgR.

---

## DST Strategy Revalidation

**Date:** 2026-02-17
**Script:** `research/research_dst_strategy_revalidation.py`
**Output:** `research/output/dst_strategy_revalidation.csv`
**Status:** COMPLETED — 1272 strategies analyzed. No validated strategies broken.

### Method

Re-validated all strategies at sessions 0900/1800/0030/2300 by splitting trade days into winter (standard time) and summer (DST). Used correct reference timezone per session: US Eastern for 0900/0030/2300, UK London for 1800.

### Verdict Summary (1272 strategies)

| Verdict | Count | % | Meaning |
|---------|-------|---|---------|
| STABLE | 275 | 22% | Edge survives in both regimes. Trustworthy. |
| WINTER-DOMINANT | 155 | 12% | Stronger in winter, still positive in summer. |
| SUMMER-DOMINANT | 130 | 10% | Stronger in summer, still positive in winter. |
| SUMMER-ONLY | 10 | 1% | Edge DIES completely in winter. RED FLAG. |
| UNSTABLE | 48 | 4% | Large split between regimes. Caution. |
| LOW-N | 654 | 51% | Cannot assess — too few trades in one regime. |

### RED FLAGS: 10 Strategies Where Edge Dies

All 10 are **MES 0900 E1** — experimental (never validated). Edge dies in winter (when 0900 = CME open exactly), lives only in summer (when 0900 is 1 hour after CME open). Confirms time scan finding that MES at the CME open is toxic.

No validated strategies have broken edges. All production strategies survive both regimes.

### MGC 1800 Validated Strategies (UK DST split)

| Strategy | Combined avgR | Winter avgR (N) | Summer avgR (N) | Verdict |
|----------|--------------|-----------------|-----------------|---------|
| E1 RR2.0 CB1 G5 | +1.56 | +1.59 (15) | +1.54 (13) | **STABLE** ✅ |
| E3 RR2.0 CB1 G5 | +1.53 | +1.57 (15) | +1.48 (11) | **STABLE** ✅ |
| E3 RR2.5 CB1 G5 | +1.71 | +1.57 (15) | +1.90 (11) | UNSTABLE (still positive both) |

The two primary 1800 plays are DST-stable. Edge is genuine year-round.

### MNQ 1800 — Winter Monster (Unvalidated)

Multiple MNQ 1800 strategies showing avgR +1.3 to +2.5 with Sharpe 2-25 in winter only (N=47-52). Needs scrutiny — many identical trade-day hashes across filter levels suggests same underlying trades counted multiple times. Investigate before acting on these numbers.

### Key Conclusions

1. **No production strategies are broken by DST.** All validated setups survive both regimes.
2. **MES 0900 confirmed toxic in winter.** All 10 edge-dies strategies are MES at the CME open. Do NOT validate MES 0900 strategies without DST split.
3. **MGC 1800 is genuinely stable.** UK DST split confirms edge in both GMT and BST periods.
4. **51% of strategies are LOW-N** — cannot be assessed. This is expected given that splitting by DST halves the sample size.
5. **MNQ 1800 winter edge needs investigation** — could be seasonal opportunity or data artifact.

---

## Alternative Strategy NO-GOs

Summary table in `TRADING_RULES.md` under "What Doesn't Work (Confirmed NO-GOs)". Full analysis scripts in `research/` directory:

| Strategy | Script | Verdict |
|----------|--------|---------|
| Gap Fade (mean reversion) | `research/analyze_gap_fade.py` | NO-GO. Gold trends, doesn't mean-revert. |
| VWAP Pullback | `research/analyze_vwap_pullback.py` | NO-GO. No edge after costs. |
| Value Area | `research/analyze_value_area.py` | NO-GO. No predictive value. |
| RSI Reversion | `research/analyze_rsi_reversion.py` | NO-GO. Negative ExpR across all settings. |
| Session Fade | `research/analyze_session_fade.py` | NO-GO. Random after costs. |
| Session Cascade | `research/analyze_session_cascade.py` | NO-GO. Prior session range does not predict. |
| Multi-day Trend | `research/analyze_multiday_trend.py` | NO-GO. No improvement on established sessions. |
| Opening Drive | `research/analyze_opening_drive.py` | NO-GO. No risk structure. |
| Range Expansion | `research/analyze_range_expansion.py` | NO-GO. Friction kills. |
| ADX Overlay | `research/analyze_adx_filter.py` | NO-GO. Low-vol ExpR negative. |
| Volume Confirmation | `research/analyze_volume_confirmation.py` | NO-GO. Volume at break does not predict. |
| MCL (Micro Crude) breakout | `research/analyze_mcl_comprehensive.py` | PERMANENT NO-GO. Oil is structurally mean-reverting (47-80% double break). |
| Crabel Contraction/Expansion (session-level) | `research/research_contraction_expansion.py` | NO-GO. Expansion ratio is proxy for absolute ORB size (confound r=0.35-0.86). Within size bands: 2/72 significant < 3.6 expected by chance. NR-contraction: 2/24 significant = chance. |

---

## Edge Structure Analysis (Feb 2026)

**Date:** 2026-02-18
**Scripts:** `research/research_edge_structure.py`, `research/research_overlap_analysis.py`
**Data:** MGC (3,114 days), MNQ (625 days), MES (626 days). Feb 2020 - Feb 2026.
**Status:** COMPLETED
**Output:** `research/output/edge_structure_window_sensitivity.csv`, `edge_structure_size_distribution.csv`, `edge_structure_size_correlation.csv`

### Background

Phase 0A overlap analysis found ALL 11 candidate session pairs fall in GREY-ZONE: 95-100% shared break-days but R-correlation only 0.03-0.24. Same days breaking, different outcomes. Three structural questions were tested.

### Q1: Break Window Sensitivity

**Question:** Is the 100% shared-break overlap an artifact of the 240-min window?
**Method:** 3 CLEAN pairs (MNQ 1000v1015, MNQ 1130v1245, MES 1000v1015), tested at 30/60/120/180/240/360 min windows.

**Result: REAL OVERLAP.** Even at 30-min window, shared break stays 98-99%.

| Pair | 30min | 60min | 240min |
|------|-------|-------|--------|
| MNQ 1000v1015 | 99.2% | 100% | 100% |
| MNQ 1130v1245 | 98.4% | 100% | 100% |
| MES 1000v1015 | 98.1% | 100% | 100% |

Implication: nearby sessions genuinely break on the same days. This is not a window artifact.

### Q2: ORB Size Distribution by Time

**Question:** Does the edge come from WHEN you trade or HOW BIG the ORB is?
**Method:** All 16 sessions x 3 instruments, size-band avgR (G4-G6, G6-G8, G8+), DST split.

**Result: TIME MATTERS TOO.** At the same G6-G8 size band, avgR spread = 0.884 across sessions.

Top/bottom at G6-G8 (N>=20):
- MGC: 1900 ALL +0.652 (N=40) vs 0900 SUMMER -0.232 (N=61)
- MNQ: 2300 WINTER +0.444 (N=27) vs 1645 SUMMER -0.207 (N=61)
- MES: 1245 ALL +0.294 (N=85) vs 1545 WINTER -0.190 (N=24)

Implication: at the SAME ORB size, session timing produces structurally different follow-through. Size is necessary but not sufficient.

### Q3: ORB Size Correlation Between Low-R-Corr Pairs

**Question:** Are different outcomes from different ORB sizes (risk structure) or noise?
**Method:** Pairs from overlap_analysis.csv with |r|<0.3 AND shared>90%. Pearson r on paired ORB sizes.

**Result: MODERATE CORRELATION.** Average orb_size_r = +0.50, mean |size_diff| = 10.73, concordance = 64.4%.

Implication: ORB sizes between nearby sessions are positively correlated (not independent, not identical). The different outcomes are not fully explained by different sizes — timing contributes independently.

### Caveats

- IN-SAMPLE analysis (~500 MNQ/MES days, ~3100 MGC days)
- E1 entry, CB1, 5min aperture, RR2.0 only
- No walk-forward validation
- DST-affected sessions split by regime (no blended numbers)

### Combined Verdict

1. **Adding new pipeline sessions is LOW PRIORITY.** Q1 shows they break on the same days. Q3 shows moderate size correlation. The theoretical diversification benefit is limited.
2. **Band filters are INCONCLUSIVE — mostly data-limited, not disproven.** `research/research_band_sensitivity.py` tested 6 candidates with ±20% boundary shifts (184 rows, `research/output/band_sensitivity_results.csv`). Results split three ways:
   - **Genuinely failed:** MGC 1100 G6-G8 — baseline avgR already negative (-0.06) with N=76. Not a data problem.
   - **Promising but boundary-unstable:** MES 1000 G4-G6 — avgR positive in 47/48 shift cells, baseline +0.345, N=89. Only failed on >50% drop criterion at extreme shifts. Best candidate for 10-year rebuild.
   - **Data-limited (N<20 in band cells):** MES 1245 (N=9), MNQ 2300 winter (N=10), MNQ 2300 summer (N=16, all positive), MGC 1900 (N=31 but band cells shrink below 20). Cannot rule out or confirm — need the 10-year rebuild.
