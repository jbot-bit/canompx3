# Research Archive

Research findings, NO-GO archive, and alternative strategy results.
Supplements `TRADING_RULES.md`. Raw numbers preserved for reproducibility.

**Last Updated:** 2026-02-17

---

## Table of Contents

1. [Cross-Instrument Lead-Lag (MGC x MES x MNQ)](#cross-instrument-lead-lag)
2. [Hypothesis Test: Multi-Instrument Filter Optimization (Feb 2026)](#hypothesis-test-multi-instrument-filter-optimization)
3. [Alternative Strategy NO-GOs](#alternative-strategy-no-gos)

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
