# Research Archive

Research findings, NO-GO archive, and alternative strategy results.
Supplements `TRADING_RULES.md`. Raw numbers preserved for reproducibility.

**Last Updated:** 2026-02-17

---

## Table of Contents

1. [Cross-Instrument Lead-Lag (MGC x MES x MNQ)](#cross-instrument-lead-lag)
2. [Alternative Strategy NO-GOs](#alternative-strategy-no-gos)

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
