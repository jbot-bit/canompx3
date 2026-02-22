# False Breakout Detection — Design

**Date:** 2026-02-22
**Scope:** MGC 0900 E1 (single instrument, single session)
**Approach:** Feature engineering + existing FDR pipeline (no ML)
**Two deliverables:** Data access CLI (`research/explore.py`) + break quality features

---

## Motivation

The system currently filters false breakouts using three deployed mechanisms: break speed (C3 at 1000), ATR contraction skip, and relative volume. This design adds a fourth layer — **break quality features** computed from 1m bar data in the pre-entry window.

**Key constraint:** E0 enters on the confirm bar itself (limit at ORB level), so there is near-zero detection window. False breakout detection is structurally suited to **E1/E3 only**, where the break bar and confirm bars are complete before entry.

**Data access gap:** Claude Code currently cannot query gold.db during conversations. A lightweight CLI tool enables real data exploration for this project and all future research.

---

## Part 1: Data Access CLI

### File: `research/explore.py`

**Dependencies:** `research.lib` (existing), `argparse`, `pandas`
**DB access:** Read-only via `query_df()`

### Subcommands

**`summary [INSTRUMENT]`** — DB health snapshot

Shows table row counts, date ranges, instruments, latest ingest date per instrument.

**`outcomes INSTRUMENT SESSION ENTRY_MODEL [--rr RR] [--cb CB] [--filter FILTER]`** — Strategy performance

Shows N, win rate, avgR, Sharpe, year-by-year breakdown for a specific session/entry/instrument combo. Optional filter arguments for G4+, direction, etc.

**`bars INSTRUMENT TRADING_DAY SESSION`** — 1m bars around breakout

Shows ORB range, break direction/time, outcome, and 1m OHLCV bars from 30 minutes before to 10 minutes after the break. Annotates bars with ORB/BREAK/CB labels.

**`compare INSTRUMENT SESSION ENTRY_MODEL [--feature FEATURE]`** — Wins vs losses

Side-by-side comparison of feature distributions for winning vs losing breakouts. Shows median, mean, delta, and p-value (Mann-Whitney U) for each feature.

**`strategies INSTRUMENT [--family]`** — Validated strategy summary

Shows validated strategy count, family count, ROBUST/WHITELISTED counts, top families by ExpR.

### Error Handling

- Zero rows: print `"No data found for {instrument} {session} {entry_model}"`, exit 0
- Missing/invalid instrument or session: print available values from DB, exit 1
- DB not found: print expected path, exit 1
- All queries use safe triple-join from `research.lib.query`

### Not Included

No writes to gold.db. No charts/matplotlib. No file output. No statistical tests beyond compare's Mann-Whitney.

---

## Part 2: False Breakout Feature Engineering

### Architecture

**New module:** `pipeline/break_quality.py`
**Called from:** `build_daily_features.py` after Module 3 (break detection)
**Storage:** New nullable columns in `daily_features` table

Rationale for separate module: `build_daily_features.py` is 800+ lines. Break quality features require reading 1m bars around the breakout (different query pattern). Follows existing pattern where DST logic lives in `pipeline/dst.py`.

### Data Reality Check (from actual queries)

Before designing features, queried the real data:

| Existing Signal | Win/Loss Split | Verdict |
|-----------------|---------------|---------|
| Break delay (speed) | Median 5.0 min for BOTH wins and losses | **Not predictive at 0900** (unlike 1000) |
| Break bar continues | 33.6% WR (continues) vs 24% WR (reverses) | Slightly predictive but reversal N=95 (thin) |
| Relative volume | Median 0.94 for BOTH wins and losses | **Not predictive at 0900** |
| ORB size | Win median 0.9, loss median 0.8 | Weak separation without G filter |

This killed two originally proposed features (volume acceleration, pre-break momentum via rel_vol) and confirmed we need **new** features from the 1m bars.

### Feature Candidates (5 initial)

| # | Feature | Column | Computation | Mechanism |
|---|---------|--------|-------------|-----------|
| 1 | Break bar body ratio | `orb_0900_break_body_pct` | `abs(close-open) / (high-low)` of break bar | Full-body = conviction, doji = indecision/noise probe |
| 2 | Break bar wick rejection | `orb_0900_break_wick_against` | Wick INTO the ORB as % of bar range (upper wick on long break, lower wick on short break) | Wick against = opposing pressure at breakout level |
| 3 | Pre-break approach speed | `orb_0900_approach_bars` | Count of last N bars before break that closed progressively closer to ORB boundary | Trending approach = pressure building. Choppy = random touch |
| 4 | ORB-window level tests | `orb_0900_level_tests` | Count of ORB-window 1m bars touching within 1 tick of the eventual break level without breaking | Multiple tests = well-known level, different dynamics than surprise break |
| 5 | Volume surge ratio | `orb_0900_vol_surge` | `break_bar_volume / avg(ORB-window bar volumes)` — compares to immediate context, not 20-day rolling | Surge = new participants. No surge = same participants, likely false |

**All five are zero-lookahead:** computed from break bar (complete before E1 entry) and ORB-window bars (pre-break).

**Dropped from consideration:**
- Relative volume (20-day baseline): data showed identical for wins/losses at 0900
- Post-break confirm bar behavior: break_bar_continues has N=95 for reversal case (too thin)
- Pre-break momentum via price slope: overlaps with approach speed, harder to define threshold

### Schema Changes

```sql
ALTER TABLE daily_features ADD COLUMN orb_0900_break_body_pct FLOAT;
ALTER TABLE daily_features ADD COLUMN orb_0900_break_wick_against FLOAT;
ALTER TABLE daily_features ADD COLUMN orb_0900_approach_bars INTEGER;
ALTER TABLE daily_features ADD COLUMN orb_0900_level_tests INTEGER;
ALTER TABLE daily_features ADD COLUMN orb_0900_vol_surge FLOAT;
```

All nullable — not all days have 0900 breaks.

### Pipeline Integration

```
build_daily_features.py
  Module 1: ORB range computation
  Module 2: Session stats
  Module 3: Break detection
  Module 4: Daily metrics
  Module 5: Break quality features (NEW) <- calls pipeline/break_quality.py
```

Idempotent: follows existing DELETE+INSERT pattern when rebuilding date ranges.

---

## Part 3: Validation Methodology

### RESEARCH_RULES.md Compliance

| Requirement | How We Meet It |
|-------------|---------------|
| Sample size | MGC 0900 E1 has N=2,470 trading days. Even filtered, expect N>200 for CORE |
| Multiple comparisons | 5 features tested individually = 5 tests. BH FDR applied. NOT combined into grid |
| p-value threshold | p<0.01 required for actionable. Exact values reported |
| Sensitivity analysis | Each feature threshold tested at +/-20%. Must survive |
| Mechanism | Each feature has stated mechanism (table above). No mechanism = rejected |
| Walk-forward | Surviving features go through `strategy_validator.py`. WFE>50% required |
| DST split | 0900 is US-DST-sensitive. All results split by `us_dst`, both halves reported |
| In-sample vs OOS | Discovery on 2016-2023. Validation on 2024-2026. Clearly labeled |

### Testing Protocol (per feature)

1. Compute feature for all MGC 0900 trading days
2. Split outcomes by feature quartiles (Q1-Q4)
3. Compare avgR and WR across quartiles — is there a monotonic relationship?
4. If monotonic: pick threshold at Q2/Q3 boundary, test at +/-20%
5. If robust: add as filter column, run through discovery + FDR
6. If FDR-surviving: walk-forward validate
7. If WFE>50%: deploy

### What We Are NOT Doing

- No combining features into composite scores (ML territory)
- No grid search across feature thresholds (Bailey Rule: 5 features x existing grid = 11,880 combos)
- No testing across other sessions until 0900 is validated (one session at a time)
- No "guilty until proven" indicators (RSI, MACD, etc.)

---

## Part 4: Success Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| At least 1 feature survives FDR | BH-adjusted p<0.01 | 5 tests is small, but rigorous |
| Surviving feature improves avgR | Delta >= +0.05R vs unfiltered | Meaningful, not noise |
| Sensitivity holds | Threshold +/-20% keeps p<0.05 | Not curve-fitted to exact cutoff |
| Walk-forward passes | WFE>50%, no catastrophic OOS periods | Not overfit |
| N remains viable | Filtered N >= 100 (PRELIMINARY) | Feature doesn't kill sample size |
| DST-stable | Works in both DST halves, or clearly labeled regime-conditional | Per RESEARCH_RULES.md mandatory |

### Failure Modes (documented, not hidden)

- **All 5 features show no quartile separation:** NO-GO. False breakout detection at 0900 isn't viable from 1m OHLCV alone. Document and move on.
- **Features work in-sample but fail walk-forward:** Overfit. Document as statistical observation, not finding.
- **Features work but kill N below 100:** REGIME class only. Conditional signal, not standalone filter.
- **Features only work in one DST half:** Regime-conditional. Label accordingly, do not blend.

---

## Deliverable Summary

| Deliverable | File | Purpose |
|-------------|------|---------|
| Data access CLI | `research/explore.py` | Permanent infrastructure for all future research |
| Break quality module | `pipeline/break_quality.py` | Compute 5 features from 1m bars |
| Schema migration | `pipeline/init_db.py` update | Add 5 nullable columns to daily_features |
| Pipeline integration | `build_daily_features.py` update | Call break_quality after Module 3 |
| Validation research script | `research/research_break_quality_validation.py` | Quartile analysis + FDR + walk-forward |

---

## Open Questions (to resolve during implementation)

1. **Approach bars definition:** "Progressively closer" needs a precise mathematical definition. Monotonically decreasing distance to ORB boundary? Or just net closer over N bars?
2. **Level test threshold:** "Within 1 tick" is instrument-specific. For MGC, 1 tick = $0.10. May need to parameterize as percentage of ORB size instead.
3. **Volume surge baseline:** Average of all ORB-window bars, or just the last 3? The 5-minute ORB has exactly 5 bars — averaging all 5 may be noisy.
