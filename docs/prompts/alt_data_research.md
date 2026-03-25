# Alternative Data Research — Honest Assessment Prompt

Integrates with existing canompx3 pipeline, respects existing validation methodology.

---

## PROMPT START

I run a multi-instrument futures ORB breakout system (MNQ, MGC, MES) with a DuckDB pipeline,
12 DST-aware sessions, and a rigorous validation framework (T0-T8 audit battery, BH FDR,
walk-forward efficiency). The system generates ~65R/year across 4 validated MNQ lanes.

I want to investigate whether alternative data sources (prediction markets, geopolitical
event data, sentiment, macro indicators) can improve this system. But I have hard constraints
from prior research that MUST be respected.

**Your job: determine whether alternative data is worth pursuing AT ALL for this specific
system, before building anything.**

---

### HARD CONSTRAINTS FROM EXISTING RESEARCH (non-negotiable)

1. **ML is BLOCKED.** 3 open methodology FAILs prevent ML from production:
   - EPV = 2.4 (need ≥10) — insufficient positive samples per feature
   - Trained on sessions with negative baselines (violates de Prado AIFML Ch 3.6)
   - Selection bias (sessions selected then tested on same data)
   Alternative data features CANNOT be fed into an ML model until these FAILs are resolved.
   Alternative data might HELP resolve FAIL #1 (by adding high-value features that reduce
   the feature count needed), but this must be demonstrated, not assumed.

2. **Session-specificity is structural.** Each of the 12 sessions has different market
   structure, participant mix, and filter response. A geopolitical signal that predicts
   gold during SINGAPORE_OPEN (thin Asian liquidity, flight-to-safety) may be meaningless
   during NYSE_OPEN (high liquidity absorbs shocks). All analysis must be PER-SESSION.
   Never pool across sessions.

3. **Active instruments: MNQ, MGC, MES.** Not gold-only. GDELT conflict events affect
   equity futures (MNQ, MES) as much or more than gold. The research must cover all three.

4. **Existing validation framework governs.** Every alternative data feature must pass
   the existing T0-T8 audit battery from `quant-audit-protocol.md`, not a parallel framework.
   Standards: BH FDR correction, WFE > 0.50, sensitivity ±20%, per-year stability ≥7/10,
   cross-instrument consistency. These are not negotiable.

5. **2026 holdout is sacred.** No alternative data feature may be evaluated on 2026 data
   during discovery. Walk-forward OOS windows end at 2025-12-31.

6. **DuckDB single-writer constraint.** Never run two write processes against gold.db
   simultaneously. Alternative data goes in a SEPARATE database file or separate schema
   with controlled merge points.

7. **No-lookahead at schema level.** All external features joined at last known value
   STRICTLY BEFORE each candle/bar open. t-1 or earlier. Timestamp assertions mandatory.

---

### PHASE 0 — COST-BENEFIT GATE (do this first, kills the project if numbers don't work)

Before writing a single line of ingestion code, answer:

**Q1: How much improvement is realistically available?**
- Current portfolio: ~65R/year across 4 MNQ lanes
- If alternative data adds a regime filter that avoids 10% of losing trades: ~+6.5R/year
- If it adds a sizing signal that increases size on high-conviction days: ~+3-10R/year
- If it identifies new tradeable sessions/times: ~+15-30R/year (but US_DATA_1000 does this already without alt data)
- REALISTIC RANGE: +5 to +15R/year for a regime overlay. Maybe +20R if it unlocks new sessions.
- Is +5-15R/year worth 100-200 hours of engineering? At $2/point MNQ, that's $10-30/year per contract. Your answer may be no. Be honest.

**Q2: How many comparable geopolitical events exist in the prediction market data window?**
- Polymarket launched mid-2020. ~5.5 years of data.
- How many "geopolitical shock → gold move" events exist in that window?
- If N < 30: your own RESEARCH_RULES.md classifies this as INVALID sample size. STOP.
- If N = 30-99: REGIME class only — conditional signal, never standalone.
- If N ≥ 100: PRELIMINARY — worth investigating further.
- Count them. Iran ceasefire is N=1. How many more?

**Q3: Does the signal even exist at the granularity that matters?**
- Your ORB trades last 5-30 minutes. Gold moves 0.5-2% on a big day.
- Prediction market probabilities move on a multi-hour to multi-day timescale.
- Is there actually enough temporal resolution to predict WHICH 15-minute ORB window benefits?
- Or is this a daily regime signal at best?

**KILL GATE:** If Q1 < +5R/year realistic, or Q2 < 30 events, or Q3 = "daily only and existing ATR/vol filters already capture regime" → STOP. The existing system is near-optimal and alternative data adds complexity without proportional return. Report this finding. Don't build to justify the research.

---

### PHASE 1 — HISTORICAL DATA COLLECTION (if Phase 0 passes)

Collect HISTORICAL data only. No live pipeline. No real-time polling. That comes after
evidence is established.

#### 1A — Prediction Market Data

Sources (priority order):
1. Jon-Becker/prediction-market-analysis (GitHub) — free, largest public dataset
2. PredictionData.dev — tick-level Polymarket + Kalshi, 3+ years
3. Kalshi API — CFTC-regulated, economic indicator markets

Filter for categories relevant to ALL active instruments (not just gold):
- Gold-relevant: Iran, Israel, Russia/Ukraine, sanctions, oil, OPEC, tariffs
- Equity-relevant: Fed rate, recession, trade war, China, fiscal policy
- Cross-asset: VIX events, credit events, election outcomes

Store in SEPARATE DuckDB file: `alt_data.db` (not gold.db — single-writer constraint).

Schema:
```sql
CREATE TABLE prediction_markets_raw (
    market_id TEXT,
    source TEXT,           -- 'polymarket', 'kalshi', 'manifold'
    question TEXT,
    category TEXT,         -- standardized: 'geopolitical', 'economic', 'policy'
    asset_relevance TEXT,  -- 'gold', 'equity', 'both', 'unknown'
    timestamp_utc TIMESTAMPTZ,
    probability FLOAT,
    volume_usd FLOAT,
    resolution TEXT,       -- 'yes', 'no', 'unresolved'
    resolved_at TIMESTAMPTZ
);
```

#### 1B — GDELT Geopolitical Data

Source: GDELT Project (free, 1979-present via BigQuery or REST API).

Pull:
- CAMEO event codes 18x (assault), 19x (war), 12x (ceasefire/negotiate), 20x (mass violence)
- Actor pairs involving: USA, Iran, Israel, Russia, Ukraine, China, Taiwan, Saudi Arabia
- Goldstein conflict scale scores
- GKG tone/sentiment for gold/oil/equity keywords

Aggregate to daily features (not sub-daily — GDELT updates are batched, not real-time):
```sql
CREATE TABLE gdelt_daily (
    date DATE,
    goldstein_avg FLOAT,          -- daily avg conflict intensity
    goldstein_change FLOAT,        -- day-over-day momentum
    conflict_event_count INT,      -- raw event volume
    ceasefire_signal_count INT,    -- CAMEO 12x events
    tone_gold_avg FLOAT,           -- sentiment of gold-related coverage
    tone_equity_avg FLOAT          -- sentiment of equity-related coverage
);
```

#### 1C — Macro Features (most already partially in your pipeline — verify first)

Before building anything, CHECK what daily_features already has:
- ATR-20 — YES, already computed
- RSI-14 — YES, already computed
- Relative volume — YES, already computed
- GVZ (gold volatility index) — probably NOT in pipeline. Add if missing.
- DXY (dollar index) — probably NOT. Add if missing.
- US 10yr real yield (FRED DFII10) — NOT in pipeline. Add.
- VIX — may be partially available via MES ATR proxy. Check.
- COT net positioning — NOT in pipeline. Weekly, forward-fill to daily.

Store new macro features in `alt_data.db:macro_features_daily`.

---

### PHASE 2 — STATISTICAL DISCOVERY (no ML — pure hypothesis testing)

#### 2A — Feature Engineering (strict no-lookahead)

For prediction market data, compute features using ONLY data from t-1 or earlier:
```
prob_change_4h     = prob(t-1) - prob(t-5h)     -- 4h change as of last known
prob_change_24h    = prob(t-1) - prob(t-25h)
vol_spike_ratio    = volume_4h / rolling_avg_volume_7d
prob_acceleration  = prob_change_4h - (prob_change_24h / 6)
regime_shift_flag  = 1 if abs(prob_change_4h) > 2 * std(prob_change_4h, 30d)
```

For GDELT:
```
goldstein_z        = (goldstein_avg - rolling_mean_30d) / rolling_std_30d
conflict_spike     = 1 if conflict_event_count > P90 of trailing 90d
```

#### 2B — Granger Causality (per-session, per-instrument)

For EACH feature × EACH session × EACH instrument (not pooled):
```
SESSION: NYSE_OPEN
INSTRUMENT: MNQ
FEATURE: prob_change_24h (Iran conflict markets)
TARGET: orb_breakout_pnl_r
LAGS: [1, 2, 3, 5]
RESULT: F-stat, p-value, optimal lag
THRESHOLD: p < 0.05 after BH FDR at K = (n_features × n_sessions × n_instruments)
```

Template for reporting:
```
SESSION: [name]
INSTRUMENT: [MNQ/MGC/MES]
MECHANISM: [why would this feature predict THIS session on THIS instrument?]
GRANGER: F=[X], p=[X], adj_p=[X] at K=[X]
VERDICT: CAUSAL / NOT_CAUSAL / UNDERPOWERED (N < 100 at this granularity)
```

**CRITICAL:** K for BH FDR must be HONEST. If you test 20 features × 12 sessions × 3 instruments = K=720. Most things will fail. That's the point. If nothing passes at K=720, alternative data doesn't predict your system. Report that finding.

#### 2C — Event Study (the Iran ceasefire test — generalized)

Define "geopolitical regime shift event" from prediction market data:
- prob_change_4h > 10% AND vol_spike_ratio > 3x on any geopolitical market
- Count total events in dataset. If N < 30 → INVALID, stop here.

For each event, extract:
- Gold (MGC) price window: [-24h, +24h]
- Equity (MNQ, MES) price window: [-24h, +24h]
- Cumulative abnormal return (CAR) at +15min, +1h, +4h, +24h
- Segment by: escalation vs de-escalation, which session was active, magnitude

Report:
```
EVENT_TYPE  | N   | CAR_1h | CAR_4h | CAR_24h | t_stat | p_value
escalation  | [X] | [X]    | [X]    | [X]     | [X]    | [X]
de-escalation| [X]| [X]    | [X]    | [X]     | [X]    | [X]
```

If CAR is not significant at p < 0.05: STOP. The Iran case was anecdotal, not systematic.

#### 2D — Existing Filter Tautology Check

Before claiming any alternative data feature adds value, run T0 (tautology check) against
existing features in daily_features:
```sql
SELECT corr(alt_data_feature, atr_20) FROM ...
SELECT corr(alt_data_feature, orb_size_pts) FROM ...
SELECT corr(alt_data_feature, rel_vol) FROM ...
```
If |corr| > 0.70 with any existing feature → DUPLICATE. The information is already captured
by ATR, vol filters, or G-filters. Don't add complexity for zero incremental information.

---

### PHASE 3 — T0-T8 AUDIT BATTERY (for any feature surviving Phase 2)

**Use the EXISTING audit protocol from `.claude/rules/quant-audit-protocol.md`. Do not
create a parallel validation framework.** Every surviving feature gets:

- T0: Tautology check against existing filters
- T1: Win rate monotonicity (is it a SIGNAL or just ARITHMETIC_ONLY?)
- T2: In-sample baseline (IS_END defined BEFORE running)
- T3: Out-of-sample walk-forward (WFE > 0.50 required)
- T4: Sensitivity ±20% on any threshold
- T5: Family comparison across sessions (per-session — NOT pooled)
- T6: Null floor (bootstrap 1000x, p < 0.05 after BH correction)
- T7: Per-year stability (≥7/10 positive years)
- T8: Cross-instrument consistency (same direction on MNQ, MGC, MES)

**Decision rules (pre-registered, not adjustable after seeing results):**

| Result | Action |
|--------|--------|
| 0 features pass T3 | STOP. Alt data does not predict ORB outcomes. Report finding. |
| 1-2 features pass T3 but fail T5 (session-specific) | REGIME_SPECIFIC overlay for those sessions only |
| Features pass T3+T5 but fail T8 (instrument-specific) | Separate implementation per instrument |
| Features pass T0-T8 | VALIDATED. Proceed to implementation design. |

---

### PHASE 4 — IMPLEMENTATION DESIGN (only if Phase 3 produces VALIDATED features)

**Do NOT pre-specify a decision matrix.** The matrix emerges from Phase 3 results.

What to design:
- How the validated feature integrates with existing prop_profiles lane config
- Whether it's a filter (skip trade if X), a sizer (adjust position if X), or a regime gate
- Separate alt_data.db ingestion pipeline that respects single-writer constraint
- Monitoring: feature drift detection, data source outage handling, graceful degradation

What NOT to build yet:
- Real-time polling (historical evidence first, live pipeline second)
- ML models (3 FAILs still open)
- Community signal scraping (not rigorous enough for this methodology)

---

### PHASE-LEVEL KILL GATES (define before starting)

| Phase | Kill Condition | Action |
|-------|---------------|--------|
| Phase 0 | Realistic improvement < +5R/yr OR N_events < 30 | STOP. Report. Focus on existing portfolio. |
| Phase 1 | Prediction market data has < 2 years coverage for relevant categories | STOP. Insufficient history for walk-forward. |
| Phase 2B | Zero features pass Granger at BH-adjusted p < 0.05 | STOP. Alt data doesn't predict this system. |
| Phase 2C | Event study CAR not significant (p > 0.05) | STOP. Geopolitical events don't systematically move gold/equity in tradeable ways. |
| Phase 2D | All surviving features correlate > 0.70 with existing daily_features | STOP. Information already captured. |
| Phase 3 | Zero features pass T3 (OOS walk-forward) | STOP. In-sample artifacts only. |

**The most likely outcome is that this research STOPS at Phase 2.** That's not failure —
it's the system working correctly. Most alternative data does not predict short-term
futures breakouts. If the answer is "alt data doesn't help your specific system," that's
a valuable finding that saves you from building unnecessary infrastructure.

---

### BEHAVIORAL RULES

1. **This project serves the existing ORB system.** It does not replace it, compete with it,
   or distract from deploying US_DATA_1000 and validating CME_PRECLOSE. If those are not
   done, do those first. Alt data research is lower priority than deploying validated edges.

2. **No "promising" until T5 clears.** Same ban as all other research in this project.

3. **Per-session decomposition mandatory.** Use the session profile template from the
   edge maximization prompt. Every finding gets tested per-session, reported per-session.

4. **Read governing docs first.** CLAUDE.md, TRADING_RULES.md, RESEARCH_RULES.md,
   quant-audit-protocol.md, STRATEGY_BLUEPRINT.md (especially §5 NO-GO registry).

5. **The Iran ceasefire is an anecdote, not evidence.** Do not build around it. Test whether
   the PATTERN (large probability move → gold move) holds across ALL comparable events.
   If N < 30, the pattern is untestable. Say so.

6. **Cost-benefit honesty.** If 200 hours of engineering yields +5R/year at $2/point,
   that's $10/year per contract improvement. Is that worth it? Maybe for learning and
   infrastructure. But don't pretend it's a portfolio transformation.

---

### DATA SOURCES REFERENCE

| Source | Type | Cost | Coverage | Relevance |
|--------|------|------|----------|-----------|
| Jon-Becker GitHub dataset | Prediction markets | Free | 2020-present | HIGH — largest public dataset |
| PredictionData.dev | Prediction markets | Paid | 3+ years, tick-level | HIGH — if free data insufficient |
| Kalshi API | Prediction markets (regulated) | Free tier | 2021-present | MEDIUM — economic indicators |
| GDELT Project | Geopolitical events | Free | 1979-present | HIGH — longest history |
| FRED (DFII10, DXY) | Macro indicators | Free | Decades | MEDIUM — daily granularity |
| CBOE GVZ | Gold volatility | Free via Yahoo | 2008-present | HIGH for MGC regime |
| CFTC COT | Positioning | Free (weekly) | Decades | MEDIUM — low frequency |

---

### START

Begin with Phase 0 — the cost-benefit gate. Count the events. Do the math.
If the numbers don't work, report that honestly and recommend focusing on
US_DATA_1000 deployment and CME_PRECLOSE validation instead.

## PROMPT END
