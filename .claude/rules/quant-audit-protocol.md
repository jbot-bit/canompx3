---
paths:
  - "trading_app/strategy_*"
  - "trading_app/outcome_*"
  - "research/**"
  - "scripts/tools/backtest*"
---
# Quant Audit Protocol — Claude Code Edition

**Role:** Institutional hypothesis validator (prop desk / Bloomberg standard)
**Objective:** Falsify claims before accepting them. Fail-closed at every step.
**Bias rule:** Do NOT conclude truth in advance. Do NOT call anything "promising", "edge", or "mechanism" until Step 5 clears.
**Authority:** Governs ALL research claim evaluation. Complementary to `RESEARCH_RULES.md` and `research-truth-protocol.md`.

---

## PRE-FLIGHT (run first, always — no exceptions)

```sql
-- 1. DB freshness (HALT if stale > 2 trading days)
SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol = 'MNQ';
SELECT MAX(trading_day) FROM daily_features WHERE symbol = 'MNQ';
SELECT MAX(ts_utc) FROM bars_1m WHERE symbol = 'MNQ';

-- 2. Row counts (compare to prior known state)
SELECT symbol, COUNT(*) FROM orb_outcomes GROUP BY symbol ORDER BY symbol;

-- 3. Schema confirmation (print before any claim references a column)
DESCRIBE orb_outcomes;
DESCRIBE daily_features;

-- 4. Instrument coverage
SELECT symbol, COUNT(*), MIN(trading_day), MAX(trading_day)
FROM orb_outcomes GROUP BY symbol ORDER BY symbol;
```

**Timezone:** All timestamps UTC. Brisbane = UTC+10, no DST. Trading day 09:00 Brisbane = 23:00 UTC prev day. Source: `pipeline/dst.py`.

**Canonical layers only:** `bars_1m`, `daily_features`, `orb_outcomes`. BANNED for truth: `validated_setups`, `edge_families`, `live_config`, docs. See `RESEARCH_RULES.md` § Discovery Layer Discipline.

**HALT conditions:** Stale DB (>2 days), row count mismatch, schema column missing, timezone mismatch. Do not proceed past Pre-Flight if any check fails.

---

## STEP 1 — CLAIM DECOMPOSITION

For each claim, output a numbered table:

| # | Claim (exact wording) | Type | Decision impact if TRUE | Decision impact if FALSE |
|---|----------------------|------|------------------------|--------------------------|

Type must be ONE of:
- `statistical_observation` — pattern seen, not yet tested for significance/OOS
- `validated_finding` — tested, has OOS + sensitivity + null floor
- `assumed_mechanism` — causal story, not directly testable from price data

**Rule:** If a claim contains "and" or "because" → split into separate claims.

---

## STEP 2 — FAILURE MODE ANALYSIS

For EACH claim, score ALL four risks: LOW / MEDIUM / HIGH.

### 2a. Multiple Testing Risk
- How many strategies/params/thresholds were scanned? State K explicitly.
- Adjusted threshold: alpha = 0.05 / K (Bonferroni) or BH FDR at honest K
- If K unknown → HIGH by default
- Ref: Bailey & de Prado (2014) `deflated-sharpe.pdf`, Chordia et al (2018) `Two_Million_Trading_Strategies_FDR.pdf`

### 2b. Selection Bias Risk
- Was date range, session, or instrument chosen AFTER seeing the result?
- If discovery window == backtest window → HIGH
- Ref: Harvey & Liu (2014) `backtesting_dukepeople_liu.pdf`

### 2c. Overfitting / Tautology Risk
- Is this claim mathematically derivable from another filter already in use?
  (e.g., cost/risk% = constant / ORB_size — is this just a G-filter reframed?)
- Would ±20% on any threshold flip the result?
- Does it rely on a single session/instrument?
- If tautology suspected → run T0 before anything else
- Ref: Aronson `Evidence_Based_Technical_Analysis.pdf` Ch 6

### 2d. Pipeline / Arithmetic Risk
- Could improved metric be purely mechanical?
  (e.g., filtering low friction inflates AvgWinR by removing cost drag — not a predictor)
- **KEY TEST:** Is WR changing, or only payoff changing?
  WR flat + payoff improving = cost screen (ARITHMETIC_ONLY), NOT signal.
- Are features computed strictly at t < entry_time? (no forward-look)
- Is WFE suspiciously clean (>0.95)? → check for data leakage or tiny OOS N.
- Known lookahead: `double_break` (code line 393: "LOOK-AHEAD relative to intraday entry")
- Trade-time-knowable: `risk_dollars`, `break_delay_min`, `rel_vol`, `atr_20`, `orb_size`

### Claim Verdict (after Step 2 — NOT after seeing results)
- `PROCEED_TO_TEST` — at least one risk is LOW, claim is testable
- `HIGH_RISK` — multiple HIGHs, require stricter thresholds (p < 0.01)
- `TAUTOLOGY_SUSPECTED` — run T0 before anything else
- `NEEDS_CLARIFICATION` — claim is ambiguous or untestable

---

## STEP 3 — TEST BATTERY (CONCRETE, EXECUTABLE)

Run in this EXACT order. Do not skip. Do not reorder to get a better result.

### T0. TAUTOLOGY CHECK (run first if 2c scored HIGH)
```sql
-- Is this metric just an existing filter expressed differently?
SELECT corr(new_metric, existing_filter_metric) FROM ...
-- If |corr| > 0.70 → TAUTOLOGY. Label as DUPLICATE_FILTER.
-- Example: corr(cost_risk_pct, 1/orb_size_pts) = -1.0 by construction
```

### T1. WIN RATE MONOTONICITY (run second — always)
```sql
-- Does WR improve across bins, or only payoff?
SELECT NTILE(5) OVER (ORDER BY filter_variable) AS bin,
       COUNT(*), AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) AS wr,
       AVG(pnl_r) AS expr, AVG(CASE WHEN outcome='win' THEN pnl_r END) AS avg_win_r
FROM orb_outcomes WHERE ... GROUP BY 1 ORDER BY 1
```
- WR flat (<3% spread) + payoff rising = **ARITHMETIC_ONLY** (cost screen)
- WR monotonic (>5% spread) = **SIGNAL** (predicts win probability)
- **CRITICAL:** Control for confounders. Hold one variable constant, vary the other.

### T2. IN-SAMPLE BASELINE
```sql
SELECT COUNT(*), AVG(pnl_r), STDDEV(pnl_r),
       AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END)
FROM orb_outcomes WHERE trading_day < '<IS_END>' AND [conditions]
```
Define IS_END BEFORE running. Do not adjust after seeing OOS.

### T3. OUT-OF-SAMPLE / WALK-FORWARD
- Use expanding window: IS grows, each year is OOS exactly once
- WFE = OOS_Sharpe / IS_Sharpe. Threshold: WFE > 0.50.
- Suspicious if WFE > 0.95 — investigate leakage.

### T4. SENSITIVITY ±20%
- 3×3 grid if 2 parameters (9 cells, all must be positive)
- KILL if ExpR flips sign anywhere in ±20% range

### T5. FAMILY COMPARISON
- Same metric across ALL sessions and ALL instruments — not just the winner
- RED FLAG if only 1-2 sessions pass

### T6. NULL FLOOR (bootstrap permutation)
- Shuffle pnl_r 1000x, compute ExpR each time
- p-value = (exceeding + 1) / (perms + 1) (Phipson & Smyth)
- Claim must beat null P95 at p < 0.05 (adjusted for K)

### T7. PER-YEAR STABILITY
- Must be positive in ≥7/10 full years
- Negative in ≥4/10 → DOWNGRADE to era-dependent

### T8. CROSS-INSTRUMENT
- Same direction on all active instruments (MNQ, MGC, MES)
- If directionally inconsistent → DOWNGRADE

---

## STEP 4 — EXECUTION ORDER

Always this order — highest decision-impact first:

```
1. PRE-FLIGHT          — DB valid, schema confirmed
2. T0 TAUTOLOGY        — Is this a known filter in disguise?
3. T1 WR MONOTONICITY  — Signal or arithmetic? (kills many claims early)
4. T2 IS BASELINE      — What does in-sample show?
5. T3 OOS / WF         — Does it hold out-of-sample?
6. T4 SENSITIVITY      — Survives ±20%?
7. T5 FAMILY           — Generalises across sessions/instruments?
8. T6 NULL FLOOR       — Beats shuffled baseline?
9. T7 PER-YEAR         — Stable across years?
10. T8 CROSS-INSTRUMENT — Directionally consistent?
```

---

## STEP 5 — DECISION RULES (DEFINE BEFORE RESULTS — NO EXCEPTIONS)

| Test Outcome | Decision | Label |
|---|---|---|
| T0: \|corr\| > 0.70 with existing filter | KILL | `DUPLICATE_FILTER` |
| T1: WR flat, payoff rising only | DOWNGRADE | `ARITHMETIC_ONLY` |
| T3: OOS ExpR < 0 or WFE < 0.50 | KILL | `OVERFIT` |
| T4: Sign flips in ±20% | KILL | `PARAMETER_SENSITIVE` |
| T5: Only 1-2 sessions pass | DOWNGRADE | `REGIME_SPECIFIC` |
| T6: bootstrap p > 0.05 | KILL | `NO_EDGE` |
| T7: Negative ≥4/10 years | DOWNGRADE | `ERA_DEPENDENT` |
| WFE > 0.95 with small OOS N | SUSPEND | `LEAKAGE_SUSPECT` |
| Passes T1+T3+T4+T5+T6+T7 | KEEP | `VALIDATED` |

**`ARITHMETIC_ONLY` means:**
Real and useful as a cost screen — implement as minimum viable trade size gate (same family as G-filters). Do NOT call it "mechanism", "structural edge", or "breakout quality predictor."

**`REGIME_SPECIFIC` means:**
Trade only when regime condition is confirmed live. Does NOT mean: re-optimise until it passes.

---

## STEP 6 — OUTPUT CONTRACT

Return structured output only. No prose conclusions.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUDIT REPORT — canompx3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIMESTAMP:        <UTC datetime>
DB FRESHNESS:     <latest trading_day>
ROW COUNT:        <N>
TIMEZONE:         <confirmed / MISMATCH>

CLAIMS:
  #1 | <claim text>    | <type>         | <verdict>

FAILURE RISKS:
  #1 | MultiTest: <H/M/L> | SelectBias: <H/M/L> | Overfit: <H/M/L> | Pipeline: <H/M/L>

TEST RESULTS:
  T0 Tautology:    corr=<X>    → PASS / DUPLICATE_FILTER
  T1 WR Monotone:  WR spread=<X>% → SIGNAL / ARITHMETIC_ONLY
  T2 IS:           n=<N>, ExpR=<X>, WR=<X>
  T3 OOS:          n=<N>, ExpR=<X>, WR=<X>, WFE=<X>
  T4 Sensitivity:  min ExpR=<X> @ <pct>% → STABLE / SENSITIVE
  T5 Family:       sessions pass=<N>/<total> → GENERALISES / REGIME_SPECIFIC
  T6 Null P95:     <X>, claim ExpR=<X> → BEATS_NULL / NO_EDGE
  T7 Per-Year:     positive=<N>/10 → STABLE / ERA_DEPENDENT
  T8 Cross-Inst:   <direction consistency> → CONSISTENT / INCONSISTENT

DECISION:
  #1 → VALIDATED | KILL:<reason> | DOWNGRADE:<label> | SUSPEND:<reason>

IMPLEMENTATION NOTE (if VALIDATED or ARITHMETIC_ONLY):
  Frame as: <exact description of what it is>
  Do NOT call it: <what to avoid saying>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## ANTI-PATTERNS — NEVER DO THESE

```
❌ "This looks promising"          → prohibited until Step 5 clears
❌ "Likely an edge"                → prohibited until VALIDATED
❌ "Probably valid"                → prohibited always
❌ "Let's implement and monitor"   → not a test result, not a decision
❌ Adjusting IS_END after seeing OOS result
❌ Skipping T5 family because "we only care about MNQ"
❌ Calling WR=58% flat "directional" because 3 bins went 57→58→61
❌ Accepting WFE=1.00 at face value without checking OOS N and leakage
❌ Framing ARITHMETIC_ONLY filter as "structural mechanism"
❌ Running T6 null test on pre-filtered subset (shuffles must use full population)
```

---

## KNOWN FAILURE PATTERNS

Confirmed findings from this project. Append new entries as discovered.

```
[2026-03-24] cost/risk% friction gate:
  VERDICT: ARITHMETIC_ONLY
  FINDING: WR flat at ~58-60% across all friction bins (quintile spread <3%).
           ExpR improvement is payoff arithmetic — bigger ORBs pay more per win
           because costs eat less of the gross win. Not a win-rate predictor.
           |corr(cost_risk_pct, 1/orb_size_pts)| = 1.0 by construction (inverse).
  CORRECT FRAMING: Minimum viable trade size gate / cost screen (same family as G-filters)
  DO NOT CALL: "breakout quality predictor" or "structural mechanism"
  IMPLEMENT AS: ORB risk$ threshold filter. Friction < 10% = risk$ > $27.40 for MNQ.

[2026-03-24] break_delay_min timeout:
  VERDICT: SIGNAL (pending full T3-T8 verification)
  FINDING: WR spread 54-64% across delay quintiles (6%+ spread).
           Early breaks (<=5m) have higher WR AND higher ExpR.
           WR improvement persists when controlling for friction.
  MECHANISM: Order flow concentration — immediate breaks = stacked stops triggering
  STATUS: Passes T1 (WR monotonic). T3-T8 pending DB unlock for verification.

[2026-03-24] double_break:
  VERDICT: BANNED — LOOKAHEAD
  FINDING: Massive signal (+0.352 vs -0.198) but computed AFTER trade entry.
           Code line 393: "LOOK-AHEAD relative to intraday entry — do NOT use as live filter."
  STATUS: Cannot be used. Correctly flagged in pipeline code.
```

---

## PROJECT-SPECIFIC ANCHORS

- DB: `gold.db` at project root. Connect: `duckdb.connect('gold.db', read_only=True, config={'access_mode': 'READ_ONLY'})`
- Cost model: `from pipeline.cost_model import COST_SPECS` — MNQ $2.74 RT, MGC $5.74, MES $3.74
- Friction: `total_friction / risk_dollars * 100` — mathematically = `constant / orb_size_pts` (tautology with G-filters)
- Break timing: `daily_features.orb_{SESSION}_break_delay_min` (minutes from ORB end to first break)
- Lookahead (BANNED): `double_break`
- Trade-time-knowable: `risk_dollars`, `break_delay_min`, `rel_vol`, `atr_20`, `orb_size`
- Session catalog: `from pipeline.dst import SESSION_CATALOG`
- Active instruments: `from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS`
- Academic refs: `resources/` directory — deflated-sharpe.pdf, Two_Million_Trading_Strategies_FDR.pdf, backtesting_dukepeople_liu.pdf, Evidence_Based_Technical_Analysis.pdf, Algorithmic_Trading_Chan.pdf, Systematic_Trading_Carver.pdf, Lopez_de_Prado_ML_for_Asset_Managers.pdf
