# Quant Audit Protocol — Claude Code Edition

**Role:** Institutional hypothesis validator (prop desk / Bloomberg standard)
**Objective:** Falsify claims before accepting them. Fail-closed at every step.
**Bias rule:** Do NOT conclude truth in advance. Stop before results.
**Authority:** This protocol governs ALL research claim evaluation. Complementary to `RESEARCH_RULES.md` and `research-truth-protocol.md`.

---

## PRE-FLIGHT (run first, always)

Before ANY analysis:

```sql
-- 1. DB freshness (HALT if stale > 2 trading days)
SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol = 'MNQ';
SELECT MAX(trading_day) FROM daily_features WHERE symbol = 'MNQ';
SELECT MAX(ts_utc) FROM bars_1m WHERE symbol = 'MNQ';

-- 2. Row counts (sanity — compare to prior known state)
SELECT symbol, COUNT(*) FROM orb_outcomes GROUP BY symbol ORDER BY symbol;
SELECT symbol, COUNT(*) FROM daily_features GROUP BY symbol ORDER BY symbol;

-- 3. Schema confirmation (print before any claim referencing a table)
DESCRIBE orb_outcomes;
DESCRIBE daily_features;
```

**Timezone:** All timestamps are UTC. Brisbane = UTC+10, no DST. Trading day starts 09:00 Brisbane = 23:00 UTC previous day. Canonical source: `pipeline/dst.py`.

**Canonical layers only:** `bars_1m`, `daily_features`, `orb_outcomes`. Do NOT use `validated_setups`, `edge_families`, `live_config`, or docs as truth sources. See `RESEARCH_RULES.md` § Discovery Layer Discipline.

**HALT conditions:** Stale DB (>2 days), row count mismatch vs expectation, schema column missing. Do not proceed past Pre-Flight if any check fails.

---

## STEP 1 — CLAIM DECOMPOSITION

For each claim submitted, output a numbered table:

| # | Claim (exact wording) | Type | Decision impact if TRUE | Decision impact if FALSE |
|---|----------------------|------|------------------------|--------------------------|

Type must be ONE of:
- `statistical_observation` — pattern seen in data, not yet tested for significance/OOS
- `validated_finding` — tested, has OOS confirmation, passes sensitivity
- `assumed_mechanism` — causal story, not directly testable from price data

**No bundling.** If a claim contains "and", split it into separate numbered claims.

---

## STEP 2 — FAILURE MODE ANALYSIS

For EACH claim, evaluate ALL four risks. Score: LOW / MEDIUM / HIGH.

### 2a. Multiple Testing Risk
- How many strategies/parameters/thresholds were scanned to produce this claim?
- State K explicitly. Apply Bonferroni-equivalent threshold: alpha_adjusted = 0.05 / K
- BH FDR alternative: use honest instrument K per `RESEARCH_RULES.md` BH K selection rule
- If K unknown → score HIGH by default
- Reference: Bailey & Lopez de Prado (2014) deflated-sharpe.pdf, Chordia et al (2018) Two_Million_Trading_Strategies_FDR.pdf

### 2b. Selection Bias Risk
- Was this claim generated AFTER looking at the data?
- Was the date range, session, or instrument selected to fit the result?
- If backtest window == discovery window → score HIGH
- Reference: Harvey & Liu (2014) backtesting_dukepeople_liu.pdf

### 2c. Overfitting Risk
- Is this claim sensitive to parameter choices?
- Would ±20% change on any threshold flip the result?
- Does it rely on a single session/instrument with no family comparison?
- Reference: Aronson Evidence_Based_Technical_Analysis.pdf Ch 6

### 2d. Pipeline Risk
- Could this result from: stale data, incorrect outcome calculation, session boundary bugs, timezone error, or lookahead in feature build?
- Check: are features computed strictly from data available at trade decision time?
- Specific check: `double_break` is LOOKAHEAD (code line 393 in build_daily_features.py). `break_delay_min` and `risk_dollars` are trade-time-knowable.

### Claim Verdict (after Step 2 only — NOT after seeing results)
- `PROCEED_TO_TEST` — at least one failure mode is LOW, claim is testable
- `HIGH_RISK` — multiple HIGH scores, test results must clear higher threshold (p < 0.01 instead of 0.05)
- `NEEDS_CLARIFICATION` — claim is ambiguous or untestable as stated

---

## STEP 3 — TEST PLAN (CONCRETE, EXECUTABLE)

For each `PROCEED_TO_TEST` claim, specify:

```
Claim #: [N]
Table(s): orb_outcomes JOIN daily_features ON trading_day + symbol
Query skeleton:
  SELECT COUNT(*), AVG(pnl_r), STDDEV(pnl_r)
  FROM orb_outcomes o
  JOIN daily_features df ON o.trading_day = df.trading_day AND o.symbol = df.symbol
  WHERE o.symbol = 'MNQ'
    AND o.entry_model = 'E2' AND o.orb_minutes = 5
    AND o.confirm_bars = 1 AND o.rr_target = 1.0
    AND o.outcome IN ('win', 'loss')
    AND [claim-specific conditions]
    AND o.trading_day BETWEEN '<IS_start>' AND '<IS_end>'

Metric: [ExpR | Sharpe | WR | p-value | WFE]
IS window: [define BEFORE running]
OOS window: [define BEFORE running — NEVER overlap with IS]
Sensitivity sweep: ±20% on [parameter] — 3x3 grid if 2 parameters
Null comparison: shuffle pnl_r labels 1000x → bootstrap p-value
Pass threshold: [exact number, e.g., OOS ExpR > 0, p < 0.05/K, WFE > 0.50]
Fail condition: [exact condition that kills the claim]
```

**Required test types** (ALL must be present for `validated_finding` upgrade):
1. In-sample baseline (IS)
2. Out-of-sample / walk-forward (OOS/WFE > 0.50, expanding window)
3. Sensitivity sweep ±20% on each parameter
4. Family comparison (same metric across ALL sessions/instruments, not just the winner)
5. Null floor (bootstrap permutation, 1000+ perms, Phipson & Smyth p-value)
6. Per-year stability (must be positive in ≥7/10 years)
7. Cross-instrument directional consistency

---

## STEP 4 — EXECUTION ORDER

List tests in order: highest decision-impact first.

```
1. [Test name] — Claim #[N] — Impact: [what decision rides on this]
2. ...
```

Highest impact = tests that would change position sizing, session selection, or go-live decision if they fail.

---

## STEP 5 — DECISION RULES (DEFINE BEFORE RESULTS)

State pass/fail criteria NOW. No post-hoc reinterpretation.

| Outcome | Action |
|---------|--------|
| Passes OOS + survives ±20% sensitivity + beats null (p < alpha_adjusted) + WFE > 0.50 + positive ≥7/10 years | **KEEP** — promote to `validated_finding` |
| Passes IS, fails OOS (WFE < 0.50) | **KILL** — overfit artifact |
| Passes OOS, fails sensitivity (±20% flips sign) | **DOWNGRADE** — regime-specific, reduce size |
| Fails null comparison (bootstrap p > 0.05) | **KILL** — no edge above noise |
| Pipeline issue found (lookahead, stale data, timezone bug) | **SUSPEND** — fix pipeline, re-run from Pre-Flight |
| Negative in ≥4/10 years | **DOWNGRADE** — era-dependent, not structural |

"Regime-specific" means: trade only when regime condition is confirmed live (e.g., cost/risk < 10%). It does NOT mean: re-optimize until it passes.

---

## STEP 6 — OUTPUT CONTRACT

Return structured output only. No prose conclusions.

```
AUDIT TIMESTAMP: <UTC datetime>
DB FRESHNESS: orb_outcomes=<date>, daily_features=<date>, bars_1m=<datetime>
ROW COUNTS: orb_outcomes=<N>, daily_features=<N>

CLAIMS:
  [#] | <claim> | <type> | <verdict>

FAILURE RISKS:
  [#] | MultiTest: <score> | SelectBias: <score> | Overfit: <score> | Pipeline: <score>

TEST PLAN:
  [#] | <query skeleton> | <metric> | <pass threshold> | <fail condition>

EXECUTION ORDER:
  1. ...

DECISION RULES:
  [#] -> KEEP if: ... | KILL if: ... | DOWNGRADE if: ...

STATUS: READY_TO_RUN | HALTED_PREFLIGHT | NEEDS_CLARIFICATION
```

**Do NOT write:** "This looks promising", "likely an edge", "probably valid", "interesting finding."
**Do NOT run tests yet.** Output the plan only. Execution is a separate step requiring explicit "go."

---

## Project-Specific Anchors

- Cost model: `from pipeline.cost_model import COST_SPECS` — MNQ $2.74 RT, MGC $5.74, MES $3.74
- Friction: `total_friction / risk_dollars * 100` (percentage of trade risk eaten by costs)
- Break timing: `daily_features.orb_{SESSION}_break_delay_min` (minutes from ORB end to first break)
- Lookahead variables (BANNED as filters): `double_break` (code line 393: "LOOK-AHEAD relative to intraday entry")
- Trade-time-knowable: `risk_dollars`, `break_delay_min`, `rel_vol`, `atr_20`, `orb_size`
- Session catalog: `from pipeline.dst import SESSION_CATALOG`
- Active instruments: `from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS`
