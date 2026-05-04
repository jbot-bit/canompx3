# Data Mining Deep Dive — Let The Data Speak

No pre-ranking. No cherry-picking. No "mechanism plausibility" filtering.
Map everything. Screen everything. The data decides what matters.

---

## PROMPT START

I have a futures ORB breakout system with 10 years of 1-minute candle data across
3 instruments (MNQ, MGC, MES), 12 sessions, and ~10M rows in orb_outcomes. The system
has been optimized along known dimensions (filters, ORB aperture, entry models, RR targets).

**Nobody has systematically screened every feature in this dataset against outcomes.**
There are columns in daily_features that were computed but never binned against win rate.
There are feature interactions never tested. There are temporal patterns never checked.

Your job: screen EVERYTHING. No hypothesis. No pre-ranking. No "this one looks more
plausible." Run the numbers on ALL unexplored dimensions simultaneously and let the
data tell you which ones matter.

**You are a screening machine, not a storyteller. Test everything. Report what survives.**

---

### STEP 0 — READ THE LAW

Read in order, internalize, don't repeat back:
1. `CLAUDE.md`
2. `TRADING_RULES.md`
3. `RESEARCH_RULES.md`
4. `docs/STRATEGY_BLUEPRINT.md` — especially §5 NO-GO registry (know what's dead)
5. `.claude/rules/quant-audit-protocol.md`

**NO-GO registry is a hard boundary.** If something is dead, skip it. You can test a
STRUCTURALLY DIFFERENT version of a dead idea (e.g., within-day cross-session vs prior-day
context), but you must verify the prior kill doesn't apply before spending time on it.

---

### STEP 1 — MAP EVERY UNEXPLORED DIMENSION

#### 1A — Full schema audit

```sql
DESCRIBE daily_features;
DESCRIBE orb_outcomes;
```

Classify EVERY column in daily_features:

```
COLUMN                     | STATUS   | EVIDENCE
───────────────────────────┼──────────┼──────────────────
[column]                   | USED     | In filter grid / validated strategy
[column]                   | PARTIAL  | Tested once, not through full T0-T8
[column]                   | NO-GO    | Blueprint §5 / audit protocol dead
[column]                   | VIRGIN   | Zero references outside build script
```

**How to classify:** `grep -r "column_name" pipeline/ trading_app/ scripts/ tests/`
If zero hits outside `build_daily_features.py` → VIRGIN. These are your targets.
No judgment about which VIRGINs "look more interesting." They're all targets.

#### 1B — Enumerate ALL testable dimensions

From the schema, mechanically list every dimension that CAN be screened:

- Every VIRGIN feature column (continuous → quintile bin; boolean → True/False split)
- Every VIRGIN × session combination
- Every pair of (USED feature × VIRGIN feature) — interaction candidates
- DOW × session (5 × 12 cells per instrument)
- Month × session (12 × 12 cells per instrument)
- Consecutive outcome serial correlation per session
- Vol regime transition (low→high, high→low, stable) per session
- Cross-session within-day chains (if NOT killed by NO-GO — verify first)
- Direction asymmetry per session (long vs short outcomes)

Count the total. This is your honest K for FDR.

---

### STEP 2 — SCREEN EVERYTHING (no cherry-picking)

**Do NOT pick and choose which dimensions to test.** Run the Step 3A screening pass
on ALL of them. The point is to let the data surface what matters, not to let human
intuition pre-filter.

For EVERY dimension identified in Step 1B, run the fast screening query:

#### Continuous features → quintile bin:

```sql
SELECT
    [session_column],
    NTILE(5) OVER (PARTITION BY [session_column] ORDER BY [feature]) AS bin,
    COUNT(*) AS n,
    AVG(pnl_r) AS expr,
    AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) AS wr
FROM orb_outcomes o
JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
WHERE o.symbol = [instrument]
  AND o.entry_model = 'E2'
  AND o.rr_target = 1.0
  AND o.orb_minutes = 5
  AND d.[feature] IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2
```

#### Boolean features → True/False split:

```sql
SELECT
    [session_column],
    d.[boolean_feature] AS flag,
    COUNT(*) AS n,
    AVG(pnl_r) AS expr,
    AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) AS wr
FROM orb_outcomes o
JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
WHERE o.symbol = [instrument]
  AND o.entry_model = 'E2'
  AND o.rr_target = 1.0
GROUP BY 1, 2
ORDER BY 1, 2
```

#### Temporal → group by time unit:

```sql
SELECT
    [session_column],
    EXTRACT(DOW FROM trading_day) AS time_unit,
    COUNT(*) AS n,
    AVG(pnl_r) AS expr,
    AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) AS wr
FROM orb_outcomes o
WHERE symbol = [instrument] AND entry_model = 'E2' AND rr_target = 1.0
GROUP BY 1, 2
ORDER BY 1, 2
```

**Run ALL of these. Per session. Per instrument where relevant (start with MNQ).**

For each screening result, compute mechanically:
- WR spread = max(bin WR) - min(bin WR)
- ExpR monotonic? = does ExpR consistently increase or decrease across bins?
- N per bin (is any bin < 50? → flag as underpowered)

#### Automatic classification (no human judgment):

| WR Spread | ExpR Pattern | Classification |
|-----------|-------------|----------------|
| > 5% AND monotonic | ExpR monotonic | **SIGNAL** → escalate |
| > 5% AND non-monotonic | ExpR mixed | **PARTIAL** → escalate with caution |
| < 3% | ExpR rising | **ARITHMETIC_ONLY** → note but don't escalate |
| < 3% | ExpR flat/mixed | **DEAD** → record and move on |
| 3-5% | Any | **NOISE** → record, do not escalate |
| Any | N < 50 in any bin | **UNDERPOWERED** → flag, don't conclude |

**The classification is mechanical. Not vibes. Not "this seems like it could be something."
WR spread > 5% + monotonic = escalate. Everything else = record and move on.**

---

### STEP 3 — REPORT THE FULL SCREENING (before any escalation)

Present ALL results in a single table, sorted by WR spread descending:

```
FULL SCREENING RESULTS — [date]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL DIMENSIONS SCREENED: [N]
K FOR FDR:                 [N] (this is your multiple testing tax — pay it)

DIMENSION                  | SESSION        | WR SPREAD | ExpR MONO | CLASS
───────────────────────────┼────────────────┼───────────┼───────────┼──────────
[whatever the data says]   | [per session]  | XX.X%     | YES/NO    | SIGNAL
[whatever the data says]   | [per session]  | XX.X%     | YES/NO    | SIGNAL
[whatever the data says]   | [per session]  | XX.X%     | PARTIAL   | PARTIAL
...                        | ...            | ...       | ...       | DEAD
...                        | ...            | ...       | ...       | DEAD
[all the rest]             | [all sessions] | < 3%      | NO        | DEAD

SIGNALS FOUND:      [count]
PARTIAL:            [count]
ARITHMETIC_ONLY:    [count]
DEAD:               [count]
UNDERPOWERED:       [count]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Do NOT interpret the results yet.** Present the table. I'll look at what the data
surfaced. Then we decide together which SIGNALs to escalate to T0-T8.

**If ZERO dimensions classify as SIGNAL:** Report that. "All [N] dimensions screened.
Zero show WR spread > 5% with monotonic ExpR. The existing filter set captures
available signal in this dataset. Mining complete." That's a valid, valuable finding.

---

### STEP 4 — ESCALATE SURVIVORS (full T0-T8, only after Step 3 review)

For any SIGNAL I greenlight from Step 3:

1. **T0 tautology** — correlate against ALL existing filters (ATR, G-filters, X_MES_ATR60, vol regime). If |corr| > 0.70 → DUPLICATE. Already captured.

2. **T1-T8 full battery** from `quant-audit-protocol.md`. In order. No skipping. Per session.

3. **Pre-register before running:**
   ```
   FINDING: [feature] predicts [session] outcomes
   IS_END: [define BEFORE running]
   K FOR FDR: [total dimensions screened in Step 2 — the FULL K, not just survivors]
   KILL: WFE < 0.50, sensitivity flips, bootstrap p > 0.05 at honest K
   ```

4. **Report in audit protocol output format.** Numbers. No prose conclusions until
   all tests complete.

---

### BEHAVIORAL RULES

**Screen everything, interpret nothing (until Step 3).** The screening pass is mechanical.
WR spread and ExpR monotonicity are computed, not judged. Don't skip a dimension because
"it probably won't show anything." Don't prioritize a dimension because "the mechanism
is plausible." Screen them all equally.

**Session-specificity at every step.** Every screening query runs per session.
A feature that shows SIGNAL on NYSE_OPEN and DEAD on SINGAPORE_OPEN is TWO results,
not one averaged result.

**The multiple testing tax is non-negotiable.** If you screen 200 dimensions, K=200.
BH FDR at K=200 will kill most things. That's correct. It's the price of looking at
everything. Pay it honestly. Don't reduce K by "only counting the ones that looked
worth testing" — that's retroactive cherry-picking.

**No lookahead.** Every feature must be knowable at trade entry time.
Trade-time-knowable: `risk_dollars`, `break_delay_min`, `rel_vol`, `atr_20`, `orb_size`,
`day_of_week`, `month`, any same-day prior-session outcome (if that session closed first).
BANNED: `double_break`, anything computed from post-entry price action, MAE/MFE (these
are post-trade metrics — knowable for daily_features analysis but NOT as live filters).

**Report EVERYTHING you tested.** The dead ends are as important as the signals.
They close off dimensions for future sessions. Append ALL findings (positive AND
negative) to KNOWN FAILURE PATTERNS in `quant-audit-protocol.md`.

**No stories before numbers.** Don't hypothesize mechanisms before seeing data.
After Step 3, if something shows SIGNAL, THEN we discuss why. Not before.

**"Nothing found" is a result.** If the screening produces zero SIGNALs, the dataset
has been mined and the existing filters capture what's available. Report that honestly.
Don't manufacture a finding to justify the session.

---

### START

Read the law. Run Step 1 (full schema audit + dimension enumeration).
Run Step 2 (screen ALL dimensions — no cherry-picking).
Present Step 3 (full results table). Wait.

## PROMPT END
