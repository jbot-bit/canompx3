# Data Mining Deep Dive — Find What Nobody Asked About

The other prompts optimize known workstreams. This one prospects for unknowns.
You are not confirming hypotheses — you are discovering what the data contains
that has never been examined. No story first. Data first.

---

## PROMPT START

I have a futures ORB breakout system with 10 years of 1-minute candle data across
3 instruments (MNQ, MGC, MES), 12 sessions, and ~10M rows in orb_outcomes. The system
has been optimized along known dimensions (filters, ORB aperture, entry models, RR targets).

**But nobody has systematically audited what features exist in the data that have NEVER
been analyzed.** There are columns in daily_features that may have never been binned.
There are patterns in bars_1m that were never aggregated. There are interaction effects
between known features that were never tested together.

Your job: find the unexplored corners of this dataset. No hypothesis. No story.
Just: what's in the data, what's never been tested, and does any of it predict outcomes?

**You are a prospector, not a storyteller. Dig first, interpret later.**

---

### STEP 0 — READ THE LAW

Same as master prompt. Read in order, internalize, don't repeat back:
1. `CLAUDE.md`
2. `TRADING_RULES.md`
3. `RESEARCH_RULES.md`
4. `docs/STRATEGY_BLUEPRINT.md` — especially §5 NO-GO registry (know what's already dead)
5. `.claude/rules/quant-audit-protocol.md`

**Critical: the NO-GO registry exists for a reason.** Some paths are dead. Before you
"discover" something, check if it was already tested and killed. If it's on the NO-GO,
skip it unless you can articulate why your test is structurally different from the one
that killed it.

---

### STEP 1 — MAP THE UNEXPLORED TERRITORY

This is the core differentiator. Before testing anything, CATALOG what exists.

#### 1A — Schema audit

Run these queries and report the FULL column list:

```sql
DESCRIBE daily_features;
DESCRIBE orb_outcomes;
DESCRIBE bars_1m;
DESCRIBE bars_5m;
```

For daily_features specifically, classify EVERY column into:

```
COLUMN                          | ANALYZED? | HOW DO YOU KNOW?
────────────────────────────────┼───────────┼──────────────────
orb_NYSE_OPEN_size_pts          | YES       | Used in G-filters, validated
orb_NYSE_OPEN_break_delay_min   | PARTIAL   | T1 done, T3-T8 pending
atr_20                          | YES       | Used in ATR filters
rel_vol                         | PARTIAL   | In feature set, not binned alone
rsi_14                          | NO-GO     | Blueprint §5: GUILTY
[some_column_you_find]          | UNKNOWN   | No references found in codebase
```

**How to determine "ANALYZED?":**
- `grep -r "column_name" pipeline/ trading_app/ scripts/ tests/` — is it used anywhere?
- Check `TRADING_RULES.md` — is it mentioned as a filter or feature?
- Check `docs/STRATEGY_BLUEPRINT.md` §5 NO-GO — is it dead?
- Check `quant-audit-protocol.md` KNOWN FAILURE PATTERNS — was it tested and killed?
- If ZERO references outside of `build_daily_features.py` → it's computed but NEVER ANALYZED.
  These are your primary targets.

#### 1B — Feature interaction matrix

List all PAIRS of known-useful features that have never been tested TOGETHER:
- break_delay × orb_size (do fast breaks on big ORBs predict differently than fast breaks on small ORBs?)
- break_delay × session (already known to reverse — but within the "good direction" sessions, does break_delay × vol regime interact?)
- day_of_week × session (are certain sessions stronger on certain days?)
- direction × session (already known SINGAPORE_OPEN is long-only — are other sessions directionally biased?)
- orb_size × vol_regime (do small ORBs in high-vol regimes behave differently than small ORBs in low-vol?)

**Don't make this list up from imagination.** Build it from what actually exists in the schema.

#### 1C — Temporal pattern audit

Check for patterns that most ORB research ignores:
- **Month-of-year effects:** Is there seasonality? (January effect, September/October volatility, FOMC cycle months)
- **Day-of-week effects PER SESSION:** Not global DOW — session-specific. Tuesday NYSE_OPEN vs Friday NYSE_OPEN.
- **Time-since-last-trade effects:** After a 2-day break (holiday), is the next ORB different?
- **Consecutive outcome effects:** After 3 losses in a row on a session, is the next trade different? (Serial correlation test)
- **Volatility regime transitions:** Not vol LEVEL (already filtered) — the CHANGE. Does the day vol regime switches from low→high predict ORB behavior differently than stable-high?

#### 1D — Cross-session within-day patterns

This is the interaction nobody looks at:
- If NYSE_OPEN ORB breaks long AND wins, does US_DATA_1000 (30 min later) also break long more often?
- If COMEX_SETTLE was a loss, does CME_PRECLOSE (75 min later) behave differently?
- Does the SIZE of the NYSE_OPEN ORB predict anything about US_DATA_1000 behavior?

**WARNING:** Cross-session lead-lag is on the NO-GO registry. CHECK what specific test was done and why it was killed. If the prior test was "prior-day context" (different-day), within-day same-session-chain may be structurally different and still testable. But verify before assuming.

---

### STEP 2 — REPORT THE MAP (before testing anything)

Present the catalog:

```
DATA MINING MAP — [date]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEVER-ANALYZED FEATURES (computed but never tested):
  1. [column] — exists in daily_features since [date], zero codebase references
  2. [column] — ...

PARTIALLY-ANALYZED (T1 done, no OOS/WF):
  1. break_delay_min — T1 SIGNAL, T3-T8 pending
  2. ...

UNTESTED INTERACTIONS:
  1. [feature_A] × [feature_B] — never tested together, structural reason to check: [why]
  2. ...

TEMPORAL GAPS (never checked):
  1. Month-of-year × session
  2. DOW × session (session-specific, not global)
  3. ...

CROSS-SESSION WITHIN-DAY (check NO-GO first):
  1. NYSE_OPEN outcome → US_DATA_1000 direction
  2. ...

NO-GO CONFIRMED DEAD (skip these):
  1. [item] — killed by [test], reason: [why]
  2. ...

ESTIMATED TOTAL UNEXPLORED DIMENSIONS: [N]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**WAIT for my direction after presenting the map.** I'll pick which veins to mine.

---

### STEP 3 — MINE (only what I greenlight)

For each dimension I greenlight:

#### 3A — Exploratory bin analysis (fast, cheap, kills early)

```sql
-- Quintile the feature, compute ExpR and WR per bin, per session, per instrument
SELECT
    [session],
    NTILE(5) OVER (PARTITION BY [session] ORDER BY [feature]) AS bin,
    COUNT(*) AS n,
    AVG(pnl_r) AS expr,
    AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) AS wr,
    STDDEV(pnl_r) AS std_r
FROM orb_outcomes o
JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
WHERE o.symbol = 'MNQ'
  AND o.entry_model = 'E2'
  AND o.rr_target = 1.0
  AND o.orb_minutes = 5
  AND d.[filter_column] IS NOT NULL  -- session-specific filter if applicable
GROUP BY 1, 2
ORDER BY 1, 2
```

Report format (PER SESSION — never pooled):

```
FEATURE: [name]
SESSION: [name] | INSTRUMENT: [MNQ/MGC/MES]

BIN | N    | ExpR   | WR    | Std
────┼──────┼────────┼───────┼──────
1   | XXX  | +0.XXX | XX.X% | X.XX
2   | XXX  | +0.XXX | XX.X% | X.XX
3   | XXX  | +0.XXX | XX.X% | X.XX
4   | XXX  | +0.XXX | XX.X% | X.XX
5   | XXX  | +0.XXX | XX.X% | X.XX

WR SPREAD: [Q5 - Q1]  → SIGNAL (>5%) / FLAT (<3%) / NOISE (3-5%)
ExpR MONOTONIC: YES / NO / PARTIAL
QUICK VERDICT: DIG_DEEPER / ARITHMETIC_ONLY / DEAD
```

**Kill rule for Step 3A:** If WR spread < 3% AND ExpR is not monotonic → DEAD. Move on.
Don't invest T0-T8 time on something that shows nothing in the exploratory bin.

#### 3B — Interaction analysis (for feature pairs)

Hold feature A constant (e.g., top quintile), vary feature B across quintiles.
Then swap: hold B constant, vary A. This isolates the marginal contribution of each.

```
INTERACTION: [feature_A] × [feature_B]
SESSION: [name]

HOLDING A=Q5 (high), VARYING B:
B_BIN | N   | ExpR   | WR    | MARGINAL EFFECT OF B
──────┼─────┼────────┼───────┼─────────────────────
1     | XX  | +0.XXX | XX.X% |
5     | XX  | +0.XXX | XX.X% | Δ = [difference]

HOLDING B=Q5 (high), VARYING A:
A_BIN | N   | ExpR   | WR    | MARGINAL EFFECT OF A
──────┼─────┼────────┼───────┼─────────────────────
1     | XX  | +0.XXX | XX.X% |
5     | XX  | +0.XXX | XX.X% | Δ = [difference]

INTERACTION VERDICT:
  A alone explains: [X]% of variance
  B alone explains: [X]% of variance
  A×B together explains: [X]% — if > A+B individually → REAL INTERACTION
  If A×B ≈ A+B → NO INTERACTION, features are additive (still useful, but not a new finding)
```

#### 3C — Temporal pattern analysis

For month/DOW/sequence effects:

```sql
-- DOW × session example
SELECT
    DAYOFWEEK(trading_day) AS dow,
    [session],
    COUNT(*) AS n,
    AVG(pnl_r) AS expr,
    AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) AS wr
FROM orb_outcomes o
WHERE symbol = 'MNQ' AND entry_model = 'E2' AND rr_target = 1.0
GROUP BY 1, 2
ORDER BY 2, 1
```

**WARNING on temporal patterns:** These have the HIGHEST multiple testing risk.
5 DOW × 12 sessions × 3 instruments = K=180 comparisons just for DOW.
BH FDR threshold at K=180 is brutal. Most things will die. That's correct.
If something survives K=180, it's real. If nothing does, temporal patterns don't matter
for this system. Report that finding honestly.

---

### STEP 4 — ESCALATE SURVIVORS (full T0-T8)

Any finding that passes Step 3A with WR spread > 5% AND monotonic ExpR gets
the FULL audit battery from `quant-audit-protocol.md`. No shortcuts. T0 through T8,
in order, per session.

Pre-register before running:
```
FINDING: [feature] predicts [session] ORB outcomes
CLAIM TYPE: statistical_observation (upgrading to validated_finding if T0-T8 pass)
IS_END: [define before running — do not adjust]
KILL CRITERIA: WFE < 0.50, sensitivity flips, bootstrap p > 0.05
K FOR FDR: [honest K — total dimensions explored in this mining session]
```

**K must include ALL dimensions you explored in Step 3, not just the one that
survived.** If you binned 15 features × 12 sessions and 1 survived, K=180.
This is the tax on data mining. Pay it honestly.

---

### BEHAVIORAL RULES

**No stories before numbers.** Don't hypothesize why DOW=Tuesday might matter
before you've seen the data. Bin first, interpret second.

**Session-specificity at every step.** A Tuesday effect on NYSE_OPEN and a Tuesday
effect on SINGAPORE_OPEN are different findings. Always decompose.

**The NO-GO registry is not a suggestion.** If something is dead, it's dead.
You can test a STRUCTURALLY DIFFERENT version of a dead idea (e.g., within-day
lead-lag vs prior-day context), but you must explain how it's different and why
the prior kill doesn't apply.

**Multiple testing tax is real.** Data mining explores many dimensions. The more
you explore, the higher K goes, the harder it is to pass FDR. This is the CORRECT
outcome — it prevents you from finding noise and calling it signal. If nothing
survives after honest FDR correction, the data has been mined and there's nothing
left. That's a valid finding.

**No lookahead.** Every feature must be knowable at trade entry time.
Trade-time-knowable: `risk_dollars`, `break_delay_min`, `rel_vol`, `atr_20`, `orb_size`,
`day_of_week`, `month`, any prior-session outcome (if that session has already closed).
BANNED: `double_break`, anything computed from post-entry price action.

**Report EVERYTHING you tested, not just what worked.** The null results are as
important as the positives. They close off dead ends for future sessions.
Append findings (positive AND negative) to the KNOWN FAILURE PATTERNS section
of `quant-audit-protocol.md` so they're never re-tested.

**The most likely outcome is that most dimensions are noise.** The existing
filter set (G-filters, ATR, X_MES_ATR60, vol regime) has already captured the
low-hanging fruit. What's left is either subtle interactions, temporal patterns,
or nothing. "Nothing new found" is a valid, useful result. Don't manufacture
a finding to justify the mining session.

---

### OUTPUT FORMAT

After mining, deliver:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MINING REPORT — canompx3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIMESTAMP:        [UTC]
DIMENSIONS EXPLORED: [N total]
K FOR FDR:           [honest total]

TERRITORY MAPPED:
  Never-analyzed features found:    [N]
  Untested interactions found:      [N]
  Temporal patterns checked:        [N]

SURVIVORS (passed Step 3A screening):
  [feature] × [session]: WR spread=[X]%, ExpR monotonic=[Y/N]
  → Escalated to T0-T8: [YES/NO]
  → T0-T8 result: [VALIDATED / KILLED at T[N] / PENDING]

DEAD ENDS (failed Step 3A — record for future sessions):
  [feature] × [session]: WR spread=[X]%, verdict=DEAD
  [feature] × [session]: ExpR non-monotonic, verdict=DEAD
  [temporal pattern]: K=[N], no survivors after FDR

INTERACTION FINDINGS:
  [feature_A × feature_B]: Real interaction=[Y/N], marginal=[X]R

NEW NO-GO ENTRIES (append to audit protocol):
  [date] [feature/pattern]: DEAD, reason=[X]

GEMS FOUND: [count]
  If 0: "Dataset mined thoroughly. Existing filters capture available signal.
         No new dimensions worth pursuing. This is a valid finding."
  If >0: [structured finding per quant-audit-protocol output format]

RECOMMENDATION:
  [what to do with findings — or "stop mining, trade what you have"]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### START

Read the law. Run Step 1 (schema audit + feature catalog). Present the map.
Don't test anything until I pick the veins.

## PROMPT END
