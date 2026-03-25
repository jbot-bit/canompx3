# Lane Deployment & Validation — Execution Prompt

Phase 0-2 complete. Rankings challenged and revised. This prompt executes the top 2 items.

---

## PROMPT START

I have a validated ORB breakout portfolio (4 MNQ lanes in Apex) and completed a multi-phase research investigation that identified two high-value actions, ranked by expected impact and readiness. Prior sessions did Phase 0 (orient), Phase 1 (challenge rankings), and Phase 2 (honest prioritization). This session executes.

**Your job: verify, then deploy. Kill fast if the numbers don't hold.**

### CONTEXT YOU MUST DISCOVER (don't trust this summary — verify everything)

Prior sessions concluded:
- **#1 US_DATA_1000** — 4 MNQ validated strategies (FDR=True), 8 cross-instrument, best MNQ ExpR=+0.138, 7/8 positive years, ~+15-30R/yr expected, near-zero correlation with existing lanes. 90% deployment-ready. Needs: edge family check, prop profile lane config, pre-session integration.
- **#2 CME_PRECLOSE** — Strongest unfiltered MNQ session (+0.117 ExpR) but ZERO MNQ validated strategies. 2160 experimental, 137 with ExpR>0.05. Failed BH FDR at K=55. Pre-registered for 2026 holdout. Needs: validation run to determine if any MNQ strategies survive.

**These conclusions may be stale or wrong.** Verify before acting.

---

### TASK A — US_DATA_1000 DEPLOYMENT VERIFICATION (do this first)

#### Step A1: Re-verify the validated strategies exist and are current

Query `validated_setups` for US_DATA_1000. For each MNQ strategy, extract:
- Strategy ID, filter_type, orb_minutes, entry_model, confirm_bars, rr_target, direction
- Sample size (N), win_rate, ExpR, Sharpe
- Walk-forward efficiency (WFE)
- FDR status (pass/fail, at what K)
- Date range covered

**Kill criterion:** If fewer than 2 MNQ strategies survive with ExpR > 0.05 and FDR=True → STOP. The prior session's claim was wrong.

#### Step A2: Session-specific deployment profile

Build the US_DATA_1000 session profile FROM LIVE DATA (not from prior session notes):

```
SESSION: US_DATA_1000
TIME: [query SESSION_CATALOG — Brisbane TZ and exchange mapping]
MECHANISM: [articulate why edge exists HERE specifically]
  - What happens at 10:00 AM ET? (economic data releases + post-open secondary flow)
  - Who is trading? (institutional desks deploying after NYSE open digest period)
  - Why would ORB breakouts work HERE vs a random time? (data catalyst + directional conviction)
  - How is this DIFFERENT from NYSE_OPEN? (they overlap in US hours but different catalysts)
  - How is this DIFFERENT from US_DATA_830? (different data releases, different liquidity)
FILTER LOGIC: [why does X_MES_ATR60 work for this session?]
  - MES ATR percentile = overall US equity volatility regime
  - US_DATA_1000 is a US-hours session → US equity vol is the right regime measure
  - This is the SAME filter used on NYSE_OPEN → structural consistency, not overfitting
  - Would this filter make sense for an overnight session? NO → session-appropriate
CORRELATION: [query pairwise correlation with each existing Apex lane]
  - Must be < 0.10 with every existing lane for clean diversification
  - If correlated > 0.15 with NYSE_OPEN → FLAG (same US hours, similar catalyst?)
```

**Kill criterion:** If correlation with NYSE_OPEN > 0.20 → PAUSE. The diversification benefit may be overstated. Investigate whether US_DATA_1000 trades are just NYSE_OPEN continuation trades in disguise.

#### Step A3: Per-year deep dive

For the BEST MNQ strategy (highest ExpR with FDR=True):
- Per-year ExpR, WR, N, and cumulative R for every full year available
- Identify the worst year — what happened? (regime shift, vol regime, specific events?)
- Identify if performance is front-loaded (early years strong, recent weak) or stable
- 2025 specifically — what does the most recent full pre-holdout year show?

**Kill criterion:** If 2024 or 2025 are both negative → DOWNGRADE to MONITOR. Recent performance matters for deployment timing.

#### Step A4: Strategy selection

If A1-A3 pass, recommend ONE strategy for deployment. Justify:
- Why this one over the other 3 MNQ validated options?
- What's the tradeoff? (higher ExpR with smaller N vs lower ExpR with larger N)
- What's the expected trade frequency? (trades/year after filter application)
- What's the expected R/year? (ExpR × trades/year — give a RANGE, not point estimate)

Present as:
```
RECOMMENDED STRATEGY: [full strategy spec]
EXPECTED TRADES/YR: [range]
EXPECTED R/YR: [range]
CONFIDENCE: [HIGH/MEDIUM/LOW with reasoning]
ALTERNATIVE: [second choice and why it's #2]
```

#### Step A5: Implementation checklist (DO NOT IMPLEMENT — just list what's needed)

Identify every file that would need to change for deployment:
- Prop profile config (which file, what to add)
- Pre-session check integration (if applicable)
- Edge family assignment (check if already done)
- Any pipeline or config changes required
- Dashboard/reporting updates

**Do NOT write any code.** List the files and changes. Wait for my go.

---

### TASK B — CME_PRECLOSE VALIDATION (do this second, only if Task A completes)

#### Step B1: Run strategy validation on MNQ CME_PRECLOSE experimentals

This is the definitive test. Prior sessions established that 2160 MNQ experimental strategies exist, with 137 having ExpR > 0.05. The question is: how many survive the full validation pipeline?

Run `strategy_validator.py` (or equivalent) on CME_PRECLOSE MNQ strategies. Report:
- How many entered validation
- How many survived each gate (walk-forward, FDR, per-year stability, etc.)
- Final count of VALIDATED strategies
- For each survivor: full strategy spec, ExpR, WR, N, WFE, FDR status

**Hard kill:** If ZERO MNQ strategies survive → CME_PRECLOSE is DEAD for MNQ deployment. Report this clearly. Do not suggest "loosening thresholds" or "re-running with different parameters." Dead means dead. The pre-registered 2026 holdout test is the only remaining path.

#### Step B2: If survivors exist — session profile and audit

Same structure as Task A Steps 2-4, but for CME_PRECLOSE:
- Session profile with mechanism (settlement-driven mechanical flows)
- Per-year stability on survivors
- Correlation with existing lanes + US_DATA_1000 (if being added)
- Strategy selection and expected R/year

#### Step B3: If survivors exist — compare to US_DATA_1000

Side-by-side:
```
                    US_DATA_1000        CME_PRECLOSE
Best MNQ ExpR:      [X]                 [X]
Sample size:        [X]                 [X]
Years positive:     [X/Y]              [X/Y]
WFE:                [X]                 [X]
Correlation w/ Apex:[X]                 [X]
Est. R/yr:          [range]             [range]
Deployment ready:   [YES/NO]            [YES/NO]
```

---

### BEHAVIORAL RULES

1. **Verify, don't assume.** Every number from the prior session must be re-queried. Memory is not truth. The DB is truth.

2. **Session-specificity at every step.** US_DATA_1000 and CME_PRECLOSE are DIFFERENT sessions with different market structures. Don't copy-paste analysis. Each gets its own profile, its own mechanism hypothesis, its own filter logic assessment.

3. **Kill fast.** Each step has an explicit kill criterion. If triggered, STOP that task. Report the kill. Move to the next task. Do not try to rescue a failing deployment candidate.

4. **No implementation without approval.** Task A Step 5 produces a checklist, not code. Task B is research only. I decide what gets built and when.

5. **Correlation is the hidden risk.** The prior session showed near-zero correlations, but that was summary-level. Check if US_DATA_1000 and NYSE_OPEN fire on the SAME DAYS (overlap percentage). Near-zero correlation in returns doesn't mean independent if they're trading the same events.

6. **2026 data is observed-not-acted-upon.** Prior sessions peeked at 2026 forward data (CME_PRECLOSE N=53, US_DATA_1000 N=56). This contamination is acknowledged. Do NOT use 2026 data for any decision threshold. If you need to reference it, label it `[2026-OBSERVED]` and do not weight it in your recommendation.

7. **The honest answer might be "just deploy US_DATA_1000 and stop."** If CME_PRECLOSE produces zero survivors, say so. If the marginal value of adding a 6th lane is negligible, say so. Don't manufacture a second deployment to fill the prompt.

8. **Read the governing docs.** Before any queries: `CLAUDE.md`, `TRADING_RULES.md`, `RESEARCH_RULES.md`, `.claude/rules/quant-audit-protocol.md`. The pipeline has specific rules about filter application, FDR thresholds, and discovery layer discipline. Violating them invalidates the work.

---

### OUTPUT FORMAT

When done, deliver:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPLOYMENT REPORT — canompx3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIMESTAMP:        [UTC]
DB FRESHNESS:     [latest trading_day]

TASK A — US_DATA_1000
  Status:         DEPLOY / PAUSE / KILL
  Strategy:       [full spec]
  ExpR:           [X] (N=[X], WR=[X]%)
  WFE:            [X]
  Per-year:       [X/Y positive]
  Correlation:    [max with any Apex lane]
  Trade overlap:  [% days shared with NYSE_OPEN]
  Est. R/yr:      [range]
  Implementation: [file list — no code]
  Kill triggers:  [any triggered? which?]

TASK B — CME_PRECLOSE
  Status:         VALIDATED / DEAD / NEEDS_MORE_DATA
  Survivors:      [count] of [tested]
  Best (if any):  [full spec, ExpR, N, WFE]
  Per-year:       [X/Y positive]
  vs US_DATA_1000:[side-by-side comparison]
  Next step:      [deploy / wait for 2026 holdout / abandon]

RECOMMENDATION:
  [1-3 sentences. What to do NOW. What to defer. What's dead.]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### START

Read governing docs. Verify Task A Step 1. Go.

## PROMPT END
