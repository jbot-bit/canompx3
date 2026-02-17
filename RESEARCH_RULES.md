# Research Rules — Analytical Standards for This Project

**Authority:** Wins for all research methodology, statistical analysis, and trading interpretation decisions.
**Conflict:** If TRADING_RULES.md says "X works" and this file says "the evidence for X is weak" → this file wins on methodology, TRADING_RULES.md wins on what to trade.

---

## Role

When analyzing data or producing research for this project, act as a **systematic futures trader and quantitative researcher**, not a data scientist. Every output must pass this test: "Would a prop desk risk manager accept this as evidence to allocate capital?"

---

## Statistical Rigor

### Sample Size Rules
| Trades | Classification | What You Can Do |
|--------|---------------|-----------------|
| < 30 | INVALID | Cannot draw conclusions. Do not report as finding. |
| 30-99 | REGIME | Conditional signal only. Never standalone. State "N=XX, regime-class only." |
| 100-199 | PRELIMINARY | Actionable with caveats. State limitations. |
| 200+ | CORE | Reliable for capital allocation if mechanism exists. |
| 500+ | HIGH-CONFIDENCE | Institutional-grade sample if multi-regime. |

**Sample count alone is insufficient.** 500 trades in 6 months of a bull market is WEAKER than 100 trades across 5 years covering bull, bear, and sideways regimes. Always report the time span alongside trade count.

### Significance Testing
- **p < 0.05:** Minimum threshold for noting a result. NOT sufficient for action.
- **p < 0.01:** Required for any finding you'd recommend trading.
- **p < 0.005:** Required for "discovery" claims (per Harvey & Liu, 2014).
- **Always report exact p-values**, not just "significant" or "not significant."
- **After grid search / multiple tests:** Apply Benjamini-Hochberg FDR correction. If you tested 50+ combinations, expect 2-3 false positives at p<0.05 even if nothing is real.

### The Bailey Rule (Multiple Variations)
With 5 years of daily data, testing more than ~45 strategy variations guarantees at least one will show Sharpe >= 1.0 purely by chance (Bailey & López de Prado, 2014). This project's grid searches 2,376 combinations. Therefore:
- **NEVER cite a single strategy's backtest Sharpe as evidence of edge.**
- **ALWAYS report family-level averages** (per TRADING_RULES.md reporting rules).
- **Walk-forward validation is mandatory** before any finding moves to "confirmed."
- If in-sample Sharpe drops by more than 50% out-of-sample, the finding is likely overfit.

### In-Sample vs Out-of-Sample
- Same data used for discovery AND testing = ONE test, not two.
- Expected Sharpe decay from in-sample to out-of-sample: ~50-63% (industry benchmark).
- Walk-forward efficiency (WFE) > 50% = strategy likely real. < 50% = likely overfit.
- **Current limitation:** Most cross-instrument research uses only 579 overlapping days (Feb 2024 - Feb 2026). This is a single in-sample window. No findings from this window are "validated" until tested on the 10-year rebuild.

### Sensitivity Analysis (Mandatory for Parameter-Based Findings)
Before calling any parameterized finding "real," test:
- Change the parameter by ±20%. Does the result survive?
- Change the threshold by ±2 units. Does the result survive?
- If small parameter changes kill the finding → it's curve-fitted, not real.
- RSI, moving average, and ATR threshold findings are ESPECIALLY prone to this.

---

## Trading Lens — Mechanism Over Numbers

### The Mechanism Test
Every statistical finding MUST have a structural market reason (mechanism) for WHY it works. Without a mechanism, it is a "statistical artifact until proven otherwise."

**Good mechanisms (structural, likely to persist):**
- Friction eats small ORBs → size filter works (cost structure). **This is THE primary edge in the project — confirmed across all instruments Feb 2026.**
- Double-break = chop, no directional conviction → exclude (market microstructure)
- All 3 instruments breaking same direction = macro flow day → concordance (institutional behavior)
- Large gold ORB at 2300 = active overnight session → equity vol at 0030 (cross-market information flow)
- 1000 shorts fail because Asian session has upward bias from physical demand (market structure)
- Large ORB = institutional participation, real conviction → breakout has energy behind it (market microstructure)

**Bad mechanisms (unlikely to persist or not mechanisms at all):**
- "The numbers show it works" (that's not a mechanism)
- "It worked in 2024-2025" (that's a regime, not a mechanism)
- "RSI below 30 means oversold" (RSI is a lagging calculation, not a cause)
- "Year-over-year improvement" on 3 data points (that's noise)
- "Low correlation between gold and equities" (that's a baseline fact, not a discovery)

### Reframing Rules
| Raw Finding | Wrong Interpretation | Right Interpretation |
|-------------|---------------------|---------------------|
| MGC 2300 large ORB → MES 0030 better | "Gold predicts equity direction" | "Gold volatility signals active overnight session → equity breakouts have more energy" |
| 3/3 concordance improves WR | "All instruments agree = strong signal" | "Macro flow day → all assets trending → breakout more likely to follow through" |
| RSI < 30 + long breakout works | "Buy oversold bounces" | "UNVALIDATED. RSI threshold sensitivity not tested. Guilty until proven innocent." |
| Edge strengthening 2024→2026 | "Strategy is getting better" | "Gold trending harder in this period. Regime-dependent, not structural." |
| MES/MNQ 87% concordance | "Equities confirm each other" | "They're the same asset class. This is baseline, not signal." |

### Volatility vs Direction
A common error in this project: confusing volatility signals with directional signals.
- MGC 2300 ORB **size** predicts MES 0030 quality. **Direction** does not.
- Concordance is about conviction (all trending), not about which way.
- Large ORBs work because they indicate a session with energy, not because "big up ORB = price goes up."

---

## Report Standards

### What to Call Things
| Label | Requirements |
|-------|-------------|
| "Confirmed edge" | Walk-forward validated, mechanism exists, N >= 100, deployed or deployable |
| "Validated finding" | Out-of-sample tested, mechanism exists, N >= 100 |
| "Promising hypothesis" | In-sample positive, mechanism plausible, needs OOS test |
| "Statistical observation" | Numbers show X, no mechanism confirmed, may be noise |
| "NO-GO" | Tested and failed. Do not revisit without new model type. |

**NEVER use:** "Institutional-grade," "alpha," "edge" (without qualifying as confirmed/validated/promising), "significant" (without exact p-value), "trend" (for < 5 data points).

### Mandatory Disclosures in Every Research Output
1. **Sample size** (N trades, not N days)
2. **Time period** (exact dates)
3. **In-sample or out-of-sample?** (be explicit)
4. **Number of variations tested** (for multiple comparison context)
5. **Mechanism** (why it should work, or "no mechanism identified")
6. **What could kill this** (regime change, parameter sensitivity, cost model change)

### Honest Summary Format
Every research output should end with:
```
SURVIVED SCRUTINY: [list what passed, with N and p-values]
DID NOT SURVIVE: [list what failed, with reasons]
CAVEATS: [honest limitations]
NEXT STEPS: [what would validate or invalidate]
```
This format already exists in the research scripts. Maintain it.

---

## NO-GO Rules

### Things That Are Always Wrong
1. **NEVER suggest "fixing" filters to increase sample size.** That's data mining, not research.
2. **NEVER treat low trade count alone as a bug.** Strict filters (G6/G8) SHOULD have few trades.
3. **NEVER present in-sample results as validated.** Words matter. "In-sample positive" ≠ "confirmed edge."
4. **NEVER recommend REGIME-class strategies for standalone trading.**
5. **NEVER speculate on WHY a session-specific pattern exists unless you have evidence.** Record the empirical fact. "0900 and 1100 show concordance benefit, 1000 and 1800 do not" is the correct statement. "Early sessions work because X" is speculation.
6. **NEVER call 3 data points a trend.** 5 is barely suggestive. 10+ with a mechanism is a trend.
7. **NEVER assume stationarity.** If gold returns to $1800, most current G5+ filters become untradeable (< 5 qualifying days per year).

### Indicators That Are Guilty Until Proven Innocent
RSI, MACD, Bollinger Bands, moving average crossovers, Stochastics. These are the most over-fitted indicators in existence. Any finding based on these requires:
- Sensitivity analysis (±2 on period/threshold)
- Out-of-sample validation on unseen data
- A mechanism beyond "indicator says overbought/oversold"
- Survival through walk-forward, not just full-period backtest

---

## Market Structure Knowledge

### Session Context (All Times Brisbane UTC+10)
| Session | Brisbane | UTC | ET (approx) | What's Happening |
|---------|----------|-----|-------------|-----------------|
| 0900 | 9:00 AM | 23:00 prev | ~6 PM prev | Asia/Sydney open. Gold physical demand. Thin liquidity. |
| 1000 | 10:00 AM | 00:00 | ~7 PM prev | Tokyo. Increasing liquidity. LONG-ONLY edge. |
| 1100 | 11:00 AM | 01:00 | ~8 PM prev | Singapore/HK. **EXCLUDED** — 74% double-break. |
| 1130 | 11:30 AM | 01:30 | ~8:30 PM prev | HK/SG continuation. |
| 1800 | 6:00 PM | 08:00 | ~3 AM | US pre-market / GLOBEX open. Data releases. Highest vol. |
| 2300 | 11:00 PM | 13:00 | ~8 AM | US morning. Position building. |
| 0030 | 12:30 AM | 14:30 | ~9:30 AM | US equity open (NYSE). Key for MES/MNQ. |

**CRITICAL:** 1100 (11:00 AM, Singapore) and 2300 (11:00 PM, US morning) are COMPLETELY DIFFERENT sessions. 24-hour time. No ambiguity.

### Cross-Asset Relationships
- **Gold ↔ Equities:** Structurally uncorrelated (~0.05 correlation). This is BASELINE, not a finding.
- **Gold ↔ USD:** Inverse correlation (~-0.60 over 5 years). Driven by dollar denomination.
- **Gold ↔ Real Yields:** Inverse. Rising real yields pressure gold (opportunity cost of holding non-yielding asset).
- **MES ↔ MNQ:** ~87-89% concordant. Same asset class. Agreement is not signal.
- **MGC 2300 → MES/MNQ 0030:** Volatility transmission, not directional. Size matters, direction doesn't. 90-minute temporal gap makes this zero-lookahead.

### Cost Reality for Micro Futures
| Instrument | Tick Value | RT Friction | Friction as % of 5pt ORB |
|-----------|-----------|-------------|--------------------------|
| MGC | $1.00/tick | $8.40 | 16.8% |
| MES | $1.25/point | $2.10 | 8.4% |
| MNQ | $2.00/point | $2.74 | 5.5% |
| MCL | $1.00/tick | ~$3.00 | varies |

**Friction is why small ORBs lose.** A 2pt MGC ORB has 42% of its risk eaten by friction before the trade starts. This is the STRUCTURAL reason ORB size filtering works — it's not a statistical artifact, it's arithmetic.

### Gold-Specific Knowledge
- Gold trends intraday more than it mean-reverts. The ONLY confirmed edges in this project are momentum breakouts with size filters.
- Physical demand (central banks, jewelry) creates baseline buying pressure, especially in Asian sessions.
- Gold had an exceptional run in 2024-2025 (central bank buying, geopolitical hedging). G5+ day frequency exploded from ~3% to 27-88%. This is regime, not permanent.
- If gold returns to $1800 levels, the entire strategy set becomes low-frequency (5 qualifying days/year at 0900 G5+).

---

## Current Research Status

### Validated and Deployed
See `TRADING_RULES.md` → Confirmed Edges table.

### Cross-Instrument Stress Test Finding (Feb 2026)
**ORB size is the primary edge across ALL instruments.** Not sessions, not parameters, not entry models. Stress test (`scripts/tools/stress_test.py`) on 11 top edges:
- MES/MGC: ALL edges die without size filter. Regime-dependent, parameter-fragile.
- MNQ 0900: SURVIVED. Positive all 3 years, bidirectional, all real neighbors profitable.
- Mechanism: friction as % of risk (arithmetic, not statistical).
- **Research priority:** Map optimal ORB size threshold per session per instrument. The threshold is NOT universal — MNQ (low friction) may profit from smaller ORBs than MGC (high friction).
- Tool: `scripts/tools/orb_size_deep_dive.py`

### Awaiting 10-Year Outcome Rebuild
These findings are in-sample only (579 days, Feb 2024 - Feb 2026). Do NOT act on them until validated on full dataset:
1. **Concordance filter (0900, 1100)** — mechanism strong, awaiting OOS
2. **MGC 2300 size gate for MES/MNQ 0030** — mechanism plausible, awaiting OOS
3. **Asia session high filter** — needs investigation after rebuild
4. **RSI oversold for long ORB** — guilty until sensitivity-tested

### Confirmed NO-GOs (Do Not Revisit)
See `TRADING_RULES.md` → What Doesn't Work table.
Also: `docs/RESEARCH_ARCHIVE.md` for full data.

### Re-Validation Trigger
Cross-instrument findings (concordance, MGC 2300 gate) should be re-validated when overlapping day count reaches **800 days** (currently 579). This is a data trigger, not a time trigger. Do not re-run at arbitrary dates.

---

## Coding Standards for Research Scripts

### Location
All research scripts go in `research/` directory. Never in `scripts/` or project root.

### Required Structure
```python
# 1. Honest header
"""One-line description. Date. Author."""

# 2. Zero-lookahead compliance
# All features computed from data available at decision time

# 3. Honest summary section at end of output
# SURVIVED / DID NOT SURVIVE / CAVEATS / NEXT STEPS

# 4. No sys.path hacks (project is pip-installable)
```

### Output Format
Research scripts must print structured output with:
- Exact N, WR, ExpR, Sharpe, p-values for every finding
- Clear labels for in-sample vs out-of-sample
- Baseline comparison (unconditional performance) for every conditional finding
- Caveats section acknowledging limitations
