# Research Rules — Analytical Standards for This Project

**Authority:** Wins for all research methodology, statistical analysis, and trading interpretation decisions.
Document-role registry: `docs/governance/document_authority.md`.
**Conflict:** If TRADING_RULES.md says "X works" and this file says "the evidence for X is weak" → this file wins on methodology, TRADING_RULES.md wins on what to trade.

> **Phase 0 literature grounding (2026-04-07).** This file's statistical thresholds are now backed by verbatim PDF extracts in [`docs/institutional/literature/`](docs/institutional/literature/). Specifically:
>
> - **The Bailey Rule** (§ Statistical Rigor below) — see [`docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md`](docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md) for MinBTL theorem and [`docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`](docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md) for the DSR formula.
> - **Multiple-testing thresholds** — see [`docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`](docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md) (t ≥ 3.79) and [`docs/institutional/literature/harvey_liu_2015_backtesting.md`](docs/institutional/literature/harvey_liu_2015_backtesting.md) (BHY haircut, profitability hurdle Exhibit 4).
> - **Theory-first principle** — see [`docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md`](docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md): *"Backtests are not a research tool. Theories are."*
> - **Live drift monitoring** — see [`docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`](docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md) for the Shiryaev-Roberts procedure.
>
> **The 12 locked criteria** that any validated strategy must meet live in [`docs/institutional/pre_registered_criteria.md`](docs/institutional/pre_registered_criteria.md) — see § *Version history* + § *Amendment procedure* in that file for the current revision state. Read it BEFORE any discovery run. Discovery without a committed pre-registered hypothesis file in `docs/audit/hypotheses/` is banned per Criterion 1. The 300-trial MinBTL bound is per Criterion 2. **Cross-reference convention:** always cite by the stable `## Criterion N` anchors; amendment subsection numbers are revision history and not stable cross-refs.
>
> **🔴 2026-04-08 CORRECTION — Mode B RESCINDED, Mode A (holdout-clean) operative.** Yesterday's autonomous Mode B declaration (committed `1aa11e5`) was WRONG per user intent. The user was operating under the assumption that 2026 was being held out and that ~3 months of real-time forward OOS data was accumulating from 2026-01-01. The Apr 2 16yr pipeline rebuild plan (`docs/plans/2026-04-02-16yr-pipeline-rebuild.md:79-83`) EXPLICITLY prescribed `--holdout-date 2026-01-01` with the comment *"CRITICAL: --holdout-date 2026-01-01 protects sacred holdout"* — the 117-strategy baseline from that rebuild was legitimately Mode A. The Apr 3 decision to re-run discovery without `--holdout-date` (gaining 7 extra strategies to reach 124) was a **5-day-old autonomous mistake that consumed a valid holdout for marginal benefit** — now rescinded. **Mode A is restored:** 2026-01-01 onwards is the sacred forward window, currently ~3.2 months deep, continuing to accumulate real-time forward OOS data. Existing 124 validated_setups are contaminated by the Apr 3 Mode B re-run and are research-provisional (Amendment 2.4 already labels them this way); they need re-validation under Mode A once the pipeline rerun happens (blocked on e2-canonical-window-fix merge). **See `Criterion 8 — 2026 out-of-sample positive` (as revised by Amendment 2.7) in [`docs/institutional/pre_registered_criteria.md`](docs/institutional/pre_registered_criteria.md) for the correction + full audit trail.** The rescinded Mode B doc is kept for audit history at [`docs/plans/2026-04-07-holdout-policy-decision.md`](docs/plans/2026-04-07-holdout-policy-decision.md).

---

## Role

When analyzing data or producing research for this project, act as a **systematic futures trader and quantitative researcher**, not a data scientist. Every output must pass this test: "Would a prop desk risk manager accept this as evidence to allocate capital?"

---

## Discovery Layer Discipline (enforced 2026-03-24)

**Discovery and research use ONLY canonical layers:** `bars_1m`, `daily_features`, `orb_outcomes`.

**Derived layers are BANNED for truth-finding:** `validated_setups`, `edge_families`, `live_config`, docs, comments, and memory files may be stale or contaminated. They reflect past deployment state, not current research truth. If a derived layer contradicts a canonical query, the canonical query wins — mark the derived layer STALE.

**Every research finding must be grounded in:**
- Exact canonical-layer query (SQL or script path)
- Sample size, p-value (exact, two-tailed), and BH FDR K
- Walk-forward efficiency (WFE) where applicable
- Data state timestamp (e.g., "orb_outcomes through 2026-03-23, 1475 MNQ trading days")

**2026 holdout is sacred.** Forward-test data informs paper-trade monitoring ONLY. It must not be used to select sessions, thresholds, filters, or entry models. Session selection must be reproducible from pre-2026 data alone. The sacred window is **2026-01-01 onwards** and is currently ~3.2 months deep (2026-01-01 → 2026-04-08), continuing to accumulate daily. Every discovery run MUST use `--holdout-date 2026-01-01` (or earlier). See [`docs/plans/2026-04-02-16yr-pipeline-rebuild.md:79-83`](docs/plans/2026-04-02-16yr-pipeline-rebuild.md) for the original Apr 2 plan that established this discipline, and `docs/institutional/pre_registered_criteria.md` § *Criterion 8 — 2026 out-of-sample positive* (as revised by Amendment 2.7 on 2026-04-08) for the Mode A restoration after the brief Apr 3 → Apr 8 Mode B deviation.

> **Audit history — the brief Mode B deviation (2026-04-03 → 2026-04-08):**
>
> Between Apr 3 and Apr 8 2026, the project briefly operated under Mode B (post-holdout-monitoring). This was an error:
> - Apr 3: autonomous decision to re-run discovery without `--holdout-date`, gaining 7 extra validated strategies (117 → 124). See `HANDOFF.md:1465-1473`.
> - Apr 7: autonomous Mode B declaration codified the Apr 3 decision as policy (commit `1aa11e5`, pre_registered_criteria.md Amendment 2.6).
> - Apr 8: user correction identified the error. The Apr 2 plan's `--holdout-date 2026-01-01` + "2026 is sacred" discipline was the correct intent. Mode B rescinded. Amendment 2.7 restores Mode A (committed `bb82add`). The earlier Mode B doc updated with a RESCINDED header at the top of `docs/plans/2026-04-07-holdout-policy-decision.md`.
>
> The 124 current validated_setups are **research-provisional** per the *Applying to current 5 deployed lanes* section of `pre_registered_criteria.md` (the lane classification language was codified by Amendment 2.4) — they were discovered with 2026 data in scope during the Mode B window and need re-validation under the restored Mode A discipline. The 5 deployed lanes continue trading operationally (no change) but carry research-provisional status until re-validated. The `e2-canonical-window-fix` worktree has merged into main (commit `8bc87f7`), lifting the scope_lock on `pipeline/check_drift.py`, `pipeline/build_daily_features.py`, `trading_app/outcome_builder.py`, and related files. The pipeline rerun to restore the clean 117-strategy baseline is now unblocked pending user approval of the rerun scope.

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
- **Discovery claims:** see the binding threshold in `docs/institutional/pre_registered_criteria.md` § *Criterion 4 — Chordia t-statistic threshold*.
- **Always report exact p-values**, not just "significant" or "not significant."
- **After grid search / multiple tests:** Apply Benjamini-Hochberg FDR correction. If you tested 50+ combinations, expect 2-3 false positives at p<0.05 even if nothing is real.
- **BH K selection:** Report both global K (full search space) and the honest instrument/family/decision K where relevant. Use the honest instrument/family/decision K for promotion or portfolio decisions; use global K for conservative full-search headline claims. Never swap K post-hoc to rescue or kill a result.

### The Bailey Rule (Multiple Variations)
With 5 years of daily data, testing more than ~45 strategy variations guarantees at least one will show Sharpe >= 1.0 purely by chance (Bailey & López de Prado, 2014). This project's grid searches 2,376 combinations. Therefore:
- **NEVER cite a single strategy's backtest Sharpe as evidence of edge.**
- **ALWAYS report family-level averages** (per TRADING_RULES.md reporting rules).
- **Walk-forward validation is mandatory** before any finding moves to "confirmed."
- If in-sample Sharpe drops by more than 50% out-of-sample, the finding is likely overfit.

### In-Sample vs Out-of-Sample
- Same data used for discovery AND testing = ONE test, not two.
- Expected Sharpe decay from in-sample to out-of-sample: ~50-63% (industry benchmark).
- Walk-forward efficiency (WFE) ≥ 0.50 = strategy meets the locked binding threshold (see `docs/institutional/pre_registered_criteria.md` § *Criterion 6 — Walk-forward efficiency*). < 0.50 = likely overfit.
- **Current limitation:** Most cross-instrument research uses only 579 overlapping days (Feb 2024 - Feb 2026). This is a single in-sample window. No findings from this window are "validated" until tested on the 10-year rebuild.

### Multi-RR Testing (Mandatory for Filter Discovery)
All filter discovery and NO-GO tests MUST test across the full RR grid {1.0, 1.5, 2.0} and both apertures {O5, O15}. Testing at a single RR target gives false reads because RR changes what the test measures: RR1.0 is direction-dominant, RR2.0 is trend-persistence-dominant. A direction signal (e.g., IBS mean-reversion) will appear at RR1.0 and vanish at RR2.0. A momentum signal (e.g., ATR_VEL) will appear at RR2.0 and be weaker at RR1.0. The IBS NO-GO (Apr 2026) was initially declared at RR2.0 only, missing a t=3.89 (N=74) in-sample signal at RR1.0 — corrected by `verify_external_ibs_nr7.py` (signal exists but holdout-killed).

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

**What this test DOES:** Screens out obvious artifacts — things that look like edge but are
explained by data bugs (E0 fill-at-open bias), cost model errors, or filter tautologies.
This is an artifact screen, not a causation proof.

**What this test does NOT DO:** Prove why price continues after a breakout. The mechanisms
below are structural plausibility arguments inferred from market microstructure theory.
They are not measured or verified from order flow data. Adversarial review (2026-03-18)
correctly identified that "friction eats small ORBs" describes what the G-filter does,
not why breakout continuation exists. The honest statement: we have statistical evidence
of edge after cost, plus plausible structural reasons it should persist, but no direct
measurement of the proposed mechanism.

**Plausible mechanisms (structural, likely to persist):**
- Friction eats small ORBs → size filter works (cost structure). **Primary artifact screen — confirmed across all instruments Feb 2026. NOTE: this explains why small ORBs fail, not why large ORBs succeed.**
- Double-break = chop, no directional conviction → exclude (market microstructure)
- All 3 instruments breaking same direction = macro flow day → concordance (institutional behavior)
- Large gold ORB at US_DATA_830 = active overnight session → equity vol at NYSE_OPEN (cross-market information flow)
- TOKYO_OPEN shorts fail because Asian session has upward bias from physical demand (market structure)
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
| MGC US_DATA_830 large ORB → MES NYSE_OPEN better | "Gold predicts equity direction" | "Gold volatility signals active overnight session → equity breakouts have more energy" |
| 3/3 concordance improves WR | "All instruments agree = strong signal" | "Macro flow day → all assets trending → breakout more likely to follow through" |
| RSI < 30 + long breakout works | "Buy oversold bounces" | "UNVALIDATED. RSI threshold sensitivity not tested. Guilty until proven innocent." |
| Edge strengthening 2024→2026 | "Strategy is getting better" | "Gold trending harder in this period. Regime-dependent, not structural." |
| MES/MNQ 87% concordance | "Equities confirm each other" | "They're the same asset class. This is baseline, not signal." |

### Volatility vs Direction
A common error in this project: confusing volatility signals with directional signals.
- MGC US_DATA_830 ORB **size** predicts MES NYSE_OPEN quality. **Direction** does not.
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
5. **NEVER speculate on WHY a session-specific pattern exists unless you have evidence.** Record the empirical fact. "CME_REOPEN and SINGAPORE_OPEN show concordance benefit, TOKYO_OPEN and LONDON_METALS do not" is the correct statement. "Early sessions work because X" is speculation.
6. **NEVER call 3 data points a trend.** 5 is barely suggestive. 10+ with a mechanism is a trend.
7. **NEVER assume stationarity.** If gold returns to $1800, most current G5+ filters become untradeable (< 5 qualifying days per year).
8. **NEVER simulate exit rules using post-resolution observations.** If a condition (e.g., "price returned inside ORB") can occur AFTER the trade already hit its stop or target, it is not actionable as an exit trigger. Simulations must verify the trigger fires while the trade is still open. The C8 break-even stop was overestimated 5x (Feb 2026) because the simulation counted post-exit price returns as if they were live-trade events. Always use `_while_open` or equivalent temporal guards.
9. **NEVER report blended results without checking DST regime.** All sessions are now event-based (DST-clean), but historical data from before the migration may still contain blended winter+summer averages. When analyzing historical data, verify whether the session was dynamic at the time. Blended averages across DST halves can mask regime-specific failures or inflate effects that only exist in one half.
10. **NEVER use R-multiples to compare time slots or session candidates.** R normalizes away ORB width, which is the mechanism being tested. When asking "which time is better?" use **dollar P&L** (`pnl_r × orb_width × point_value`). The DST wrong-time audit (Mar 2026) found 33% of families flip direction between R and $ — CME_PRECLOSE wrong time wins in R but right time wins in dollars because event-time ORBs are 19% wider. ExpR remains correct for strategy-vs-strategy evaluation within a fixed session. @research-source `scripts/tmp_dst_wrong_time_audit.py` @revalidated-for E1/E2 event-based (Mar 2026).

### Indicators That Are Guilty Until Proven Innocent
RSI, MACD, Bollinger Bands, moving average crossovers, Stochastics. These are the most over-fitted indicators in existence. Any finding based on these requires:
- Sensitivity analysis (±2 on period/threshold)
- Out-of-sample validation on unseen data
- A mechanism beyond "indicator says overbought/oversold"
- Survival through walk-forward, not just full-period backtest

---

## Market Structure Knowledge

### Session Context (Event-Based — All Times Dynamic)
| Session | Event | Reference TZ | What's Happening |
|---------|-------|-------------|-----------------|
| CME_REOPEN | CME daily reopen | 5:00 PM CT | Asia/Sydney open. Gold physical demand. Thin liquidity. |
| TOKYO_OPEN | Tokyo open | 10:00 AM AEST | Tokyo. Increasing liquidity. LONG-ONLY edge. |
| SINGAPORE_OPEN | Singapore/HK open | 11:00 AM AEST | Singapore/HK. **MGC EXCLUDED** (74% double-break). MNQ ACTIVE (SGX/HKEX cross-market flow). |
| LONDON_METALS | London metals open | 8:00 AM UK | London gold market. Data releases. Highest vol. |
| US_DATA_830 | US economic data | 8:30 AM ET | US morning. Position building. |
| NYSE_OPEN | NYSE equity open | 9:30 AM ET | US equity open. Key for MES/MNQ. |
| US_DATA_1000 | US 10 AM data | 10:00 AM ET | Post-open data releases. |
| CME_PRECLOSE | CME pre-close | 2:45 PM CT | End-of-day positioning. |
| COMEX_SETTLE | COMEX gold settle | 1:30 PM ET | Gold settlement window. |
| NYSE_CLOSE | NYSE equity close | 4:00 PM ET | Equity close. |

All sessions are now event-based with per-day time resolution from `pipeline/dst.py`.

### Cross-Asset Relationships
- **Gold ↔ Equities:** Structurally uncorrelated (~0.05 correlation). This is BASELINE, not a finding.
- **Gold ↔ USD:** Inverse correlation (~-0.60 over 5 years). Driven by dollar denomination.
- **Gold ↔ Real Yields:** Inverse. Rising real yields pressure gold (opportunity cost of holding non-yielding asset).
- **MES ↔ MNQ:** ~87-89% concordant. Same asset class. Agreement is not signal.
- **MGC US_DATA_830 → MES/MNQ NYSE_OPEN:** Volatility transmission, not directional. Size matters, direction doesn't. 90-minute temporal gap makes this zero-lookahead.

### Cost Reality for Micro Futures
| Instrument | Tick Value | RT Friction | Friction as % of 5pt ORB |
|-----------|-----------|-------------|--------------------------|
| MGC | $1.00/tick | $5.74 | 11.5% |
| MES | $1.25/tick | $3.74 | 14.96% |
| MNQ | $0.50/tick | $2.74 | 5.5% |
| M2K | $0.50/tick | $3.24 | 12.96% |
| MCL | $1.00/tick | $5.24 | varies |

**Friction is why small ORBs lose.** A 2pt MGC ORB has 42% of its risk eaten by friction before the trade starts. This is the STRUCTURAL reason ORB size filtering works — it's not a statistical artifact, it's arithmetic.

### Gold-Specific Knowledge
- Gold trends intraday more than it mean-reverts. The ONLY confirmed edges in this project are momentum breakouts with size filters.
- Physical demand (central banks, jewelry) creates baseline buying pressure, especially in Asian sessions.
- Gold had an exceptional run in 2024-2025 (central bank buying, geopolitical hedging). G5+ day frequency exploded from ~3% to 27-88%. This is regime, not permanent.
- If gold returns to $1800 levels, the entire strategy set becomes low-frequency (5 qualifying days/year at CME_REOPEN G5+).

---

## Current Research Status

### Validated and Deployed
See `TRADING_RULES.md` → Confirmed Edges table.

### Cross-Instrument Stress Test Finding (Feb 2026)
**ORB size is the primary edge across ALL instruments.** Not sessions, not parameters, not entry models. Stress test (`scripts/tools/stress_test.py`) on 11 top edges:
- MES/MGC: ALL edges die without size filter. Regime-dependent, parameter-fragile.
- MNQ CME_REOPEN: SURVIVED. Positive all 3 years, bidirectional, all real neighbors profitable.
- Mechanism: friction as % of risk (arithmetic, not statistical).
- **Research priority:** Map optimal ORB size threshold per session per instrument. The threshold is NOT universal — MNQ (low friction) may profit from smaller ORBs than MGC (high friction).
- Tool: `scripts/tools/orb_size_deep_dive.py`

### Awaiting 10-Year Outcome Rebuild
These findings are in-sample only (579 days, Feb 2024 - Feb 2026). Do NOT act on them until validated on full dataset:
1. **Concordance filter (CME_REOPEN, SINGAPORE_OPEN)** — mechanism strong, awaiting OOS
2. **MGC US_DATA_830 size gate for MES/MNQ NYSE_OPEN** — mechanism plausible, awaiting OOS
3. **Asia session high filter** — needs investigation after rebuild
4. **RSI oversold for long ORB** — guilty until sensitivity-tested

### Confirmed NO-GOs (Do Not Revisit)
See `TRADING_RULES.md` → What Doesn't Work table.
Also: `docs/RESEARCH_ARCHIVE.md` for full data.

### Re-Validation Trigger
Cross-instrument findings (concordance, MGC US_DATA_830 gate) should be re-validated when overlapping day count reaches **800 days**. Query `gold.db` for current count — do not cite a number from memory. This is a data trigger, not a time trigger. Do not re-run at arbitrary dates.

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
# Exit rule simulations: trigger must fire WHILE TRADE IS OPEN
# (not post-resolution — see NO-GO rule #8)

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
