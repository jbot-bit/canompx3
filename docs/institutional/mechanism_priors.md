# Mechanism Priors — What We Think Drives ORB Edge

**Date:** 2026-04-15
**Status:** LIVE (baked into session routing via `CLAUDE.md` + `institutional-rigor.md`)
**Purpose:** so future sessions understand WHAT we're trying to build and WHY, not just WHICH filters to test.

This is a prior-beliefs document. It is NOT a validated model. Every claim here must survive empirical testing under `docs/institutional/pre_registered_criteria.md` gates before becoming a deployed rule. But these priors are what we're TESTING against.

---

## 1. Core premise — why our strategy works at all

ORB (opening range breakout) is an intraday trend-following strategy. It works BECAUSE:

1. **Commodities and stock indices trend on intraday bars.** (Fitschen 2013 Ch 3, grounded in `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md`.) MGC/MNQ/MES all fit. Our ORB catches the first clean break of early-session range as momentum entry.

2. **Stops cluster at obvious levels.** When price breaks ORB high, stops above prior-day high get hit next, creating momentum. When ORB forms IN consolidation, the breakout has to fight level-failure friction.

3. **Liquidity pools are knowable ex-ante.** PDH, PDL, session H/L, pivot (H+L+C)/3, round numbers are visible before the trade.

The strategy premise is literature-grounded (Fitschen). The **level-based refinement** (this doc) is a prior belief being tested.

---

## 2. The mechanism prior: liquidity + displacement

Each trade is fundamentally about ONE thing: **which side is trapped when price moves through an obvious level.**

- **Stops at PDH/PDL:** retail puts stops just above yesterday's high (for shorts) or below yesterday's low (for longs). Break of PDH forces short-cover buying = momentum up.
- **Value traders defend pivot:** pivot is where institutional value/mean-reversion orders sit. LONG breakouts near pivot have to clear institutional selling.
- **Failed breakouts = trapped longs:** price breaks above PDH then fails back inside = longs trapped, shorts get a trend move down.
- **Gap direction = overnight conviction:** large gap = positioning built overnight, momentum follows.

This framework predicts which signals should matter:

| Signal | Prior sign of effect | Why |
|--------|---------------------|-----|
| ORB mid INSIDE prior-day range | NEGATIVE on breakouts | Breakout fights two opposing levels → reduced quality |
| ORB mid NEAR pivot | NEGATIVE on LONG | Institutions defend pivot, LONG faces selling |
| ORB mid BELOW PDL | POSITIVE on LONG | Oversold → LONG catches mean-reversion bounce |
| ORB mid NEAR PDH | NEGATIVE on SHORT | Shorting into resistance; stops above PDH fuel trap-and-squeeze |
| Gap up on trend-continuation day | POSITIVE on LONG | Overnight conviction confirms direction |
| Gap down into pivot (bounce setup) | POSITIVE on LONG | Flush + reversion pattern |

**These are PRIORS, not validated claims.** They become claims only after passing `pre_registered_criteria.md` gates.

---

## 3. What this framework IS and IS NOT

### IS:
- A set of testable priors about WHY price moves
- A filter/signal menu anchored in mechanism, not data-mining
- Guidance for what to test next vs what's been settled
- A shared vocabulary so sessions don't redundantly rediscover

### IS NOT:
- A validated model (priors must clear institutional gates)
- A replacement for `pre_registered_criteria.md` or statistical rigor
- An excuse to loosen t-stat bars (Chordia t≥3.79 still binds unless literature extract says otherwise)
- A justification for untested deployment

---

## 4. Signal → Role mapping (the implementation menu)

The authoritative workflow for deciding which role to test first now lives in
`docs/institutional/conditional-edge-framework.md`. This section remains the
mechanism-side menu of possible roles.

Each signal can be deployed in multiple ROLES. Binary skip is the least sophisticated. Continuous-scaling position sizing is the most sophisticated. Our roadmap is staged.

| Role | What it does | Pipeline location | Grounded by |
|------|--------------|-------------------|-------------|
| **R1 FILTER (skip/take)** | Binary decision: don't take this trade | `trading_app/config.py` `StrategyFilter` subclass, registered in `ALL_FILTERS`, routed via `get_filters_for_grid()` | Current deployed pattern (PDR, Gap, ATR70, OVNRNG) |
| **R2 DIRECTION filter** | Take only LONG or only SHORT when signal fires | `ALL_FILTERS` combined with `DIR_LONG`/`DIR_SHORT` | Current pattern |
| **R3 POSITION-SIZE modifier** | Continuous size 0.5x/1.0x/1.5x based on signal strength | `trading_app/risk_manager.py` + new `forecast_combiner.py` | **Carver 2015 Ch 9-10** (extract at `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`) |
| **R4 STOP modifier** | Tighter stop (0.75x) when near level; wider (1.0x) when clear | `trading_app/execution_engine.py` `compute_stop_price()` | Partial — needs extraction of Carver Ch 11-12 or literature on level-failure speed |
| **R5 TARGET modifier** | Target = next liquidity level, not fixed RR | `trading_app/execution_engine.py` `compute_target_price()` | NOT grounded — Dalton/Steidlmayer needed (not in `resources/`) |
| **R6 ENTRY-MODEL switch** | E2 (stop-market) when far from level; E1 (market next bar) for confirmation near level | Strategy discovery layer; schema change required | Partial — Carver framework compatible |
| **R7 CONFLUENCE score** | Combine multiple signals into continuous -20/+20 forecast | `pipeline/build_daily_features.py` + `forecast_combiner` | **Carver 2015 Ch 8** — forecast combination with diversification multiplier |
| **R8 PORTFOLIO allocator weight** | Daily lane allocation scales with signal confluence | `trading_app/lane_allocation.py` | Carver Ch 4 + Ch 11 (not yet extracted) |

**Default assumption:** all 8 roles are lookahead-safe IF the signal inputs (prev_day_*, atr_20, gap_type, orb_high/low) are pre-ORB. Entry on E2 stop-market happens AFTER ORB window closes, so ORB-close values are pre-entry.

**BANNED feature inputs** (from `.claude/rules/daily-features-joins.md` § Look-Ahead Columns):
- `double_break`, `took_pdh_before_1000`, `took_pdl_before_1000`, `overnight_range_pct`
- `break_dir`, `break_ts`, `break_delay_min`, `outcome`, `mae_r`, `mfe_r`

---

## 5. Staged deployment roadmap

### STAGE 1 — Binary filters (lowest risk, fastest deploy)
- Role R1/R2 on validated patterns
- No schema change, no sizing math
- Existing `StrategyFilter` framework
- Each pattern = new pre-reg + T0-T8 + add to `ALL_FILTERS`
- Expected uplift: 10-30% on affected lanes via losing-trade removal

### STAGE 2 — Continuous position-size scaling
- Role R3/R7 using Carver framework
- **Requires:** new `trading_app/forecast_combiner.py`, extension of `risk_manager.py`
- Half-Kelly vol target (25-37% annualized for our Sharpe range)
- Each signal becomes a -20..+20 forecast, weighted combined
- Expected uplift: 1.5x to 2x on pattern-aligned trades, 0.5x on pattern-averse trades → compound advantage

### STAGE 3 — Stop/target geometry modification
- Role R4/R5 — changes the TRADE itself, not just whether to take it
- **Requires:** `outcome_builder.py` re-simulation logic, schema change on `orb_outcomes`
- Stop: 0.75x when near level (level-failure is fast)
- Target: next level, not fixed RR
- **GATED:** needs Dalton/Steidlmayer literature for target-at-level theory (not yet extracted)

### STAGE 4 — Portfolio-level allocator integration
- Role R8 — confluence score drives daily lane weighting
- **Requires:** `lane_allocation.py` refactor, new drift check, correlation-gate recomputation
- **Requires:** Carver Ch 11 extract (not yet done)
- Furthest out; only pursue if Stages 1-3 prove level-proximity is stable edge

---

## 6. Hypothesis testing discipline (the gates)

Every prior in §2 must pass these before becoming a deployed rule:

1. **Pre-registration** at `docs/audit/hypotheses/YYYY-MM-DD-<slug>.md` BEFORE any test run
2. **MinBTL cap:** N_trials ≤ 48 (per-instrument strict Bailey at E=1.2 for 6.65 clean years) or ≤ 120 for relaxed E=1.2
3. **Protocol B by default:** Chordia t≥3.79 unless a committed `literature/` extract supports the mechanism theory → t≥3.00 allowed
4. **BH-FDR:** q<0.05 both local (study K) and global (project K ~640 as of 2026-04-15)
5. **Cross-instrument coherence:** direction match AND min(|delta|) ≥ 0.05 R for features tested across instruments
6. **5-era stability:** ExpR ≥ -0.05 in each era with N ≥ 50 (Criterion 9)
7. **Holdout direction + effect:** OOS direction matches IS AND OOS_ExpR ≥ 0.40 × IS_ExpR
8. **Controls:** destruction shuffle (must fail) + known-null RNG (must fail) + positive control (must pass)
9. **Single locked evaluation date:** no peeking

Full spec: `docs/institutional/pre_registered_criteria.md`. Enforcement: `.claude/rules/research-truth-protocol.md` + `.claude/rules/quant-audit-protocol.md` T0-T8.

---

## 7. Things we know DON'T work (don't re-discover)

From `docs/STRATEGY_BLUEPRINT.md` § 5 NO-GO and prior research:

- **ML on ORB outcomes** — DEAD (V1/V2/V3 all failed, `docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md`)
- **`prev_day_range/atr × NYSE_OPEN × MNQ`** — KILLED (OOS sign flip, `docs/specs/presession-volatility-filters.md:78`)
- **`took_pdh × US_DATA_1000`** — QUARANTINED (WFE>1.89 leakage suspect)
- **Multi-timeframe full-scope chaining** — structural MinBTL-dead (`docs/audit/hypotheses/2026-04-15-multi-timeframe-chain-full-scope-nogo.md`)
- **Gap-direction-alignment on CME micros** — DEAD, near-zero overnight gaps make signal fire rate too low
- **Crabel contraction/expansion session-level** — NO-GO, ORB-size proxy confound
- **Regime-conditional rolling window discovery** — NO-GO, friction-drag kill mechanism not regime
- **Vol-regime adaptive parameter switching** — DEAD
- **Pyramiding, breakeven trails, RR ≥ 4.0 unfiltered** — DEAD

---

## 8. Things we are actively testing (live hypotheses)

As of 2026-04-15:

| Pattern | Status | Notes |
|---------|--------|-------|
| F3 NEAR_PIVOT LONG → negative across 5+ O5 sessions | Exploratory, needs T0-T8 | Multiple sessions, cross-instrument coherent. `docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md` |
| F1 NEAR_PDH SHORT NYSE_CLOSE MES → negative | Exploratory, needs T0-T8 | Cross-RR consistent (1.0/1.5/2.0). Same results file. |
| F5 BELOW_PDL LONG US_DATA_1000 MNQ → positive | Exploratory, needs T0-T8 | HOT at RR1.0, weaker at 1.5/2.0. OOS degradation concern. |

All three are in **Stage 1 candidate** state. Next action: T0-T8 audit battery per `.claude/rules/quant-audit-protocol.md` → if survive, pre-reg Stage 1 binary filter deployment.

Do NOT propose Stage 2 (size scaling) on these patterns until Stage 1 is validated and deployed.

---

## 9. What's NOT yet grounded (explicit gaps)

### Auction theory literature

Dalton (*Mind Over Markets*, 1990s-2000s editions) and Steidlmayer (*Markets and Market Logic*, 1989) are the canonical level-theory references. **Neither is in `resources/`**, neither is available as free public PDF (both copyrighted commercial publications). CME Group publishes free "Trading with Market Profile" educational material (citation pending — webfetch timed out 2026-04-15).

**Operational consequence:** mechanism claims about "level-target theory" (Role R5) can be stated as priors (this doc) but cannot invoke Chordia t≥3.00 Protocol A pathway. Until grounded, all level-based signal tests run at Protocol B t≥3.79.

**Acquisition options:**
- User-provided PDF of Dalton or Steidlmayer (if they buy them)
- CME Group public education (fetch when available)
- Academic papers on market microstructure (SSRN/ScienceDirect search returned nothing specific 2026-04-15)

### Speed-of-trade / turnover-cost scaling

Carver Ch 12 (p 177-203) covers this. Not yet extracted. Needed if we deploy Stage 3 (stop/target modifiers) because those change turnover.

### Portfolio-level correlation-gating under signal scaling

Carver Ch 11 (p 165-175). Not yet extracted. Needed for Stage 4.

---

## 10. Vocabulary (use this, not loose terms)

| Loose term | Use this instead | Why |
|-----------|------------------|-----|
| "Signal" | Feature (if raw), Forecast (if -20..+20 scaled), Filter (if binary) | Carver's framework distinguishes these |
| "Edge" | ExpR uplift delta (quantified) or Sharpe ratio (annualized) | "Edge" is vague |
| "Sophisticated sizing" | Half-Kelly volatility-targeted with combined forecast | Points at the actual framework |
| "Market structure" | Auction process / liquidity distribution | "Structure" is ambiguous |
| "Trap" | Stop-cluster trigger / failed-breakout reversal | Mechanism-named |
| "Works on NYSE_OPEN" | PASSED T0-T8 on NYSE_OPEN at X pre-registered scope | Validation-gated |

---

## 11. Self-audit — what this doc is NOT

Per `institutional-rigor.md` rule 7 ("Ground in local resources before training memory"):

- This is a **prior-beliefs document**, not a validated model
- Mechanism claims in §2 are **testable hypotheses**, not facts
- Citations in §4 are to LITERATURE extracts that exist; anything without extract is labeled ungrounded
- All thresholds in §6 imported from `pre_registered_criteria.md`, not inlined
- Staged roadmap in §5 requires EACH stage to clear its own validation before the next

If future sessions use this doc to justify a deployment, they must still produce:
1. A pre-registered hypothesis file for the specific test
2. T0-T8 audit results
3. Reviewer pass
4. Commit trail

Priors don't validate; testing does.

---

## 12. Referenced from

- `CLAUDE.md` § Research methodology grounding — **ADD POINTER**
- `.claude/rules/institutional-rigor.md` § 7 — **ADD POINTER**
- `docs/institutional/README.md` — **ADD POINTER**
- `docs/STRATEGY_BLUEPRINT.md` — **consider adding §11 pointer**
- All pre-registration templates via `docs/institutional/hypothesis_registry_template.md` — mechanism claims must cite this doc + a literature extract

## Change log

| Date | Change | Reason |
|------|--------|--------|
| 2026-04-15 | Initial write | Session HTF: user explicit request to bake trading logic into project so it doesn't pigeonhole |
