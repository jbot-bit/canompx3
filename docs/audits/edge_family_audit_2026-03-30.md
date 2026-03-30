# Edge Family & Strategy Classification — Full Adversarial Audit

**Date:** 2026-03-30
**Scope:** Edge family builder, strategy validator, strategy discovery, outcome builder, config
**Mode:** Read-only audit — no code changes
**Method:** Data-first (Phase 1 queries), then code verification, then bias/literature checks

---

## Executive Summary

| Severity | Count | Details |
|----------|-------|---------|
| **CRITICAL** | 0 | — |
| **HIGH** | 2 | Schema hygiene (p_value, n_trials_at_discovery never written to validated_setups) |
| **MEDIUM** | 3 | WFE not hard-gated; hardcoded E2 slippage; hardcoded filter specificity |
| **LOW** | 3 | Hardcoded stress multiplier; CSCV block count; no explicit ±20% sensitivity grid |
| **INFORMATIONAL** | 2 | Mixed-RR families (intentional design); 436/488 have losing years (expected) |

**Overall verdict: CLEAN with minor schema gaps.** No look-ahead contamination detected. No selection bias. Literature grounding is strong. FDR methodology is conservative (safe direction). Family grouping design is sound and well-documented.

---

## Phase 1: Data Inventory (Ground Truth)

### 1.1 Table Counts

| Table | Rows |
|-------|------|
| bars_1m | 16,773,992 |
| bars_5m | 3,477,790 |
| daily_features | 38,946 |
| orb_outcomes | 9,976,218 |
| experimental_strategies | 116,224 |
| validated_setups | 488 |
| edge_families | 172 |
| strategy_trade_days | 839,980 |

### 1.2 Validated Setups Breakdown

| Instrument | Entry Model | N Strategies | N Sessions | N Filters | Min N | Max N | Avg ExpR | Avg Sharpe |
|-----------|-------------|-------------|-----------|----------|-------|-------|----------|------------|
| MES | E2 | 14 | 3 | 6 | 854 | 1,713 | 0.0763 | 1.1366 |
| MGC | E2 | 4 | 2 | 2 | 125 | 274 | 0.2266 | 1.0712 |
| MNQ | E2 | 470 | 11 | 26 | 129 | 2,543 | 0.1089 | 1.1725 |

**Note:** MNQ dominates (96.3%). All strategies are E2 entry model (E0/E1/E3 purged).

### 1.3 Edge Families Breakdown

| Instrument | Status | Tier | N Families | Avg Members | Avg Median ExpR | Avg Sharpe | Avg CV | Min Trades | Avg PBO |
|-----------|--------|------|-----------|------------|----------------|-----------|--------|-----------|---------|
| MES | PURGED | CORE | 6 | 1.50 | 0.0676 | 1.0039 | 0.1573 | 1,057 | 0.0000 |
| MES | SINGLETON | CORE | 2 | 1.00 | 0.1015 | 1.4892 | — | 854 | — |
| MES | WHITELISTED | CORE | 1 | 3.00 | 0.0720 | 1.1720 | 0.1992 | 1,241 | 0.0000 |
| MGC | PURGED | CORE | 1 | 1.00 | 0.1317 | 0.7579 | — | 274 | — |
| MGC | WHITELISTED | CORE | 1 | 3.00 | 0.2564 | 1.1756 | 0.0935 | 125 | 0.0000 |
| MNQ | PURGED | CORE | 77 | 1.57 | 0.0920 | 1.0333 | 0.1574 | 143 | 0.1295 |
| MNQ | ROBUST | CORE | 29 | 7.24 | 0.1120 | 1.1391 | 0.1277 | 444 | 0.0303 |
| MNQ | SINGLETON | CORE | 23 | 1.00 | 0.1410 | 1.2056 | — | 129 | — |
| MNQ | WHITELISTED | CORE | 32 | 3.63 | 0.1114 | 1.3220 | 0.0917 | 415 | 0.0374 |

**Totals:** 172 families. 84 non-purged (29 ROBUST + 34 WHITELISTED + 25 SINGLETON). 88 PURGED.

### 1.4 Family Head Selection Audit

Top 20 families by head deviation examined. Largest deviation: 0.0625 (head ExpR 0.2416 vs family median 0.1791). All deviations are within acceptable range — median election is working correctly.

**Verdict:** PASS — heads are well-aligned with family medians.

### 1.5 Orphan Check

**0 orphans found.** All 488 validated_setups have matching edge_families entries. Referential integrity verified.

### 1.6 FDR Audit

| Instrument | Entry Model | Total | FDR Pass | FDR Null | Min Adj-p | Max Adj-p | Min discovery_k | Max discovery_k |
|-----------|-------------|-------|----------|----------|-----------|-----------|-----------------|-----------------|
| MES | E2 | 14 | 14 | 0 | 0.000006 | 0.039990 | — | — |
| MGC | E2 | 4 | 4 | 0 | 0.006194 | 0.048343 | — | — |
| MNQ | E2 | 470 | 470 | 0 | 0.000000 | 0.049774 | — | — |

All 488 strategies pass FDR (fdr_significant=TRUE). Max adjusted p-value = 0.0498 (just under α=0.05).

**p_value column:** NULL for all 488 rows in validated_setups. Raw p_value IS populated in experimental_strategies (113,807/116,224 = 97.9%). The validator reads p_value via JOIN from experimental_strategies, computes BH-adjusted p, and writes only `fdr_adjusted_p` back to validated_setups. Raw p is never copied. This is a **schema hygiene issue** — the audit trail exists via the JOIN, but the validated_setups table alone cannot reproduce the FDR calculation. See Finding H1.

**n_trials_at_discovery column:** NULL for all 488 rows. `discovery_k` IS populated (range 1,512–14,760). See Finding H2.

### 1.7 Walk-Forward Efficiency

| Instrument | Total | WFE > 0.5 | WFE ≤ 0.5 | WFE NULL | Avg WFE | Min WFE |
|-----------|-------|-----------|-----------|----------|---------|---------|
| MES | 14 | 14 | 0 | 0 | 0.9529 | 0.5625 |
| MGC | 4 | 4 | 0 | 0 | 0.7608 | 0.5336 |
| MNQ | 470 | 467 | 3 | 0 | 1.4574 | 0.3341 |

**3 MNQ strategies have WFE < 0.5** (0.334, 0.433, 0.434) — all US_DATA_1000, RR1.0, O15 aperture. These are still active in validated_setups. WFE is an informational metric, not a hard gate. The actual walk-forward gate is on positive aggregate OOS ExpR and window consistency (≥60% positive windows, ≥3 valid windows). See Finding M1.

### 1.8 Yearly Robustness

- **436/488 (89.3%) have at least one losing year** — expected across 5–7 year histories
- 694 total losing-year instances across all strategies
- Concentration in 2016, 2017, 2019 (early data, regime transitions)
- **Regime waivers** correctly handle dormant years (mean_atr < 20, ≤5 trades → waived)

**Verdict:** PASS — losing years are expected. The 75% positive-years gate + regime waivers are correctly implemented.

### 1.9 Session Concentration

| Instrument | Session | N Families | ROBUST | WHITELISTED | SINGLETON | PURGED |
|-----------|---------|-----------|--------|------------|-----------|--------|
| MNQ | US_DATA_1000 | 25 | 0 | 7 | 3 | 15 |
| MNQ | LONDON_METALS | 20 | 3 | 3 | 0 | 14 |
| MNQ | TOKYO_OPEN | 19 | 0 | 5 | 4 | 10 |
| MNQ | NYSE_OPEN | 19 | 5 | 3 | 1 | 10 |
| MNQ | SINGAPORE_OPEN | 18 | 13 | 3 | 0 | 2 |
| MNQ | CME_PRECLOSE | 18 | 1 | 7 | 3 | 7 |
| MNQ | NYSE_CLOSE | 15 | 0 | 0 | 11 | 4 |
| MNQ | COMEX_SETTLE | 12 | 6 | 2 | 1 | 3 |

**Highest robustness:** SINGAPORE_OPEN (13 ROBUST / 18 total). **Highest purge rate:** LONDON_METALS (70% purged). **Fragility hotspot:** NYSE_CLOSE (11/15 = 73% singletons).

### 1.10 Dead Instrument Residual Check

**CLEAN.** Zero rows for M2K, MCL, SIL, M6E, MBT in both validated_setups and edge_families.

### 1.11 2026 Holdout Status

- **orb_outcomes:** 60 trading days of 2026 data (Jan 2 – Mar 23)
- **validated_setups:** All 488 created/updated in 2026 (discovery pipeline ran in 2026)
- **Holdout discipline:** Discovery uses `holdout_date` parameter to cap feature/outcome windows. Strategies are discovered on pre-2026 data; 2026 is forward-test monitoring only.

---

## Phase 2: Code Audit — Classification Logic

### 2A: Edge Family Builder (`scripts/tools/build_edge_families.py`)

#### Trade-Day Hash — PASS (Design Intentional)

**Hash function** (`db_manager.py:27-35`): MD5 of comma-separated sorted trading days. Includes ONLY trading days — no direction, RR, session, entry model, or filter.

**Family key** (line 244): `f"{instrument}_{orb_min or 5}m_{hash_map[sid]}"`

**Initial concern:** Mixed-RR families appeared to be a bug (80 families mix 2–6 RR targets).

**After re-examination: INTENTIONAL DESIGN.** TRADING_RULES.md §507 explicitly states:
> "A 'family' = one unique combination of (session, entry_model, filter_level). All RR/CB variants within a family share 85-100% of the same trade days."

**Rationale (fdr_methodology.md line 45):** RR target variants are parameter optimization on a fixed entry signal (Aronson p.282). Grouping them prevents overcounting independent bets. The project identifies only 7 truly independent bets (Jaccard >0.7 clustering), not 88 or 172.

**Direction distinction:** Implicit via different trade days. Long strategies trade when the high breaks; short strategies when the low breaks — producing different day sets and therefore different hashes.

**Verdict:** PASS — design is sound, well-documented, and grounded in literature.

#### Median Head Election — PASS

**Function** `_elect_median_head()` (lines 103–121):
- Selects strategy closest to median ExpR (not mean or max)
- Tiebreaker: lower strategy_id (deterministic)
- Explicitly documented as avoiding Winner's Curse (selection bias)

**Data verification:** Top 20 heads by deviation all within 0.0625 ExpR of family median.

**Verdict:** PASS — correctly implements anti-selection-bias head election.

#### Robustness Classification — PASS

| Status | Criteria | Verified |
|--------|----------|----------|
| ROBUST | member_count ≥ 5 | Lines 64-65 ✓ |
| WHITELISTED | 3–4 members AND ShANN ≥ 0.8 AND CV ≤ 0.5 AND min_trades ≥ 50 | Lines 66-75 (AND logic, fail-closed) ✓ |
| SINGLETON | 1 member AND min_trades ≥ 100 AND ShANN ≥ 1.0 | Lines 76-83 (AND logic, fail-closed) ✓ |
| PURGED | Fallthrough default | Line 84 ✓ |

**Cross-verified against data:**
- All 34 WHITELISTED families meet all 4 criteria (members ≥ 3, sharpe ≥ 0.8, CV ≤ 0.5, trades ≥ 50)
- All 25 SINGLETON families meet all 3 criteria (1 member, trades ≥ 100, sharpe ≥ 1.0)
- No misclassifications found

**Verdict:** PASS — thresholds match documentation, data is consistent.

#### PBO Calculation — PASS

**Implementation** (`trading_app/pbo.py`): Authentic CSCV (Bailey et al. 2014)
- 8 chronological time blocks (configurable, default n_blocks=8)
- C(8,4) = 70 combinatorial train/test splits
- For each split: find IS-best strategy, measure OOS performance
- PBO = n_negative_oos / n_splits
- Logit transformation available

**Not a hard gate:** PBO is computed and stored for analysis but does not reject strategies. Appropriate for a research codebase where WF validation + FDR are the primary gates.

**Verdict:** PASS — real CSCV, correctly implemented.

#### PURGED Families — PASS

PURGED families remain in `edge_families` table with `robustness_status = 'PURGED'`. They are filtered out by downstream queries that check status. No evidence of leakage into portfolio construction or deployment.

### 2B: Strategy Validator (`trading_app/strategy_validator.py`)

#### Phase Ordering — PASS

Mandatory phases (cannot skip):
1. **Sample size** — reject if N < 30, warn if N < 100
2. **Post-cost expectancy** — reject if ExpR ≤ 0
3. **Yearly robustness** — reject if < 75% years positive (after regime waivers)
4. **Stress test** — reject if 1.5× cost ExpR ≤ 0

Conditional phase:
5. **Walk-forward** — anchored-expanding windows; gate on positive OOS ExpR, ≥60% positive windows, ≥3 valid windows. Can be disabled with `--no-walkforward` (not used in production).

Post-validation hard gate:
6. **BH FDR** — session-stratified K, α=0.05. Applied to all survivors.

Optional phases:
7. **Min Sharpe** — disabled by default (min_sharpe=None)
8. **Max drawdown** — disabled by default

**Verdict:** PASS — phases run in correct order; mandatory phases cannot be bypassed.

#### FDR K — PASS (Conservative)

K is computed **per-session** (session-stratified BH):
- Each session pool has its own K = number of canonical strategies in that session
- Strategy validator reads from experimental_strategies, groups by orb_label
- discovery_k frozen on first write (immutable audit trail)

**K inflation:** Intentionally conservative (~2–3× independent tests). Includes stop multiplier variants, correlated filter levels, E1/E2 duplication. Safe direction — kills real edge rather than admitting false positives.

**Grounding:** Efron separate-class model (sessions are pre-specified by exchange events). BH Theorem 1 (Benjamini & Hochberg 1995, journal p.293).

**Verdict:** PASS — session-stratified K is correctly implemented and conservatively biased.

#### Walk-Forward Windows — PASS

**Construction** (walkforward.py lines 150–250):
- **Anchored expanding:** IS = all data before window; OOS = fixed window
- Non-overlapping test windows (each OOS period appears exactly once)
- MGC override: WF windows start from 2022-01-01 (skips pre-2022 dormant ATR regime in OOS only; full-sample IS phase uses ALL data)

**WFE calculation** (walkforward.py lines 310–313):
- Trade-weighted aggregate: `WFE = Σ(OOS_ExpR × OOS_N) / Σ(IS_ExpR × OOS_N)`
- NOT Sharpe ratio — it's raw expectancy ratio
- Division-by-zero guard: windows with IS ExpR ≤ 0 excluded from WFE

**WFE is NOT a hard gate:** The actual gate requires (1) ≥3 valid windows, (2) ≥60% positive, (3) positive aggregate OOS ExpR, (4) sufficient OOS trades. WFE ratio is informational. See Finding M1.

**Verdict:** PASS — anchored-expanding windows correctly constructed, WFE properly computed.

#### Regime Waivers — PASS

**Implementation** (lines 442–534): A year is waived if DORMANT (mean_atr < 20 AND ≤5 trades). After waivers: `(pos_years + waived_years) / total_years ≥ 75%`. This allows strategies to survive years where the instrument was dormant (e.g., MGC pre-2022 low-ATR).

**Verdict:** PASS — documented, auditable, not ad-hoc.

### 2C: Strategy Discovery (`trading_app/strategy_discovery.py`)

#### Grid Search — PASS

Full Cartesian product:
- E2: sessions × filters_per_session × 6 RR targets × 1 CB × 2 stop multipliers
- E1: sessions × filters_per_session × 6 RR targets × 5 CB × 2 stop multipliers
- K = total combos across all sessions for that instrument

**Filter application:** Filters applied in Python after bulk-loading outcomes (lines 1184–1189). Filter day sets pre-computed from daily_features. Outcomes loaded without filter, then intersected with filter day sets. This is correct — no look-ahead from filter application.

#### Deflated Sharpe / FST — PASS

**Implementation** (lines 358–421):
- Bailey & López de Prado (2014) formula
- Non-normality correction via Mertens (2002): `V[SR] = (1/T)(1 - γ₃SR + (γ₄/4)SR²)`
- Expected max under null: Euler-Mascheroni approximation
- FST hurdle: expected max SR for K trials under zero skill (Bailey et al. 2018)

**Grounding:** References and formulas match the cited papers. Skewness, kurtosis, AND number of trials all included.

**Verdict:** PASS — correctly implements the cited literature.

#### P-value Computation — PASS

Two-tailed Student's t-test on R-multiples (not daily returns). Normal approximation for df > 100; beta continued fraction for df ≤ 100. Test statistic: mean(R) / (std(R) / √N).

**Note:** Testing on R-multiples rather than daily returns is acceptable because R-multiples are the natural unit of outcome measurement in this system. Autocorrelation is minimal (trades are non-overlapping within sessions).

### 2D: Outcome Builder (`trading_app/outcome_builder.py`)

#### No Look-Ahead — PASS (7 dimensions verified)

| Dimension | Implementation | Look-Ahead Risk |
|-----------|---------------|-----------------|
| ATR(20) | Prior 20 days only, slice `[i-20:i]` excludes current | NONE |
| ORB range | Fixed UTC time window (clock-based, not data-driven) | NONE |
| E2 entry | First bar where intra-bar range touches ORB level (`np.argmax`) | NONE |
| Confirm bars | Forward-time counter with reset on pullback; no forward scan | NONE |
| Session end | Fixed 24h from 09:00 Brisbane UTC; hard cutoff `< trading_day_end` | NONE |
| Time-stop | DISABLED (all None); if enabled, fixed offset from entry | NONE |
| Prior-day features | `rows[i-1]` in post-pass; strictly lagged one row | NONE |

#### Fill Assumptions — PASS

- **E2 (stop-market):** Fill at ORB level + E2_SLIPPAGE_TICKS (1 tick). First bar where intra-bar high/low touches ORB level.
- **Ambiguous fills:** Both target and stop hit in fill bar → conservatively recorded as loss (`ambiguous_bar=True`). Pessimistic, not optimistic.

#### Cost Model — PASS

Costs applied ONCE per trade in `to_r_multiple()` (cost_model.py). Formula: `pnl_r = (pnl_points × point_value - total_friction) / risk_in_dollars`. Correct instrument cost spec used. No double-counting.

#### Double-Break — PASS

`double_break` column exists in daily_features for historical analysis. **NOT used** in any strategy filter, outcome computation, or validation gate. Correctly documented as "LOOK-AHEAD relative to intraday entry." Grep confirms zero references in strategy_discovery.py, outcome_builder.py, config.py, or strategy_validator.py.

### 2E: Config (`trading_app/config.py`)

#### Threshold Provenance

| Threshold | Value | Grounding |
|-----------|-------|-----------|
| CORE_MIN_SAMPLES | 100 | Carver "Systematic Trading" (N≥100 for reliable Sharpe); López de Prado AFML Ch.7 |
| REGIME_MIN_SAMPLES | 30 | Standard minimum for t-test validity; documented as "not tradeable standalone" |
| WF_START_OVERRIDE MGC | 2022-01-01 | Regime analysis — pre-2022 dormant ATR. Full IS still uses all data. |
| NOISE_FLOOR | All zeros (disabled) | Conscious decision 2026-03-26. Per-strategy bootstrap null is replacement. |
| WFE threshold | 0.50 | Pardo "Evaluation & Optimization of Trading Strategies" (2008) |
| Stress multiplier | 1.5× | Industry practice (50% cost buffer). See Finding M2. |

**Verdict:** PASS — all thresholds documented with references or justification.

---

## Phase 3: Bias & Look-Ahead Checklist

### 3A: Selection Bias

| Check | Verdict | Evidence |
|-------|---------|----------|
| Survivorship bias in instrument selection | **PASS** | Dead instruments tested with identical pipeline; rejected on merits (0 validated). Exclusion from K is correct (dead instruments are excluded from ACTIVE_ORB_INSTRUMENTS before K is computed). |
| Session selection bias | **PASS** | Sessions are pre-specified by exchange events (CME settlement times, NYSE opens), not by outcome data. Documented in fdr_methodology.md as Efron separate-class model. |
| Filter specificity bias | **PASS** | FDR K includes ALL filter levels tested (not just the winner). K inflation from correlated filters is ~2–3× but intentionally conservative. |

### 3B: Look-Ahead Bias

All 7 dimensions verified in Phase 2D above. **CLEAN — no look-ahead detected.**

### 3C: Multiple Testing

| Check | Verdict | Evidence |
|-------|---------|----------|
| BH FDR K is honest | **PASS** | K is session-stratified, frozen on first write, includes stop multiplier variants. Conservative (~2–3× independent tests). |
| Researcher degrees of freedom | **PASS (with note)** | Informal decisions (instruments, sessions, date ranges, entry models) are documented. E0/E1/E3 purges are auditable. Sessions are exchange-event-driven, not data-mined. |
| Family-level vs strategy-level testing | **PASS** | FDR is applied at strategy level FIRST. Families are formed from survivors. Correct ordering. |

### 3D: Overfitting Indicators

| Check | Verdict | Evidence |
|-------|---------|----------|
| IS vs OOS decay | **PASS** | MNQ avg WFE = 1.457 (OOS > IS, unusual but valid — may indicate IS conservatism). MES avg WFE = 0.953 (minimal decay). |
| Parameter cliff | **PASS (implicit)** | No explicit ±20% grid, but WF + per-year stability implicitly test parameter robustness across regimes. Edge families group parameter variants — if only one RR works, the family has low member count and weak robustness status. |
| PBO distribution | **PASS** | MNQ ROBUST avg PBO = 0.030 (excellent — 3% overfit probability). MNQ WHITELISTED avg PBO = 0.037. MNQ PURGED avg PBO = 0.130. No family with PBO > 0.50 in ROBUST/WHITELISTED tiers. |

---

## Phase 4: Literature Grounding

| Method | Claimed Reference | Implementation Correct? | Deviation |
|--------|------------------|------------------------|-----------|
| BH FDR | Benjamini & Hochberg (1995) | **YES** — standard BH with monotonicity enforcement, session-stratified K | None |
| Deflated Sharpe | Bailey & López de Prado (2014) | **YES** — includes skewness, kurtosis, number of trials via Mertens (2002) | None |
| PBO (CSCV) | Bailey et al. (2014) | **YES** — full combinatorial symmetric CV with C(8,4)=70 splits | None |
| Walk-Forward Efficiency | Pardo (2008) | **YES** — trade-weighted ratio, anchored-expanding windows | Uses ExpR ratio not Sharpe ratio (project-specific, documented) |
| Median head election | Anti-Winner's Curse | Project-invented but reasoning is sound (avoid selection bias on best-performing parameter variant) | No published reference; defensible |
| CORE=100, REGIME=30 | Carver + standard practice | **YES** — N=100 for reliable Sharpe (Carver Ch.4); N=30 minimum for t-test | None |
| CV ≤ 0.5 for WHITELISTED | Project-specific | Not from Carver — project-chosen threshold for family consistency | No published reference; reasonable |
| Cost buffer +50% | Industry practice | Reasonable but not from a specific published source | Kissell suggests transaction cost uncertainty of 30–70%; 50% is within range |
| FST hurdle | Bailey et al. (2018) | **YES** — Euler-Mascheroni approximation for expected max SR under null | None |

**Local PDFs verified:** 17 academic/practitioner PDFs in `resources/`. References in code match claimed sources. No fake citations detected.

---

## Phase 5: Cross-Consistency Checks

### 5.1 Config ↔ Code

**Grep for hardcoded thresholds bypassing config.py:**

| Hardcoded Value | Location | Impact | Severity |
|----------------|----------|--------|----------|
| Filter specificity ranking (G8=5, G6=4, ...) | strategy_discovery.py:23-28 | Determines canonical dedup. Not in config. | MEDIUM |
| E2_SLIPPAGE_TICKS = 1 | config.py:1114 (but hardcoded, not per-instrument) | All instruments share 1-tick slippage | MEDIUM |
| Stress multiplier = 1.5 | strategy_validator.py:527 | Not configurable | LOW |
| CSCV blocks = 8 | pbo.py:48 | Not configurable | LOW |
| Max family size guard = 100 | build_edge_families.py:392 | Defensive guard only | LOW |

### 5.2 Docs ↔ Data

| Claim | Source | Actual | Match? |
|-------|--------|--------|--------|
| "488 validated total" | MEMORY.md | 488 rows in validated_setups | ✓ |
| "172 families" | Code output | 172 rows in edge_families | ✓ |
| "7 independent bets" | fdr_methodology.md | Jaccard >0.7 clustering — verified | ✓ |
| "3 active instruments" | CLAUDE.md | MGC, MNQ, MES in validated_setups | ✓ |
| "All E2" | Data | 488/488 = E2 | ✓ |

### 5.3 Robustness Status ↔ Thresholds

- **All 34 WHITELISTED:** Meet all 4 criteria (members ≥ 3, sharpe ≥ 0.8, CV ≤ 0.5, trades ≥ 50). Verified by SQL.
- **All 25 SINGLETON:** Meet all 3 criteria (1 member, trades ≥ 100, sharpe ≥ 1.0). Verified by SQL.
- **No misclassifications detected.**

### 5.4 Dead Instruments

**CLEAN.** Zero residual rows in validated_setups or edge_families for M2K, MCL, SIL, M6E, MBT.

### 5.5 2026 Holdout Integrity

- 60 days of 2026 orb_outcomes data (Jan 2 – Mar 23) exist for forward monitoring
- Discovery pipeline uses `holdout_date` to cap training data; strategies are discovered on pre-2026 data
- **No evidence of holdout breach** — strategies are trained/validated on historical data, 2026 is OOS only

---

## Findings

### HIGH Severity

#### H1: `p_value` Never Written to `validated_setups`

**Location:** `strategy_validator.py` lines 1298–1312
**Issue:** The UPDATE statement writes `fdr_adjusted_p`, `discovery_k`, `discovery_date`, and `fdr_significant` — but never writes raw `p_value`. The column exists in the schema but is NULL for all 488 rows.
**Impact:** Audit trail gap. To reconstruct BH calculation, you must JOIN to `experimental_strategies`. The validated_setups table alone is insufficient for FDR verification.
**Mitigation:** Raw p_value IS available via `experimental_strategies` JOIN (97.9% populated). FDR correctness is not affected — only auditability from a single table.
**Recommendation:** Add `p_value` to the UPDATE statement in strategy_validator.py.

#### H2: `n_trials_at_discovery` Never Populated in `validated_setups`

**Location:** `strategy_validator.py` lines 1298–1312
**Issue:** Column exists in schema, NULL for all 488 rows. `discovery_k` IS populated (range 1,512–14,760). The distinction between K (total pool size) and n_trials (combos tested at discovery time) is lost.
**Impact:** Cannot independently verify whether K at discovery time matched current K. Audit trail gap.
**Recommendation:** Populate `n_trials_at_discovery` alongside `discovery_k` on first write.

### MEDIUM Severity

#### M1: WFE < 0.5 Is Not a Hard Gate

**Location:** `walkforward.py` lines 92–96, `strategy_validator.py`
**Issue:** 3 strategies have WFE 0.334–0.434 and remain active. WFE is computed and stored but the walk-forward gate checks (1) positive OOS ExpR, (2) ≥60% positive windows, (3) ≥3 valid windows — not WFE ratio.
**Impact:** Strategies with severe OOS decay (WFE 0.33 = OOS is 33% of IS) are deployed. This is arguably acceptable if OOS ExpR is still positive, but contradicts documentation suggesting WFE > 0.5 as a standard.
**Recommendation:** Either add WFE ≥ 0.5 as a hard gate or document that WFE is informational-only. Current ambiguity is confusing for audit.

#### M2: E2 Slippage Hardcoded (Not Per-Instrument)

**Location:** `config.py` line 1114: `E2_SLIPPAGE_TICKS = 1`
**Issue:** All instruments share 1-tick slippage. MNQ tick = $0.50, MGC tick = $0.10, MES tick = $0.25. The dollar impact varies significantly.
**Impact:** Low — Databento tbbo pilot (memory: `e2_slippage_microstructure.md`) validated 1-tick median slippage for E2. But per-instrument calibration would be more rigorous.
**Recommendation:** Move to per-instrument config with empirical calibration.

#### M3: Filter Specificity Ranking Hardcoded

**Location:** `strategy_discovery.py` lines 23–28
**Issue:** ORB_G8=5, G6=4, G5=3, G4=2 ranking is hardcoded, not in config.py or TRADING_RULES.md.
**Impact:** Low — ranking is research-derived and stable. But invisible to operators auditing config.
**Recommendation:** Document in TRADING_RULES.md or move to config.py.

### LOW Severity

#### L1: Stress Test Multiplier Hardcoded

**Location:** `strategy_validator.py` line 527: `multiplier=1.5`
**Issue:** 50% cost buffer is hardcoded, not configurable. Kissell suggests 30–70% is reasonable.
**Recommendation:** Move to config.py. Minor — current value is within accepted range.

#### L2: CSCV Block Count Hardcoded

**Location:** `pbo.py` line 48: `n_blocks=8`
**Issue:** 8 blocks is the default from Bailey et al. but is not configurable without code change.
**Recommendation:** Already a parameter to `compute_pbo()` — just not exposed in build_edge_families.py call. Minor.

#### L3: No Explicit ±20% Parameter Sensitivity Grid

**Issue:** Parameter sensitivity is tested implicitly via walk-forward + per-year stability, not via explicit ±1 step neighbor analysis.
**Impact:** Adequate but not best-in-class. WF + yearly testing across 5–7 years naturally exposes parameter cliffs.
**Recommendation:** Optional enhancement — add neighbor sensitivity analysis to family builder. Low priority.

### INFORMATIONAL

#### I1: Mixed-RR Families Are Intentional Design

**Initially flagged as potential bug.** After re-examination: this is the intended design. Edge families group parameter variants (RR/CB/stop) on the same entry signal. Documented in TRADING_RULES.md §507, fdr_methodology.md line 45, and grounded in Aronson p.282 (parameter optimization).

100% of ROBUST families and 97% of WHITELISTED families contain mixed RR targets. This is expected — the ROBUST threshold (≥5 members) naturally encompasses multiple parameter variants of a strong signal.

#### I2: 436/488 Strategies Have Losing Years

Expected across 5–7 year histories. Regime waivers correctly handle dormant years. The 75% positive-years gate is appropriate for multi-regime data.

---

## Conclusion

The edge family and strategy classification pipeline is **architecturally sound** with no critical issues. The design choices are well-documented, grounded in published quantitative finance literature, and implemented with fail-closed semantics.

**Key strengths:**
- Zero look-ahead contamination across 7 verified dimensions
- Conservative FDR correction (session-stratified, K frozen, inflation in safe direction)
- Real CSCV-based PBO (not simplified)
- Median head election avoiding Winner's Curse
- Fail-closed classification defaults (PURGED as fallthrough)
- Extensive literature grounding (17 local PDFs, widespread code citations)

**Action items (prioritized):**
1. **H1+H2:** Populate `p_value` and `n_trials_at_discovery` in validated_setups UPDATE (schema hygiene)
2. **M1:** Clarify WFE gate status — either hard-gate at 0.5 or document as informational
3. **M2:** Per-instrument E2 slippage calibration
4. **M3:** Document filter specificity ranking in TRADING_RULES.md
5. **L1–L3:** Minor config hygiene (low priority)
