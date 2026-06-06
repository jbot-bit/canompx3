# C11 Live-Gate Clearance Audit — topstep_50k_mnq_auto

**Date:** 2026-06-03 | **Decision class:** LIVE_CAPITAL_RISK | **Status:** READ-ONLY (no mutation; awaiting approval)
**Author worktree:** `canompx3-live-trade-diag` (isolated, off origin/main)

## Verdict

**`FIX_MODEL_DIVERGENCE` + `CLEAR_WITH_LANE_DROP`.** The C11 gate is honest and correctly grounded — it is NOT a bug and must NOT be weakened. But it is **more pessimistic than live reality**: `account_survival` does not model the per-trade `max_orb_size_pts` caps that `session_orchestrator` already enforces live. Closing that divergence is correct on its own merits (institutional-rigor §4). After closing it, the cleanly-passing live book is **COMEX_SETTLE + TOKYO_OPEN (drop US_DATA_1000)**.

## Scope

**Question:** Is the C11 account-survival gate FAIL on `topstep_50k_mnq_auto` a real risk, a calc bug, or a model-vs-live divergence — and what is the cleanly-passing live book? Read-only diagnosis of the C11/C12 live-launch blocker; no config/profile/broker mutation. Bounded to one profile's survival sim (`account_survival.py`) vs the live `session_orchestrator` cap enforcement.

## Evidence labels: MEASURED / CODE_TRACE / INFERRED / UNSUPPORTED

## Phase 1 — Source-of-truth chain (CODE_TRACE)

| Claim | Canonical source | Verified |
|---|---|---|
| PnL = pnl_r × risk_in_dollars(cost_spec) | account_survival.py:364-365 | YES |
| Costs from COST_SPECS | pipeline.cost_model.get_cost_spec | YES |
| 1 micro contract/trade | account_survival.py:352,575 | YES |
| Daily-loss limit $450 | profile.daily_loss_dollars | YES (MEASURED) |
| DD limit $2000, strict budget $1600 = ×0.80 | tier.max_dd × STRICT_DD_BUDGET_FRACTION | YES |
| Source = canonical orb_outcomes (NOT live_journal) | account_survival.py:9, _load_strategy_outcomes | YES |
| Live per-trade caps enforced | session_orchestrator.py:2442 (orb_cap pts), 2466 (max_risk $) | YES |
| **Survival sim models the caps** | — | **NO — divergence (root cause)** |

## Phase 2-3 — Reproduce + calc-bug hunt (MEASURED)

`python -m trading_app.account_survival --profile topstep_50k_mnq_auto`:
- gate=FAIL @ 70% | DD survival=73.3% (clears ≥70%) | strict deployability=FAIL
- daily_loss breach rate (MC)=26.5% | historical breach days=7 | max 90d DD=$2,788 / $1,600 budget
- **No calc bug found.** Costs applied, 1 contract, intraday lows via MAE, lanes aggregated per-day-then-combined, rolling-DD over real calendar days. Survived prior F-1 audit (ceiling-then-sum fixed). CALC IS CORRECT.

## Phase 5 — Breach-day attribution (MEASURED)

| Date | close $ | intraday min $ | lanes | worst lane | classification |
|---|---|---|---|---|---|
| 2022-05-12 | -240 | -558 | 3 | US -374 | MULTI_LANE_STACKING |
| 2025-04-04 | -129 | -453 | 3 | US -338 | MULTI_LANE_STACKING |
| **2025-04-07** | **-2,050** | **-2,321** | 3 | **US -1,438** | **OUTLIER (tariff crash, 957-pt stop)** |
| 2025-04-09 | -674 | -674 | 2 | COMEX -535 | MULTI_LANE_STACKING |
| 2025-04-23 | -354 | -527 | 3 | US -241 | MULTI_LANE_STACKING |
| 2026-02-20 | -383 | -451 | 2 | US -339 | MULTI_LANE_STACKING (intraday) |
| 2026-05-19 | -302 | -544 | 3 | US -326 | MULTI_LANE_STACKING |

**The single worst US trade = −$1,917 (957-pt stop × $2/pt). Next-worst US trade = −$500.** One outlier dominates.

## Phase 6 — Book decomposition (MEASURED)

| Lane | trades | total PnL | standalone gate | breach days | standalone 90d DD |
|---|---|---|---|---|---|
| COMEX_SETTLE OVNRNG_100 | 601 | +$6,898 | FAIL | 1 | $712 |
| US_DATA_1000 VWAP_MID O15 | 853 | +$12,552 | FAIL | 4 | **$3,679** |
| TOKYO_OPEN COST_LT08 | 504 | +$4,679 | FAIL | 1 | $669 |

US standalone DD ($3,679) > whole book ($2,788) — US is the single worst contributor, dominated by 2025-04-07.

## Phase 7 — Scenario families (MEASURED)

- **Drop-one:** drop US → COMEX+TOKYO = $927 DD, 2 breach, op 0.918 (best subset).
- **Apply EXISTING live orb_caps in sim** → 7 breach→2, DD $2,788→$2,039, op 0.733→0.910. **Still FAIL** (caps real but US cap 143pts too loose; 2026-02-20 stacking remains).
- **COMEX-only + $300-400 cap** → GATE PASS (op 1.0, 0 breach, $712 DD).
- **COMEX+TOKYO (drop US) @ 60-80pt caps** → **GATE PASS (op 1.0, 0 breach, $788-892 DD).** ✅
- Full 3-lane book: no cap config clears — 2026-02-20 is irreducible multi-lane intraday stacking (needs daily-loss-halt, not a per-trade cap).

## Phase 4 — No-lookahead (CODE_TRACE)

`max_orb_size_pts` cap and daily-loss-halt are both **known at entry time** (stop distance computed from the ORB at signal). `LOOKAHEAD_SAFE`. No banned columns. Selection among existing lanes = ALLOCATION optimization, not new alpha discovery (no MTC issue; no new thresholds fit on 2026).

## Root cause (INFERRED)

1. **C11/C12 state was `reason: missing`** — DailyRefresh rebuilt gold.db; envelope guard correctly rejected stale state. This is what blocked the live launch tonight. Fixed by regen.
2. **account_survival ignores the live per-trade caps** → gate replays trades the live engine would skip → gate fails on phantom risk. This is the institutional-rigor §4 model-divergence.

## Recommendation

1. **Close the model divergence:** teach `account_survival` to apply `max_orb_size_pts` (the live cap source) when replaying trades, so C11 == live reality. Add a drift check asserting both consume the same cap registry.
2. **Auto-regen C11/C12 after DB rebuild** so the live gate never silently blocks on missing state again.
3. **Cleanly-passing live book** post-fix: COMEX_SETTLE + TOKYO_OPEN with tightened orb caps (60-80pts), 0 breach days, $788-892 DD.

## Anti-tunnel audit (2026-06-03, operator-requested before Part 1)

**Cap is EDGE-IMPROVING, not drawdown-suppression — confirmed 3 ways:**
1. Capped-out trades (>80pt stop) are NEGATIVE-EV per lane: US cuts 9% @ −$25 avg (−$1,853 total losses removed), COMEX 1% @ −$1, TOKYO 0.4% @ −$317. Kept ExpR rises (US +$14.7→+$18.5).
2. Annual capacity goes UP: cap@80 $3,801/yr vs baseline $3,445 (+$356) while DD drops $2,788→$1,770.
3. Lost-opportunity is NEGATIVE (we remove money-losing trades, not winners).

**4-remediation-class comparison (full 3-lane book unless noted):**
| Class | N | ann$ | MaxDD | C11 |
|---|---|---|---|---|
| Account TS100k + cap80 | 1874 | $3,801 | $1,770 | **PASS** ($630 headroom) |
| Account Bulenox50k + cap80 | 1874 | $3,801 | $1,770 | **PASS** ($230 headroom) |
| Drop US + cap80 (TS50k) | 1096 | $1,744 | $892 | **PASS** but −$2,057/yr lost edge |
| Cap@80 alone (TS50k) | 1874 | $3,801 | $1,770 | fail by $170 |
| Cap@60 (too tight) | 1746 | $3,326 | $2,194 | fail (worse both axes) |
| Baseline (no fix) | 1958 | $3,445 | $2,788 | fail |

**Verdict:** Build the cap (edge-improving + closes gate-vs-live divergence, independent of account). C11 clearance requires budget ≥$1,770 → Bulenox$50k / TS$100k / self-funded. Staying TS$50k forces dropping US (−$2k/yr). Account choice is a separate capital decision.

**Part 1 = teach account_survival to model the live orb_cap. APPROVED to build (operator: 'Part 1 only, then review').**

## Provenance audit of the 3 gates (2026-06-03, operator-requested, READ-ONLY)

Commit `b3e768c5` ("fix(account): enforce strict survival gate", Josh Lees, today 18:46) added the strict overlay.

| Gate | Source | Classification | Your status |
|---|---|---|---|
| **1. C11 MC survival ≥70%@90d** | `pre_registered_criteria.md:228` (LOCKED), Pepelyshev-Polunchenko 2015 | **LOCKED CRITERION** | **PASS (73.3%)** |
| **2. Strict DD budget $1,600 = 0.80×$2,000** | `account_survival.py:57` bare constant, commit `b3e768c5` | **DISCRETIONARY — UNGROUNDED** | FAIL ($2,788) |
| **3. Daily-loss breach = 0 days @ −$450** | belt `$450` grounded (prop_profiles.py:687 Carver Tbl20 ≤25%); "0 breaches/7yr" requirement = commit `b3e768c5` | belt GROUNDED, **0-breach rule DISCRETIONARY** | FAIL (7 days = 0.34%) |

**Key findings (MEASURED):**
- `STRICT_DD_BUDGET_FRACTION = 0.80` has **NO rationale comment, no @research-source, no literature citation**. Commit body is one line. Test asserts `==1600.0` circularly. Not in pre_registered_criteria.md or TRADING_RULES.md.
- The official LOCKED criterion is ONLY "≥70% MC survival" — which the book PASSES at 73.3%.
- Both failing gates (2 and 3's 0-breach rule) are DISCRETIONARY additions stacked on top of the locked criterion by today's peer commit. Neither is a Topstep rule (Topstep allows the full $2,000) nor a locked criterion.
- The 7 breach days (0.34% of 2,048 days) are the $450 belt catching its designed ~1% tail — the belt WORKING, framed as a failure by the 0-breach overlay.

**Implication:** the cheapest HONEST fix may be to ground/calibrate the 0.80 (and the 0-breach rule) rather than contort the book. This does NOT mean weaken-to-pass — it means decide deliberately whether these discretionary overlays are wanted, and if so, cite them. Operator decision pending.

## Buffer calibration — OBJECTIVE, result-independent (2026-06-03, operator hard-rule: do NOT pick 0.90 because it passes)

Calibrated from the MC rolling-90d-DD tail distribution (same method as the grounded $450 belt, prop_profiles.py:687 "~1% tail of real risk distribution, 100k-day MC"). This is a property of the LOSS DISTRIBUTION, computed before checking whether the book passes:

| Book | MC 90d-DD p50 | p90 | p95 | p99 | buffer to cover p95 | p99 |
|---|---|---|---|---|---|---|
| **Uncapped** | $710 | $1,403 | $1,976 | $2,852 | 0.99 | **1.43 (exceeds $2,000!)** |
| **Capped@80** | $562 | $986 | $1,154 | $1,513 | 0.58 | **0.76** |

**Findings:**
- Uncapped book's p99 90d-DD ($2,852) EXCEEDS the raw $2,000 DD limit — no buffer ≤1.0 is safe uncapped. The cap is REQUIRED, not optional, for any $50k account.
- Capped book's p99 ($1,513) fits inside $2,000 at fraction 0.76. The current 0.80 sits just ABOVE the capped p99 → 0.80 is reasonable-to-conservative FOR THE CAPPED book.
- **The $1,600 (0.80) passing the capped book is NOT post-hoc rescue** — it passes because the capped tail genuinely fits ($1,513 p99 < $1,600), not because the threshold was tuned to the result.

**Carver grounding** (`carver_2015_volatility_targeting_position_sizing.md:83`): prop implication is "worst daily loss ≪ DD limit" (much-less-than), and ≤25% vol target → <1% chance of losing half capital (Table 23). Carver frames the buffer as DISTRIBUTION-driven (keep the tail well inside DD), NOT a fixed fraction — supporting a percentile-calibrated buffer over a round 0.80.

## Decision tree — ONE policy per option (post-hoc-rescue flags inline)

| Policy | Buffer rule | Capped book on TS$50k | Grounding | Post-hoc rescue? |
|---|---|---|---|---|
| **A. Official C11 only** | none (just ≥70% MC) | PASS (73.3%) | LOCKED criterion | NO — but removes ALL DD-tail protection (uncapped p99 $2,852 would deploy) |
| **B. Keep 0.80 strict overlay** | fixed 0.80 → $1,600 | FAIL by $170 (capped DD $1,770... wait — historical $1,770 > p99-MC $1,513) | ungrounded constant | NO (not tuned) but UNCITED |
| **C. Calibrated buffer (p99 of capped dist)** | budget = MC p99 of CAPPED 90d-DD ≈ $1,513-1,600 | PASS (capped historical max $1,770 vs... see note) | DISTRIBUTION-derived, Carver-aligned, belt-method-consistent | NO — derived from tail, not result |
| **D. Account upgrade** | keep 0.80, bigger DD | TS$100k $2,400 budget → PASS | capacity doctrine | NO |
| **E. Lane reduction** | keep 0.80, drop US | COMEX+TOKYO $892 DD → PASS | conservative | NO but −$2k/yr edge |

**NOTE / honest discrepancy to resolve:** the historical max-observed 90d-DD of the capped book is $1,770 (single realized worst window), while the MC p99 is $1,513. The strict gate uses the HISTORICAL max (`_max_observed_rolling_drawdown`), not the MC percentile. So calibrating to MC-p99 ($1,513) would NOT pass the historical $1,770. A calibrated policy must decide: gate on historical-max (more conservative, $1,770) or MC-p99 ($1,513). This is the real open question — NOT which round number to pick.

## RECOMMENDATION: Policy C (calibrated) gated on HISTORICAL max, which means buffer must cover $1,770 → fraction 0.885 of $2,000.
A buffer of 0.885 ($1,770) is the SMALLEST honest budget that admits the capped book's realized worst case on TS$50k — and it is calibrated to the actual historical tail, not chosen to pass. If that feels too tight a margin (only exactly covers the worst case, no headroom), then Policy D (TS$100k, $2,400 budget, $630 headroom over the $1,770 historical max) is the conservative capacity-preserving choice. Both keep the cap (required). Policy A (official-only) is REJECTED — it would deploy the uncapped $2,852-p99 tail.

**Flagged post-hoc rescue risk:** picking 0.885 specifically because it equals the capped historical max IS a mild result-fit. The cleaner version: gate on a pre-committed percentile (e.g. p99 of MC) of the CAPPED distribution, accept that the historical single-worst-window may exceed it, and size account headroom to cover historical-max separately. Operator decision needed on historical-max vs MC-percentile gating.

**APPROVAL NEEDED before Parts 2-3, before grounding/relaxing any gate, and before any live config / profile / broker mutation.**

## Reproduction

- Gate run: `python -m trading_app.account_survival --profile topstep_50k_mnq_auto` (Phase 2-3 numbers above).
- Cap-emulation scenarios (cap@60/80, drop-US, account-tier comparisons): `scripts/tools/c11_clearance_scenarios.py` (companion script preserved alongside this doc).
- Buffer-calibration percentiles: MC rolling-90d-DD tail distribution from the same `account_survival` MC path.
- Canonical sources traced inline (account_survival.py / session_orchestrator.py line refs in Phase 1 + Provenance tables).

## Limitations

- **READ-ONLY diagnosis — no fix landed.** Recommendations (close the model divergence, auto-regen, account/cap decision) are proposals pending operator approval. None of the gate-grounding decisions (0.80 vs calibrated buffer, historical-max vs MC-p99 gating) are resolved here.
- Open adversarial-audit gate: any actual `account_survival` change to model the live cap is a capital-path edit and must pass the adversarial-audit gate before merge — NOT done in this audit.
- The historical-max vs MC-p99 discrepancy ($1,770 vs $1,513) is flagged as an unresolved open question, not a settled number.
- Cap-emulation in `c11_clearance_scenarios.py` approximates the live `max_orb_size_pts` skip in replay; it is an emulation, not a re-run through the live `session_orchestrator`, so the exact live book may differ at the margin.
- Single-profile scope: findings are specific to `topstep_50k_mnq_auto` and its 3-lane book; not a pooled cross-lane claim (pooled-finding front-matter not owed).
