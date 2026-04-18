# Decision Log — Why We Chose What We Chose

**Purpose:** Every major architectural and methodological choice has a *why*. Without those, ChatGPT will stress-test our rules incorrectly (either blindly accepting, or blindly challenging without understanding the tradeoff). Read this to understand the space of alternatives we rejected and the reasons.

**Last updated:** 2026-04-18.

---

## 1. Holdout policy — Mode A (sacred boundary 2026-01-01), not Mode B

**Alternatives considered:**
- **Mode A (chosen):** Fixed sacred boundary. Discovery NEVER touches trading_day ≥ 2026-01-01. Forward OOS window accumulates in real time. Amendment 2.7 (2026-04-08).
- **Mode B (rescinded):** Rolling walk-forward with synthetic re-split. Used briefly Apr 3-8 2026.

**Why Mode A won (per the code + docs, not quoted statements):**
- **Statistical honesty over data efficiency.** Mode B permits synthetic OOS splits during discovery. The docstring in `trading_app/holdout_policy.py` and the RESCINDED header at the top of `docs/plans/2026-04-07-holdout-policy-decision.md` record the correction; `pre_registered_criteria.md` Amendment 2.7 (2026-04-08) locks Mode A.
- The override mechanism in `holdout_policy.py::enforce_holdout_date()` documents the principle: "ANY STRATEGY DISCOVERED WITH THIS OVERRIDE IS RESEARCH-PROVISIONAL — its OOS-clean property is destroyed."
- Cost of Mode A: ~3 months of accumulating forward OOS data before any new strategy can clear Criterion 8 (OOS N ≥ 30). Accepted as the price of holdout integrity.

**Implications:**
- 124 existing `validated_setups` rows are **grandfathered as research-provisional** (NOT OOS-clean). They were discovered under Mode B before the rescission.
- New discovery MUST use `--holdout-date 2026-01-01`. Enforcement: `trading_app/holdout_policy.py::enforce_holdout_date()`.
- Override token `"3656"` exists as speed-bump but destroys OOS-clean property; anything produced with it is research-provisional only.

**Canonical source:** `trading_app/holdout_policy.py`. Amendment trail: `pre_registered_criteria.md` Amendment 2.7.

---

## 2. ORB apertures — {5, 15, 30}, not more

**Why these three:**
- **5 minutes:** tight breakout, higher signal/noise, lower N of valid-range days (ORB-within-ORB collapses).
- **15 minutes:** middle ground — 3 5-minute bars is the historical institutional ORB definition (per Fitschen 2013).
- **30 minutes:** slower, avoids "fakeout" characteristic of early breakouts, captures end-of-opening-range value.

**Alternatives rejected:**
- **1-minute:** too noisy, single-bar ORB degenerates. No institutional grounding.
- **60+ minutes:** morning-session too long; by the time the ORB forms, news volatility collapses.
- **7 / 12 / 20 / 45:** arbitrary, no literature support. Would multiply K (multiple-testing penalty) without reward.

**Institutional grounding:** Fitschen 2013 ch3 establishes intraday trend-following with natural breakpoint at 15-30 minutes. Multiple apertures within this range are mutually independent enough to be tested (with BH-FDR) without becoming silly.

---

## 3. RR ratios — {1.0, 1.5, 2.0}, not wider

**Why these three:**
- 1.0, 1.5, 2.0 span the usable range for E2 stop-market entries.
- Below 1.0: rewards less than risk. Win-rate would need >50% + edge to overcome costs. Hard on real intraday data.
- Above 2.0: targets frequently out of reach in the window. WR collapses.

**Alternatives considered and rejected (and now re-considered):**
- **RR 2.5 / 3.0** was in earlier strategy set. Removed because MNQ COMEX_SETTLE RR2.5 showed C9 era-failure (2024 ExpR=-0.062) per the `cross_session_momentum_research` audit.
- **RR 0.5** — tested in earlier 2026-02 scans. Universally failed BH-FDR at any lane.

**Ongoing exceptions:** `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_VOL_RV12_N20` is in the book at TRANSITIONING status. Not killed, but sized-down.

---

## 4. Session catalog — 12 sessions, not arbitrary

**Why these 12:**
Each session corresponds to a **real market event** (CME reopen, Tokyo open, London metals open, NY data release, etc.) — NOT arbitrary clock times. See `CANONICAL_VALUES.md` §3 for the full list + Brisbane local time + exchange DOW alignment.

**One exception:** `BRISBANE_1025` is a fixed 10:25 Brisbane session with NO event anchor. It's in the catalog because discovery (2026-03-01) found a strong MNQ FDR survivor there that's independent from the 09:25 cluster (inverted seasonality). Retained for research-grade monitoring but not justified by event flow.

**Alternatives rejected:**
- **Arbitrary 30-min grid covering 24h:** would multiply K massively and capture zero edge at non-event hours (tested 2025-12, all negative).
- **Only US RTH sessions:** misses Asian / European structural edges (MGC LONDON_METALS, MNQ SINGAPORE_OPEN).
- **Single "global" session:** conflates regimes; obscures session-specific edges.

**DST handling:** All sessions use dynamic resolvers in `pipeline/dst.py`. NYSE_OPEN crosses Brisbane midnight in winter → DOW-alignment exception documented in `DOW_ALIGNMENT.md`. `validate_dow_filter_alignment()` fail-closes on DOW filters for NYSE_OPEN.

---

## 5. Active instruments — {MGC, MNQ, MES}, not more

**Why these three:**
- **MNQ** (Nasdaq micro): highest liquidity, longest data history (2019-05-06 launch), most live-tradeable on all prop firms.
- **MES** (S&P micro): same launch date, equity-index twin of MNQ, distinct event exposure (COMEX-correlated via VIX).
- **MGC** (Gold micro): 2022-06-13 launch. Captures commodity flow. LONDON_METALS edge.

**Dead for ORB:** `MCL`, `SIL`, `M6E`, `MBT`, `M2K` — tested and killed. Per `pipeline/asset_configs.py::DEAD_ORB_INSTRUMENTS`.
- M2K null test (2026-03-18): 0/18 families survive noise screening at any threshold.
- MCL, M6E, MBT, SIL: tested in waves, no ORB edge passed BH-FDR.

**Research-only (not for ORB):** `2YY`, `ZT` (rates) — event-window macro work. `GC`, `NQ`, `ES` exist as parent-contract data for pre-micro history only; discovery never uses parents as tradeable instruments.

**MGC special case:** `deployable_expected=False` because real-micro data horizon is ~3.8yr (launch 2022-06), below T7 era-discipline threshold (~5yr needed). Discovery runs on GC proxy (16yr) for research, but validator T7 correctly kills resulting candidates. Expected to flip to deployable ~2027-06.

---

## 6. Entry model — E2 (stop-market) canonical

**Why E2:**
- **Simple, replayable.** Stop-market order placed above/below ORB high/low at ORB end. No ambiguity in backtest reconstruction.
- **Live-executable on all prop firms.** No limit-on-retest waiting logic required.
- **Canonical ORB-window invariant** (Chan 2013 p4): `pipeline.dst.orb_utc_window()` is single source shared by backtest + live + feature builder. No backtest/live divergence possible.

**Alternatives:**
- **E0** (hypothetical market-on-close entry) — purged 2025-Q4 as survivorship bias contributor.
- **E1** (close-based break detection) — retained for M2K LONDON_METALS (REGIME-type). Known to produce artifacts on small-ORB-size instruments; M2K EUROPE_FLOW scan caught this.
- **E_RETEST** (limit-on-retest after failed break) — Phase C stub at `docs/audit/hypotheses/phase-c-e-retest-entry-model.md`. NOT YET BUILT. Requires outcome_builder changes (2-3 weeks infra).

**CB (Confirm Bars):** 0, 1, 2, 3. CB1 is the most common. CB0 = enter on break (aggressive). CB2/CB3 = require 2-3 bar close beyond ORB (slow).

---

## 7. Multiple-testing correction — BH-FDR, not Bonferroni

**Why BH-FDR:**
- Bonferroni is too conservative for exploratory discovery (family-wise error rate). With K=14,261 cells, Bonferroni p-threshold = 0.05/14,261 = 3.5e-6 — practically nothing passes.
- BH-FDR controls the expected FALSE DISCOVERY RATE at 5%, which is the statistically correct framing for "which of my N hypotheses are real."

**Literature grounding:** Benjamini-Hochberg 1995 (original FDR paper), extended by Benjamini-Yekutieli 2001, applied to finance by Harvey-Liu 2015.

**Multi-framing:** per `backtesting-methodology.md` Rule 4, we report survivors at K_global, K_family, K_lane, K_session, K_instrument, K_feature. A cell is a legitimate discovery if it passes BH-FDR at K_family OR K_lane (not just K_global — Harvey-Liu argue the family is the natural hypothesis unit).

---

## 8. t-stat thresholds — 3.0 general, 3.79 no-prior-theory

**Why two bars:**
- **t ≥ 3.0** for features with economic prior (e.g., ORB compression Z, ATR regime, volume confirmation — all grounded in market microstructure literature).
- **t ≥ 3.79** (Chordia 2018 strict) for features with no prior theory (e.g., novel ORB pattern discovered via scan). Guards against data mining.
- Below 3.0: ambiguous even with prior; publication cutoff.

**Source:** Chordia et al 2018 (LIT_chordia_2018_two_million_t379.md) — tested 2M strategies on US equities, derived 3.79 threshold as the t-stat beyond which false discoveries drop to ~5%.

---

## 9. Cost model — TopStep Rithmic baseline (conservative)

**Why Rithmic not Tradovate:**
- TopStep Rithmic commission is higher than Tradovate on every instrument (e.g., MNQ $1.42 Rithmic vs $1.34 Tradovate).
- Conservative bias: backtest can't UNDER-estimate friction. Real-world costs floor the backtest claims.
- Rithmic-based prop firms (Bulenox, Elite Trader Funding) use these rates directly — baseline works for all deployments.

**Known limitation (adversarial review 2026-03-18):** Flat per-instrument slippage. Real slippage probably correlates with ORB size + session liquidity. MGC tbbo pilot measured mean=6.75 ticks (vs 1 modeled), std=41.57, max=263. MNQ pilot NOT YET RUN. Kill criterion: if live slippage > 2× modeled, all backtest results overstated.

**Alternatives considered:**
- **Broker-specific per-firm costs:** implemented as `SESSION_SLIPPAGE_MULT` for live only. Not used in backtests.
- **Flat commission across firms:** rejected — TopStep's Express clearing commission is consistently higher than Interactive Brokers; backtesting on IB rates would overstate.

---

## 10. Position sizing — Fixed-fractional at 0.5-1.0x, not Kelly yet

**Current:** STABLE → 1.0x risk/trade; TRANSITIONING → 0.5x; DEGRADED → OFF.

**Why not Kelly / vol-targeted yet:**
- Kelly-linked sizing (Carver 2015 ch10) requires reliable edge estimates. Our criteria-8 N_OOS ≥ 30 barely clears for individual lanes.
- Kelly's pre-condition is stationary edge distribution. Our per-year stability tests (T7) show era-sensitivity on several lanes.
- Phase D plan (`docs/audit/hypotheses/phase-d-carver-forecast-combiner.md`) proposes forecast-combiner framework for MNQ TOKYO_OPEN pilot (4-5 signals). Infrastructure not yet built.

**Alternatives considered:**
- **Full Kelly:** too aggressive with uncertain edge.
- **Half-Kelly:** on the Phase D roadmap but gated by F-1 XFA dormant.
- **Vol-targeting (Carver 2015 ch9):** on Phase D roadmap after forecast combiner.

---

## 11. Strategy classification — ExpR primary sort, not Sharpe

**Why ExpR:**
- Sharpe can rank a low-ExpR high-consistency strategy above a high-ExpR moderate-consistency one. For capital compounding on prop-firm accounts with daily loss limits, ExpR (per-trade expected dollar in R-units) matters more.
- Sharpe of ORB strategies is dominated by one or two regime periods per 3-year window → noisy.
- `mechanism_priors.md` connects: Sharpe is a summary of the edge × cost structure; ExpR is the raw edge.

**When Sharpe still matters:** fitness classifier (STABLE / TRANSITIONING / DEGRADED) uses Sharpe_ann ≥ 0.1 in 60%+ of last 6 windows as the decay detector. Different purpose — detecting edge erosion, not ranking.

---

## 12. One DB — canonical `gold.db`, no scratch copies

**Why:**
- Scratch DB at `C:\db\gold.db` was deprecated 2026-03. Caused stale-data bugs across sessions.
- `pipeline.paths.GOLD_DB_PATH` now blocks scratch path.
- Drift check #37 verifies canonical exists; drift check #62 blocks hardcoded scratch defaults.

**Trade-off accepted:** single file = single point of failure for backups. Daily copy to offsite is responsibility of user (not automated).

---

## 13. Canonical code modules — never re-implement

**Why:**
- Every re-encoding drifts. E2 canonical-window fix (2026-04-07) caught a divergence between backtest and live that was a look-ahead bias risk per Chan 2013 p4.
- Maintenance burden of parallel implementations compounds.

**Canonical list (from `CANONICAL_VALUES.md` §6):**
- Sessions → `pipeline/dst.py::SESSION_CATALOG`
- ORB window → `pipeline/dst.py::orb_utc_window()`
- Costs → `pipeline/cost_model.py::COST_SPECS`
- Instruments → `pipeline/asset_configs.py::ACTIVE_ORB_INSTRUMENTS`
- Holdout → `trading_app/holdout_policy.py::HOLDOUT_SACRED_FROM`
- Profiles → `trading_app/prop_profiles.py::ACCOUNT_PROFILES`
- DB path → `pipeline/paths.py::GOLD_DB_PATH`

---

## 14. Why we trade prop-firm accounts (not just self-funded)

**Capital scaling.** Starting self-funded would require $50K+ risk capital for meaningful per-trade dollar edge. Prop firms offer $50K-$150K funded accounts for a one-time eval fee (~$100-300). Copy-trading 5-10 accounts multiplies edge without multiplying personal capital risk.

**Risk asymmetry.** Prop firms absorb drawdown losses past the trailing max. Personal loss cap = eval fee + monthly sub.

**Scaling landscape (2026-04-15 analysis):**
- Phase 1: 1-2 TopStep XFA ($5-10K/yr net)
- Phase 2: +3 Bulenox ($15-25K/yr)
- Phase 3: +5 MFFU Rapid ($30-50K/yr)
- Phase 4: +self-funded 1 NQ mini @ $50K ($65-105K/yr total)

**Dead firms for our plan:** Apex (bots banned), Tradeify (exclusivity), FundedNext (red flags), Elite/ETF (TradersPost gate).

**Key constraints:**
- TopStep XFA ↔ LFA mutually exclusive (LFA promotion destroys XFAs)
- Bulenox 3 → 11 account unlock ladder
- MFFU 5 sim + 5 DTF
- Rithmic integration unlocks Bulenox + AMP/EdgeClear self-funded
- 1 NQ mini vs 10 MNQ = 77% commission cut at self-funded scale

Canonical: `prop-firm-official-rules.md` (bundled) + `CANONICAL_VALUES.md` §5.

---

## 15. Why we pre-register hypotheses

**To defend against our own confirmation bias.** Pre-registration is the user's countermeasure against the natural drift of "scan → see what looks good → post-hoc rationalize."

- Each discovery scan writing to `experimental_strategies` or `validated_setups` requires a pre-reg file at `docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml|md` BEFORE running.
- File must include: numbered hypotheses with economic citations, exact filter/feature dimensions + threshold ranges, K_budget (pre-committed), kill criteria, expected N per cell.
- Enforcement: `pipeline/check_drift.py` #94 validates hypothesis_file_sha on validated_setups.

**Literature grounding:** Bailey et al 2013 (MinBTL) + Harvey-Liu 2015. Discovery without pre-reg = data mining until proven otherwise.

---

## 16. Why this is ADHD-friendly collaboration, not solo grind

The user is aware of his own failure modes: hyperfocus, freshness bias, narrative capture. Design protections:
- Claude Code does drift checks / tests / audits on every file edit (hook-enforced)
- ChatGPT (you) challenges hypotheses and refuses shortcuts
- NO-GO registry persists across sessions (STRATEGY_BLUEPRINT)
- Pre-registration locks the hypothesis before the scan
- Two-track decision rule (`02_USER_PROFILE.md`) checks new work against existing queue

You are part of this defense. Your main job when asked "should I try X?" is to check if X is new, if X is already tested-dead, and if X has been pre-registered. Not to enthuse.
