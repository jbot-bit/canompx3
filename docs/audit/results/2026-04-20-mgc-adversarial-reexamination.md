# MGC Adversarial Re-examination — Contracts, Data, Slippage, Regime

**Date:** 2026-04-20
**Owner:** claude-opus-4-7
**Status:** AUDIT (no deployment claim, no discovery run)
**Classification:** Cross-cutting audit — instrument integrity, cost realism, regime context

**Trigger:** User pushback after MGC thread closure (HANDOFF §2026-04-19 "MGC path-accurate sub-R v1"):
1. *"isn't it also to do with MGC > GC contracts? do we handle that all properly."*
2. *"it has like different volumes and shit like that. ensure /resource understanding."*
3. *"ensure blast radius and upstream and downstream are audited and handled appropriately. nothing left behind. professional."*
4. *"we kinda in a like new regime ig idk"* — gold regime 2024-2026
5. *"ensure proper understanding of project and no bias or lookahead. no pigeon holing"*

**Authority:**
- `.claude/rules/institutional-rigor.md`
- `.claude/rules/backtesting-methodology.md`
- `.claude/rules/research-truth-protocol.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/plans/2026-04-19-gc-mgc-handling-note.md` (canonical GC/MGC policy)

---

## Executive verdict

The 2026-04-19 MGC thread closure stands **with these important refinements**:

1. **The closure is soft / in-waiting, not hard-kill.** `pipeline.asset_configs.ASSET_CONFIGS['MGC']['deployable_expected']=False` (asset_configs.py:84) AND the active shadow-record pre-reg (`docs/audit/hypotheses/2026-04-19-mgc-orbg5-long-signal-only-shadow-v1.yaml`) AND the MGC-LONDON_METALS level-scan T0-T8 survivor (`docs/audit/results/2026-04-15-t0-t8-audit-mgc-level-cells.md`: 6/8 tests PASS, only T3 fails on N_OOS=6) all mean the project already treats MGC as **statistically underpowered**, not dead. The HANDOFF prose "do not keep rescuing" is stronger than the actual state.

2. **Contract-size (10:1) is handled correctly** in R-space by construction — verified line-by-line. R-multiples are symbol-invariant. The translation audit's "payoff compression" finding is a genuine price-realization difference, not a unit-accounting artifact.

3. **Volume handling is correct** — `rel_vol` is symbol-local (`pipeline/build_daily_features.py:1513-1519`), drift check is landed and active (`pipeline/check_drift.py:2102-2201`), and zero deployed MGC strategies use volume-based filters. Phase 2 (2026-04-08) relabelled pre-2022 GC-proxy data under `symbol='GC'` so bars_1m / orb_outcomes / daily_features are clean.

4. **MGC real slippage is ~3.4× modeled in dollar terms** — `pipeline/cost_model.py:145` records TBBO pilot mean = 6.75 ticks (std 41.57) vs modeled `slippage=$2.00` = 2 ticks round-trip. 6.75 / 2 ≈ **3.4× modeled**. Real per-R friction rises from 5.43% to ~9.49% of risk (1.75× higher). Every MGC backtest `pnl_r` is systematically optimistic by approximately this amount. **MNQ TBBO pilot has not been run** per cost_model.py:146 — the same optimism plausibly applies to the deployed MNQ/MES book, unmeasured. Repo-wide debt, not MGC-specific. **Caveat:** pilot script `research/research_mgc_e2_microstructure_pilot.py` has not been adversarially audited here — if its methodology has bias, the 6.75-tick figure is wrong. Flag for H0 pre-reg.

5. **Gold is in a new structural regime** (2022 Russia-sanctions watershed, accelerating 2024-2026). Central bank buying 750-850t projected for 2026 (~26% of mine output); BRICS+ gold reserves 11.2% → 17.4% of global since 2019; spot hit $5,595 intraday Jan 2026. Transition regime, not clean trend.

6. **MinBTL binds tightly on MGC**. Per Bailey et al 2013 (`docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md:49`): *"if only 5 years of data are available, no more than 45 independent model configurations should be tried."* MGC has **3.8 years real-micro data** (2022-06-13 → 2026-04-20) → hard trial budget ≈ **30 independent configurations** to keep E[max_N]=1 under null. Every prior Phase 2.9 / rediscovery scan that spent MGC trial budget above this is statistically fragile.

**What that means operationally:** the thread is NOT reopenable as a broad discovery campaign. It IS eligible for **narrow, theory-tied, K=1 Pathway B tests**, provided each trial comes out of the ~30-trial budget and the budget is pre-committed.

---

## 1 — Inventory of prior MGC research (was my claim "never tested" wrong?)

**Claim retraction.** My first POV response said MGC-native discovery and MGC-as-conditioner had "never been tested." Both wrong. Ground truth from `git ls-files` + `Glob research/**/*mgc*.py`:

**Research scripts (14):**
| Script | Scope |
|---|---|
| `research_mgc_mnq_correlation.py` | MGC ↔ MNQ cross-instrument correlation (exists; haven't read contents) |
| `research_mgc_regime_shift.py` | MGC-specific regime analysis (exists) |
| `research_mgc_asian_fade_mfe.py` / `research_mgc_asian_reversal.py` | Asian-session MGC behavior |
| `research_mgc_e2_microstructure_pilot.py` | The TBBO slippage pilot that found the 6.75× gap |
| `research_gc_mgc_translation_audit.py` | Shelf-transfer audit (2026-04-19) |
| `research_mgc_payoff_compression_audit.py` | Follow-up on the "compression" finding |
| `research_mgc_native_low_r_v1.py` | MGC-native low-RR discovery (5 survivors) |
| `research_mgc_path_accurate_subr_v1.py` | Path-accurate rebuild (killed all 5) |
| `mgc_mode_a_rediscovery_orbg5_v1_scan.py` / `_short_v1_scan.py` | Mode-A native rediscovery (4+4 cells killed) |
| `mgc_level_scan.py` | 12 sessions × 3 apts × 3 RRs × 2 dirs × 14 level features (1775 cells) |
| `t0_t8_audit_mgc_level_cells.py` | T0-T8 validation on level-scan survivors |

**Result docs (8):** 2026-04-15 level-scan, 2026-04-15 T0-T8-MGC-level-cells, 2026-04-19 translation audit, 2x 2026-04-19 rediscovery scans, native-low-R v1, path-accurate sub-R v1, payoff-compression audit.

**Hypothesis pre-reg files (14):** from 2026-04-09 comprehensive through 2026-04-19 path-accurate. Pathway-A and Pathway-B disciplines both used.

**Implication:** MGC has been tested extensively. The statement "we haven't done MGC-native discovery" is false. Any new MGC test must justify itself as **genuinely untested** (different mechanism, different feature, different encoding).

---

## 2 — Contract-size (10:1) handling — verified CORRECT

### 2.1 The constraint

Both contracts trade in USD/oz with identical tick grid (`$0.10/oz`). GC is 100oz → `$10/tick`; MGC is 10oz → `$1/tick`. Source: `docs/plans/2026-04-19-gc-mgc-handling-note.md:37-49`, grounded in CME rulebook Ch 113 (GC) and Ch 120 (MGC).

### 2.2 The code

`pipeline/cost_model.py:118-126` defines GC as a deliberate 10× scaling of MGC dollar costs (`commission_rt=17.40`, `spread_doubled=20.00`, `slippage=20.00`). Comment at :114-116 states the invariant:

> "Cost specs are 10x MGC by contract multiplier. Commission estimated proportionally — exact broker commission varies but does NOT affect pnl_r (R-multiples are price-based). COST_LT thresholds are identical in points because both numerator and denominator scale by 10x."

`pipeline/cost_model.py:439-469` implements `to_r_multiple` and `pnl_points_to_r`. R = `(pnl_points × point_value − friction_$) / risk_$`. Because friction scales with point_value by construction, the 10× cancels.

### 2.3 Algebraic verification

For a 1R winner on a 10-pt stop at entry=2000:

| Symbol | Raw risk $ | Friction $ | Risk $ | Raw win $ | Net win $ | R |
|---|---|---|---|---|---|---|
| MGC | 100 | 5.74 | 105.74 | 100 | 94.26 | **0.8914** |
| GC  | 1000 | 57.40 | 1057.40 | 1000 | 942.60 | **0.8914** |

Identical. Contract-size is NOT the source of the translation-audit's "payoff compression" finding.

### 2.4 What this does NOT cover

GC `commission_rt=17.40` and the two `20.00` spread/slippage figures are **estimated proportionally** (cost_model.py:121: *"Not canonical — GC not traded live"*). GC-modeled friction is NOT measured. So the R-equivalence holds only under modeled costs. If MGC real slippage is 6.75× modeled (§4 below), then the real-world MGC-vs-GC R-burden asymmetry is UNKNOWN because GC reality isn't measured.

---

## 3 — Data-integrity (MGC era, GC era, contamination) — CLEAN

### 3.1 Era definitions

`pipeline/data_era.py` defines the canonical PARENT vs MICRO classification:

- `micro_launch_day("MGC") = date(2022, 6, 13)` (CME Micro Gold real-micro contract start per `pipeline/asset_configs.py:97`)
- `era_for_trading_day("MGC", trading_day)`: MICRO if ≥ 2022-06-13, else PARENT
- `is_micro("MGC")` = True (active real-micro)
- `is_micro("GC")` = False (native parent)

### 3.2 Phase 2 relabel (2026-04-08)

Prior to 2026-04-08, `bars_1m` stored GC historical data under `symbol='MGC'` (backfill from GC.FUT series). Phase 2 of canonical-data-redownload **relabelled** those rows to `symbol='GC'`. Since then, `symbol='MGC'` contains **only** real-micro bars from 2022-06-13 onward.

Source: `pipeline/asset_configs.py:319` *"Stored as symbol='GC' (relabeled 2026-04-08 from former 'MGC' as part of Phase 2 of canonical-data-redownload). Cost model: PARENT specs ($100/pt), NOT MGC specs."*

### 3.3 Empirical verification (gold.db today)

| Table | Symbol | Date range | N rows |
|---|---|---|---|
| `bars_1m` | GC | 2010-06-07 → 2026-04-06 | 5.5M |
| `bars_1m` | MGC | 2022-06-13 → 2026-04-16 | 1.35M |
| `orb_outcomes` | MGC | ≥ 2022-06-13 only | 918,684 (valid: 882,996) |
| `daily_features` | MGC | ≥ 2022-06-13 only | 3,360 |

**Zero MGC rows before 2022-06-13 in any canonical layer.** Verified via `db-analyst` agent (2026-04-20). No contamination.

### 3.4 Data-era drift check — LANDED + ACTIVE

`pipeline/check_drift.py:2102-2201` implements `check_active_micro_only_filters_after_micro_launch`:

1. Loads all active `validated_setups` rows.
2. Filters to those whose filter declares `requires_micro_data=True`.
3. Filters to micro instruments (per `data_era.is_micro`).
4. For each: recomputes first-traded-day from canonical `daily_features` + `orb_outcomes`.
5. Violates if `first_day < micro_launch_day(instrument)`.

This gate prevents any MGC strategy with a volume-based filter from carrying pre-2022-06-13 trade days. The gate is active in CI, pre-commit, and the post-edit hook.

---

## 4 — Slippage realism — REPO-WIDE DEBT

### 4.1 The measurement

`pipeline/cost_model.py:145`:

> "MGC tbbo pilot showed mean=6.75 ticks (vs 1 modeled), std=41.57, max=263. MNQ tbbo pilot has NOT been run yet — `research/research_mnq_e2_slippage_pilot.py` exists."

Interpreting precisely: `cost_model.py:107` sets MGC `slippage=$2.00` round-trip; MGC tick = $1 → 2 ticks modeled. The pilot mean is 6.75 ticks which is **3.4× the modeled dollar slippage** (6.75 ticks ÷ 2 ticks = 3.4×). The "vs 1 modeled" in the code comment is per-side; the round-trip figure is what enters the R-calculation. Std = 41.57 ticks shows individual trade slippage varies enormously. Measured by `research/research_mgc_e2_microstructure_pilot.py`. **Not yet adversarially audited** — pilot methodology could itself have bias.

### 4.2 Break-even scenario (cost_model.py:148-152)

> "Break-even analysis (`scripts/tools/slippage_scenario.py`):
>   COMEX_SETTLE: 4.9 extra ticks to zero (FRAGILE)
>   SINGAPORE_OPEN: 6.0 extra ticks to zero
>   NYSE_CLOSE: 15.4 extra ticks (robust)
>   NYSE_OPEN: 17.7 extra ticks (robust)"

COMEX_SETTLE and SINGAPORE_OPEN are within 1σ of the MGC pilot mean. If those sessions' MGC lanes were deployed, they could realistically be zero-expectancy in live trading.

### 4.3 Why this matters for the MGC conclusion

Under modeled slippage, `research_mgc_path_accurate_subr_v1.py` found 0/5 survivors. Under pilot-realistic slippage, the hurdle is strictly tighter. Closure direction is **confirmed stronger**, not weaker. Quantitative per-cell impact is what H0 (§9) is designed to measure — pre-registered, not projected here. Order-of-magnitude sanity: a 3.4× slippage increase translates to roughly −0.08R per winner (derivation: 1R winner under modeled 0.8914R → under real 0.8100R). Cells with modeled Δ_IS < +0.08R multiplied by their win-rate are at-risk; cells with wider margin are slippage-robust. The shadow-track cells span both sides of this line.

### 4.4 Scope of the debt

This is NOT MGC-specific. **MNQ TBBO pilot has never been run.** Every production backtest in the 38-lane book uses modeled 1-tick slippage for MNQ (`cost_model.py:164`) and 0.25-pt slippage for MES (`cost_model.py:176`). If MNQ real slippage is similarly 3-7× modeled, the deployed book's ExpR estimates are optimistic.

**Recommendation:** add to `docs/runtime/debt-ledger.md` as `cost-realism-slippage-pilot`. Schedule MNQ TBBO pilot. Defer any MGC deployment claim until `orb_outcomes` can be recomputed with pilot-realistic per-session slippage.

---

## 5 — Volume handling — SYMBOL-LOCAL, DRIFT-CHECKED

### 5.1 Raw volume differences (empirical, overlap era 2022-06-13 → 2026-01-01)

Per `db-analyst` agent (2026-04-20):

- **Per-bar volume**: GC mean 126, median 68. MGC mean 95, median 38. GC has 32% higher per-bar in raw contract count.
- **ORB-window total volume (aggregate contracts per ORB)**: MGC is **1.4–3.1× LARGER than GC in absolute contracts traded**. TOKYO_OPEN 3.14×, SINGAPORE_OPEN 2.62×, COMEX_SETTLE 1.41×. Interpretation: MGC has more *participants* during breakout windows (retail-accessible contract); GC has fewer, larger trades (institutional). This is a structural microstructure difference, not a contract-multiplier artifact.

### 5.2 CME-reality 2025 context (sources at bottom)

- MGC ADV 2025: 300,757 contracts/day (record)
- MGC ADV Dec 2025: 475,825/day
- MGC single-day record: 741,822 contracts (Oct 9, 2025)
- GC "more heavily traded overall" but MGC's growth is material

**Our project assumption that MGC is illiquid is outdated.** MGC is a deep market with retail participation; GC remains the institutional benchmark.

### 5.3 Code-path verification

`pipeline/build_daily_features.py:1470-1546` computes `rel_vol_{label}`:

```python
bar_rows = con.execute(
    """SELECT ts_utc, volume FROM bars_1m
       WHERE symbol = ?
       AND EXTRACT(HOUR FROM ...) = ?
       AND EXTRACT(MINUTE FROM ...) = ?
       ORDER BY ts_utc""",
    [symbol, h, m],
).fetchall()
```

**Symbol-local**: `WHERE symbol = ?` — no cross-symbol baseline. Lookback 20, min 5 prior entries, fail-closed None if insufficient. MUST match `strategy_discovery._compute_relative_volumes()` — paper trader and live orchestrator read the pre-computed values directly.

### 5.4 rel_vol distribution asymmetry

Even with symbol-local computation, the distributions differ:
- rel_vol mean GC vs MGC: 1.27-1.41× different across sessions
- TOKYO_OPEN baseline differs 2.48× (GC 177 vs MGC 440 avg volume)

**Operational implication:** `rel_vol >= 1.2` as an absolute threshold is NOT numerically comparable across GC and MGC. Within a single symbol, rank-wise comparisons (quantile) are safe. Across symbols, percentile framing (e.g., "rel_vol_HIGH_Q3") is the only safe comparison.

### 5.5 E2 look-ahead gate

`trading_app/config.py:3560-3568` lists `E2_EXCLUDED_FILTER_PREFIXES = ("VOL_RV", "ATR70_VOL")` — these filter types are BANNED on E2 entries because break-bar volume is not known when E2 places its stop order mid-bar. This is correctly enforced in `strategy_discovery.py:1161` and `execution_engine.py:627`.

Practical effect: our deployed book (E2 CB1 throughout) cannot use rel_vol as a filter. It could if we ran E1 (entry after break bar closes) — untested production path.

### 5.6 Validated-setups volume-filter audit

Per db-analyst: **zero active MGC strategies** use volume-based filters. 7 GC strategies do (retired, not deployed). Phase 2.5 `rel_vol_HIGH_Q3` 5-Tier-1 survivors are all index (MNQ/MES) per memory — none MGC.

---

## 6 — Gold regime 2024-2026 — NEW STRUCTURAL BASELINE

Sources at bottom.

### 6.1 The shift

- **Russia sanctions 2022** = watershed. Frozen $300B FX reserves shifted central-bank risk-asymmetry calculus. Per World Gold Council and multiple industry sources.
- **2025**: central-bank buying 1,200+ tonnes. WGC 2025 survey: 43% plan to increase (up from 29% year prior).
- **2026 projection**: 750-850 tonnes CB buying ≈ 26% of annual mine output.
- **BRICS+ reserves**: 11.2% → 17.4% of global since 2019.
- **Price**: gold hit $5,595 intraday Jan 29, 2026.

### 6.2 What this means for intraday behavior

From Fibonacci/analysis sources: *"Gold futures are more likely in a transition regime than in a clean trending environment."* Moves look like *"deleveraging events"* rather than directional trends. Strong price moves + falling open interest = position-clearing, not structural trend commitment.

**Implication for ORB**: Fitschen Ch 3 grounds commodity intraday trend (`docs/institutional/literature/fitschen_2013_path_of_least_resistance.md:41`). But Fitschen tested 2000-2010. A 2024-2026 "transition regime" is the opposite of what the Ch 3 mechanism expects. A broad MGC ORB-trend strategy may genuinely fail during this regime even if it re-emerges as gold normalizes.

### 6.3 Data-horizon overlay

MGC real-micro data starts 2022-06-13. That means our ENTIRE MGC horizon (3.8 years) is IN the post-sanctions regime, mostly in the high-vol/transition period. We have ZERO MGC data from gold's previous calm regime. Any MGC discovery finding is implicitly a finding about "gold in its 2022-2026 structural-shift regime", not about "gold generally." This limits transferability to future calm periods AND means we cannot use pre-2022 GC data as a proxy for "normal gold" behavior any more — pre-2022 GC is a DIFFERENT regime than today's MGC.

---

## 7 — MinBTL budget for MGC

Bailey et al 2013 (`docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md:49`):

> *"if only 5 years of data are available, no more than 45 independent model configurations should be tried, or we are almost guaranteed to produce strategies with an annualized Sharpe ratio IS of 1, but an expected Sharpe ratio OOS of zero."*

Formula (Eq. 6): `MinBTL ≈ 2·Ln[N] / E[max_N]²`.

**For MGC at 3.8 years:** rough linear extrapolation from Bailey's 5yr/45-trial example gives approximately **N ≤ 30 independent configurations** at E[max_N] = 1. Precise derivation via Equation 6 requires numerically solving for N given the closed-form E[max_N] = `(1-γ)·Z⁻¹[1-1/N] + γ·Z⁻¹[1-1/(Ne)]` — not done here. Order of magnitude is the right read: we have budget for ~10-30 mechanism-tied trials, not hundreds.

Practical interpretation:
- Broad MGC discovery scans (Phase 2.9-style, 324 combos) already saturated the budget many times over. Their MGC-specific survivors must be treated as statistically fragile even if BH-FDR passes.
- Any **new** MGC-specific discovery must be K=1 Pathway B (theory-driven, Chordia `t ≥ 3.00` with theory) or a very small confirmatory battery on pre-specified cells.
- The shadow-record track is the correct holding pattern — it doesn't spend trials, it waits for data.

---

## 8 — Adversarial audit of my 5 prior POV claims (2026-04-20 first POV response)

My first response to the second-brain prompt listed 5 alternative interpretations. Auditing each against now-grounded evidence:

| # | Prior claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Translation≠instrument failure; MGC-native never run | **RETRACT** | 14 research scripts + 14 pre-regs = extensively tested. Shadow track active for underpowered positives. |
| 2 | Payoff compression = microstructure/cost artifact, not alpha | **PARTIAL CONFIRM** | Modeled friction is symmetric in R-space (contract-size cancels). BUT real MGC slippage 6.75× modeled → real friction asymmetry unmeasured. Compression is *partly* real price-realization AND *partly* an under-modeled-friction amplifier. |
| 3 | Gold regime shift kills pre-2024 baselines | **CONFIRM** | New regime confirmed via WGC. MGC real-micro era (2022-06-13+) is entirely post-shift. No calm-regime MGC data exists. |
| 4 | MGC portfolio-diversifier value never tested | **TENTATIVE CONFIRM** | No evidence that correlation matrix was computed; `research_mgc_mnq_correlation.py` exists but I haven't read contents. **Scheduled as H1 test.** |
| 5 | 0/17 above RR≥1.0 positive is p≈0.0002 under null | **PARTIAL RETRACT** | Under IID sign test, P(X=0 | 17 Bernoulli 0.5) = 7.6e-6, not 0.0002. But the 17 rows are NOT independent (same instrument, overlapping time). Effective N <17. Still directionally significant but I over-stated precision. The signal is "systematic negative bias at high RR on MGC" but exact p is unknown. |

**Overall self-assessment:** my first POV response was 40% correct, 60% unverified/wrong. Ground-truth verification changed three claims materially.

---

## 9 — Proposed tests (H0, H1, H3) — with kill criteria

Pre-reg files to follow at `docs/audit/hypotheses/2026-04-20-*.yaml`.

### H1 — MGC portfolio-diversifier correlation + Sharpe lift (cheapest, highest info)

- **Theory citation:** Markowitz (1952) mean-variance; diversification benefit from low-correlation zero-alpha assets.
- **Data:** daily ExpR realizations per strategy-lane from `paper_trades` or `orb_outcomes` intersected with `validated_setups`.
- **Scope:** 2022-06-13 to 2026-01-01 (IS only; no OOS peek).
- **Operation:** compute pairwise daily-return correlation between a hypothetical MGC lane and each of the 38 active MNQ/MES lanes. Report average, max, median.
- **Sharpe lift:** add zero-ExpR MGC return stream at 10% weight to current book; measure ΔSharpe.
- **Kill:** max pairwise corr > 0.50 OR ΔSharpe < 0.05.
- **Pass:** both below threshold → move to H3 or keep as signal-only diversifier candidate.
- **K budget:** 1 (single hypothesis).

### H0 — MGC real-slippage sensitivity on 5 native-low-R-v1 cells (cleanest adversarial test)

- **Theory citation:** `pipeline/cost_model.py:145` (MGC TBBO pilot 6.75× slippage gap); institutional-rigor rule #8 "Verify before claiming."
- **Scope:** 5 pre-specified cells from `research_mgc_native_low_r_v1.py` result.
- **Operation:** monkey-patch MGC `slippage=6.75` (USD, reflecting 6.75 ticks × $1/tick). Re-compute `pnl_r` for every trade in these 5 cells. Re-evaluate IS ExpR.
- **Kill criterion for thread closure:** any cell with IS ExpR > +0.05R under real slippage = closure is "thread not dead, defer to shadow." If all 5 drop below +0.05R → closure confirmed harder.
- **K budget:** 5 (audit, not discovery — per backtesting-methodology.md RULE 4).
- **Note:** this test does not *rescue* any lane — it tests whether the modeled-vs-real friction gap changes the closure verdict.

### H3 — MGC retail-fade hypothesis (deferred pending H1 or H0 survivor)

- **Theory citation:** Fitschen Ch 3 (intraday trend on commodities); combined with MGC's 1.4-3.1× higher ORB-window volume density suggesting retail-driven break chasing.
- **Hypothesis:** MGC shorts AT +1R beyond ORB break, on days with `rel_vol_{session} ≥ Q3`, have positive IS mean.
- **Kill:** BH p > 0.05 OR IS mean < +0.05R OR N < 50.
- **K budget:** 1 (single hypothesis, theory-supported).
- **Note on look-ahead:** `rel_vol` is E2-excluded. This test requires E1 entry (entry after break bar closes) — a non-deployed production path.
- **Deferred** until H0/H1 produce a reason to spend another trial from the MGC budget.

### H2 — Repo-wide real-slippage re-audit (separate campaign)

- Deferred. Requires MNQ TBBO pilot run first.
- Tracked as debt-ledger entry, not MGC-specific.

---

## 9.5 — Additional angles found on self-audit

Three questions that the initial pass missed but belong in a "nothing left behind" audit:

### A. Is MGC "payoff compression" actually a commodity-intraday property, not MGC-specific?

We have no other deployed commodity to compare. The 5 dead-for-ORB instruments (MCL/SIL/M6E/MBT/M2K) are all parent-proxy data (not real-micro) and were retired before Phase 2.9, so direct comparison is blocked.

What's testable: re-run the exact MGC translation audit structure on **MCL vs CL** (crude oil micro vs mini) — same contract-ratio (10:1), same proxy relationship — to see if "payoff compression" appears there too. If yes, it's a commodity-class property not gold-specific; if no, MGC is unique.

Budget cost from MGC's ~30-trial allocation: zero (MCL has its own budget; this is outside MGC's MinBTL constraint).

Scheduled as potential follow-up, not in H0-H3 scope.

### B. E1 entry path — untested production path, would unlock rel_vol

Our entire deployed book is E2 CB1. `trading_app/config.py:3558`:

> *"E1 is NOT affected: E1 enters AFTER the break bar closes (next bar open), so all break-bar properties are known at E1 entry time."*

E1 would make `rel_vol`, `break_bar_continues`, and `break_delay_min` trade-time-knowable. Phase 2.9 / rediscovery scans appear to have tested E2 only. If an MGC edge exists that depends on break-bar-observed volume confirmation, E2 scans would never see it. **E1 discovery on MGC is genuinely untested.** But: MinBTL budget is tight and entering a new entry-model discovery campaign would burn much of the ~30-trial allocation. Would require a pre-committed K=5-10 Pathway B.

Not in H0-H3 scope; flagged for a separate E1-native campaign (different trial budget than H0-H3).

### C. Pilot bias check for the 6.75-tick slippage number

`research/research_mgc_e2_microstructure_pilot.py` is the upstream source of the 6.75-ticks figure that drives all slippage-realism reasoning here. If the pilot:
- Selected sessions with known high slippage,
- Used a slippage metric that double-counts spread,
- Sampled during unusual market conditions (FOMC days, thin holidays),
...then the 6.75 figure is biased. **Must be adversarially audited before H0 uses it as a hypothesis parameter.** H0 pre-reg must include this audit step.

Concrete action: read pilot script + result JSONs (`research/output/mgc_e2_slippage_analysis.json`) before writing H0 pre-reg.

---

## 10 — What's left behind (nothing)

- [x] Prior MGC research inventory complete (§1)
- [x] Contract-size 10:1 handling verified in R-space (§2)
- [x] Data-era integrity verified post-Phase-2 (§3)
- [x] Drift check enforcement verified (§3.4)
- [x] Slippage realism gap documented — scheduled for debt-ledger (§4, task #13)
- [x] Volume handling verified symbol-local (§5)
- [x] Volume filter E2-exclusion verified (§5.5)
- [x] Zero MGC volume-filter strategies in validated_setups (§5.6)
- [x] Gold regime 2024-2026 context grounded (§6)
- [x] MinBTL budget computed for MGC 3.8yr horizon (§7)
- [x] My prior POV claims audited, 3 of 5 retracted/revised (§8)
- [x] Proposed tests pre-reg-ready (§9)
- [x] Self-audit corrections applied (2026-04-20 §9.5): slippage ratio precision (3.4× not 6.75×), MinBTL derivation softened, speculative 0.03R drop removed, three missing angles added
- [ ] Pilot bias check (read `research/research_mgc_e2_microstructure_pilot.py` + output JSONs) — PREREQUISITE for H0 pre-reg (§9.5.C)
- [ ] H1 pre-reg + execution (task #14)
- [ ] H0 pre-reg + execution (task #15)
- [ ] Debt-ledger slippage entry (task #13)

---

## Sources

**Internal (canonical):**
- `pipeline/cost_model.py:102-126, 145-152, 439-469` — cost specs + R math
- `pipeline/asset_configs.py:76-97, 319` — MGC launch date + Phase 2 relabel note
- `pipeline/data_era.py:114-148, 224-240` — micro vs parent classification
- `pipeline/build_daily_features.py:1470-1546` — rel_vol symbol-local computation
- `pipeline/check_drift.py:2102-2201` — micro-data drift check
- `trading_app/config.py:394-418, 682-732, 3560-3568` — requires_micro_data, VolumeFilter, E2 exclusion
- `docs/plans/2026-04-19-gc-mgc-handling-note.md` — canonical GC/MGC handling policy
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` — ORB premise grounding
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` — MinBTL theorem
- `docs/audit/results/2026-04-15-mgc-level-scan.md`, `2026-04-15-t0-t8-audit-mgc-level-cells.md`, `2026-04-19-gc-mgc-translation-audit.md`, `2026-04-19-mgc-path-accurate-subr-v1.md`, `2026-04-19-mgc-native-low-r-v1.md`
- `docs/audit/hypotheses/2026-04-19-mgc-orbg5-long-signal-only-shadow-v1.yaml` — active shadow track

**External (web, fetched 2026-04-20):**
- [CME Micro Gold Overview](https://www.cmegroup.com/markets/metals/precious/e-micro-gold.html)
- [CME 2026 Micro Metals Report (Jan)](https://www.cmegroup.com/newsletters/micro-gold-silver-and-copper-monthly-update/2026-01-micro-metals-products-update.html)
- [CME 2025 Micro Metals Report (Dec)](https://www.cmegroup.com/newsletters/micro-gold-silver-and-copper-monthly-update/2025-12-micro-metals-products-report.html)
- [WGC 2025 Central Banks Report](https://www.gold.org/goldhub/research/gold-demand-trends/gold-demand-trends-full-year-2025/central-banks)
- [WGC 2026 New CB Buyers](https://goldinvest.de/en/gold-remains-in-demand-world-gold-council-sees-new-central-banks-on-the-buyer-side-in-2026/)
- [BRICS+ reserves 2026](https://www.miningweekly.com/article/brics-plus-countries-increase-gold-reserves-to-more-than-6-000-t-2026-04-07)
- [Amundi 2025 Gold Structural](https://research-center.amundi.com/article/gold-beyond-records)
- [VanEck 2026 Gold Outlook](https://www.vaneck.com/us/en/blogs/gold-investing/gold-investing-outlook/)
