# Finite-Data Institutional Framework for ORB Futures Research

**Established:** 2026-04-07
**Authority:** Derived from literature extracts in `docs/institutional/literature/`. Supersedes any prior ad-hoc methodology notes in memory or CLAUDE.md for research decisions.

**Scope:** All strategy discovery, validation, and deployment decisions for ORB breakout trading on CME micro futures (MGC, MNQ, MES). Does not apply to feature engineering for signals already in the validated set.

> **⚠️ 2026-04-07 v2 AMENDMENTS — READ BEFORE APPLYING SECTIONS 2.2, 2.3, 4.4, 5.**
>
> After this framework was written, a parallel Codex audit (`docs/audits/2026-04-07-finite-data-orb-audit.md`) identified three places where the framework was ahead of what the live codebase has actually solved. Those findings were integrated into `pre_registered_criteria.md` as v2 amendments 2.1-2.5. **The criteria file is the binding current state; this framework file is the rationale backing.** Where the two disagree, follow the criteria file.
>
> Specific deltas to apply mentally when reading this framework:
> - **§ 2.2 DSR:** Framework says "binding rule: DSR > 0.95". Amendment 2.1 downgrades DSR to **cross-check only** until `N_eff` is formally solved in-repo (`strategy_validator.py:583-589, :1400-1453` explicitly removed DSR/FST as hard gates for this reason). DSR must still be computed and reported; it must not override BH-FDR or WFE as a deploy switch.
> - **§ 2.3 Chordia:** Framework says "t ≥ 3.00 OR theory". Amendment 2.2 refines to a **banded** threshold: t ≥ 3.79 flag / t ≥ 3.00 with pre-registered theory / t in [2.0, 3.0] passes only under strict co-requirements (pre-reg + theory + BH FDR + WFE ≥ 0.50 + 2026 OOS or N/A).
> - **§ 4.4 Holdout OOS:** Framework assumes 2026 is clean. Amendment 2.3 made this contingent on a pre-run policy declaration; **Amendment 2.7 (2026-04-08) declared Mode A operative** — sacred holdout window is 2026-01-01 onwards, currently ~3.2 months deep. The framework's § 4.4 OOS protocol is now fully active under Mode A. Every discovery run must use `--holdout-date 2026-01-01` (now CLI-enforced via `trading_app.holdout_policy.enforce_holdout_date()`). The 124 existing validated_setups are grandfathered as research-provisional per Amendment 2.4.
> - **§ 5 Currently deployed lanes:** Framework flags lanes against DSR > 0.95 and t < 3.00. Amendment 2.4 reclassifies the current 5 deployed MNQ/MGC lanes as **research-provisional + operationally deployable**, NOT production-grade institutional proof. Passing all 12 criteria remains the bar for production-grade.
> - **§ 7 check_drift.py new check:** ~~Blocked until e2-canonical-window-fix worktree merges.~~ **UNBLOCKED 2026-04-08:** the e2-fix merged in `8bc87f7` and the 6-stage holdout enforcement refactor (`81a7079`...`0edf8e8`) landed all four deferred items. New canonical source: `trading_app.holdout_policy`. New drift check: `check_holdout_policy_declaration_consistency` (#83). CLI enforcement: `strategy_discovery.main()`. Validator gate: `strategy_validator._check_mode_a_holdout_integrity()`.
>
> Amendment 2.5 adds a new rule not in this framework: execution overlays (calendar skip, ATR velocity, E2 timeout, market-state gating) must be reported separately from discovery-filter evidence. See `pre_registered_criteria.md` § v2 Amendment 2.5.
>
> **Amendment 2.6 (RESCINDED):** A brief Mode B declaration on 2026-04-07 was rescinded the next day per user correction. The Apr 2 16yr rebuild plan explicitly prescribed `--holdout-date 2026-01-01` and that discipline is restored under Amendment 2.7.

---

## 1. The problem we are solving

Our discovery pipeline historically brute-forced ~35,000+ (session × RR × entry_model × filter × filter_threshold) combinations on ~6 years of clean MNQ data (2.2 years actual MNQ + ~14 years NQ-parent proxy). Per Bailey et al (2013) Theorem 1 (see `literature/bailey_et_al_2013_pseudo_mathematics.md`), the MinBTL bound for 5 years of data is approximately 45 independent trials. We over-tested by ~600x. Per the False Strategy Theorem (see `literature/lopez_de_prado_bailey_2018_false_strategy.md`), the expected maximum Sharpe under zero true edge for K=35,000 trials is approximately 3.87. Our best deployed strategy has an annualized Sharpe of 1.23. **Our deployed strategies may contain real edge, but the discovery methodology alone cannot prove it.**

This framework corrects the discovery methodology while preserving any real edge that may exist in the currently deployed set. It does NOT claim the current 5 deployed lanes are invalid — it claims the statistical evidence for them is weaker than prior reports suggested, and establishes the procedure for stronger evidence going forward.

---

## 2. Core constraints derived from literature

### 2.1 MinBTL constraint (Bailey et al 2013)

Reference: `literature/bailey_et_al_2013_pseudo_mathematics.md` § Theorem 1.

For a discovery run testing N independent trials on T years of data, the minimum backtest length bound is:

```
MinBTL < 2·Ln[N] / E[max_N]²
```

where E[max_N] is the expected maximum Sharpe ratio under the null of zero true edge, approximated by Equation 1 of Bailey-Lopez de Prado 2014.

**Binding rule:** Before any discovery run, compute MinBTL for the intended N with E[max_N] set to the minimum Sharpe we would accept as evidence. If MinBTL > available clean data years, REDUCE N until the constraint holds.

### 2.2 Deflated Sharpe Ratio constraint (Bailey-LdP 2014)

Reference: `literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` § Equation 2.

Every strategy claimed as validated must compute DSR using the full Equation 2 formula:

```
DSR = Z[ ((ŜR - ŜR_0)·√(T-1)) / √(1 - γ̂₃·ŜR + (γ̂₄-1)/4·ŜR²) ]
```

where ŜR_0 is computed via Eq. 1 with the effective N̂ (not raw M) per Appendix A.3.

**Binding rule:** A strategy is only "validated" if DSR > 0.95.

### 2.3 Multiple testing threshold (Chordia et al 2018)

Reference: `literature/chordia_et_al_2018_two_million_strategies.md`.

For adaptive MHT methods appropriate to finance data (BH-FDR and FDP-StepM), the t-statistic threshold at 5% FDP is approximately 3.79. Conventional 1.96 is insufficient; even HLZ's 3.00 is slightly lenient.

**Binding rule:** BH-FDR at q=0.05 is the minimum filter. Strategies for deployment should additionally meet t ≥ 3.00 (HLZ threshold from `literature/harvey_liu_2015_backtesting.md`) OR have a strong pre-registered economic hypothesis.

### 2.4 Harvey-Liu profitability hurdle

Reference: `literature/harvey_liu_2015_backtesting.md` § Exhibit 4.

At our typical data horizon (~240 monthly obs equivalent for proxy-extended, ~26 months for clean MNQ), the BHY-adjusted profitability hurdle at 10% annualized volatility with 300 tests assumed is approximately 7.4% annualized return. This is the economic significance bar that survives multiple-testing correction.

**Binding rule:** Deployed strategies should project an annualized return (net of costs) above the BHY hurdle computed for the actual (N, T, volatility) of the candidate.

### 2.5 Theory-first principle (Lopez de Prado 2020)

Reference: `literature/lopez_de_prado_2020_ml_for_asset_managers.md` (verbatim extract committed 2026-04-07 in `0028333`, covering printed pp 1-23 / PDF pp 6-28 of the Cambridge Elements short-form Chapter 1 "Introduction").

Key principle from LdP 2020 § 1.10 Conclusions (printed p 22, verbatim): *"Successful investment strategies are specific implementations of general theories. An investment strategy that lacks a theoretical justification is likely to be false. Hence, a researcher should concentrate her efforts on developing a theory, rather than of backtesting potential strategies."*

Companion principle from § 1.2.1 Lesson 1 (printed p 3, verbatim): *"Contrary to popular belief, backtesting is not a research tool. Backtests can never prove that a strategy is a true positive, and they may only provide evidence that a strategy is a false positive."*

**Binding rule:** Every pre-registered hypothesis must cite specific economic or microstructural theory. Purely data-mined patterns without theoretical grounding are not eligible for validation.

### 2.6 Live monitoring (Pepelyshev-Polunchenko 2015)

Reference: `literature/pepelyshev_polunchenko_2015_cusum_sr.md`.

For any deployed strategy, the Shiryaev-Roberts procedure (Equation 11) provides multi-cyclic optimal drift detection:

```
R_n = (1 + R_{n-1})·Λ_n, R_0 = 0
Alarm when R_n ≥ A, where A is calibrated to ARL to false alarm ~ 60 trading days
```

**Binding rule:** All deployed strategies must have SR-procedure drift monitoring. On alarm: pause that strategy pending manual review.

---

## 3. Effective independent trials (N̂)

Reference: `literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` § Appendix A.3, Equation 9.

Our discovery tests are NOT independent. Many are correlated variations (same session different RR, same filter different session). The effective number of independent trials is:

```
N̂ = ρ̂ + (1 - ρ̂)·M
```

where M is raw trial count and ρ̂ is the average correlation between trial Sharpe ratios.

**Estimation procedure:** After any discovery run, compute the correlation matrix of per-strategy Sharpe ratios within the hypothesis family. Estimate ρ̂ as the average off-diagonal correlation. Apply Eq. 9 to get N̂. Use N̂ (not M) in all subsequent DSR and MinBTL calculations.

**Expected order of magnitude:** For our tight hypothesis families (e.g., OVNRNG across all sessions and RRs), ρ̂ likely ≈ 0.7-0.9, so N̂ is roughly 10-30% of M. For a 1,000-trial hypothesis family, expect N̂ in range [100, 300].

---

## 4. Practical methodology for future discovery runs

### 4.1 Pre-registration phase (before any code runs)

1. **Write hypothesis file** using `hypothesis_registry_template.md`. Each hypothesis includes:
   - Name and numeric id
   - Economic theory basis (citation required)
   - Exact filter specification (which column, which threshold range)
   - Intended sessions, instruments, RR targets, entry models, confirm bars
   - Expected trial count for this hypothesis
   - Kill criteria (what outcome would refute it)
2. **Sum trial counts across hypotheses** to get total N.
3. **Compute MinBTL** with N, required E[max_N] = 0.5 annualized (conservative).
4. **Check MinBTL ≤ available clean data years**. If not, cut hypotheses until it does.
5. **Commit hypothesis file** to `docs/audit/hypotheses/` before running any discovery. This is the pre-registration.

### 4.2 Discovery phase

1. Modify `strategy_discovery.py` to accept `--hypotheses docs/audit/hypotheses/YYYY-MM-DD.yaml`.
2. Run discovery with `--holdout-date 2026-01-01` (or appropriate holdout).
3. Discovery pipeline outputs to `experimental_strategies` as before but ONLY for pre-registered specifications.

### 4.3 Validation phase

For each strategy in `experimental_strategies`:
1. Compute sample size N_trades.
2. Compute raw annualized Sharpe.
3. Compute N̂ effective (from hypothesis family correlation).
4. Compute DSR per Bailey-LdP Eq. 2.
5. Compute walk-forward efficiency (WFE) per existing pipeline.
6. Compute bootstrap null p-value via permutation.
7. Apply all criteria in `pre_registered_criteria.md`.

### 4.4 Holdout OOS validation

For strategies passing validation:
1. Compute 2026 OOS performance (data held out from discovery).
2. Require OOS positive and OOS / IS degradation ≤ acceptable range.
3. Flag any strategy showing era-specific behavior (per era-split per `literature/harvey_liu_2015_backtesting.md` stability requirement).

### 4.5 Deployment phase

Only strategies passing Phase 4.4 may be deployed. Initial deployment:
- 1 contract per lane
- 1 copy per profile
- Shiryaev-Roberts drift monitor active
- Live paper or small-size live for 90 days before scaling

---

## 5. Handling the currently deployed 5 lanes

The 5 MNQ/MGC lanes deployed in `topstep_50k_mnq_auto` were selected via the old brute-force methodology. Per this framework, they are NOT automatically grandfathered.

**Procedure:**
1. Treat them as "provisional deployment" until re-validated under this framework.
2. Retroactively compute for each:
   - N̂ effective (from April rebuild correlation structure)
   - DSR per Eq. 2
   - Harvey-Liu BHY-adjusted hurdle comparison
   - Chordia t-statistic (from stored fdr_adjusted_p)
3. Flag any lane that fails DSR > 0.95 or t < 3.00.
4. Flagged lanes move to "monitoring only" (Shiryaev-Roberts active, size reduced).
5. Passing lanes continue deployment with monitoring.

Based on the April 7 audit, COMEX_SETTLE OVNRNG_100 is the only lane clearing Chordia's t ≥ 3.79. SINGAPORE_OPEN, EUROPE_FLOW, TOKYO_OPEN COST_LT filters are below. MGC ORB_G6 is marginal. This is an expected outcome — brute-force-selected strategies rarely survive institutional MHT thresholds.

---

## 6. What this framework does NOT do

- Does not ban backtesting. Backtests remain useful for falsification (per LdP 2020 Lesson 1). They are insufficient for positive validation alone.
- Does not require abandoning current deployed lanes. It requires re-validating them under proper criteria.
- Does not limit research creativity. It limits brute-force enumeration. Creative hypothesis generation is encouraged; each hypothesis just needs a theoretical basis and pre-registration.
- Does not apply retroactively to already-killed NO-GO hypotheses. Those remain dead.

---

## 7. Auto-invocation points

For this framework to bind decisions, it must be invoked automatically at key moments. The integration points are:

1. `CLAUDE.md` Document Authority section — references `docs/institutional/` for research methodology
2. `.claude/rules/research-truth-protocol.md` — requires reading `pre_registered_criteria.md` before any discovery
3. `.claude/rules/institutional-rigor.md` — already updated with extract-before-dismiss; should also reference this framework
4. `pipeline/check_drift.py` — new check to verify validated_setups meet locked criteria
5. A hook on `strategy_discovery.py` edits to warn if pre-registration file is missing

---

## 8. Open questions for future research

The following are explicitly NOT resolved by this framework and require targeted work:

1. What is the actual ρ̂ for our discovery hypothesis space? Must be measured empirically.
2. Is the 2026 "hot regime" for Asian sessions a real edge or a data-source transition artifact (NQ parent → MNQ micro)?
3. Does the Shiryaev-Roberts procedure with score function (Eq. 17-18) work for our per-trade R-multiple distribution or do we need a different score?
4. What is the proper hypothesis budget for ORB discovery on 2 years of real MNQ data? (Lower bound from MinBTL; practical minimum from having enough candidates to find one.)
5. How should the CPCV methodology (mentioned in LdP 2020) be adapted for time-series with regime changes?

Each of these is a candidate for a pre-registered research question once the framework itself is in place.

---

## Related files

- `README.md` — master index for this directory
- `HANDOFF.md` — session state as of 2026-04-07
- `literature/` — verbatim source extracts
- `pre_registered_criteria.md` — LOCKED thresholds (v2 locked 2026-04-07 in `0028333`, with 5 amendments incorporating Codex audit findings)
- `hypothesis_registry_template.md` — template for pre-registration (committed in `0028333`)
- `../audit/hypotheses/README.md` — workflow + infrastructure directory (committed in `7bc47a7`)
- `../audits/2026-04-07-finite-data-orb-audit.md` — Codex audit that produced the v2 amendments (committed in `390e408`)
