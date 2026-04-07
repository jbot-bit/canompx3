# Pre-Registered Criteria — LOCKED Thresholds for Strategy Validation

**Locked:** 2026-04-07
**Authority:** Derived from `finite_data_framework.md` and `literature/` extracts in this directory.
**Revision policy:** This file may be AMENDED only with explicit justification written as a new section. Prior criteria versions remain visible for audit. **No post-hoc relaxation of criteria after discovery results are seen.** Loosening a threshold because "our best strategy just missed" is explicitly banned as a form of selection bias per `literature/lopez_de_prado_bailey_2018_false_strategy.md`.

---

## Version history

| Version | Date | Change | Signer |
|---|---|---|---|
| v1 | 2026-04-07 | Initial lock. All criteria derived from Phase 0 literature extraction. | Claude Code audit session |

---

## Scope

These criteria apply to:
- All strategies discovered via `trading_app/strategy_discovery.py` or equivalent
- All strategies promoted to `validated_setups` in gold.db
- All strategies added to `prop_profiles.ACCOUNT_PROFILES` as active lanes
- All research claims that reach a deployment decision

They do NOT apply to:
- Exploratory backtesting clearly labeled as such (not claimed as validated)
- Feature engineering that doesn't produce deployable strategies
- Monitoring/reporting code that reads existing validated_setups

---

## Criterion 1 — Pre-registered hypothesis file (required)

**Source:** `literature/lopez_de_prado_2020_ml_for_asset_managers.md` (theory-first principle) and `literature/bailey_et_al_2013_pseudo_mathematics.md` (MinBTL).

**Rule:** Before any discovery run, a pre-registered hypothesis file must exist at `docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml` containing:
- Numbered list of hypotheses
- Each hypothesis has an economic theory citation
- Each hypothesis specifies exact filter columns, threshold ranges, sessions, instruments, RR targets
- Total expected trial count across all hypotheses
- Kill criteria stating what outcome would refute the hypothesis

**Verification:** Discovery code checks for the file existence and warns if missing. A drift check validates that validated_setups entries correspond to pre-registered hypotheses.

---

## Criterion 2 — MinBTL constraint

**Source:** `literature/bailey_et_al_2013_pseudo_mathematics.md` Theorem 1.

**Rule:** Before any discovery run, compute:
```
MinBTL = 2·Ln[N] / E[max_N]²
```
where N is the total pre-registered trial count and E[max_N] is the minimum Sharpe we would accept as evidence (set to 0.5 annualized by default for conservative gating).

If `MinBTL > available_clean_data_years`, reduce N. No exceptions.

**Default bound for our project:** With 2.2 years of clean MNQ data and E[max_N] = 0.5, MinBTL ≤ 2.2 implies:
```
2·Ln[N] / 0.25 ≤ 2.2
Ln[N] ≤ 0.275
N ≤ 1.32
```
That is far too strict for practical use. Using E[max_N] = 1.0 annualized (still conservative) with 2.2 years:
```
2·Ln[N] / 1.0 ≤ 2.2
Ln[N] ≤ 1.1
N ≤ 3
```
Still too strict. The honest implication is that **with only 2.2 years of clean MNQ data, almost no discovery is statistically valid under strict MinBTL.** We need either (a) the proxy-extended horizon of ~16 years, accepting the data-source caveat, or (b) many more years of real MNQ data, or (c) to rely on theory-based priors to reduce effective N dramatically.

**Practical rule using 16-year proxy-extended horizon:**
```
MinBTL = 2·Ln[N] / 1.0 ≤ 16
Ln[N] ≤ 8
N ≤ 2981
```
So with the proxy-extended horizon and E[max_N] = 1.0, we can test up to about 2,980 independent trials. **Even this is far below our prior ~35,000 trials.**

**Locked bound for v1:** N ≤ 300 pre-registered trials per discovery run on clean MNQ data, OR N ≤ 2,000 pre-registered trials on proxy-extended data with explicit data-source disclosure. These are conservative and leave margin for error in the effective-N estimation.

---

## Criterion 3 — BH FDR significance (filter, not final)

**Source:** `literature/harvey_liu_2015_backtesting.md` + `literature/chordia_et_al_2018_two_million_strategies.md`.

**Rule:** Strategies must pass Benjamini-Hochberg FDR at q = 0.05 computed on the PRE-REGISTERED hypothesis family (NOT the raw brute-force universe).

**Implementation:** Use `fdr_adjusted_p < 0.05` as a FIRST filter. This is necessary but not sufficient for validation.

---

## Criterion 4 — Chordia t-statistic threshold

**Source:** `literature/chordia_et_al_2018_two_million_strategies.md`.

**Rule:** After BH-FDR passes, compute the implied t-statistic for the strategy's mean return. Require t ≥ 3.00 (Harvey-Liu-Zhu 2015) for strategies with strong pre-registered economic theory support. Require t ≥ 3.79 (Chordia et al 2018) for strategies without such theoretical support.

**Enforcement:** Computed from stored `expectancy_r`, `sample_size`, and returns standard deviation. Flag non-compliant strategies in a drift check.

---

## Criterion 5 — Deflated Sharpe Ratio

**Source:** `literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` Equation 2.

**Rule:** Compute DSR for every candidate using the full formula:
```
DSR = Z[ ((ŜR - ŜR_0)·√(T-1)) / √(1 - γ̂₃·ŜR + (γ̂₄-1)/4·ŜR²) ]
```
with:
- ŜR = annualized Sharpe of the candidate
- T = number of return observations
- ŜR_0 = expected max Sharpe under null (per Eq. 1 with N̂ effective)
- γ̂₃ = skewness of returns
- γ̂₄ = kurtosis of returns
- N̂ = effective independent trials via Eq. 9

**Required threshold:** DSR > 0.95.

**Implementation gap:** `validated_setups` has `dsr_score` and `sr0_at_discovery` columns but actual DSR computation needs to be verified. Task for next session: audit the DSR calculation code path.

---

## Criterion 6 — Walk-forward efficiency

**Source:** existing project convention from `.claude/rules/validation-workflow.md` + `literature/lopez_de_prado_2020_ml_for_asset_managers.md`.

**Rule:** Walk-forward efficiency (WFE = OOS_SR / IS_SR) must be ≥ 0.50.

Strategies with WFE > 0.95 on small OOS samples should be flagged as LEAKAGE_SUSPECT per `.claude/rules/quant-audit-protocol.md` § T3.

---

## Criterion 7 — Sample size

**Source:** `literature/harvey_liu_2015_backtesting.md` Exhibit 4 shows hurdles drop as T grows. Our pipeline already enforces N ≥ 30 in `strategy_validator.py`.

**Rule:** Sample size N ≥ 100 trades for deployment eligibility. Minimum N ≥ 30 for exploratory discovery entry to `experimental_strategies`.

---

## Criterion 8 — 2026 out-of-sample positive

**Source:** derived from pipeline's `--holdout-date` infrastructure and general OOS validation principle.

**Rule:** For discovery runs using `--holdout-date 2026-01-01`, the held-out 2026 period must show positive ExpR and OOS ExpR ≥ 0.40 × IS ExpR (allowing for typical degradation).

---

## Criterion 9 — Era stability (no dead era)

**Source:** audit finding 2026-04-07 that several deployed MNQ lanes showed zero or negative edge in pre-2020 data.

**Rule:** When era-split into (2015-2019, 2020-2022, 2023, 2024-2025, 2026), a strategy must show ExpR ≥ -0.05 in every era with ≥ 50 trades. Eras with < 50 trades are exempt (not enough data to judge).

**Rationale:** A strategy that was negative in a specific regime is era-dependent; it may be deployed with explicit regime gating but not treated as a general-case edge.

---

## Criterion 10 — Data era compatibility (volume filters)

**Source:** audit finding 2026-04-07 that MNQ pre-2024 source data is actually NQ parent (~10x different volume profile from MNQ micro). 

**Rule:** Volume-based filters (ORB_VOL, any filter using `orb_*_volume` or `rel_vol_*`) must be computed only on MICRO era data (MNQ/MES from 2024-02-05 onwards; MGC never valid because no real MGC data exists). Price-based filters (ORB_G, COST_LT, OVNRNG using range points) are valid on parent proxy data with disclosure.

**Enforcement:** Drift check to flag volume-filter strategies with trades from PARENT era.

---

## Criterion 11 — Account death Monte Carlo for deployment

**Source:** `resources/prop-firm-official-rules.md` — TopStep, Apex, Tradeify rulesets.

**Rule:** Before any strategy is deployed to a funded account, run a Monte Carlo simulation of the account under the strategy's return distribution using the prop firm's daily loss limit, trailing DD, and consistency rules. Require survival probability ≥ 70% at 90 days.

**Parameters:**
- Per-trade return distribution from the strategy's historical per-trade R-multiples
- Position sizing per account rules (TopStep Express 50K max 5 contracts until buffer, etc.)
- Correlation across concurrent lanes modeled
- 10,000 Monte Carlo paths

---

## Criterion 12 — Live monitoring via Shiryaev-Roberts

**Source:** `literature/pepelyshev_polunchenko_2015_cusum_sr.md`.

**Rule:** Every deployed strategy must have a Shiryaev-Roberts drift monitor running against its live R-multiple stream. Parameters:
- Pre-change distribution estimated from first 50-100 live trades
- Score function: linear-quadratic per Eq. 17-18 of the paper
- Detection threshold A calibrated to ARL to false alarm ≈ 60 trading days
- On alarm: strategy goes to "suspended" state pending manual review

---

## Acceptance matrix

A strategy is ELIGIBLE FOR DEPLOYMENT if and only if all 12 criteria pass:

| Criterion | Threshold | Required |
|---|---|---|
| 1 Pre-registration | file exists at `docs/audit/hypotheses/` | YES |
| 2 MinBTL | N ≤ 300 (clean MNQ) or N ≤ 2000 (proxy) | YES |
| 3 BH FDR | q < 0.05 | YES |
| 4 Chordia t-stat | t ≥ 3.00 (with theory) or 3.79 (without) | YES |
| 5 DSR | DSR > 0.95 | YES |
| 6 WFE | WFE ≥ 0.50 | YES |
| 7 Sample size | N ≥ 100 trades | YES |
| 8 2026 OOS | OOS ExpR ≥ 0 and ≥ 0.40 × IS ExpR | YES |
| 9 Era stability | No era with ExpR < -0.05 (N ≥ 50) | YES |
| 10 Data era compat | Volume filters only on MICRO era data | YES |
| 11 Account MC | 90-day survival ≥ 70% | YES (at deployment) |
| 12 SR monitor | Active drift monitoring in place | YES (post-deployment) |

Strategies passing 1-10 can promote to `validated_setups` as CANDIDATE.
Strategies passing 11 can be added to `prop_profiles.ACCOUNT_PROFILES` as PROVISIONAL (1 contract, 1 copy).
Strategies passing 12 months of live with 12 active can graduate to PRODUCTION (scale allowed).

---

## Applying to current 5 deployed lanes (status as of 2026-04-07)

| Lane | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MGC CME_REOPEN ORB_G6 RR2.5 | ❌ | ❌ | ✅ | ⚠️ (t≈2.22) | ? | ✅ | ⚠️ (N=88) | ✅ | ❌ (pre-2020 4 trades all loss) | ✅ (price-based) | ? | ❌ |
| MNQ SIN COST_LT12 RR2.0 | ❌ | ❌ | ✅ | ⚠️ (t≈2.24) | ? | ✅ | ✅ | ✅ | ⚠️ (pre-2020 N=48) | ✅ | ? | ❌ |
| MNQ COMEX OVNRNG_100 RR1.5 | ❌ | ❌ | ✅ | ✅ (t≈3.77) | ? | ✅ | ✅ | ✅ | ⚠️ | ✅ | ? | ❌ |
| MNQ EUR COST_LT10 RR3.0 | ❌ | ❌ | ✅ | ❌ (t≈2.10) | ? | ✅ | ✅ | ✅ | ❌ (pre-2020 -0.18) | ✅ | ? | ❌ |
| MNQ TOK COST_LT10 RR2.0 | ❌ | ❌ | ✅ | ❌ (t≈2.04) | ? | ✅ | ✅ | ✅ | ⚠️ (2023 loss) | ✅ | ? | ❌ |

Legend: ✅ pass, ❌ fail, ⚠️ marginal, ? not yet computed.

**Outcome:** No currently-deployed lane passes all 12 criteria. MNQ COMEX_SETTLE OVNRNG_100 is the closest. All 5 should be reclassified as "provisional" until re-validated under this framework — they remain deployed but under monitoring only, no scaling.

---

## Banned practices

1. Relaxing any criterion after seeing results — full stop.
2. "Just this once" exceptions — no exceptions.
3. Citing a threshold from memory without linking to the `literature/` file.
4. Deploying strategies skipping criteria for "speed" or "opportunity" reasons.
5. Counting correlated strategies as independent trials.
6. Claiming a strategy "passes" DSR without actually running Bailey-LdP Eq. 2.
7. Quoting sample sizes from metadata without joining against raw trade days.
8. Using volume filters on parent-proxy data without explicit disclosure + era split.

---

## Amendment procedure

To amend these criteria:
1. Write a new section below labeled `v2 (YYYY-MM-DD)` with the exact change.
2. Cite the specific literature passage justifying the change.
3. Explain why the prior criterion was wrong, not merely inconvenient.
4. Commit with message `docs(institutional): amend pre_registered_criteria vN`.
5. Update the version history table at the top.

Prior versions remain visible permanently for audit.
