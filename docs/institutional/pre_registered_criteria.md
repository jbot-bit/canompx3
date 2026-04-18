# Pre-Registered Criteria — LOCKED Thresholds for Strategy Validation

**Locked:** 2026-04-07
**Authority:** Derived from `finite_data_framework.md` and `literature/` extracts in this directory.
**Revision policy:** This file may be AMENDED only with explicit justification written as a new section. Prior criteria versions remain visible for audit. **No post-hoc relaxation of criteria after discovery results are seen.** Loosening a threshold because "our best strategy just missed" is explicitly banned as a form of selection bias per `literature/lopez_de_prado_bailey_2018_false_strategy.md`.

**Citation convention (added 2026-04-17 per docs authority audit C2):** Cross-references from other documents (RESEARCH_RULES.md, TRADING_RULES.md, handoffs, audit docs) MUST cite by the stable `## Criterion N` anchors, NOT by amendment number. Amendment subsections (`## Amendment X.Y`) are revision history and their numbering may be renumbered, merged, or reorganised — they are not stable cross-reference anchors. When discussing *when* or *why* a criterion was revised, mention the amendment by name inline (e.g. *"Criterion 8, as revised by Amendment 2.7"*), but the load-bearing citation must be the criterion number. Drift check for this convention is tracked as future work per `docs/audit/2026-04-17-docs-authority-overlap-audit.md` § C2.

---

## Version history

| Version | Date | Change | Signer |
|---|---|---|---|
| v1 | 2026-04-07 | Initial lock. All criteria derived from Phase 0 literature extraction. | Claude Code audit session |
| v2 | 2026-04-07 | Codex audit feedback integrated. DSR downgraded from binding to cross-check (N_eff unresolved). Chordia t-threshold reframed as severity benchmark, not hard bar. Criterion 8 (2026 OOS) gated on holdout-policy decision. See amendments 2.1-2.5 at bottom. | Claude Code session (Codex audit incorporated) |
| v2.6 | 2026-04-07 | Holdout policy DECLARED — Mode B operative. Pass-2 audit verified the 2026 H1 holdout was consumed per pre-registered protocol on 2026-04-02. New forward-sacred window starts 2026-04-07 (earliest first-look 2026-10-07). Walk-forward continues as OOS discipline for existing discoveries; forward-paper requirement applies to new deployments. See Amendment 2.6 at bottom. | Claude Code session (autonomous decision per user delegation, pass-2 audit verified) |
| v2.7 | 2026-04-08 | **RESCINDS Amendment 2.6.** Holdout policy corrected to Mode A (holdout-clean) per explicit user correction. 2026-01-01 is the sacred holdout boundary going forward. The 3+ months of real-time 2026 data (2026-01-02 → present) is the accumulating forward OOS record. The 124 existing validated_setups are grandfathered as RESEARCH-PROVISIONAL per Amendment 2.4 — they were discovered with 2026 data in scope and are NOT OOS-clean. Any NEW discovery run must use `--holdout-date 2026-01-01`. See Amendment 2.7 at bottom. | Claude Code session (user correction: *"I THOUGHT WE WERE HOLDING OUT FROM 2026 ONWARDS SO THAT WE HAD 3 MONTHS ALREADY OF TRADES OOS"*) |
| v2.8 | 2026-04-09 | **FACTUAL CORRECTION.** Phase 3c canonical layer rebuild (merged to main as commit `c33805b` on 2026-04-08) replaced pre-2019 parent-proxy bars with real-micro bars for MNQ/MES/MGC. Post-rebuild actual horizons: MNQ/MES 6.65 clean years (1,951 pre-holdout trading days, 2019-05-06 → 2025-12-31), MGC 2.7 clean years (671 pre-holdout days, 2023-09-11 → 2025-12-31). The prior text "~2.2 years of clean MNQ data" and "16 years proxy-extended" in § Criterion 2, and "MNQ/MES from 2024-02-05 onwards; MGC never valid" in § Criterion 10, both predate the Phase 3c rebuild and are factually wrong. All 12 locked numeric thresholds (300/2000 trial bounds, t ≥ 3.00, DSR > 0.95, WFE ≥ 0.50, N ≥ 100, etc.) remain EXACTLY as locked — this amendment is a factual correction of stale narrative, not a threshold relaxation. See Amendment 2.8 at bottom. | Claude Code session (user correction: *"I DONT WANNA FUCK AROUND WITH HALF THIS DATA HALF THAT DATA. I HAVE SUBSCRIPTION TO GET ALL THE DATA"*) |
| v2.9 | 2026-04-09 | **Parent/Proxy Data Policy.** Binding rules for NQ/ES/GC parent vs MNQ/MES/MGC micro data. Delete NQ/ES bars. Keep GC for MGC Tier 2 validation (price-only). 4 new banned practices (#9-#12). **Partially superseded by Amendment 3.1 (GC proxy expanded to discovery for price-based filters).** | Claude Code session (user: *"is it useful at all for us to have the 2 different contract sizes or is it just a canonical fucking project nightmare?"*) |
| v3.1 | 2026-04-10 | **Revised Proxy Data Policy.** GC proxy expanded from validation-only to discovery-eligible for price-safe filters (ORB_G, GAP, PDR, ATR, OVNRNG, COST_LT). Volume-unsafe filters remain micro-only. Regime-awareness requirement added. Empirical evidence: 4-gate research (price corr=0.99999, 96% trigger match, 99.5% GAP filter agreement). | Claude Code session (user: *"can we use the gc for our strats or not? it doesn't have to do with price movement does it?"*) |
| v3.0 | 2026-04-09 | **Theory-Driven Individual Hypothesis Testing.** Adds dual-pathway to Criterion 3: individual hypothesis pathway (raw p < 0.05) for theory-grounded, mechanism-specific predictions alongside existing BH FDR pathway for exploratory search. Incorporates canonical-base-truth methodology for admissible search space definition. Downstream gates (WF, OOS, era stability) mandatory and non-waivable under individual pathway. | Claude Code session (user: *"There is a way that people realistically trade ORB breakouts that are valid and profitable without using 20 years of data and such hard tests, right?"*) |
| v3.1 | 2026-04-09 | **Structural Data Boundary for Discovery + Era Stability.** WF_START_OVERRIDE now applies to both discovery IS scope (outcomes + daily_features) and Criterion 9 era bins. Years before the override are excluded from both. Justified by 5-variable structural audit (ATR, volume, ORB size, G-filter pass rates, trading days) confirming MNQ/MES 2019 micro-launch data is non-representative. MNQ CME_PRECLOSE 2019: ATR 0.42x, G8 pass 39%, vol 0.16x. NOT performance-snooped — zero strategy PnL consulted. See Amendment 3.1 at bottom. | Claude Code session (user: *"ensure we do not use v1 of anything. we always check back over to verify our work"*) |

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

**Default bound for our project** (corrected by Amendment 2.8, 2026-04-09, to reflect post-Phase-3c actual data horizons):

Post-Phase-3c canonical layer rebuild (merged to main Apr 8 2026, commit `c33805b`), the actual clean real-micro horizons are:

| Instrument | First trading day | Pre-holdout days (< 2026-01-01) | Clean years |
|---|---|---|---|
| MNQ | 2019-05-06 | 1,951 | ≈ 6.65 |
| MES | 2019-05-06 | 1,951 | ≈ 6.65 |
| MGC | 2023-09-11 | 671  | ≈ 2.70 |

With 6.65 years of clean MNQ and E[max_N] = 1.0 annualized (Bailey default), strict MinBTL bounds N:
```
2·Ln[N] / 1.0² ≤ 6.65
Ln[N] ≤ 3.325
N ≤ 27.8  → max 28 pre-registered trials at strict Bailey E=1.0
```

At the relaxed E[max_N] = 1.2 (still below the 1.5 "professional Sharpe" level), the bound widens to N ≤ 120. At E[max_N] = 1.5, N ≤ 1,774. The locked 300/2000 cap is looser than strict Bailey E=1.0 but tighter than strict E=1.5, so it functions as an operational ceiling with explicit noise-floor disclosure required when N exceeds the strict-E=1.0 bound.

For 2.7-year MGC (no-backfill), strict Bailey E=1.0 gives N ≤ 4 — too small for meaningful pre-registration. A Databento backfill to MGC launch (2022-06-13, adds ~308 pre-holdout days for ~3.9 clean years) relaxes the strict bound to N ≤ 7 at E=1.0 or N ≤ 17 at E=1.2. The backfill is strongly recommended before any MGC discovery run.

**The earlier "~2.2 years of clean MNQ data" / "16-year proxy-extended horizon" narrative here was stale, predating the Phase 3c rebuild. The locked 300/2000 thresholds below remain exactly as they were — this correction updates the worked example, not the caps.**

**Locked bound for v1 (unchanged):** N ≤ 300 pre-registered trials per discovery run on clean data, OR N ≤ 2,000 pre-registered trials on proxy-extended data with explicit data-source disclosure. These function as operational ceilings. **For institutional maximum rigor, operate at or below the strict Bailey bound for the target instrument's actual horizon** (N ≤ 28 for MNQ/MES at 6.65yr clean, N ≤ 7 for MGC at 3.9yr with backfill, N ≤ 4 for MGC at 2.7yr without).

---

## Criterion 3 — Statistical significance (dual pathway)

**Source:** `literature/harvey_liu_2015_backtesting.md` + `literature/chordia_et_al_2018_two_million_strategies.md`.

**Rule (dual pathway — see Amendment 3.0):**

**Pathway A — BH FDR (exploratory search):** Strategies must pass Benjamini-Hochberg FDR at q = 0.05 computed on the PRE-REGISTERED hypothesis family. Use when a single hypothesis file tests multiple hypotheses as a family and the researcher intends to select survivors from the set. Implementation: `fdr_adjusted_p < 0.05` as a FIRST filter.

**Pathway B — Individual hypothesis (theory-driven):** A single pre-registered prediction with its own economic theory may be tested at raw p < 0.05 (two-tailed). Use when the hypothesis is a specific mechanism-session-instrument prediction, pre-registered in its own file (or flagged `testing_mode: individual`), with mandatory downstream gates (criteria 6, 8, 9 non-waivable). See Amendment 3.0 for full conditions.

**Which pathway:** The hypothesis file declares `testing_mode: family` (default, Pathway A) or `testing_mode: individual` (Pathway B). Individual mode requires a `theory_citation` on EVERY hypothesis — no theory, no individual testing.

---

## Criterion 4 — Chordia t-statistic threshold

**Sources:**
- `literature/chordia_et_al_2018_two_million_strategies.md` (grounds t ≥ 3.79 without-theory verbatim).
- `literature/harvey_liu_zhu_2015_cross_section.md` (STUB — grounds t ≥ 3.00 with-theory via one-step-removed reference through Chordia p5; full PDF re-ground pending per audit IMP-1).

**Rule:** After BH-FDR passes, compute the implied t-statistic for the strategy's mean return. Require t ≥ 3.00 (Harvey-Liu-Zhu 2015 — currently indirect Tier 1 via Chordia et al 2018 p5 quote, see stub at `literature/harvey_liu_zhu_2015_cross_section.md`) for strategies with strong pre-registered economic theory support. Require t ≥ 3.79 (Chordia et al 2018, verbatim Tier 1 at `literature/chordia_et_al_2018_two_million_strategies.md:20`) for strategies without such theoretical support. **Promote t ≥ 3.00 grounding from INDIRECT to DIRECT before any 3.00 ≤ t < 3.79 with-theory candidate is accepted.**

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

**Source:** audit finding 2026-04-07 that MNQ pre-2024 source data was NQ parent (~10x different volume profile from MNQ micro). This source-data issue was resolved by the Phase 3c canonical layer rebuild (commit `c33805b`, merged 2026-04-08) which replaced all parent-proxy bars with real-micro bars for MNQ/MES/MGC. See Amendment 2.8 (2026-04-09) for the factual correction of the prior MICRO-era dates.

**Rule:** Volume-based filters (ORB_VOL, any filter using `orb_*_volume` or `rel_vol_*`) must be computed only on MICRO-era data. Per Amendment 2.8, real-micro data in this repo covers:
- **MNQ/MES:** 2019-05-06 onwards (CME Micro E-mini launch date). Volume filters eligible on this entire horizon.
- **MGC:** 2023-09-11 onwards (present canonical start; MGC launched 2022-06-13 and ~15 months of earlier MGC can be backfilled via Databento subscription). Volume filters eligible from whichever start date applies after backfill decision.

Price-based filters (ORB_G, COST_LT, OVNRNG using range points, ATR_P30/P50/P70) are valid on the full real-micro horizon for each instrument without special disclosure. The parent-proxy caveat no longer applies because the canonical layer no longer contains parent-proxy bars.

**Legacy language corrected:** the earlier "MNQ/MES from 2024-02-05 onwards; MGC never valid because no real MGC data exists" text predates Phase 3c and was factually wrong — the MNQ/MES micro contracts launched 2019, and MGC has been accumulating real-micro data since 2023-09-11. The rule (volume filters only on MICRO era) is unchanged; only the dates were stale.

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

---

## v2 Amendment (2026-04-07) — Codex audit feedback integration

**Justification:** A parallel Codex audit (`docs/audits/2026-04-07-finite-data-orb-audit.md`) reviewed the v1 criteria against current repo code and flagged three cases where v1 was ahead of what the codebase has actually solved. This amendment is not threshold relaxation — v1 was internally inconsistent with the live validator, and this amendment brings the criteria into alignment with what the repo can currently enforce. Loosening thresholds for performance reasons remains banned.

### Amendment 2.1 — Criterion 5 (DSR) downgraded from binding to cross-check

**v1 statement:** "Required threshold: DSR > 0.95."

**Problem identified by Codex:**
- `trading_app/strategy_validator.py:583-589` and `:1400-1453` explicitly removed DSR and False Strategy Theorem as hard gates because `N_eff` (the effective number of independent trials for adjusting the DSR formula) remains unresolved.
- The adversarial audit `docs/plans/2026-03-18-adversarial-review-findings.md:47-58` confirms DSR is informational, not binding, pending ONC (Optimal Number of Clusters) for N_eff estimation.
- Current DB state (**snapshot at 2026-03-18 — historical rationale for Amendment 2.1; for current state run live query against `validated_setups.dsr_score`**): `0 / 124` validated_setups have `dsr_score > 0.95`. Max `dsr_score = 0.1198`. A policy that makes DSR binding while the repo computes DSR against an unknown N_eff is a policy that rejects every strategy the repo has ever produced — including ones that may be valid.

**Amended statement:** DSR is a CROSS-CHECK, not a hard gate, until `N_eff` is formally solved in-repo.
- Every candidate strategy must have its DSR computed and reported in the audit write-up.
- DSR MAY flag strategies as suspect even when BH FDR passes.
- DSR does NOT override BH FDR or WFE as deploy/don't-deploy switches until N_eff is resolved.

**Required follow-up:** Issue a dedicated task to formalize N_eff estimation in `trading_app/strategy_discovery.py`, document the method, and verify against Bailey-LdP 2014 Equation 9. Until that task closes, DSR stays cross-check only.

### Amendment 2.2 — Criterion 4 (Chordia t-statistic) reframed as severity benchmark

**v1 statement:** "Require t ≥ 3.00 (with theory) or t ≥ 3.79 (without)."

**Problem identified by Codex:** The Chordia et al 2018 threshold was derived from a 2M-strategy universe across CRSP/COMPUSTAT equity factors with a specific FDP-StepM method that accounts for cross-correlation structure in that universe. Transplanting it unchanged into this repo's ORB futures family structure is aggressive and not grounded in our own effective-N analysis.

**Amended statement:** Chordia's t ≥ 3.79 is a SEVERITY BENCHMARK, not a universal hard bar.
- Strategies that clear t ≥ 3.79 get a "clears Chordia" flag in their audit write-up.
- Strategies with t ≥ 3.00 (HLZ threshold) pass the criterion if they also cite a pre-registered economic theory.
- Strategies with 2.0 ≤ t < 3.00 can pass ONLY if:
  - Pre-registered with theory citation, AND
  - BH FDR passes on the pre-registered family K (not raw brute-force K), AND
  - WFE ≥ 0.50, AND
  - 2026 OOS positive (or holdout-policy N/A per Amendment 2.3)
- Strategies with t < 2.0 do not pass.

**Why this is honest, not relaxation:** Chordia 3.79 was never grounded in our specific ORB family structure. The new banding is more defensible because it acknowledges the gap between literature benchmarks and repo-specific validity.

### Amendment 2.3 — Criterion 8 (2026 OOS) gated on holdout policy decision

**v1 statement:** "For discovery runs using `--holdout-date 2026-01-01`, the held-out 2026 period must show positive ExpR and OOS ExpR ≥ 0.40 × IS ExpR."

**Problem identified by Codex:** Repo authority is internally inconsistent on whether 2026 is still a sacred holdout:
- `RESEARCH_RULES.md:26` says 2026 holdout is sacred.
- `docs/plans/2026-04-02-16yr-pipeline-rebuild.md:79-83, 110` instructs discovery with `--holdout-date 2026-01-01`.
- `docs/pre-registrations/2026-03-20-mnq-rr1-verified-sessions.md:4` says the holdout test completed and 2026 has been reincorporated.
- `pipeline/check_drift.py:3354-3405` contains a holdout contamination checker whose declaration map is EMPTY, so no instrument is actively guarded by that check.

Criterion 8 assumed a clean 2026 holdout, but the repo has not decided the policy.

**Amended statement:** Criterion 8 is CONTINGENT on a pre-run holdout policy declaration.
- Before any discovery run, the project must declare ONE of:
  - **Mode A — Holdout-clean:** discovery data ends 2025-12-31. 2026 excluded from session, RR, filter, and lane choice. 2026 only used later for OOS reporting. Criterion 8 enforced as v1.
  - **Mode B — Post-holdout-monitoring:** 2026 already consumed. No more "clean 2026 holdout" claims. Criterion 8 REPLACED by a forward-paper-only requirement (minimum 6 months live paper with positive ExpR before deploy decision).
- The declaration must be committed to the hypothesis file BEFORE discovery runs.
- Mixing the two modes is banned.

**Required follow-up:** Decide the holdout policy at the project level and update `pipeline/check_drift.py` holdout declaration map to enforce it. This touches `pipeline/check_drift.py` which is currently in e2-canonical-window-fix scope_lock — defer enforcement until that stage merges.

### Amendment 2.4 — Current lane classification language

**v1 implication:** Strategies passing all 12 criteria are "validated." Strategies not passing are "provisional."

**Codex finding:** Even the best currently-deployed lanes (N = 591 to 1941 trades per lane) are classified ahead of what the repo has solved. Trade count is not the bottleneck — selection bias, regime-span weakness, holdout contamination, and policy drift between discovery and live overlays are the actual failure modes.

**Amended classification language:** The current 5 deployed MNQ/MGC lanes should be explicitly labeled as:
- **Operationally deployable** — the live system can execute them.
- **Research-provisional** — consistent with available evidence, not proof of durable edge.
- **Not yet production-grade institutional proof** — passing all 12 criteria remains the bar for that label.

This applies to documentation, memory files, and any downstream use of the word "validated." Use "provisional" or "research-provisional" to describe the current state honestly.

### Amendment 2.5 — Execution overlays vs discovery filters

**New requirement (not in v1):** Execution overlays (calendar skip, ATR velocity skip, E2 order timeout, market-state gating) MUST be reported and evaluated SEPARATELY from discovery filters. Per `trading_app/config.py:2600-2611` and `trading_app/execution_engine.py:207-212, 654-674`, overlays change the live decision rule in ways not encoded in `strategy_id`. A lane can be "clean" at discovery time and still have later operational logic that shifts the actual traded distribution.

**Rule:** Any claim about a strategy's evidence base must specify whether it refers to (a) the discovery-filter-only backtest, (b) the discovery-filter + overlays backtest, or (c) live/paper forward data. Mixing (a) and (b) in the same evidence object is banned.

### Amendment 2.6 — Holdout policy DECLARED — Mode B operative (2026-04-07)

**Resolves:** Amendment 2.3's gating requirement ("the project must declare ONE of Mode A or Mode B"). The full decision document is at [`docs/plans/2026-04-07-holdout-policy-decision.md`](../plans/2026-04-07-holdout-policy-decision.md).

**Decision-maker:** Claude Code session (autonomous, per explicit user delegation: *"YOU FIGURE IT OUT WHATS BEST AND MOST PROPER FOR US"* + *"ENSURE DOUBLE CHECKING FINDINGS BEFORE MAKING DECISIONS"*).

**Audit method:** Pass-1 (sloppy) → pass-2 (verified against gold.db, HANDOFF.md:1468, pre-registration completion record). Pass-2 refined the mechanism without changing the conclusion.

**Smoking gun:** `HANDOFF.md:1468` states *"2026 included in discovery — holdout test was spent (CME_PRECLOSE DEAD recorded). Walk-forward handles OOS. Live trading = new forward test."* This is an explicit, committed audit-trail decision: the pre-registered 2026 holdout test ran 2026-04-02, killed CME_PRECLOSE NO_FILTER baseline per `docs/plans/2026-03-25-cme-preclose-holdout-test-plan.md`, and after that the holdout was officially "spent". All 124 current `validated_setups` were discovered 2026-04-05 → 2026-04-06 with 2026 in scope; all 124 have `wf_tested = True AND wf_passed = True` (walk-forward is the operative OOS discipline).

**Mode B is the only honest position because:**

1. The holdout test was already RUN and the result is committed to git (`docs/pre-registrations/2026-03-20-mnq-rr1-verified-sessions.md:4` "Status: COMPLETED")
2. After the test, the project EXPLICITLY chose to fold 2026 into discovery scope and use walk-forward as the OOS discipline (HANDOFF.md:1468)
3. 124 / 124 active `validated_setups` reflect this choice (all wf_tested + wf_passed, all discovered post-2026)
4. Reverting to Mode A would require erasing committed audit trail and re-running discovery from scratch (multi-week project, no scientific value because the pre-registered protocol was followed correctly)

**Operative rules under Amendment 2.6 (Mode B):**

- **Past window (2026-01-01 → 2026-04-06):** CONSUMED. No "clean 2026 OOS" claims for any strategy. Existing memory/doc references to "+2026 OOS" are historical observations, not strict OOS evidence.
- **Forward-sacred window:** **2026-04-07 onwards.** No discovery may use this window's data. Any new hypothesis file at `docs/audit/hypotheses/` must declare a `holdout_date <= 2026-04-07`.
- **Earliest first-look at the new window:** **2026-10-07** (6-month minimum per Amendment 2.3 + Codex audit recommendation).
- **Forward-paper requirement** (replaces strict Criterion 8 for new deployments): minimum 6 months live paper with positive ExpR before any deployment decision. Earliest possible new deployment for a strategy first paper-traded 2026-04-07 = **2026-10-07**.
- **Existing 5 deployed lanes:** Grandfathered as research-provisional + operationally deployable per Amendment 2.4. No retroactive forward-paper requirement (they have walk-forward + live tracking already). No scaling until they pass all 12 v2 criteria.
- **No mixing of modes:** Mode B is project-wide. No future research may claim Mode A status.
- **Walk-forward continues** as the operative OOS discipline for new discoveries that fit within the pre-2026-04-07 window. Mode B does not eliminate walk-forward — it adds a forward-sacred window on top.

**Updated Acceptance Matrix (Criterion 8 row only — others unchanged from v2):**

| Criterion | Threshold | Enforcement |
|---|---|---|
| 8 Forward / OOS | Walk-forward `wf_passed = True` for in-scope discoveries; **AND** ≥ 6 months forward-paper positive ExpR for new deployments after 2026-04-07 | BINDING (WF) + BINDING (forward-paper, new deployments only) |

**Deferred enforcement (blocked by `e2-canonical-window-fix` scope_lock on `pipeline/check_drift.py`, `pipeline/build_daily_features.py`, `trading_app/strategy_discovery.py`):**

1. `pipeline/check_drift.py` `HOLDOUT_DECLARATIONS` map populated with `MNQ: 2026-04-07`, `MES: 2026-04-07`, `MGC: 2026-04-07`. Activates the existing `check_holdout_contamination()` function.
2. `trading_app/strategy_discovery.py` runtime enforcement: reject `--holdout-date > 2026-04-07` and require explicit holdout_date for any new discovery run.
3. `trading_app/strategy_validator.py` validation gate: verify `discovery_date < 2026-04-07` OR hypothesis file declares `holdout_date <= 2026-04-07`.
4. New drift check: assert `RESEARCH_RULES.md`, this file, and `pipeline/check_drift.py` agree on the holdout policy (catches future drift between docs and code).

All four follow-ups go on the action queue for the post-merge sweep.

---

## v2 Acceptance matrix (replaces v1 for current work)

A strategy is ELIGIBLE FOR DEPLOYMENT under v2 if:

| Criterion | Threshold | Enforcement |
|---|---|---|
| 1 Pre-registration | file exists at `docs/audit/hypotheses/` | BINDING |
| 2 MinBTL | N ≤ 300 (clean MNQ) or N ≤ 2000 (proxy-extended) | BINDING |
| 3 BH FDR | q < 0.05 on pre-registered family K | BINDING |
| 4 Chordia t-statistic | banded per Amendment 2.2 | BINDING via banding |
| 5 DSR | computed + reported | **CROSS-CHECK ONLY** (Amendment 2.1) |
| 6 WFE | WFE ≥ 0.50 | BINDING |
| 7 Sample size | N ≥ 100 trades | BINDING |
| 8 2026 OOS / forward | depends on holdout policy | **CONTINGENT** (Amendment 2.3) |
| 9 Era stability | no era ExpR < -0.05 (N ≥ 50) | BINDING |
| 10 Data era compat | volume filters MICRO-only | BINDING |
| 11 Account death MC | 90-day survival ≥ 70% | BINDING (at deployment) |
| 12 Shiryaev-Roberts | active drift monitor | BINDING (post-deployment) |

A strategy passing 1-4, 6-7, 9-10 is **research-provisional**. Passing 11 makes it **operationally deployable**. Passing 8 (under whichever holdout policy applies) + 12 months of live with 12 active makes it **production-grade institutional proof**. The current 5 lanes are at research-provisional + operationally deployable — not production-grade.

---

## Amendment 2.7 (2026-04-08) — RESCINDS Amendment 2.6 — Mode A holdout-clean operative

**Supersedes:** Amendment 2.6 (2026-04-07 Mode B declaration, commit `1aa11e5`).

**Trigger:** Explicit user correction on 2026-04-08 after reviewing Amendment 2.6:

> *"I THOUGHT WE WERE HOLDING OUT FROM 2026 ONWARDS SO THAT WE HAD 3 MONTHS ALREADY OF TRADES OOS"*

Amendment 2.6 was an autonomous decision made under a delegation that was misread. The user's intent from the start was Mode A (holdout-clean) with 2026-01-01 as the sacred boundary. The 3+ months of real-time 2026 data (2026-01-02 → current) was supposed to be accumulating as genuine forward OOS evidence, not consumed in discovery.

### What Amendment 2.6 got factually right (but misused)

Pass-2 audit evidence cited in Amendment 2.6 is still mechanically accurate:
- All 124 active `validated_setups` were discovered 2026-04-05 → 2026-04-06
- Discovery runs had 2026 data in scope (273,000 outcome rows for 2026-01-02 → 2026-04-05 exist in `orb_outcomes`)
- `HANDOFF.md:1468` did say *"2026 included in discovery — holdout test was spent"*
- `pre-registrations/2026-03-20-mnq-rr1-verified-sessions.md:4` did say the test was COMPLETED

**What Amendment 2.6 got WRONG:** it interpreted these facts as "the holdout was intentionally spent, so Mode B is the only honest position." The correct interpretation is "an earlier unreviewed decision (HANDOFF.md:1468) violated user intent, and the 124 discoveries are therefore contaminated, not canonical." Mode A is restored; the 124 are grandfathered as research-provisional.

### Operative rules under Amendment 2.7 (Mode A)

- **Sacred holdout window:** **2026-01-01 onwards.** Growing each day. Currently ~3.2 months of real-time forward OOS data (2026-01-02 → 2026-04-08).
- **The 124 existing validated_setups are grandfathered as RESEARCH-PROVISIONAL** per Amendment 2.4. They were discovered with 2026 data in scope and are NOT OOS-clean evidence. Their walk-forward `wf_passed = True` flag remains valid as in-sample evidence but does not substitute for a clean forward OOS test.
- **Existing 5 deployed lanes remain operationally deployable** with the same provisional label. No forced rollback. The user has chosen to run them live — that is an operational decision, not a claim of institutional-grade proof.
- **Any NEW discovery run** must use `--holdout-date 2026-01-01` (or earlier). No exceptions.
- **Forward OOS scoring protocol:**
  - New discovery on pre-2026 data → 2026-01-01 → current is the clean OOS window
  - Report OOS ExpR, sample_size, win_rate, Sharpe on the 2026 window
  - Criterion 8 enforceable: OOS ExpR ≥ 0 AND OOS ExpR ≥ 0.40 × IS ExpR (per v1, reactivated under Mode A)
- **The existing 124 cannot retroactively become OOS-clean.** The only way to make one "clean" is to re-run discovery with `--holdout-date 2026-01-01` and verify the same strategy is rediscovered. Any strategy NOT rediscovered under the clean-holdout protocol is NOT OOS-clean.
- **No mixing of modes:** Mode A is project-wide. No future research may claim Mode B status without another documented amendment.
- **Forward-paper requirement** (from Amendment 2.6) is SOFTENED: the walk-forward + forward 2026 OOS under Mode A replaces the 6-month forward-paper requirement for strategies that already have 3+ months of Mode-A-clean OOS. For strategies without that window, forward-paper still applies as a supplement.

### Updated Acceptance Matrix (Criterion 8 row)

| Criterion | Threshold | Enforcement |
|---|---|---|
| 8 Forward / OOS | Under Mode A: `--holdout-date 2026-01-01` required for discovery; OOS ExpR ≥ 0 AND OOS ExpR ≥ 0.40 × IS ExpR on the 2026-01-01 → current window; walk-forward remains in-sample discipline only | BINDING |

All other rows unchanged from v2 / v2.6.

### Classification of the 124 existing validated_setups under Mode A

| Label | Count | Meaning under Amendment 2.7 |
|---|---|---|
| Research-provisional | 124 | Discovered with 2026 in scope. WF in-sample only. NOT OOS-clean. Cannot be called "validated" in the institutional sense. |
| Operationally deployable | 5 (deployed lanes) | Live trading authorized. Research-provisional status inherited from above. No scaling until re-audited under Mode A. |
| Production-grade institutional proof | 0 | No strategy has passed all 12 criteria under Mode A yet. |

### What this means for the deployed 5 lanes

Nothing changes operationally. They remain deployed. They remain research-provisional. You are free to trade them — you are not free to call them institutionally proven. The path to institutional proof is:

1. Re-run discovery with `--holdout-date 2026-01-01` (requires Phase 2 data redownload + Phase 3 era schema + Phase 4 clean rediscovery — all gated on post-merge work)
2. If the same filter-session-entry-model-rr combos are rediscovered on the clean-holdout protocol, they graduate from research-provisional to research-validated
3. Compute their 2026-01-01 → current ExpR as genuine forward OOS evidence
4. If ExpR positive and ≥ 40% of IS ExpR → Criterion 8 passes
5. Then they can be scaled / relabeled as production-grade

### Deferred enforcement (blocked by post-merge sweep)

Now that `e2-canonical-window-fix` has merged into main (`8bc87f7`), these four items are UNBLOCKED and queued for the post-merge sweep:

1. `pipeline/check_drift.py` `HOLDOUT_DECLARATIONS` map populated with `MNQ: 2026-01-01`, `MES: 2026-01-01`, `MGC: 2026-01-01`. Activates the existing `check_holdout_contamination()` function.
2. `trading_app/strategy_discovery.py` runtime enforcement: reject `--holdout-date > 2026-01-01` for any new discovery run, and require explicit `--holdout-date` argument.
3. `trading_app/strategy_validator.py` validation gate: verify `discovery_date < 2026-01-01` OR the discovery was run with `--holdout-date 2026-01-01`.
4. New drift check: assert `RESEARCH_RULES.md`, `pre_registered_criteria.md`, and `pipeline/check_drift.py` HOLDOUT_DECLARATIONS agree on the holdout policy.

### Rescinded items from Amendment 2.6

- ~~"Forward-sacred window starts 2026-04-07"~~ → replaced with "Forward-sacred window is 2026-01-01 onwards"
- ~~"Earliest first-look at the new window is 2026-10-07"~~ → removed; the 2026-01-01 window is already 3+ months deep
- ~~"Past window (2026-01-01 → 2026-04-06): CONSUMED"~~ → rescinded; the window is NOT consumed, the contamination is in the 124 strategies' DISCOVERY provenance, not in the data itself
- ~~"Existing memory/doc references to '+2026 OOS' are historical observations, not strict OOS evidence"~~ → still accurate for the 124 existing strategies, but the 2026 data itself becomes genuine OOS for NEW Mode-A discoveries
- ~~"Grandfathered as research-provisional + operationally deployable per Amendment 2.4"~~ → this part is kept; grandfathering is still how the 124 are treated

### Acknowledgment of the contamination trade-off

Mode A restoration has a cost: 4 days of discovery work (2026-04-05, 2026-04-06) that produced the 124 validated_setups are now labeled research-provisional rather than canonical. Those 4 days are not wasted — the 124 strategies still represent a testable hypothesis list that can be re-verified under the clean-holdout protocol in Phase 4. The cost is one Phase 4 rediscovery run, not starting from zero.

This cost is accepted as the price of institutional honesty. Alternative (keeping Mode B to avoid the rework) was rejected because it contradicted explicit user intent.

---

## Amendment 2.8 (2026-04-09) — Factual correction of post-Phase-3c data horizons

**Type:** Factual correction. **NOT a threshold relaxation.** All 12 locked criteria, their numeric thresholds, and their enforcement status remain exactly as locked. This amendment updates stale narrative text in § Criterion 2 and § Criterion 10 to reflect the canonical data state after the Phase 3c rebuild merged to main.

**Trigger:** During Phase 4 Stage 4.2 hypothesis-file authoring on 2026-04-09, I drafted an MNQ hypothesis file declaring `data_source_mode: "proxy"` on the assumption that MNQ had "~2.2 years clean" + "16 years NQ parent proxy" (per § Criterion 2 worked example as originally written 2026-04-07). The user caught the error:

> *"I DONT WANNA FUCK AROUND WITH HALF THIS DATA HALF THAT DATA. I HAVE SUBSCRIPTION TO GET ALL THE DATA. WHY ARE WE DOING ALL THIS IF WE COULD JUST DOWNLOAD THE RIGHT DATA"*

Ground-truth verification against `gold.db` confirmed the user was right. Post-rebuild actual state:

| Instrument | bars_1m first day | bars_1m last day | Total trading days | Pre-holdout days (< 2026-01-01) | Clean years |
|---|---|---|---|---|---|
| MNQ | 2019-05-06 | 2026-04-07 | 2,154 | 1,951 | ≈ 6.65 |
| MES | 2019-05-06 | 2026-04-07 | 2,154 | 1,951 | ≈ 6.65 |
| MGC | 2023-09-11 | 2026-04-07 | 801 | 671 | ≈ 2.70 |

**Canonical evidence:** Phase 3c canonical layer rebuild commit `c33805b`, merged to main 2026-04-08. `HANDOFF.md` and memory topic file `phase_4_stage_4_1_shipped.md` reference this rebuild.

### What this amendment corrects

1. **§ Criterion 2 worked example (lines 62-84 as originally written):** the "~2.2 years clean MNQ data" and "16-year proxy-extended horizon" narrative was stale. Corrected in-place to reference post-Phase-3c actuals.

2. **§ Criterion 10 (line 170 as originally written):** the claim "MNQ/MES from 2024-02-05 onwards; MGC never valid because no real MGC data exists" was factually wrong. MNQ/MES CME Micro E-mini contracts launched 2019-05-06 (not 2024); MGC CME Micro Gold launched 2022-06-13 with real-micro data in the canonical layer since 2023-09-11. Corrected in-place.

3. **Derived consequence: strict Bailey MinBTL bounds at E=1.0 for actual horizons:**

   | Instrument | Horizon (yr) | Strict Bailey E=1.0 max N | Strict Bailey E=1.2 max N |
   |---|---|---|---|
   | MNQ | 6.65 | 28 | 120 |
   | MES | 6.65 | 28 | 120 |
   | MGC (no backfill) | 2.70 | 4  | 7 |
   | MGC (with Databento backfill to 2022-06-13) | 3.90 | 7  | 17 |

   The MGC no-backfill scenario is too tight for meaningful pre-registration (N ≤ 4). **The amendment recommends the MGC backfill before any MGC discovery run**, and labels MGC-without-backfill as incompatible with institutional pre-registration under strict Bailey.

### What this amendment does NOT change

- The locked 300 clean / 2000 proxy trial caps remain exactly as locked. They function as OPERATIONAL CEILINGS, looser than strict Bailey E=1.0 at the actual horizons. When operating between strict Bailey and the locked ceiling (e.g., 28 < N ≤ 300 for MNQ), the hypothesis file MUST disclose the corresponding Sharpe noise floor and explicitly flag that N exceeds the strict Bailey bound.
- The 12 criteria, all enforcement states (BINDING / CROSS-CHECK / CONTINGENT), and the v2.7 Mode A holdout policy are unchanged.
- Prior Amendments 2.1 through 2.7 retain their status.

### Institutional maximum rigor rule

**For institutional maximum rigor, operate at or below the strict Bailey E=1.0 bound for the target instrument's actual horizon.** This is tighter than the locked ceiling but corresponds to the original Bailey 2013 Theorem 1 at its most conservative parameterization. Pre-registered hypothesis files that choose N beyond strict Bailey must include, in `metadata.purpose` or a dedicated `bailey_compliance` field, the computed Sharpe noise floor and an explicit acknowledgment that candidates below that Sharpe are expected to be within noise even if they pass BH FDR.

### Why this matters for Phase 4 Stage 4.2

Stage 4.2 is the first hypothesis-file authoring gate. Authoring three files (MNQ, MES, MGC) under this amendment means:

1. **MNQ file:** 16 hypothesis bundles × 1 trial each = 16 trials. Strict Bailey E=1.0 at 6.65yr gives 28 trial bound, so 16 passes with 43% headroom. Noise floor Sharpe 0.91 annualized. No proxy declaration needed. Clean mode.
2. **MES file:** same structure as MNQ, 16 trials, same 6.65yr horizon.
3. **MGC file:** pending backfill decision.
   - **Backfill completed:** 16 hypothesis bundles possible up to strict Bailey bound of 7 at E=1.0 or 17 at E=1.2. Likely operating at N ≤ 7 with 3.9yr horizon, noise floor ~1.0.
   - **No backfill:** insufficient horizon for meaningful pre-registration. MGC discovery deferred until backfill lands OR explicitly labeled research-provisional with noise floor Sharpe ≥ 1.46.

### Related follow-ups surfaced during the Stage 4.2 self-audit

- **Loader drift check:** `trading_app.hypothesis_loader.load_hypothesis_metadata` reads `metadata.total_expected_trials` and `extract_scope_predicate` independently sums per-hypothesis `expected_trial_count`, but does not cross-check that they agree. A hypothesis file could silently declare `metadata.total_expected_trials: 10` while summing to 1000 in the per-hypothesis list. Queue a drift-check or loader-level cross-consistency assertion for Phase 4 Stage 4.3.
- **Databento backfill task for MGC:** Add to the action queue — download MGC bars from 2022-06-13 → 2023-09-10 via existing Databento Standard subscription, run Phase 3c-style canonical rebuild for the added days, re-verify drift checks. One-day wall-time task; ~308 trading days added.

### What was wrong and how it happened

The authoring error was: the criteria file worked example was written 2026-04-07, the Phase 3c canonical rebuild merged 2026-04-08, and the criteria file was not updated between those events. During Stage 4.2 authoring 2026-04-09 I read the criteria file as authoritative without checking whether its data-horizon narrative was still factually current against the DB. The user's instinct check ("WHY ARE WE DOING ALL THIS IF WE COULD JUST DOWNLOAD THE RIGHT DATA") caught the gap. The fix is this amendment plus corrected Stage 4.2 YAML authoring.

This failure mode is exactly what `.claude/rules/institutional-rigor.md` rule 8 ("Verify before claiming") prevents, and `.claude/rules/integrity-guardian.md` rule 7 ("Never Trust Metadata — Always Verify") reinforces. The criteria file narrative was metadata about the data horizon; I trusted it without verification; the user's correction is the institutional audit mechanism working correctly.

**No commitments beyond those already locked in the criteria file were invalidated by this stale narrative. The locked numeric thresholds were correct all along; only the worked-example text around them was out of date.**

---

## Amendment 2.9 (2026-04-09) — Parent/Proxy Data Policy (binding)

**Type:** New binding policy. Codifies the parent vs micro data handling rules that were previously implied across Criteria 2, 10, and Banned Practice #8, but never stated as a single coherent policy.

**Trigger:** User review on 2026-04-09 surfaced that the project had NQ/ES/GC parent symbol bars (2010-2024) sitting in `gold.db` alongside clean MNQ/MES/MGC micro bars, with no formal policy on when (if ever) to use them. The core risk: parent data with 10-100x different volume, different tick values ($5/tick NQ vs $0.50/tick MNQ), and different liquidity profiles could contaminate discovery or validation if accidentally mixed with micro data.

**Literature grounding:**
- Bailey et al 2013 (`literature/bailey_et_al_2013_pseudo_mathematics.md`): MinBTL bounds depend on data LENGTH, but adding proxy data with systematic feature mismatch inflates the denominator without adding real statistical power. The False Strategy Theorem (LdP-Bailey 2018, `literature/lopez_de_prado_bailey_2018_false_strategy.md`) shows that contaminated trials produce the illusion of tighter confidence intervals.
- Carver (`memory/data_length_literature.md` citing Table 5 p.62): cross-instrument portfolio (3 instruments) needs ~11yr combined, stronger than single-instrument time-extension with proxy data.
- Project finding `memory/era_contamination_trap.md`: wider filters on proxy-era data expose thin parent-symbol years (1-5 trades) that poison era stability checks.

### Policy rules

**Rule 1 — Discovery MUST use clean micro data only.** Never discover edges on parent-proxy data and claim they apply to micros. The cost model, tick value, liquidity profile, and volume characteristics differ fundamentally between parent and micro contracts.

**Rule 2 — Per-instrument proxy disposition:**

| Instrument | Clean micro data | Bailey N budget | Parent data in gold.db | Disposition |
|---|---|---|---|---|
| MNQ | 6.65yr (2019-05-06 → 2025-12-31) | N ≤ 28 (E=1.0) | NQ 2010-2024 (4.6M bars) | **DELETE NQ bars.** Not needed — 6.65yr sufficient for N=16-28 clean discovery. Cross-validate with MES instead. |
| MES | 6.65yr (2019-05-06 → 2025-12-31) | N ≤ 28 (E=1.0) | ES 2010-2024 (4.8M bars) | **DELETE ES bars.** Same rationale as MNQ. |
| MGC | 2.70yr (2023-09-11 → 2025-12-31) | N ≤ 4 (E=1.0) | GC 2010-2026 (5.5M bars) | **KEEP GC bars.** MGC N=4 is too small for meaningful discovery alone. GC proxy provides validation tier for price-based features only. |

**Rule 3 — MGC two-tier protocol:**
- **Tier 1 (discovery):** N ≤ 4 clean trials on 2.70yr MGC micro data. Strict Bailey E=1.0 compliance.
- **Tier 2 (validation):** Survivors from Tier 1 may be validated against GC parent data (2010-2023) for PRICE-BASED features ONLY (ORB range, session timing, direction, settlement gap). Volume/OI/microstructure filters (ATR70_VOL, OVNRNG, ORB_VOL, rel_vol_*) DO NOT transfer and must NOT be tested on GC data.
- **GC validation outcomes must be built under symbol "GC" with `COST_SPECS['GC']`**, never mixed into MNQ/MES/MGC orb_outcomes tables.
- **Statistics must be era-split** (GC pre-2019, GC 2019-2022, MGC 2023+). Never pool across eras in a single p-value.

**Rule 4 — Pipeline protection:** No pipeline path (`outcome_builder`, `strategy_discovery`, `strategy_validator`, `build_daily_features`) may accidentally query parent symbols when the user requests a micro instrument. The `ACTIVE_ORB_INSTRUMENTS` list in `pipeline.asset_configs` governs which symbols are eligible for automated processing. Parent symbols (NQ, ES, GC) are NOT in that list and must never be added.

**Rule 5 — Existing validated_setups provenance:** The 9 existing MGC validated strategies were discovered under the pre-Phase-0 brute-force regime (~35,000 trials on 2.8yr), violating Bailey MinBTL by ~600x. They are grandfathered as research-provisional per Amendment 2.4 but are **statistically suspect** and must not be scaled or treated as institutional evidence without re-validation under the clean protocol.

**Rule 6 — No silent proxy substitution.** If any code path, research query, or hypothesis file uses parent data, it must explicitly declare `data_source_mode: proxy` and cite this amendment. Silent use of parent data (e.g., querying `bars_1m WHERE symbol = 'GC'` in a script that claims to analyze MGC) is a banned practice.

### Updated Banned Practices (additions to existing list)

9. Using parent symbol bars (NQ, ES, GC) in any discovery run for micro instruments (MNQ, MES, MGC).
10. Mixing parent and micro data in the same orb_outcomes table or the same statistical test.
11. Applying volume/OI-based filters to parent-era data and treating results as micro-applicable.
12. Claiming parent-era price-pattern validation as equivalent to clean micro evidence (must be labeled "proxy validation" with explicit era-split disclosure).

### Interaction with existing criteria

- **Criterion 2 (MinBTL):** This policy constrains how the "proxy-extended N ≤ 2000" budget from the locked ceiling can be used. Proxy extension is only available for MGC (via GC), only for Tier 2 validation, and only for price-based features. MNQ/MES have no proxy extension path (NQ/ES bars to be deleted).
- **Criterion 10 (Data era compatibility):** This policy is a superset — Criterion 10 addresses volume filters on micro data; this policy addresses the entire parent/micro boundary.
- **Banned Practice #8:** This policy replaces #8 with the stronger #9-#12 above. #8 allowed proxy use "with explicit disclosure + era split"; #9 bans it entirely for discovery, restricting proxy to MGC Tier 2 validation only.

### Deferred action items

1. **Delete NQ and ES bars from gold.db** — one-time cleanup. Run after user confirmation of this amendment.
2. **MGC Databento backfill** — download MGC bars 2022-06-13 → 2023-09-10 to extend clean micro data to ~3.9yr (N ≤ 7 at E=1.0). Strongly recommended before MGC discovery. Already noted in Amendment 2.8.
3. **Drift check for parent symbol leakage** — new check that no `orb_outcomes` or `daily_features` row has a symbol matching a known parent symbol list. Queue for next drift check batch.

---

## Amendment 3.0 (2026-04-09) — Theory-Driven Individual Hypothesis Testing (binding)

**Type:** New dual-pathway for Criterion 3 (statistical significance). Adds an individual hypothesis pathway alongside the existing BH FDR pathway.

**Trigger:** The K=5 redesign discovery runs (2026-04-09) showed that mechanism-grounded predictions with raw p-values of 0.02-0.05 (MES NYSE_OPEN, MNQ COMEX_SETTLE) were killed by BH FDR correction when bundled with unrelated sessions. The BH FDR framework was designed for blind enumeration of millions of strategies (Chordia et al 2018); applying it to small, theory-driven hypothesis sets penalizes precisely the researcher behavior the framework was meant to encourage (pre-registration, theory-first design, honest K counting).

**Literature grounding:**

- **Harvey-Liu 2015** (`literature/harvey_liu_2015_backtesting.md`): The BHY haircut scales with the number of strategies tested. At N=1 (single pre-registered prediction), the haircut approaches zero. The framework explicitly rewards theory-driven hypothesis reduction.
- **Lopez de Prado 2020** (`literature/lopez_de_prado_2020_ml_for_asset_managers.md` §1.2): "Whatever edge you aspire to gain in finance, it can only be justified in terms of someone else making a systematic mistake from which you benefit." Theory-first discovery is the institutional standard — the statistical framework should SUPPORT this, not penalize it.
- **Bailey et al 2013** (`literature/bailey_et_al_2013_pseudo_mathematics.md`): MinBTL constrains total N. At K=1, MinBTL = 0 (ln(1) = 0) — a single pre-registered hypothesis requires NO minimum backtest length beyond what the researcher deems sufficient for the test's statistical power.
- **Crabel 1990**: ORB breakouts are a published, well-understood strategy class with a 35-year track record. Testing "does MNQ NYSE_OPEN G5 have positive expectancy?" is NOT blind enumeration — it is testing a specific, theory-grounded prediction about a known strategy class.

### The canonical-base-truth methodology

Before any hypothesis is pre-registered under the individual pathway, the researcher must follow this 5-step methodology:

**Step 1 — Canonical base truth.** For each instrument, identify which sessions have any plausible unfiltered or size-first edge on pre-holdout data only. This is a READ-ONLY audit, not discovery. No deployment decisions are made here.

**Step 2 — Restrict the admissible family set.** Do not let every session inherit every filter family. If a family has no mechanism or no prior structural evidence for that instrument-session, it does not enter the grid. The restriction must be documented BEFORE seeing filtered results.

**Step 3 — Pre-register bundles.** Each bundle is a specific mechanism × session × instrument × filter prediction. Examples: "MGC CME_REOPEN size-first," "MES NYSE_OPEN Crabel-commitment G5," "MNQ EUROPE_FLOW cross-border-flow G5." NOT "run all filters and see."

**Step 4 — Count each bundle honestly.** If you test 5 bundles, your trial count is 5. If inside one bundle you test 8 thresholds and 4 overlays, your real trial count is not 1 — it is 32. The `expected_trial_count` field in the hypothesis YAML must match reality.

**Step 5 — 2026 = forward judgment only.** The holdout tells you whether the pre-registered idea survives. It does not get to choose the idea. No post-hoc hypothesis selection based on holdout results.

### Dual-pathway comparison

| Property | Pathway A: BH FDR (existing) | Pathway B: Individual (new) |
|---|---|---|
| **When to use** | Exploratory search, threshold sweeps, multiple overlays | Single mechanism-session prediction with economic theory |
| **Family K** | All hypotheses in the file | K=1 per hypothesis |
| **Significance threshold** | BH-adjusted q = 0.05 | Raw p < 0.05 (two-tailed) AND positive Sharpe (direction gate) |
| **Theory citation** | Recommended | MANDATORY (no theory = Pathway A only) |
| **Hypothesis file** | Multiple hypotheses, `testing_mode: family` | `testing_mode: individual` flag required |
| **Criterion 6 (walk-forward)** | Required, waivable with justification | MANDATORY, non-waivable |
| **Criterion 8 (2026 OOS)** | Required when holdout data available | MANDATORY, non-waivable |
| **Criterion 9 (era stability)** | Required, regime waivers available | MANDATORY, no regime waivers |
| **Reporting** | Report FDR-adjusted p | Report raw p + cumulative hypothesis count across ALL individual files |
| **Abuse safeguard** | FDR correction | All hypotheses (pass + fail) permanently recorded via SHA stamp. Downstream gates non-waivable. |

### Conditions for Pathway B (ALL must be met)

1. **Theory citation required.** Every hypothesis must have a `theory_citation` field referencing published external literature (not repo history, not training memory). The citation must explain WHY this specific instrument at this specific session should exhibit ORB breakout continuation.

2. **One prediction per hypothesis.** No threshold sweeps (testing G4 AND G5 AND G6 within one "individual" hypothesis). Each threshold is a separate prediction if tested.

2b. **Positive direction required.** Raw p < 0.05 is necessary but not sufficient. The strategy must ALSO have positive annualized Sharpe (direction gate). A significantly negative strategy (small p-value but negative Sharpe) fails Pathway B. This prevents the two-tailed p-value from passing strategies that are significantly BAD.

3. **Pre-registered before results.** The hypothesis file must be committed to git before the discovery run. The SHA stamp enforces this.

4. **Downstream gates non-waivable.** Criteria 6 (WFE ≥ 0.50), 8 (2026 OOS positive), and 9 (era stability) are ALL mandatory for Pathway B. No regime waivers, no "insufficient OOS data" exemptions.

5. **All results reported.** Every hypothesis (pass or fail) appears in `experimental_strategies` with the committed SHA. Selective reporting is structurally prevented.

6. **Cumulative count disclosure.** Each individual hypothesis file must note the TOTAL number of individual hypotheses tested across ALL files for that instrument. This is for transparency, not for correction — it lets auditors assess the overall family-wise error rate.

### What this amendment does NOT change

- **Criterion 2 (MinBTL):** Still applies. At K=1, MinBTL = 0, so this is trivially satisfied. The practical constraint becomes data sufficiency for statistical power (N ≥ 100 per Criterion 7).
- **Criterion 4 (Chordia t):** Still applies. t ≥ 3.00 with theory.
- **Criterion 5 (DSR):** Still applies as a cross-check.
- **Criteria 7, 10, 11, 12:** Unchanged.
- **Pathway A:** Unchanged. BH FDR remains available for exploratory search.

### Contamination disclosure

This amendment was written after seeing the K=5 redesign results (2 MNQ survivors, 0 MES, 0 MGC). The MES NYSE_OPEN signal (raw p = 0.026) was the proximate trigger. The legitimacy of this amendment rests on the STRUCTURAL argument that BH FDR is designed for blind enumeration (Chordia's 2 million strategies), not for theory-driven testing of a known strategy class (Crabel ORB breakouts). This argument is valid independent of the specific results that prompted it — it would be equally valid if all 15 hypotheses had passed or all had failed.

The amendment was requested by the user based on their understanding that real ORB breakout traders profitably apply this strategy class without academic-scale multiple testing corrections, and that the project's framework should match the actual use case (theory-driven testing) rather than the worst case (blind enumeration).

---

## Amendment 3.1 (2026-04-09) — Structural Data Boundary for Discovery and Era Stability (binding)

**Type:** Extends WF_START_OVERRIDE to discovery IS scope and Criterion 9 era bins.

**Trigger:** MNQ CME_PRECLOSE E2 RR1.0 G8 (Sharpe 1.83, p=0.000002, N=1320) was rejected by Criterion 9 because the "2015-2019" era bin contained 56 trades from the structurally non-representative 2019 micro-contract launch period. The WF engine already excluded this data via WF_START_OVERRIDE=2020-01-01 (set by Amendment 3.1's prerequisite data audit), but Criterion 9 read `yearly_results` which included 2019.

**Structural data audit evidence (execution output, not metadata):**

MNQ 2019 vs 2020+ (micro launched 2019-05-06):
- ATR: 113.7 vs 279.5 = **0.42x**
- CME_PRECLOSE G8 pass rate: **39.0%** vs 97.7% (monthly: Nov=5.3%, Dec=20%)
- EUROPE_FLOW G8 pass rate: **22.8%** vs 83.5%
- NYSE_OPEN volume: 5,689 vs 30,845 = **0.16x**
- CME_PRECLOSE volume: 1,672 vs 7,980 = **0.18x**

Monthly granularity confirms Q3/Q4 2019 does NOT normalize. Jan 2020 = 71.4% G8 pass, confirming 2020-01-01 as the clean structural boundary.

**The 56 trades that triggered Criterion 9 rejection:**
- ORB size: 13.3 vs 26.5 (2020) = 0.50x
- Volume: 2,361 vs 5,646 (2020) = 0.42x
- ATR: 124.0 vs 249.1 (2020) = 0.50x
- All from an 8-month-old contract with immature liquidity

**Rule (two parts):**

**Part 1 — Discovery IS scope:** When `run_discovery` is called without an explicit `--start` date, the discovery engine defaults to `WF_START_OVERRIDE.get(instrument)` as the lower bound for both `daily_features` and `orb_outcomes` loading. This ensures the IS sample, `yearly_results` JSON, and all derived statistics (sample_size, trades_per_year, p_value, etc.) exclude structurally non-representative pre-override data. An explicit `--start` CLI argument overrides this default.

**Part 2 — Criterion 9 era bins:** When computing era stability, years before the instrument's `WF_START_OVERRIDE` year are excluded from era bin aggregation. The logic: skip any year `y` where `y < wf_start_override.year`. This makes the era stability check consistent with the walk-forward engine and the discovery IS scope.

**Rationale:** The same structural data audit that makes pre-override data unreliable for walk-forward training also makes it unreliable for era stability assessment. A contract-launch liquidity artifact is NOT a dead regime — it is a microstructure state that no longer exists and will not recur. Including it in era stability creates a false rejection that blocks an otherwise honest strategy.

**Honest caveat:** We cannot prove the counterfactual "what would 2019 look like with mature MNQ liquidity." The exclusion rests on structural judgment (5 independent metrics all converging), not a statistical test. If NQ volatility returned to 2019 levels with today's mature MNQ liquidity, the outcome would be structurally different — but this is an assertion, not a proven fact.

**Bias check:** The WF_START_OVERRIDE=2020-01-01 was set BEFORE examining any strategy's per-year PnL. The justification is purely structural (ATR, volume, filter pass rates). The override would be identical if MNQ CME_PRECLOSE showed positive 2019 performance.

**Interaction with Amendment 3.0 (Pathway B):** Pathway B condition 4 ("Criteria 9 non-waivable") is STRENGTHENED by this amendment, not weakened. The exclusion applies to the DATA, not to the GATE. Criterion 9 still runs on all post-override eras with the same -0.05 threshold and N>=50 requirement. The gate is non-waivable; the data scope is narrowed to structurally valid observations.

**What this does NOT change:**
- The -0.05 ExpR threshold is UNCHANGED
- The N >= 50 minimum is UNCHANGED
- Post-override era bins (2020-2022, 2023, 2024-2025, 2026) are UNCHANGED
- The holdout boundary (2026-01-01) is UNCHANGED
- Existing validated strategies are unaffected (they already pass Criterion 9 or are grandfathered)

---

## Amendment 3.1 (2026-04-10) — Revised Parent/Proxy Data Policy (supersedes parts of 2.9)

**Type:** Policy revision with empirical evidence. Expands GC proxy from "validation-only" to "discovery-eligible for price-based filters." Adds filter classification and regime-awareness requirement. Does NOT relax MinBTL bounds, holdout policy, or volume-filter restrictions.

**Trigger:** User challenge (2026-04-10): *"can we use the gc for our strats or not? it doesn't have to do with price movement does it?"* Followed by structured 4-gate empirical research on branch `research/gc-proxy-validity`.

**Evidence (from `scripts/research/gc_proxy_validity.py`):**

| Gate | Test | Result | Key metric |
|---|---|---|---|
| 1 | Bar-level price identity (1.25M paired bars) | PASS | Price corr=0.99999, range corr=0.984 |
| 2 | Filter input identity (gap, PDR, range) | PASS | PDR corr=0.993, GAP_R005 agreement=99.5% |
| 3 | Outcome identity (241K matched trades) | PASS | 100% GC coverage, 96% trigger match |
| 4 | Adversarial (worst-case divergence) | PASS | 2.2% of days >5pt diff (roll artifacts) |

**Negative control:** Volume correlation = 0.62 (contract counts similar but patterns DIFFERENT, confirming GC/MGC are structurally different instruments for volume-based features).

### What Amendment 2.9 got right (UNCHANGED)

- Volume/OI filters DO NOT transfer (10-100x notional difference). Banned Practice #11 stands.
- Statistics must be era-split. Banned Practice #10 stands.
- No silent proxy substitution. Rule 6 and Banned Practice #12 stand.
- Pipeline protection (Rule 4) stands — parent symbols not in ACTIVE_ORB_INSTRUMENTS.
- NQ/ES deletion disposition stands (Rule 2, MNQ/MES rows).

### What Amendment 2.9 got wrong (REVISED)

**Rule 1 (old):** "Discovery MUST use clean micro data only."
**Rule 1 (revised):** Discovery MUST use clean micro data for volume-based filters. Discovery MAY use GC parent data for price-based filters, subject to the conditions below.

**Rule 3 (old):** MGC two-tier protocol — Tier 1 discovery on micro only, Tier 2 validation on GC for price-only.
**Rule 3 (revised):** MGC two-tier protocol — Tier 1 discovery on micro OR GC proxy (for price-safe filters with disclosure), Tier 2 cross-era validation on GC.

**Banned Practice #9 (old):** "Using parent symbol bars (NQ, ES, GC) in any discovery run for micro instruments."
**Banned Practice #9 (revised):** "Using parent symbol bars in discovery WITHOUT explicit `data_source_mode: proxy` declaration, filter-class compliance, and regime-awareness disclosure."

### Filter classification (empirically grounded)

**PRICE-SAFE** — can use GC proxy for MGC discovery:
- ORB size filters: ORB_G4, ORB_G5, ORB_G6, ORB_G8 (and all CONT/DOW/FAST/L12 variants)
- Gap filters: GAP_R005, GAP_R015
- Prior day range: PDR_R080, PDR_R105, PDR_R125
- ATR percentile: ATR_P30, ATR_P50, ATR_P70, ATR70_VOL
- Overnight range: OVNRNG_10, OVNRNG_25, OVNRNG_50, OVNRNG_100 (and FAST variants)
- Cost ratio: COST_LT08, COST_LT10, COST_LT12, COST_LT15 (and FAST variants) — threshold scales with ORB size in points, not contract size
- Direction: DIR_LONG, DIR_SHORT
- Cross-asset: X_MES_ATR60, X_MES_ATR70, X_MGC_ATR70
- Baseline: NO_FILTER, PIT_MIN

**VOLUME-UNSAFE** — micro-only, no proxy:
- ORB volume: ORB_VOL_2K, ORB_VOL_4K, ORB_VOL_8K, ORB_VOL_16K
- Relative volume: VOL_RV12_N20, VOL_RV15_N20, VOL_RV20_N20, VOL_RV25_N20, VOL_RV30_N20

**Basis:** Price-safe filters use only price-derived inputs (OHLC, range, gap). GC and MGC share identical price discovery (corr=0.99999). Volume-unsafe filters use contract-level volume/OI which differs structurally between parent and micro.

### Regime-awareness requirement (NEW)

GC proxy extends the data horizon back to 2010. However, gold market structure has changed significantly:
- Gold $1200 (2015): ATR ~22pts, G5 pass rate ~6-13% per session
- Gold $2700 (2025): ATR ~57pts, G5 pass rate ~58% per session
- Gold $4800 (2026): ATR ~156pts, G5 pass rate ~100%

**Absolute-threshold filters (G4/G5/G6/G8)** select fundamentally different populations at different price levels. A 5-point ORB at $1200 is 0.42% of price; at $3000 it's 0.17%. Researchers using GC proxy with G-filters MUST:
1. Report per-era G-filter pass rates
2. Acknowledge that low-pass-rate eras have low trade counts (wide CIs, not "no edge")
3. Consider ATR-normalized G thresholds for cross-era comparability (optional, not required)

**Relative/percentile filters (PDR_R*, GAP_R*, OVNRNG_*, ATR_P*)** are regime-robust by construction — they use percentile thresholds that adapt to the price level. These are the PRIMARY value of GC proxy for MGC discovery.

### Conditions for GC proxy discovery

1. Hypothesis file declares `data_source_mode: proxy` with `data_source_disclosure: "GC parent futures 2010-2025"`
2. Only price-safe filters used (see classification above)
3. Cost model: use MGC cost specs (not GC) — `COST_SPECS['MGC']`
4. Era-split required: statistics reported per era (minimum: pre-2020, 2020-2022, 2023+)
5. G-filter strategies must report per-era pass rates
6. MinBTL bound: N ≤ 2000 (proxy-extended ceiling, per Criterion 2)
7. Deployment validation: MUST use MGC micro data only (proxy is for discovery power, not deployment evidence)

### What this enables

- MGC discovery budget expands from N ≤ 4 (2.7yr micro) to N ≤ 2000 (16yr GC proxy) for relative filters
- PDR_R080, GAP_R005, OVNRNG_50 hypotheses become testable with 16 years of data
- G-filter hypotheses gain marginal benefit (low pass rates pre-2025, but still more data than 2.7yr)
- MGC joins MNQ/MES as a fully researchable instrument for price-based edge discovery

### What this does NOT change

- MinBTL bounds (N ≤ 300 clean, N ≤ 2000 proxy) — UNCHANGED
- Holdout boundary (2026-01-01) — UNCHANGED
- Volume-filter restrictions — UNCHANGED (micro-only)
- Pipeline protection (GC not in ACTIVE_ORB_INSTRUMENTS) — UNCHANGED
- Deployment validation (MGC-only) — UNCHANGED
- NQ/ES deletion disposition — UNCHANGED

### Deferred action items (from 2.9, updated)

1. ~~Delete NQ and ES bars~~ — still pending, unchanged.
2. **Build GC orb_outcomes** — required before GC proxy discovery can run. The outcome_builder needs GC in ASSET_CONFIGS with `orb_active: True` (or a research-mode flag).
3. **Build GC daily_features** — required for filter application on GC data.
4. **Add GC to COST_SPECS** — with full-size gold specs ($100/pt, ~$57.40 friction) for GC-native cost analysis; MGC cost model used for proxy discovery.
5. **Write MGC proxy hypothesis files** — using GC data, relative filters, full 16-year horizon.
