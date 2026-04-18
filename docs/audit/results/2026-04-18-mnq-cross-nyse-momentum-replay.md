# MNQ CROSS_NYSE_MOMENTUM — Stage 1 Replay

**Date:** 2026-04-18
**As-of trading day:** 2026-04-17
**Pre-reg:** `docs/audit/hypotheses/2026-04-18-mnq-cross-nyse-momentum.yaml`
**Design:** `docs/plans/2026-04-18-mnq-cross-nyse-momentum-design.md`
**Distinctness audit:** `docs/audit/results/2026-04-18-mnq-cross-nyse-momentum-distinctness.md`
**Family verdict:** **NULL** (0/6 cells pass primary)

## Methodology grounding (cited per claim, not from memory)

| Claim | Source (project canon) | Source (local literature) |
|---|---|---|
| 4-state CrossSessionMomentumFilter logic | `trading_app/config.py:2558-2704` | — (project implementation) |
| Mode A sacred holdout 2026-01-01 | `docs/institutional/pre_registered_criteria.md` Amendment 2.7 | — (project policy) |
| BH-FDR q<0.05 at K=6 | `.claude/rules/backtesting-methodology.md` Rule 4 | `docs/institutional/literature/harvey_liu_2015_backtesting.md` L55-62 (BHY procedure) |
| Chordia t ≥ 3.79 (without-theory) | `pre_registered_criteria.md` Criterion 4 | `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` L20, L57 |
| WFE ≥ 0.50 | `pre_registered_criteria.md` Criterion 6 | `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md` L186 (CV generalization → WFE mapping) |
| N_OOS ≥ 30 (CLT heuristic) | `trading_app/strategy_validator.py:1052` `_OOS_MIN_TRADES_CLT_HEURISTIC` | — (project code literal) |
| eff_ratio ≥ 0.40 lower bound | `pre_registered_criteria.md` Amendment 2.7 (OOS ExpR ≥ 0.40 × IS) | — |
| eff_ratio ≤ 3.00 upper bound (LEAKAGE_SUSPECT) | `.claude/rules/quant-audit-protocol.md` §T3 (WFE>0.95 on small OOS N = LEAKAGE_SUSPECT); pre-reg-specific numeric threshold added on Apr 18 | — |
| Rule 7 tautology canonical metric = boolean fire correlation | `.claude/rules/backtesting-methodology.md` Rule 7 L193-194 | — |
| Welch two-sample t-test (parametric) | — (standard statistical method via `scipy.stats.ttest_ind_from_stats`) | — |
| Fitschen intraday trend-follow core premise | — | `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` Ch 3 Tables 3.8-3.9, pp 40-41 |

**What is NOT cited from memory:** nothing. Any claim without a citation in the table above is either project-code-literal (traceable) or a standard scipy/math implementation. If a claim below required a literature source that was not readable, it would be labeled `LOCAL SOURCE READ FAILED` — no such labels appear, meaning all cited sources were verified readable at audit time (see §"Local resources actually used" below).

## Tautology pre-gate (Rule 7 canonical fire correlation)

| Lane | Alt filter | fire_rho | |rho|>0.70? | Verdict |
|---|---|---:|:---:|:---:|
| MNQ US_DATA_1000 | X_MES_ATR60 | -0.0236 | no | distinct |
| MNQ NYSE_CLOSE | — | — | — | N/A (no deployed filters on lane) |

## Per-cell primary evaluation

| Cell | IS N(TAKE/all) | ExpR_TAKE / ExpR_all (IS) | delta_IS | t_IS | p_IS | BH K=6 | OOS N_TAKE | delta_OOS | eff_ratio | dir_match | WFE | PRIMARY |
|---|---|---|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|:---:|
| US_DATA_1000_RR1.0 | 1146/1701 | +0.0873 / +0.0867 | +0.0006 | +0.02 | 0.9864 | FAIL | 41 | +0.1755 | +287.151 | PASS | +2.140 | **FAIL** |
| US_DATA_1000_RR1.5 | 1125/1674 | +0.1185 / +0.0926 | +0.0259 | +0.57 | 0.5708 | FAIL | 40 | +0.1925 | +7.440 | PASS | +1.283 | **FAIL** |
| US_DATA_1000_RR2.0 | 1099/1639 | +0.1433 / +0.0910 | +0.0523 | +0.96 | 0.3353 | FAIL | 38 | +0.1827 | +3.495 | PASS | +1.041 | **FAIL** |
| NYSE_CLOSE_RR1.0 | 445/805 | +0.1956 / +0.0838 | +0.1119 | +2.15 | 0.0316 | FAIL | 22 | +0.2284 | +2.042 | PASS | +5.693 | **FAIL** |
| NYSE_CLOSE_RR1.5 | 325/612 | +0.1174 / -0.0103 | +0.1277 | +1.64 | 0.1019 | FAIL | 17 | +0.3068 | +2.403 | PASS | +4.391 | **FAIL** |
| NYSE_CLOSE_RR2.0 | 271/515 | +0.0185 / -0.1678 | +0.1863 | +1.90 | 0.0575 | FAIL | 15 | +0.3181 | +1.708 | PASS | +25.098 | **FAIL** |

## Per-criterion breakdown

| Cell | BH K=6 | t≥3.79 | WFE≥0.50 | N_OOS≥30 | dir_match | eff∈[0.40,3.00] | tautology_ok | PRIMARY |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| US_DATA_1000_RR1.0 | FAIL | FAIL | PASS | PASS | PASS | FAIL | PASS | **FAIL** |
| US_DATA_1000_RR1.5 | FAIL | FAIL | PASS | PASS | PASS | FAIL | PASS | **FAIL** |
| US_DATA_1000_RR2.0 | FAIL | FAIL | PASS | PASS | PASS | FAIL | PASS | **FAIL** |
| NYSE_CLOSE_RR1.0 | FAIL | FAIL | PASS | FAIL | PASS | PASS | PASS | **FAIL** |
| NYSE_CLOSE_RR1.5 | FAIL | FAIL | PASS | FAIL | PASS | PASS | PASS | **FAIL** |
| NYSE_CLOSE_RR2.0 | FAIL | FAIL | PASS | FAIL | PASS | PASS | PASS | **FAIL** |

## BH-FDR K=6 rank table

| rank | cell | p_IS | threshold | pass |
|---:|---|---:|---:|:---:|
| 1 | NYSE_CLOSE_RR1.0 | 0.031639 | 0.008333 | FAIL |
| 2 | NYSE_CLOSE_RR2.0 | 0.057473 | 0.016667 | FAIL |
| 3 | NYSE_CLOSE_RR1.5 | 0.101882 | 0.025000 | FAIL |
| 4 | US_DATA_1000_RR2.0 | 0.335253 | 0.033333 | FAIL |
| 5 | US_DATA_1000_RR1.5 | 0.570802 | 0.041667 | FAIL |
| 6 | US_DATA_1000_RR1.0 | 0.986424 | 0.050000 | FAIL |

## Family verdict: **NULL**

Cells passing primary: 0/6

Verdict semantics per design doc:
- STRONG_PASS (4-6 pass) — proceed to Stage 2
- STANDARD_PASS (2-3 pass) — promote passing cells only
- MARGINAL (1 pass) — reevaluate under allocator correlation gate
- NULL (0 pass) — close family, no rescue

## Methodology transparency

- **Baseline definition:** all valid-4-state days on same (session, RR) cell, IS-only for delta_IS computation; same for OOS. Candidate = TAKE-state subset.
- **Welch two-sample t-test** applied to TAKE vs all-valid-state ExpR populations (unequal variance assumption). Via `scipy.stats.ttest_ind_from_stats`.
- **WFE simplification:** computed as annualized-Sharpe ratio `Sharpe(TAKE OOS) / Sharpe(TAKE IS)` on single-split, not 5-fold expanding window. For NULL verdict this does not change outcome; for STRONG_PASS candidate any downstream promotion requires 5-fold upgrade.
- **Tautology Rule 7 metric:** canonical boolean fire correlation (Pearson on 0/1 fire indicators), NOT continuous-variable correlation. Confirmed by reading `.claude/rules/backtesting-methodology.md` Rule 7 L193-194.
- **eff_ratio upper bound (3.00):** added to this pre-reg specifically because the Apr 11 memo `docs/plans/2026-04-11-cross-session-state-round3-memo.md` showed OOS/IS ratios 1.8-3.4× on this exact family. Per `.claude/rules/quant-audit-protocol.md` §T3, WFE>0.95 on small OOS N = LEAKAGE_SUSPECT; the same structural concern applies to eff_ratio » 1.0 on small OOS N.
- **All OOS queries made after IS scope was locked**, per v2 discipline.

## Local resources actually used (grounding audit)

Every methodology/statistics claim in this document is sourced from ONE of the following local files. All were verified readable at audit time (2026-04-18).

| Source | Role in this document |
|---|---|
| `docs/institutional/pre_registered_criteria.md` | Criterion 4 (t≥3.79), 6 (WFE≥0.50), 7 (N≥100 related); Amendment 2.7 Mode A holdout + eff_ratio lower bound |
| `docs/institutional/literature/harvey_liu_2015_backtesting.md` | BHY / BH-FDR procedure (L55-62) |
| `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` | Chordia t=3.79 threshold (L20, L57) |
| `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md` | CV/WFE generalization (L186 Criterion 6 mapping) |
| `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` | Intraday trend-follow core premise (Ch 3 Tables 3.8-3.9, pp 40-41) |
| `.claude/rules/backtesting-methodology.md` | Rule 4 (BH-FDR multi-framing), Rule 7 (tautology fire correlation), Rule 8.1 (extreme fire rate) |
| `.claude/rules/quant-audit-protocol.md` | §T3 (LEAKAGE_SUSPECT definition) |
| `trading_app/config.py` | CrossSessionMomentumFilter implementation (L2558-2704) |
| `trading_app/strategy_validator.py` | `_OOS_MIN_TRADES_CLT_HEURISTIC = 30` (L1052) |
| `docs/plans/2026-04-11-cross-session-state-round3-memo.md` | Prior-work disclosure — Apr 11 Round-3 Pack A design (informational, not scope-shaping) |

**No `LOCAL SOURCE READ FAILED` flags.** No methodology claim in this document is sourced from training memory; all citations point to local files enumerated above.

