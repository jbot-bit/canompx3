# Phase D D4 — MNQ COMEX_SETTLE Pathway B K=1 — Result

**Pre-reg:** docs/audit/hypotheses/2026-04-28-mnq-comex-settle-garch-pathway-b-v1.yaml
**Pre-reg commit:** f7e2c9218b4b7c6a2147591d1127b6c143f5dba8
**Run timestamp:** 2026-04-28
**DB freshness:** orb_outcomes max trading_day = 2026-04-26
**Holdout boundary (Mode A):** 2026-01-01

## Scope

Pathway B K=1 confirmatory test of B-MNQ-COX (the highest-EV PATHWAY_B_ELIGIBLE candidate from the 2026-04-28 Phase B per-candidate evidence pass). The question this run answers: under the locked schema, do all KILL criteria pass and do the C5/C6/C7/C8/C9 gates support promotion to CANDIDATE_READY, or does the OOS power floor force a PARK verdict, or does any KILL criterion fire?

## Locked schema (verbatim from pre-reg)

- instrument: MNQ
- session: COMEX_SETTLE
- orb_minutes: 5
- rr_target: 1.0
- entry_model: E2
- confirm_bars: 1
- direction: long
- feature: garch_forecast_vol_pct
- feature_threshold: 70.0
- feature_op: >
- scratch_policy: include-as-zero (commit 68ee35f8)

## IS / OOS reproduction (Mode A)

| Metric | IS | OOS |
|---|---:|---:|
| N total | 879 | 35 |
| N on (signal=true) | 199 | 17 |
| N off | 680 | 18 |
| ExpR on | +0.2453 | +0.1344 |
| ExpR off | +0.0166 | -0.1194 |
| Δ (on − off) | +0.2286 | +0.2538 |
| Sharpe per-trade | +0.2746 | +0.1373 |
| Sharpe annualised | +1.6924 | — (UNVERIFIED if N_OOS<50) |
| Skewness γ̂₃ | -0.6707 | — |
| Kurtosis γ̂₄ (excess) | -1.5313 | — |

## Significance (IS, K=1 — see pre-reg testing_discipline)

- Welch t-stat: +3.1812
- Welch p (two-tailed): 0.001610
- Block-bootstrap p (block=5, B=10000): 0.001900
- Cohen's d (IS effect): +0.2574
- OOS power for d at N_OOS_on=17: 0.1059

## Phase 0 gates (post-Amendment 3.0/3.2)

| Gate | Threshold | Computed | Verdict |
|---|---|---|---|
| C5 DSR (Pathway A K_family=2700) [provenance] | ≥ 0.95 | 0.0000 | DSR_PA_FAIL |
| C5 DSR (Pathway B K=1) [primary gate] | ≥ 0.95 | 0.9998 | DSR_PB_PASS |
| C6 WFE | ≥ 0.5 (if N_OOS≥50) | GATE_INACTIVE_LOWPOWER | C6_GATE_INACTIVE_LOWPOWER |
| C7 N_IS_on | ≥ 100 | 199 | C7_PASS |
| C8 dir-match (power-floor aware) | dir_match AND power≥0.5 | dir=True, power=0.106 | C8_GATE_INACTIVE_LOWPOWER |
| C9 era stability (N_era≥20, no era < -0.05) | PASS | C9_PASS | C9_PASS |

## Per-year IS breakdown

| Year | N | ExpR |
|---:|---:|---:|
| 2020 | 10 | +0.3313 |
| 2021 | 23 | +0.1927 |
| 2022 | 70 | +0.2472 |
| 2023 | 7 | +0.3180 |
| 2024 | 48 | +0.1006 |
| 2025 | 41 | +0.4073 |

## Kill-criterion results (locked from pre-reg)

| ID | Threshold | Computed | Verdict |
|---|---|---|---|
| KILL_RAWP | >= 0.05 | welch_p = 0.001610 | PASS |
| KILL_DIR | negative on IS | sign(delta_IS) = + | PASS |
| KILL_T | < 3.0 | |t| = 3.1812 | PASS |
| KILL_N | < 100 | N_IS_on = 199 | PASS |
| KILL_ERA | < -0.05 (any era N>=20) | min_era_ExpR(eligible) = +0.1006 | PASS |
| KILL_SHARPE | <= 0.0 | sharpe_ann_IS = +1.6924 | PASS |
| KILL_BASELINE_SANITY | > 0.001 | |delta_IS - expected| = 0.000014 | PASS |

## RULE 7 lane-correlation (descriptive — does not gate verdict)

- Pearson correlation of daily pnl_r between this lane's IS_on trades and the deployed `MNQ COMEX_SETTLE O5 RR1.5 ORB_G5` lane's IS trades: **0.7733**
- Pre-reg flag: Phase B reported +0.773 vs deployed COMEX_SETTLE ORB_G5. RULE 7 threshold |corr| > 0.70 → portfolio overlap concern. Phase E admission requires an additivity audit before any capital allocation. This pre-reg's verdict is necessary, not sufficient, for live deployment.

## Decision rule outcome

**VERDICT: PARK_PENDING_OOS_POWER**

All non-conditional gates PASS (C5 DSR_PB=0.9998, C7 N=199, C9 era stable, Sharpe_ann_IS=+1.6924). C6/C8 are GATE_INACTIVE_LOWPOWER because N_OOS_on=17 < 50 power floor (Amendment 3.2). UNVERIFIED ≠ KILL — cell parks until N_OOS_on accrues to ≥50 (estimate Q3-2026 at current trade rate). Pre-reg locked; no re-tuning permitted on accrual. RULE 7 lane-correlation flag stands; Phase E admission requires separate additivity audit.

## Audit pressure-test (RULE 13)

- Pressure test: PASS

## Reproduction

```
DUCKDB_PATH=C:\Users\joshd\canompx3\gold.db python research/phase_d_d4_mnq_comex_settle_pathway_b.py
```

- DB: `pipeline.paths.GOLD_DB_PATH`
- Holdout: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` (2026-01-01)
- Pre-reg: locks all schema parameters and kill criteria
- Helpers: imported from `research/phase_b_candidate_evidence_v1.py` (canonical)

## Caveats and limitations

- N_OOS_on is small (3-month sacred holdout). C8 power floor invokes UNVERIFIED, not KILL.
- This is one Pathway B test, K=1 by design. The decision rule's PARK case is the legitimate
  verdict when OOS is underpowered; UNVERIFIED is not failure per Amendment 3.2.
- DSR Pathway A K_family is approximated at 2700 (volatility family proxy from Phase B). Sensitivity
  not exhaustive — true K may differ; the Pathway B K=1 path is what governs promotion here.
- Lane-correlation +0.773 vs deployed COMEX_SETTLE ORB_G5 (RULE 7 overlap concern). A
  CANDIDATE_READY or PARK verdict here does NOT clear Phase E portfolio admission.
  Additivity audit (residualised vs deployed lane) required before any capital deployment.
- This Pathway B test verifies the PHASE B EVIDENCE was correctly summarized; an honest
  PARK or CANDIDATE_READY here does not assert independent edge over the deployed COMEX
  ORB_G5 lane.

## Not done by this run

- No write to validated_setups, edge_families, lane_allocation, live_config
- No paper trade simulation
- No CPCV — deferred to future amendment if N_OOS power floor remains active in Q3-2026
- No portfolio additivity audit (RULE 7 overlap concern; deferred to Phase E)
- No capital deployment — that requires Phase E + capital-review skill + explicit user GO