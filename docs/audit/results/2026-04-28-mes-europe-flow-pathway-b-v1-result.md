# Phase D D1 — MES EUROPE_FLOW Pathway B K=1 — Result

**Pre-reg:** docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml
**Pre-reg commit:** d58e5ce295fb5fcddb918e83084f41a4fd72b7d5
**Run timestamp:** 2026-04-28
**DB freshness:** orb_outcomes max trading_day = 2026-04-26
**Holdout boundary (Mode A):** 2026-01-01

## Scope

Pathway B K=1 confirmatory test of B-MES-EUR (the highest-EV PATHWAY_B_ELIGIBLE candidate from the 2026-04-28 Phase B per-candidate evidence pass). The question this run answers: under the locked schema, do all KILL criteria pass and do the C5/C6/C7/C8/C9 gates support promotion to CANDIDATE_READY, or does the OOS power floor force a PARK verdict, or does any KILL criterion fire?

## Locked schema (verbatim from pre-reg)

- instrument: MES
- session: EUROPE_FLOW
- orb_minutes: 15
- rr_target: 1.0
- entry_model: E2
- confirm_bars: 1
- direction: long
- feature: overnight_range_pct
- feature_threshold: 80.0
- feature_op: >
- scratch_policy: include-as-zero (commit 68ee35f8)

## IS / OOS reproduction (Mode A)

| Metric | IS | OOS |
|---|---:|---:|
| N total | 888 | 37 |
| N on (signal=true) | 186 | 9 |
| N off | 702 | 28 |
| ExpR on | +0.1434 | +0.2331 |
| ExpR off | -0.1024 | -0.0389 |
| Δ (on − off) | +0.2459 | +0.2721 |
| Sharpe per-trade | +0.1658 | +0.2516 |
| Sharpe annualised | +0.8874 | — (UNVERIFIED if N_OOS<50) |
| Skewness γ̂₃ | -0.5515 | — |
| Kurtosis γ̂₄ (excess) | -1.6568 | — |

## Significance (IS, K=1 — see pre-reg testing_discipline)

- Welch t-stat: +3.4698
- Welch p (two-tailed): 0.000602
- Block-bootstrap p (block=5, B=10000): 0.000500
- Cohen's d (IS effect): +0.2914
- OOS power for d at N_OOS_on=9: 0.1063

## Phase 0 gates (post-Amendment 3.0/3.2)

| Gate | Threshold | Computed | Verdict |
|---|---|---|---|
| C5 DSR (Pathway A K_family=1850) [provenance] | ≥ 0.95 | 0.0000 | DSR_PA_FAIL |
| C5 DSR (Pathway B K=1) [primary gate] | ≥ 0.95 | 0.9845 | DSR_PB_PASS |
| C6 WFE | ≥ 0.5 (if N_OOS≥50) | GATE_INACTIVE_LOWPOWER | C6_GATE_INACTIVE_LOWPOWER |
| C7 N_IS_on | ≥ 100 | 186 | C7_PASS |
| C8 dir-match (power-floor aware) | dir_match AND power≥0.5 | dir=True, power=0.106 | C8_GATE_INACTIVE_LOWPOWER |
| C9 era stability (N_era≥20, no era < -0.05) | PASS | C9_PASS | C9_PASS |

## Per-year IS breakdown

| Year | N | ExpR |
|---:|---:|---:|
| 2019 | 12 | +0.0034 |
| 2020 | 37 | +0.2761 |
| 2021 | 23 | +0.0948 |
| 2022 | 24 | +0.1699 |
| 2023 | 28 | +0.1464 |
| 2024 | 29 | +0.1738 |
| 2025 | 33 | +0.0310 |

## Kill-criterion results (locked from pre-reg)

| ID | Threshold | Computed | Verdict |
|---|---|---|---|
| KILL_RAWP | >= 0.05 | welch_p = 0.000602 | PASS |
| KILL_DIR | negative on IS | sign(delta_IS) = + | PASS |
| KILL_T | < 3.0 | |t| = 3.4698 | PASS |
| KILL_N | < 100 | N_IS_on = 186 | PASS |
| KILL_ERA | < -0.05 (any era N>=20) | min_era_ExpR(eligible) = +0.0310 | PASS |
| KILL_SHARPE | <= 0.0 | sharpe_ann_IS = +0.8874 | PASS |
| KILL_BASELINE_SANITY | > 0.001 | |delta_IS - expected| = 0.000048 | PASS |

## Decision rule outcome

**VERDICT: PARK_PENDING_OOS_POWER**

All non-conditional gates PASS (C5 DSR_PB=0.9845, C7 N=186, C9 era stable, Sharpe_ann_IS=+0.8874). C6/C8 are GATE_INACTIVE_LOWPOWER because N_OOS_on=9 < 50 power floor (Amendment 3.2). UNVERIFIED ≠ KILL — cell parks until N_OOS_on accrues to ≥50 (estimate Q3-2026 at current trade rate). Pre-reg locked; no re-tuning permitted on accrual.

## Audit pressure-test (RULE 13)

- Pressure test: PASS

## Reproduction

```
DUCKDB_PATH=C:\Users\joshd\canompx3\gold.db python research/phase_d_d1_mes_europe_flow_pathway_b.py
```

- DB: `pipeline.paths.GOLD_DB_PATH`
- Holdout: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` (2026-01-01)
- Pre-reg: locks all schema parameters and kill criteria
- Helpers: imported from `research/phase_b_candidate_evidence_v1.py` (canonical)

## Caveats / limitations

- N_OOS_on is small (3-month sacred holdout). C8 power floor invokes UNVERIFIED, not KILL.
- This is one Pathway B test, K=1 by design. The decision rule's PARK case is the legitimate
  verdict when OOS is underpowered; UNVERIFIED is not failure per Amendment 3.2.
- DSR Pathway A K_family is approximated at 1850 (overnight family proxy from Phase B). Sensitivity
  not exhaustive — true K may differ; the Pathway B K=1 path is what governs promotion here.

## Not done by this run

- No write to validated_setups, edge_families, lane_allocation, live_config
- No paper trade simulation
- No CPCV — deferred to future amendment if N_OOS power floor remains active in Q3-2026
- No capital deployment — that requires Phase E + capital-review skill + explicit user GO