# D6 Result — MNQ COMEX_SETTLE GARCH>70 sizing-overlay (Pathway B K=1)

**Date:** 2026-04-29
**Pre-reg:** [`docs/audit/hypotheses/2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1.yaml`](docs\audit\hypotheses\2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1.yaml)
**Pre-reg commit_sha:** `54b7b948`
**DB max trading_day:** 2026-04-26
**Holdout boundary (Mode A):** 2026-01-01
**Verdict:** **PARK_CONDITIONAL_DEPLOY_RETAINED**

## Scope

This document is the locked Pathway B K=1 confirmatory test result for D6: the question of whether `garch_forecast_vol_pct > 70` stratifies per-trade R on the deployed `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` lane sufficiently to warrant a CONDITIONAL_DEPLOY (Phase 1 shadow-only observation overlay). Scope is exactly the deployed lane (single instrument, single session, single ORB aperture, single RR target, single confirm-bars setting, single base filter), partitioned by the locked binary regime predicate (garch>70). No threshold sweeps. No alternative predictors. No cross-instrument or cross-session expansion. No post-hoc rescue permitted. Amendment 3.2 NOT_OOS_CONFIRMABLE classification (computed at pre-reg write time, MinTRL 5.10y on IS-rate basis or 2.37y on observed-OOS-rate basis) drove the choice of CONDITIONAL_DEPLOY over KEEP_PARKED_INDEFINITELY at write time. This result confirms whether the locked decision_rule's `continue_if`, `park_if`, or `kill_if` clause fires.

## Pressure test (RULE 13 — required pre-verdict)

- corr(pnl_r, pnl_r) = 1.000000  (must be ~1.0)
- corr(is_win, pnl_r) = 0.989028  (post-trade label association)
- T0 threshold: 0.7
- Triggered correctly: **True**

## IS partition (deployed lane, ORB_G5 base, both sides, garch>70 gate)

| Subset | N | ExpR | sd |
|---|---:|---:|---:|
| baseline | 1389 | 0.104908 | 1.145453 |
| gate-on (garch>70) | 370 | 0.247738 | 1.171162 |
| gate-off (garch<=70) | 1019 | 0.053047 | 1.132103 |
| **lift (on - off)** | — | **0.194691** | — |

- Welch t (lift) = 2.7631, df = 635.4, p = 0.005892
- IS gate-on vs zero: t = 4.0689

## C9 era stability (per IS year, gate partition)

| year | gate-on N | gate-on ExpR | gate-off N | gate-off ExpR | flags |
|---|---:|---:|---:|---:|---|
| 2020 | 15 | 0.3798 | 132 | -0.0194 | counted_positive |
| 2021 | 48 | 0.4564 | 200 | -0.0463 | counted_positive |
| 2022 | 138 | 0.0478 | 112 | -0.0543 | counted_positive |
| 2023 | 10 | -0.3115 | 238 | 0.1590 |  |
| 2024 | 83 | 0.2857 | 166 | 0.0566 | counted_positive |
| 2025 | 76 | 0.4852 | 171 | 0.1446 | counted_positive |

- C9 strict fails (N>=50, ExpR<-0.05): []
- Positive years (N>=10): [2020, 2021, 2022, 2024, 2025] — count 5
- Required: >= 4

## OOS partition (accruing forward shadow)

| Subset | N | ExpR | sd |
|---|---:|---:|---:|
| gate-on | 37 | 0.358235 | 1.176225 |
| gate-off | 33 | -0.335176 | 1.045810 |
| **OOS lift** | — | **0.693411** | — |

- OOS Welch t = 2.6109, df = 68.0, p = 0.011101
- dir_match: **True**
- OOS_lift >= 0.4 * IS_lift: **True**
- C8 status: **GATE_INACTIVE_LOWPOWER** (Amendment 3.1; Amendment 3.2 NOT_OOS_CONFIRMABLE classification at pre-reg time, CONDITIONAL_DEPLOY chosen)

## Decision

**PARK_CONDITIONAL_DEPLOY_RETAINED**

- |IS t|=2.7631 in [2.5, 3.0): below Chordia with-theory but above raw p<0.05 floor
- IS lift=0.1947 OK; positive_mean_floor OK; C9 OK
- CONDITIONAL_DEPLOY (Phase 1 shadow-only) retained; Phase 2 size-multiplier pre-reg gated

## Reproduction

```
python research/phase_d_d6_mnq_comex_settle_overlay_pathway_b.py
```

Inputs: gold.db at `pipeline.paths.GOLD_DB_PATH`, Python with duckdb+scipy+pandas+numpy.
Canonical layers only (`bars_1m`, `daily_features`, `orb_outcomes`).
Filter delegation via `research.filter_utils.filter_signal` (RULE 9; no inline ORB_G5 re-encoding).

## Limitations

- Single cell, single lane, single feature. K=1 by Pathway B / Amendment 3.0.
- OOS still accruing — C8 N_OOS<50 floor not met; Amendment 3.2 already classified NOT_OOS_CONFIRMABLE at pre-reg write time.
- Phase 1 shadow-only carrier wiring is OUT OF SCOPE for this runner (separate commit).
- Phase 2 size-multiplier pre-reg gated on this verdict; not authored.
- SR monitor on the deployed lane is the live circuit-breaker (60-day ARL); KILL_SR_ALARM action defined in pre-reg.

## Cross-references

- D4 result: `docs/audit/results/2026-04-28-mnq-comex-settle-pathway-b-v1-result.md` (PARK_PENDING_OOS_POWER stands)
- D5 result: `docs/audit/results/2026-04-28-mnq-comex-settle-d5-both-sides-pathway-b-v1-result.md` (KILL stands)
- Additivity triage: `docs/audit/results/2026-04-29-parked-pathway-b-additivity-triage.md` § Addendum (Path 3 origin)
- Amendment 3.2: `docs/institutional/pre_registered_criteria.md` § Amendment 3.2 (MinTRL classification)
