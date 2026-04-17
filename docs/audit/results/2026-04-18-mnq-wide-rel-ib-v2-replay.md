# MNQ Wide-Rel-IB v2 — Stage 1 Replay

**Date:** 2026-04-18
**As-of trading day:** 2026-04-17
**Pre-reg:** `docs/audit/hypotheses/2026-04-18-mnq-wide-rel-ib-v2.yaml`
**Design:** `docs/plans/2026-04-18-mnq-wide-rel-ib-v2-design.md`
**Family verdict:** **NULL** (0/6 cells pass primary)

## Tautology pre-gate (Rule 7 canonical fire correlation)

| Lane | Alt filter | fire_rho | |rho|>0.70? | Verdict |
|---|---|---:|:---:|:---:|
| MNQ CME_PRECLOSE | X_MES_ATR60 | +0.0045 | no | distinct |
| MNQ TOKYO_OPEN | COST_LT12 | +0.4743 | no | distinct |

## Per-cell primary evaluation

| Cell | IS N(W+G5/G5only) | ExpR IS (W+G5 / G5only) | delta_IS | t_IS | p_IS | BH K=6 | OOS N | delta_OOS | eff_ratio | dir_match | WFE | PRIMARY |
|---|---|---|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|:---:|
| CME_PRECLOSE_RR1.0 | 497/913 | +0.1960 / +0.0572 | +0.1387 | +2.74 | 0.0063 | PASS | 27 | +0.0895 | +0.645 | PASS | -0.043 | **FAIL** |
| CME_PRECLOSE_RR1.5 | 413/843 | +0.1632 / +0.0574 | +0.1058 | +1.51 | 0.1320 | FAIL | 23 | -0.2649 | -2.504 | FAIL | -2.568 | **FAIL** |
| CME_PRECLOSE_RR2.0 | 346/770 | +0.0757 / +0.0315 | +0.0442 | +0.50 | 0.6164 | FAIL | 23 | -0.1257 | -2.844 | FAIL | -3.607 | **FAIL** |
| TOKYO_OPEN_RR1.0 | 690/909 | +0.1313 / +0.0193 | +0.1120 | +2.53 | 0.0114 | PASS | 37 | +0.1340 | +1.196 | PASS | +0.618 | **FAIL** |
| TOKYO_OPEN_RR1.5 | 690/909 | +0.1618 / +0.0414 | +0.1204 | +2.13 | 0.0335 | FAIL | 37 | +0.4330 | +3.598 | PASS | +2.119 | **FAIL** |
| TOKYO_OPEN_RR2.0 | 689/909 | +0.1173 / +0.0646 | +0.0527 | +0.79 | 0.4300 | FAIL | 37 | +0.4465 | +8.471 | PASS | +3.159 | **FAIL** |

## Per-criterion breakdown

| Cell | BH K=6 | t>=3.79 | WFE>=0.50 | N_OOS>=30 | dir_match | eff>=0.40 | tautology_ok | PRIMARY |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CME_PRECLOSE_RR1.0 | PASS | FAIL | FAIL | FAIL | PASS | PASS | PASS | **FAIL** |
| CME_PRECLOSE_RR1.5 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | **FAIL** |
| CME_PRECLOSE_RR2.0 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | **FAIL** |
| TOKYO_OPEN_RR1.0 | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | **FAIL** |
| TOKYO_OPEN_RR1.5 | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | **FAIL** |
| TOKYO_OPEN_RR2.0 | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | **FAIL** |

## BH-FDR K=6 rank table

| rank | cell | p_IS | threshold | pass |
|---:|---|---:|---:|:---:|
| — | CME_PRECLOSE_RR1.0 | 0.006279 | 0.008333 | PASS |
| — | TOKYO_OPEN_RR1.0 | 0.011426 | 0.016667 | PASS |
| — | TOKYO_OPEN_RR1.5 | 0.033528 | 0.025000 | FAIL |
| — | CME_PRECLOSE_RR1.5 | 0.132016 | 0.033333 | FAIL |
| — | TOKYO_OPEN_RR2.0 | 0.429969 | 0.041667 | FAIL |
| — | CME_PRECLOSE_RR2.0 | 0.616432 | 0.050000 | FAIL |

## Family verdict: **NULL**

Cells passing primary: 0/6

Verdict semantics per design doc:
- STRONG_PASS (4-6 pass) — proceed to Stage 2, propose promotion
- STANDARD_PASS (2-3 pass) — promote passing cells only
- MARGINAL (1 pass) — reevaluate under allocator correlation gate
- NULL (0 pass) — close family, no rescue

## Binding-constraint analysis

Per-criterion pass rates across 6 cells:

| Criterion | Cells passing | Binding? |
|---|---:|:---:|
| BH-FDR K=6 | 2/6 | no |
| **t ≥ 3.79 (without-theory)** | **0/6** | **YES** — binding for all cells |
| WFE ≥ 0.50 | 3/6 | no |
| N_OOS ≥ 30 | 3/6 | no |
| dir_match | 4/6 | no |
| eff_ratio ≥ 0.40 | 4/6 | no |
| tautology_ok | 6/6 | no |

**Binding criterion: Chordia t ≥ 3.79 kills all 6 cells.** Strongest cell (CME_PRECLOSE_RR1.0) has t=2.74; strongest TOKYO_OPEN cell (RR1.0) has t=2.53. Had the pre-reg used the with-theory threshold (t ≥ 3.00), CME_PRECLOSE_RR1.0 would still fail (2.74 < 3.00), as would TOKYO_OPEN_RR1.0 (2.53 < 3.00). The NULL is not a close call.

## Qualitative signal on TOKYO_OPEN (noted, not rescued)

TOKYO_OPEN cells show OOS `eff_ratio` > 1 on all 3 RR (OOS delta stronger than IS delta). That pattern triggers `LEAKAGE_SUSPECT` per `.claude/rules/quant-audit-protocol.md` § T3 (WFE > 0.95 on small OOS samples). N_OOS=37 is close to the 30 threshold — thin. The apparent OOS strengthening is more likely small-sample noise than regime evidence.

## Methodology transparency

- **WFE:** computed as single-split annualized-Sharpe ratio `Sharpe(OOS) / Sharpe(IS)` on `WIDE+G5` cells (not the 5-fold expanding-window WFE the pre-reg specified). For a NULL verdict this simplification does not change outcome (the t-threshold fails unambiguously), but for any future STRONG_PASS candidate this would need to be upgraded to proper 5-fold expanding WFE.
- **Tautology Rule 7:** canonical boolean fire correlation on the full trading-day universe (not IS-only). The rho values (0.005 and 0.474) are below 0.70 threshold; both lanes clear.
- **BH-FDR:** standard Benjamini-Hochberg at K=6 family framing, q=0.05. Secondary K_lane=3 framing would have passed 2 lanes at q=0.05 (both RR1.0 cells). Reporting the family framing as primary per the pre-reg.
- **All OOS queries made after IS scope was locked.** Scope was set from IS distinctness only per v2 pre-reg.

## Doctrine action

**Close the family at current scope.** Per pre-reg scope hard boundaries and kill criteria:
- No threshold sweep on 1.0x
- No trailing-window sweep on 20d
- No aperture expansion to O15 without Stage 2 authorization
- No instrument expansion to MES

**Stage 2 NOT triggered.** Stage 2 required ≥ 3 cells passing Stage 1 primary; actual count is 0. Per the pre-reg, no aperture/window/threshold rescue is permitted.

MNQ wide-rel-IB at O5 is CLOSED pending a fundamentally new mechanism claim.

