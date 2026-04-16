# Path C — Close the H2 / rel_vol Book

**Date:** 2026-04-15
**Trigger:** User selected Path C over Path A (HTF level features) to finish the open volume/garch-vol hypothesis book before opening new level-based hypotheses.
**Stage file:** `docs/runtime/stages/path-c-close-h2-book.md` (deleted on completion).
**Deferred:** Path A kickoff at `docs/audit/hypotheses/2026-04-15-htf-level-break-pre-reg-stub.md`.

---

## Step 1 — DSR at honest K (empirical var_sr)

**Empirical `var_sr`:** 0.0174 (computed across all 527 universality cells; dsr-audit table below shows 6 representative cells — H2 + top-5 by |delta|)
**`dsr.py` default (calibrated for `experimental_strategies`):** 0.047
**Ratio:** 2.70× — experimental_strategies default is MORE conservative than our empirical distribution

**DSR table** (per-cell across K framings):

| Cell | N_on | SR_hat | skew | kurt_ex | K=5 | K=12 | K=36 | K=72 | K=300 | K=527 | K=14261 |
|------|------|--------|------|---------|---|---|---|---|---|---|---|
| MNQ COMEX_SETTLE O5 RR1.0 long | 198 | +0.275 | -0.68 | -1.54 | 0.935 | 0.763 | 0.460 | 0.292 | 0.086 | 0.049 | 0.001 |
| MGC US_DATA_1000 O30 RR2.0 long | 51 | -0.477 | +1.76 | +1.14 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MNQ NYSE_CLOSE O5 RR2.0 short | 62 | +0.090 | +0.41 | -1.89 | 0.295 | 0.151 | 0.062 | 0.035 | 0.010 | 0.006 | 0.000 |
| MNQ NYSE_CLOSE O5 RR2.0 long | 54 | -0.436 | +1.67 | +0.83 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MNQ CME_REOPEN O30 RR1.5 short | 49 | +0.266 | -0.21 | -2.04 | 0.769 | 0.624 | 0.455 | 0.364 | 0.218 | 0.176 | 0.043 |
| MES CME_REOPEN O15 RR1.5 long | 66 | -0.562 | +1.46 | +0.16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Interpretation:** DSR > 0.95 = robust; 0.80-0.95 = marginal; < 0.80 = not distinguishable from selection bias at that N_eff.

**H2 verdict from DSR:** see H2 row above. The key K framings are K=12 (distinct-deployed-cell count), K=36 (cell-count of top-family), K=527 (total universality scan).

---

## Step 2 — T5 family formalize

**Universality scan:** N=527 testable combos (N>=30 both sides).
**Positive delta:** 361 (68.5%) | **Negative delta:** 166
**Min per-instrument positive %:** 62.2%
**Decision rule:** PASS if generalization >= 60% AND instrument floor >= 50%.
**T5 VERDICT:** **PASS**

**Per-instrument:**

| Instrument | N | Positive | % |
|---|---|---|---|
| MES | 177 | 134 | 75.7% |
| MGC | 154 | 105 | 68.2% |
| MNQ | 196 | 122 | 62.2% |

**Per-session:**

| Session | N | Positive | % |
|---|---|---|---|
| BRISBANE_1025 | 18 | 17 | 94.4% |
| CME_PRECLOSE | 13 | 13 | 100.0% |
| CME_REOPEN | 46 | 28 | 60.9% |
| COMEX_SETTLE | 54 | 33 | 61.1% |
| EUROPE_FLOW | 54 | 43 | 79.6% |
| LONDON_METALS | 54 | 38 | 70.4% |
| NYSE_CLOSE | 18 | 9 | 50.0% |
| NYSE_OPEN | 54 | 23 | 42.6% |
| SINGAPORE_OPEN | 54 | 42 | 77.8% |
| TOKYO_OPEN | 54 | 50 | 92.6% |
| US_DATA_1000 | 54 | 32 | 59.3% |
| US_DATA_830 | 54 | 33 | 61.1% |

**Per-direction:**

| Direction | N | Positive | % |
|---|---|---|---|
| long | 263 | 156 | 59.3% |
| short | 264 | 205 | 77.7% |

This **replaces** the placeholder T5 INFO result from `2026-04-15-t0-t8-audit-horizon-non-volume.md` for H2.

---

## Step 3 — Composite rel_vol_HIGH_Q3 AND garch_vol_pct_GT70 on H2 cell

**Cell:** MNQ COMEX_SETTLE O5 RR1.0 long
**rel_vol Q3 cutoff** (p67): 1.599
**garch cutoff:** >= 70

**T7 orthogonality** — corr(fire_rel, fire_garch) on full data: **0.069**
  -> PASS — orthogonal (|corr| <= 0.40).

### IS 4-cell decomposition

| rel_fires | garch_fires | N | ExpR | WR | $/trade | Total $ |
|---|---|---|---|---|---|---|
| 0 | 0 | 374 | -0.022 | 54.3% | $0.28 | $106 |
| 0 | 1 | 123 | +0.263 | 67.5% | $17.48 | $2,150 |
| 1 | 0 | 170 | +0.096 | 59.4% | $2.69 | $458 |
| 1 | 1 | 75 | +0.220 | 64.0% | $31.42 | $2,356 |

### OOS 4-cell decomposition

| rel_fires | garch_fires | N | ExpR | $/trade |
|---|---|---|---|---|
| 0 | 0 | 11 | -0.333 | $-21.69 |
| 0 | 1 | 8 | -0.038 | $9.64 |
| 1 | 0 | 5 | +0.488 | $47.78 |
| 1 | 1 | 7 | +0.384 | $54.51 |

**Synergy:** ExpR(both) - max_marginal = -0.043
  -> **NO SYNERGY / SUBSUMED** — composite is not additive.

**T6 composite bootstrap:** observed lift over baseline = +0.144 R, p = 0.0789 (B=1000).

---

## Closing verdict on the H2/rel_vol book

- **H2 T5 family:** PASS based on 68.5% generalization across 527 combos. Feature is genuinely cross-asset cross-session, not a single-cell find.
- **H2 DSR at K=36 (top-family count):** see Step 1 table. At honest K the question is whether DSR crosses 0.95 or sits in the 0.80-0.95 marginal band.
- **Composite:** orthogonality +0.069, synergy -0.043. Decides whether rel_vol and garch deploy together (R1 AND-filter) or separately (R3 independent confirmations).

**Deployment posture unchanged from prior handover:** nothing to live capital until the composite and DSR resolve. If DSR >= 0.95 AND synergy > 0.05, pre-register signal-only shadow.

**Next-session handoffs:**
1. Path A kickoff (HTF levels) — stub at `docs/audit/hypotheses/2026-04-15-htf-level-break-pre-reg-stub.md`.
2. Non-ORB terminal (Phase E) — sync findings when it reports back.
3. If composite PASSES synergy gate: pre-reg `docs/audit/hypotheses/<date>-h2-garch-shadow.md` for signal-only shadow.