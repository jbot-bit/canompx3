# Path C — Self-Audit Addendum (Honest Correction)

**Date:** 2026-04-15
**Parent:** `docs/audit/results/2026-04-15-path-c-h2-closure.md`
**Trigger:** User requested rigorous bias/bug review — "ensure we handled correctly and no bias or bugs etc. logical and honest. ensure profit extracted."

This addendum corrects the Path C closing verdict in one material way and reinforces it with more rigorous tests.

---

## Bug checks

| Check | Result |
|---|---|
| Sample count discrepancy 198 vs 213 in prior horizon audit | **Consistent.** 213 was IS+OOS union; 198 is IS-only. Difference = 15 OOS fires, which matches the Path C OOS 4-cell table (garch=1 → 8 + 7 = 15). No bug. |
| `var_sr` label in report — "from N=6 row sample" | **Text bug, not computation bug.** `step1['var_sr']` was computed as `universality["sr_on"].var(ddof=1)` across all 527 cells. Label printed `len(step1['rows'])` = 6 (the audited cells). Fix applied to `research/close_h2_book_path_c.py`. |
| IS-OOS leakage | Clean — `trading_day.year < 2026` applied consistently. rel_vol Q3 cutoff derived from IS only, applied to both windows. |
| Look-ahead in features | `garch_forecast_vol_pct` is computed at prior close (per `pipeline/build_daily_features.py` — forecast-at-t-1 feature). `rel_vol_SESSION` uses break-bar volume + prior 20-day median (cleared in rel_vol v2 stress-test). Both trade-time-knowable. |

---

## Rigor upgrades on the three Path C tests

### Step 2 T5 — binomial vs arbitrary threshold

The 60% threshold was cosmetic. Proper null test:

- Under H0 (signal is random): P(positive delta) = 0.5
- Observed: 361/527 positive
- **Binomial z-score: 8.49**
- **p-value (one-sided): 6.16 × 10⁻¹⁸**

The 60% threshold is not load-bearing — the finding crushes any reasonable threshold by 17 orders of magnitude. T5 PASS stands.

### Step 3 composite — additive interaction > max-marginal

My original "synergy" test asked "does BOTH beat the better single?" That's valid but conservative. The standard econometric test is an additive interaction term:

```
Both_additive_predicted = Neither + (garch_only - Neither) + (rel_only - Neither)
                        = -0.022 + 0.285 + 0.118
                        = +0.381
Both_actual            = +0.220
Interaction            = -0.161
```

**Interaction = −0.161 R per trade** — signals actively COMPETE, not merely subsume. Stronger negative than my max-marginal test (−0.043). Composite verdict reinforced.

Why do they compete? Likely because high garch forecast vol correlates with market regimes where actual realized volume is LESS informative (noise regime) — signals point in opposite informational directions in specific conditions.

### Step 1 DSR — var_sr sensitivity check

My var_sr=0.0174 is computed on 527 universality cells (all IS, N≥30 both sides). This is the cleanest population available for THIS experiment. But DSR is sensitive to var_sr choice. Sensitivity bracket:

| var_sr | Source | H2 DSR at K=36 |
|---|---|---|
| 0.012 | v2 rel_vol stress test (different population) | 0.71 |
| 0.0174 | Path C 527-cell universality (this analysis) | 0.46 |
| 0.047 | `dsr.py` default (experimental_strategies, wrong population) | 0.04 |

At any reasonable var_sr, H2 fails DSR at K=36 (top-family count) and higher K. DSR verdict "ambiguous-to-failing at honest K" stands across the sensitivity range.

---

## Honest profit extraction (cost-normalized)

**This is where the prior Path C report was INCOMPLETE.** The dollar-per-trade comparison across 4 cells was misleading because garch-fire days correlate with bigger ORB sizes, so `risk_dollars` per trade is inflated. Per-R is the correct cost-normalized metric.

### H2 raw cell (MNQ COMEX_SETTLE O5 RR1.0 long E2, IS 2020-05 to 2025-12, 5.58 years)

Unfiltered baseline:
- N=876 trades, ExpR=+0.069, SR_per_trade=+0.077, WR=59.1%
- Total = $5,476 at avg $6.25/trade = $981/yr per 1 MNQ contract
- **The raw cell is already genuinely profitable without any filter.**

OOS (2026-01 to 2026-04): N=34, ExpR=+0.002 (thin, near-flat).

### MNQ COMEX_SETTLE E2 RR1.0 is ALREADY validated with 5 filters in `validated_setups`

| Deployed filter | N | ExpR | SR | t/yr | **Ann R/yr** |
|---|---|---|---|---|---|
| ORB_G5 | 1473 | +0.089 | 0.098 | 246 | **21.9** |
| COST_LT12 | 1247 | +0.110 | 0.121 | 208 | **22.9** |
| OVNRNG_100 | 520 | +0.172 | 0.189 | 87 | **15.0** |
| X_MES_ATR60 | 673 | +0.151 | 0.167 | 114 | **17.2** |
| ORB_G5_NOFRI | 1182 | +0.104 | 0.115 | 198 | **20.5** |

Note: these N include BOTH long AND short breaks (my H2 was long-only → half the trade count by construction).

### Path C "validated" garch-only on long-only subset

- N=123 IS, ExpR=+0.263, 22 t/yr → **Ann R/yr = 5.8**

### Honest comparison

**Garch-only delivers 5.8 R/yr. ORB_G5 delivers 21.9 R/yr on the same cell.** At fixed-risk position sizing (the institutional standard), garch-only would REDUCE portfolio R/yr vs the deployed status quo by ~74%.

The $/trade gap (garch $17.48 vs COST_LT12 ~$26) is a position-size effect — garch-fire days have bigger ORBs, not better edges. Once you cost-normalize, ORB_G5 and COST_LT12 both dominate garch-only.

### Profit-extraction ranking (cost-normalized, raw cell, IS)

| Deploy rule | N | Ann R/yr | Status |
|---|---|---|---|
| **COST_LT12** (deployed) | 1247 | **22.9** | Existing validated |
| **ORB_G5** (deployed) | 1473 | **21.9** | Existing validated (primary) |
| **ORB_G5_NOFRI** (deployed) | 1182 | 20.5 | Existing validated |
| **X_MES_ATR60** (deployed) | 673 | 17.2 | Existing validated |
| **OVNRNG_100** (deployed) | 520 | 15.0 | Existing validated |
| unfiltered raw long | 876 | 10.8 | Not deployed |
| garch-only long | 123 | 5.8 | Path C "validated" — UNDERPERFORMS |
| rel-only long | 245 | 4.6 | UNDERPERFORMS |
| composite AND long | 75 | 3.0 | WORST |

**Verdict:** H2 does not beat the incumbent. Deploying garch-only at this cell would be a portfolio downgrade.

---

## Corrected verdicts

1. **H2 garch filter is NOT better than the deployed ORB_G5 / COST_LT12 filters on this cell.** Prior Path C implied "ship garch alone if anything ships" — that framing was incomplete because it didn't compare against what's already live. Corrected: **garch-only under-delivers vs status quo at the H2 cell by ~74% on R/yr.**
2. **rel_vol × garch composite is doubly dead** — additive interaction −0.161 (stronger than max-marginal test). Not deploying.
3. **T5 universality finding stands** at p=6×10⁻¹⁸. Garch IS a real signal cross-session cross-instrument. But "real" ≠ "beats incumbent on any given cell."
4. **DSR ambiguous-to-failing at honest K** across var_sr sensitivity bracket 0.012-0.047.

## What would actually improve the portfolio?

The correct next test is NOT "deploy garch as filter" — that's proven to underperform here. The two open questions:

1. **Does garch work at cells that DON'T already have a deployed filter?** Run the universality scan survivors that are NOT in `validated_setups`. Deployable-but-novel cells.
2. **Does garch work as a SIZER on top of ORB_G5-filtered trades?** Carver forecast: when garch=HIGH on an ORB_G5 fire day, size up; when LOW, size down. Per-R metric on sized returns vs flat. This is the R5 deployment role (`docs/institutional/mechanism_priors.md`) — we tested R1 (filter) and got a negative result; R5 is a different hypothesis.

Neither is a trivial test; both should be pre-registered before execution. Added to next-session queue.

---

## Files touched

- `research/close_h2_book_path_c.py` — text label fix (no compute change)
- `docs/audit/results/2026-04-15-path-c-self-audit-addendum.md` — this file

No production code touched. No validated_setups writes.
