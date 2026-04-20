# Vol-Regime Confluence Portfolio-Generalization Test

**Date:** 2026-04-20
**Pre-reg:** `docs/audit/hypotheses/2026-04-20-vol-regime-confluence-portfolio-test.yaml` (locked + 2 amendments)
**Branch:** `research/vol-regime-confluence-2026-04-20`
**Scan script:** `research/vol_regime_portfolio_scan_v1.py`
**Verification scripts:** `research/vol_regime_independent_sql.py` (Stage K), `research/vol_regime_pressure_test.py` (Stage L), `research/vol_regime_gates_g_h_i_j.py` (G/H/I/J)
**Outputs:** `research/output/vol_regime_portfolio_scan_v1.{json,csv}`, `research/output/vol_regime_gates_results.json`
**Data cutoff:** MNQ/MES `orb_outcomes` through 2026-04-16 (69 OOS trading days)

---

## TL;DR

**Pre-committed verdict: COEXISTS_BOTH.**

After running the full institutional verification stack (13 gates across Stages D-L), **2 of 6 MNQ live allocator lanes** have confluence variants that add edge **beyond what the base filter already captures**:

- **MNQ COMEX_SETTLE E2 RR1.5 ORB_G5 × OVNRNG_100** — CONFIRMED (p_boot=0.013, all gates pass)
- **MNQ EUROPE_FLOW E2 RR1.5 ORB_G5 × OVNRNG_100** — CONFIRMED (p_boot=0.011, all gates pass)

**4 cells preliminarily passed BH-FDR + Chordia but fail the block-bootstrap null.** This is the key finding — the initial scan-v1 "PORTFOLIO_GENERAL" label (4/6 lanes) was *arithmetically correct* but *institutionally wrong*: the XMES_60 variants do not distinguish from random subsampling of the base-fire population.

**The vol-regime effect as a portfolio-level conditioner is NOT validated.** It is a 2-lane-specific, OVN-only finding. Phase 5 should pre-register these 2 OVN confluence candidates as per-lane additions, NOT an allocator-level regime conditioner.

---

## Hypothesis

Vol-regime confluence (F_lane ∧ OVNRNG_100, F_lane ∧ X_MES_ATR60, or F_lane ∧ session_london_range ≥ 100) adds marginal per-trade information to the 6 MNQ live allocator lanes.

**Theory grounding** (verbatim local extracts, verified 2026-04-20):
- Chan 2008 Ch 7 pp.106-107: "For stop loss to be beneficial, we must believe that we are in a momentum, or trending, regime. [...] In mean reverting regimes, stop-loss behavior is anti-edge."
- Chan 2008 Ch 7 p.120: "high- vs. low-volatility regimes [...] volatility regime switching seems to be most amenable to classical econometric tools such as the generalized autoregressive conditional heteroskedasticity (GARCH) model."
- Chan 2008 Ch 7 p.121: state-classification via current-volatility features.
- Fitschen 2013 Ch 3 Table 3.8 p.41: stock indices trend-follow on hourly bars.
- Chan 2013 Ch 7 p.155: ORB = stop-triggered breakout; cascading stop mechanism.
- Bailey 2013 Theorem 1: MinBTL bound (17 trials, 6.65yr available > 5.67yr required).
- Harvey-Liu 2015 p.16: BHY dependence-aware FDR.
- Chordia 2018: t ≥ 3.00 with theory.

---

## Scope (17 cells)

Temporal-alignment gate (RULE 1.2) restricts which features are valid per lane:

| Lane | ORB start (Brisbane) | OVN valid | London valid | Variants tested |
|---|---|:-:|:-:|---|
| EUROPE_FLOW | 17:00-18:00 | ✓ | ✗ (starts during London) | OVN, XMES, OVN∨XMES |
| SINGAPORE_OPEN | 11:00 | ✗ (during Asia) | ✗ | XMES only |
| COMEX_SETTLE | 03:30-04:30 next day | ✓ | ✓ | OVN, XMES, OVN∨XMES, LON |
| NYSE_OPEN | 23:30-00:30 next day | ✓ | ✓ | OVN, XMES, OVN∨XMES, LON |
| TOKYO_OPEN | 10:00 | ✗ | ✗ | XMES only |
| US_DATA_1000 | 00:00-01:00 next day | ✓ | ✓ | OVN, XMES, OVN∨XMES, LON |

Total: 17 cells. Pathway A (BH-FDR family), K_primary = 17.

MinBTL check: N=17, E=1.0 → MinBTL = 2·ln(17)/1² = 5.67yr < 6.65yr available. WITHIN strict Bailey E=1.0 bound.

---

## Stage D — Primary scan

Results (IS 2019-01-01 to 2025-12-31, OOS 2026-01-02 to 2026-04-16):

| # | Lane | Variant | N_IS | ExpR_IS | t_IS | p_IS | N_OOS | ExpR_OOS | dir_match | C8 ratio | fire rate | T0 corr | BH-FDR K=17 |
|---|---|---|---:|---:|---:|---:|---:|---:|:-:|---:|---:|---:|:-:|
| 1 | EUROPE_FLOW | OVN | 535 | +0.1776 | +3.54 | 0.0004 | 66 | +0.3750 | ✓ | 2.11 | 34% | +0.18 | PASS |
| 2 | EUROPE_FLOW | XMES | 712 | +0.0435 | +1.02 | 0.31 | 45 | +0.4073 | ✓ | 9.37 | 45% | +0.18 | pass |
| 3 | EUROPE_FLOW | OVN∨XMES | 882 | +0.0658 | +1.71 | 0.09 | 67 | +0.3545 | ✓ | 5.39 | 56% | +0.23 | pass |
| 4 | SINGAPORE | XMES | 706 | +0.1424 | +3.27 | 0.001 | 45 | −0.1475 | ✗ | −1.04 | 77% | +0.75 | pass |
| 5 | COMEX_SETTLE | OVN | 517 | +0.2099 | **+4.07** | **0.0001** | 60 | +0.1063 | ✓ | 0.51 | 33% | +0.14 | PASS |
| 6 | COMEX_SETTLE | XMES | 682 | +0.1592 | +3.57 | 0.0004 | 40 | +0.1918 | ✓ | 1.20 | 44% | +0.12 | pass |
| 7 | COMEX_SETTLE | OVN∨XMES | 844 | +0.1524 | +3.81 | 0.0002 | 61 | +0.0882 | ✓ | 0.58 | 54% | +0.16 | pass |
| 8 | COMEX_SETTLE | LON | 660 | +0.1537 | +3.37 | 0.0008 | 58 | +0.0610 | ✓ | 0.40 | 42% | +0.19 | pass |
| 9 | NYSE_OPEN | OVN | 535 | +0.0670 | +1.60 | 0.11 | 65 | +0.0898 | ✓ | 1.34 | 32% | +0.08 | pass |
| 10 | NYSE_OPEN | XMES | 717 | +0.1180 | +3.29 | 0.001 | 44 | +0.2525 | ✓ | 2.14 | 43% | +0.06 | PASS |
| 11 | NYSE_OPEN | OVN∨XMES | 887 | +0.0879 | +2.72 | 0.007 | 66 | +0.1032 | ✓ | 1.17 | 53% | +0.09 | pass |
| 12 | NYSE_OPEN | LON | 673 | +0.0879 | +2.36 | 0.02 | 62 | +0.1121 | ✓ | 1.28 | 40% | +0.10 | pass |
| 13 | TOKYO_OPEN | XMES | 516 | +0.1763 | +3.45 | 0.0006 | 45 | +0.1143 | ✓ | 0.65 | 54% | +0.26 | PASS |
| 14 | US_DATA_1000 | OVN | 484 | +0.0989 | +1.79 | 0.07 | 52 | +0.2279 | ✓ | 2.30 | 33% | +0.04 | pass |
| 15 | US_DATA_1000 | XMES | 644 | +0.0743 | +1.56 | 0.12 | 36 | +0.2307 | ✓ | 3.10 | 43% | +0.05 | pass |
| 16 | US_DATA_1000 | OVN∨XMES | 797 | +0.0909 | +2.12 | 0.03 | 53 | +0.2047 | ✓ | 2.25 | 54% | +0.06 | pass |
| 17 | US_DATA_1000 | LON | 593 | +0.0788 | +1.58 | 0.11 | 49 | +0.1043 | ✓ | 1.32 | 40% | +0.04 | pass |

Cells passing all 8 primary gates: **6/17**. Lanes with ≥1 passing variant: **4/6**.

Preliminary verdict: **PORTFOLIO_GENERAL** (if stopped here — WRONG).

---

## Stage K — Independent SQL verification

Pure-SQL reproduction of the 6 survivors, bypassing `research.filter_utils.filter_signal`. Filter equivalences derived from canonical source:

| Filter | SQL equivalent |
|---|---|
| ORB_G5 | `orb_{session}_size >= 5` |
| COST_LT12 | `orb_{session}_size >= (88/12) × total_friction / point_value` = 10.7067 for MNQ |
| OVN_100 | `overnight_range >= 100` |
| XMES_60 | `mes_atr_20_pct >= 60` |

**All 6 survivors reproduce to exact 4dp** on N, ExpR, and t. Δ_N = 0 on every cell. No filter-delegation drift.

---

## Stage L — Pressure test (RULE 13)

Validated the block-bootstrap infrastructure on MNQ COMEX_SETTLE E2 RR1.5:

1. **Known survivor (OVN_100)**: null_mean = +0.0949 ≈ base_fire_ExpR (+0.0915) ✓. Observed +0.2099 > null p95 +0.1782. p_boot = 0.013. **Correctly rejects null on known-real signal.**
2. **Random-mask null (30 trials × 500 perms)**: mean p_boot = 0.535 (uniform), FP rate at p<0.05 = 0.00. **No false positives.**
3. **Lookahead injection (pnl≥0 as variant)**: t = 335 >> RULE 12 red-flag of 10. **Surfaces correctly.**

**Design lesson recorded:** Initial pressure test used "random subsample with same fire rate" as null — produced mean t = 1.74 and FP rate 40%. Root cause: any random subsample of a positive-base-population inherits sqrt(N_sub/N_base) × t_base bias. The correct null is block-bootstrap on pnl_r within base-fires with mask FIXED.

---

## Stages G + H + I + J — Full verification stack

### Stage G — T4 Sensitivity (±20% threshold grids)

All 6 cells: **PASS**. No sign flips. Grid min t-stats stay ≥ 2.65 (EUROPE_FLOW at OVN=80) and ≥ 3.29 at OVN=100.

### Stage H — T6 Null bootstrap (moving-block, 5000 perms, block size = √N)

Correct null: "does variant add info beyond base filter?" Resample pnl_r in blocks within base-fires; mask FIXED.

| # | Cell | observed | null_mean | null_p95 | p_boot | Gate |
|---|---|---:|---:|---:|---:|:-:|
| 5 | COMEX_SETTLE × OVN | +0.2099 | +0.0942 | +0.1772 | **0.0126** | **PASS** |
| 1 | EUROPE_FLOW × OVN | +0.1776 | +0.0634 | +0.1465 | **0.0112** | **PASS** |
| 6 | COMEX_SETTLE × XMES | +0.1592 | +0.0937 | +0.1663 | 0.0718 | FAIL |
| 7 | COMEX_SETTLE × OVN∨XMES | +0.1524 | +0.0937 | +0.1596 | 0.0684 | FAIL |
| 10 | NYSE_OPEN × XMES | +0.1180 | +0.0826 | +0.1386 | 0.1466 | FAIL |
| 13 | TOKYO_OPEN × XMES | +0.1763 | +0.1296 | +0.2143 | 0.1804 | FAIL |

**Critical finding: all XMES-only variants fail the correct null.** Their marginal edge over the base filter is indistinguishable from random block-bootstrap subsampling.

**Interpretation:** Under BH-FDR + Chordia alone, these cells look significant *as standalone tests*. Under the proper null that asks "does the variant mask add information beyond what the base filter already captures?" — XMES does not. The OVN-variant effect is distinct.

### Stage I — T7 per-year stability + 2019-exclude sensitivity

All 6 cells: **PASS**. Positive years dominate (5-6 of 7 IS years). 2019-exclude re-run produces t-stat within ±0.1 of full-IS t-stat on every cell — no 2019-drag dependence.

### Stage J — OVN_P75 non-stationarity reframe

Rolling 252-day percentile ≥ 75 substituted for absolute OVN ≥ 100 (IS-only per 2026-04-19 feature-lookahead addendum). Only applicable to OVN-based variants.

| Cell | Absolute OVN_100 t | OVN_P75 t | OVN_P75 fire_rate | Verdict |
|---|---:|---:|---:|---|
| 5 COMEX_SETTLE OVN | +4.07 | +3.60 | 28% | Edge persists |
| 1 EUROPE_FLOW OVN | +3.54 | +3.16 | 28% | Edge persists |
| 7 COMEX_SETTLE OVN∨XMES | +3.81 | +3.81 | (same-rate combined) | (n/a — mixed) |

The OVN edge is **not an artifact of distribution drift** — it holds under a regime-standardized relative-percentile threshold with stable fire rate.

---

## Integrated verdict per cell

| # | Cell | Stage D BH | Stage K SQL | Stage G ±20% | Stage H null | Stage I year+2019 | Stage J OVN_P75 | Integrated |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 5 | COMEX_SETTLE × OVN | ✓ | ✓ | ✓ | **✓** | ✓ | ✓ | **CONFIRMED** |
| 1 | EUROPE_FLOW × OVN | ✓ | ✓ | ✓ | **✓** | ✓ | ✓ | **CONFIRMED** |
| 6 | COMEX_SETTLE × XMES | ✓ | ✓ | ✓ | ✗ | ✓ | n/a | Fails H |
| 7 | COMEX_SETTLE × OVN∨XMES | ✓ | ✓ | ✓ | ✗ (borderline 0.068) | ✓ | edge similar | Fails H |
| 10 | NYSE_OPEN × XMES | ✓ | ✓ | ✓ | ✗ | ✓ | n/a | Fails H |
| 13 | TOKYO_OPEN × XMES | ✓ | ✓ | ✓ | ✗ | ✓ | n/a | Fails H |

---

## Final verdict per pre-committed decision tree

- `PORTFOLIO_GENERAL` trigger: 4-6 of 6 lanes have passing variants. **Not met.** Only 2 lanes (COMEX_SETTLE, EUROPE_FLOW) have integrated-confirmed variants.
- `COEXISTS_BOTH` trigger: 2-3 lanes pass. **MET.**
- `COMEX_SETTLE_SPECIFIC` / `DEAD`: not met.

**Pre-committed verdict: COEXISTS_BOTH.** Effect is lane-dependent, restricted to COMEX_SETTLE and EUROPE_FLOW under the OVNRNG_100 variant only.

### Implications for Phase 5 rediscovery

**Phase 5 (re-audit v2, clean Mode A rediscovery of active families)** should:

1. **Queue COMEX_SETTLE × OVNRNG_100 confluence** as a per-lane pre-reg candidate (K=1 Pathway B) alongside the 6 active family rediscoveries.
2. **Queue EUROPE_FLOW × OVNRNG_100 confluence** similarly.
3. **Do NOT queue an allocator-level vol-regime conditioner** — portfolio-general verdict not met.
4. **Do NOT queue XMES_60 variants** — failed block-bootstrap null on all 4 lanes tested.
5. **Consider OVN_P75 threshold** as an alternative specification for the two confirmed variants — edge persists and fire-rate is regime-stable (addresses the fire-rate non-stationarity caveat that surfaced in the phase-2.9 framing audit).

### Does this change the live book?

**No direct change.** The 2 confirmed confluence variants are candidates for Phase 5 rediscovery, not immediate replacement. Before any change to `lane_allocation.json`:
- Phase 5 clean Mode A rediscovery must complete (per re-audit v2 locked plan).
- Correlation-gate analysis must clear (OVN fires correlate with cross-lane vol regime on the live book — impact on portfolio Sharpe unquantified in this test).
- OOS must reach N≥150 for meaningful significance (current OOS N is 60-67 on the two confirmed cells — underpowered).

---

## Audit trail

| Stage | File | Commit |
|---|---|---|
| A Pre-reg | `docs/audit/hypotheses/2026-04-20-vol-regime-confluence-portfolio-test.yaml` | `76b20f5e`, `ff5c565f`, `293bd522` |
| D Primary scan | `research/vol_regime_portfolio_scan_v1.py` | `9799cb77` |
| K Independent SQL | `research/vol_regime_independent_sql.py` | `f78a41a3` |
| L Pressure test | `research/vol_regime_pressure_test.py` | `58e8abaa` |
| G H I J | `research/vol_regime_gates_g_h_i_j.py` | `096f5ec1` |
| M Result doc | `docs/audit/results/2026-04-20-vol-regime-confluence-portfolio-test.md` | (this commit) |

All theory claims cite local extracts in `docs/institutional/literature/` verbatim. No training-memory citations per `institutional-rigor.md` § 7.

Database fresh through 2026-04-16 (MNQ/MES orb_outcomes). Canonical delegation via `research.filter_utils.filter_signal` verified by independent SQL to 4dp. Block-bootstrap null validated by pressure test before application.

---

## What the scan did NOT test (honest scope)

- **Portfolio-level correlation under the 2 confirmed variants.** OVN fires are concentrated on high-vol days, which likely also drives cross-lane co-fire. Marginal portfolio Sharpe with OVN confluence replacing the base lane at COMEX_SETTLE + EUROPE_FLOW is not computed. Required before any deployment.
- **Role-R3 continuous sizing (Carver Ch 9-10) vs Role-R1 binary filter.** This test is R1-binary. Whether vol-regime as a continuous position-size modifier would outperform the binary gate is a separate, Stage-2-deployment question.
- **Cross-instrument generalization.** This test is MNQ-only. Whether the confluence generalizes to MES (equity-index peer) or MGC (different asset class) would require a separate pre-reg.
- **Prop-firm DD survival Monte Carlo (Criterion 11).** Confirmed variants lower trade count by ~half vs base lane — affects account survival probability. Not computed here.

These are legitimate follow-ups — each requires its own pre-reg if pursued.
