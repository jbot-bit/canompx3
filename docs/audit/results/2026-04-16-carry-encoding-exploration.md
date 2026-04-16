# Carry Encoding Exploration Results

**Date:** 2026-04-16
**Pre-registered:** `docs/audit/hypotheses/2026-04-16-carry-encoding-exploration.yaml`
**Boundary:** validated shelf only, IS < 2026-01-01 for quintile gates, OOS descriptive only

## Scope

- Validated shelf rows: **26**
- Target population (after filter + start_ts): **24200**
- Prior trade rows: **114797**
- Main cells (3 encodings × 3 session groups): **9**
- E2 sensitivity cells: **6**

## Main results

| Encoding | Session group | N total | N with feature | Coverage | Spearman rho | rho p | WR spread | ExpR spread | ARITH_ONLY | Garch-high rho | Garch-low rho | OOS rho | OOS match | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| E1 | late_day | 8999 | 8999 | 1.00 | +1.000 | 0.0000 | 0.082 | 0.196 | no | +0.500 | +1.000 | +1.000 | yes | **encoding_monotonic** |
| E2 | late_day | 8999 | 8999 | 1.00 | +0.800 | 0.1041 | 0.130 | 0.275 | no | +0.400 | +0.800 | +0.900 | yes | **encoding_flat** |
| E3 | late_day | 8999 | 8999 | 1.00 | -0.500 | 0.6667 | 0.010 | 0.008 | no | -0.400 | -0.500 | -0.400 | yes | **encoding_flat** |
| E1 | mid_day | 10369 | 10297 | 0.99 | +1.000 | 0.0000 | 0.032 | 0.089 | YES | -0.200 | +0.500 | +0.400 | yes | **encoding_arithmetic_only** |
| E2 | mid_day | 10369 | 10297 | 0.99 | +0.500 | 0.3910 | 0.072 | 0.156 | no | +0.000 | +0.100 | -0.400 | no | **encoding_flat** |
| E3 | mid_day | 10369 | 10297 | 0.99 | -0.200 | 0.8000 | 0.040 | 0.097 | YES | -0.600 | -0.800 | -0.600 | yes | **encoding_flat** |
| E1 | early_day | 4832 | 784 | 0.16 | +nan | nan | nan | nan | no | — | — | — | no | **encoding_sparse** |
| E2 | early_day | 4832 | 784 | 0.16 | +nan | nan | nan | nan | no | — | — | — | no | **encoding_sparse** |
| E3 | early_day | 4832 | 784 | 0.16 | +nan | nan | nan | nan | no | — | — | — | no | **encoding_sparse** |

## Quintile detail (IS only)

### E1 × late_day

| Quintile | N | ExpR | WR |
|---:|---:|---:|---:|
| Q1 | 5103 | +0.051 | 0.508 |
| Q2 | 1704 | +0.139 | 0.548 |
| Q3 | 1696 | +0.248 | 0.590 |

### E2 × late_day

| Quintile | N | ExpR | WR |
|---:|---:|---:|---:|
| Q1 | 1706 | -0.071 | 0.446 |
| Q2 | 1696 | +0.138 | 0.548 |
| Q3 | 1700 | +0.116 | 0.537 |
| Q4 | 1708 | +0.204 | 0.576 |
| Q5 | 1693 | +0.153 | 0.555 |

### E3 × late_day

| Quintile | N | ExpR | WR |
|---:|---:|---:|---:|
| Q1 | 3431 | +0.111 | 0.530 |
| Q2 | 1685 | +0.103 | 0.539 |
| Q3 | 3387 | +0.107 | 0.532 |

### E1 × mid_day

| Quintile | N | ExpR | WR |
|---:|---:|---:|---:|
| Q1 | 5815 | +0.080 | 0.511 |
| Q2 | 1963 | +0.100 | 0.520 |
| Q3 | 1911 | +0.169 | 0.543 |

### E2 × mid_day

| Quintile | N | ExpR | WR |
|---:|---:|---:|---:|
| Q1 | 1941 | +0.080 | 0.511 |
| Q2 | 1938 | +0.062 | 0.502 |
| Q3 | 1936 | +0.049 | 0.499 |
| Q4 | 1939 | +0.205 | 0.571 |
| Q5 | 1935 | +0.111 | 0.514 |

### E3 × mid_day

| Quintile | N | ExpR | WR |
|---:|---:|---:|---:|
| Q1 | 3902 | +0.143 | 0.538 |
| Q2 | 1954 | +0.046 | 0.498 |
| Q3 | 1897 | +0.079 | 0.507 |
| Q4 | 1936 | +0.096 | 0.514 |

## E2 sensitivity (half-life variants)

| Variant | Session group | Coverage | Spearman rho | WR spread | ExpR spread | Verdict |
|---|---|---:|---:|---:|---:|---|
| E2_hl1h | late_day | 1.00 | +0.700 | 0.128 | 0.271 | **encoding_flat** |
| E2_hl1h | mid_day | 0.99 | +0.600 | 0.070 | 0.170 | **encoding_flat** |
| E2_hl1h | early_day | 0.16 | +nan | nan | nan | **encoding_sparse** |
| E2_hl4h | late_day | 1.00 | +0.700 | 0.103 | 0.223 | **encoding_flat** |
| E2_hl4h | mid_day | 0.99 | +0.600 | 0.051 | 0.103 | **encoding_flat** |
| E2_hl4h | early_day | 0.16 | +nan | nan | nan | **encoding_sparse** |

## BH-FDR at encoding level (K=3)

| Rank | Encoding | Best p | BH threshold (q=0.05) | Pass |
|---:|---|---:|---:|---|
| 1 | E1 | 0.0000 | 0.0167 | YES |
| 2 | E2 | 0.3910 | 0.0333 | no |
| 3 | E3 | 0.6667 | 0.0500 | no |

## Guardrails

- **Chronology:** all prior trades must have `exit_ts < target_start_ts` (dynamic per `pipeline.dst.orb_utc_window`).
- **Canonical sources:** `orb_outcomes` + `daily_features` only. `validated_setups` used to enumerate targets, each verified by raw query.
- **IS/OOS split:** quintile gates use IS (pre-2026-01-01) only. OOS (2026+) is descriptive and not used for promotion.
- **Binary gates banned:** no threshold on any encoding. Quintile analysis only.
- **K budget:** 3 encodings × 3 session groups = 9 main cells. BH-FDR applied at encoding level K=3.
- **No deployment claims.**

---

## Self-review: skeptical audit of these results

### E1 late_day "monotonic" — what's actually happening

The quintile table shows only 3 bins because E1 is **bimodal**: 42% of rows
are exactly -1.000 (prior session stopped out) and 58% are clustered above
+0.5 (prior session won). `pd.qcut` with `duplicates='drop'` collapses 5
requested quintiles to 3.

The "perfect Spearman rho = +1.000" is trivially 3 monotonic points — any
weak positive trend would get rho=1 on N=3. This is a **binning artefact**
that mechanically inflates the monotonicity score.

### What the deeper decile analysis shows

Independent 10-bin analysis (6 actual bins after deduplication) reveals:

| Bin | N | E1 range | ExpR | WR |
|---|---:|---|---:|---:|
| D1 (prior stopped) | 4,290 | [-1.000, +0.896] | +0.021 | 0.493 |
| D2 | 835 | [+0.896, +0.919] | +0.184 | 0.577 |
| D3 | 832 | [+0.920, +0.934] | +0.180 | 0.567 |
| D4 | 874 | [+0.934, +0.948] | +0.121 | 0.539 |
| D5 | 850 | [+0.949, +0.962] | +0.147 | 0.547 |
| D6 (prior near target) | 822 | [+0.962, +0.986] | +0.356 | 0.636 |

**WR spread is real:** 49.3% → 63.6% across the full range = 14.3% WR spread.
This is NOT arithmetic-only. The most recent prior session's outcome
genuinely predicts the target session's win probability.

**But the gradient is NOT monotonic.** D2-D5 are flat (~0.12-0.18 ExpR, 54-58%
WR). The action is at the extremes: prior-stopped-out (bad) vs prior-near-
target (very good). The middle is noise.

**Implication:** E1 is better understood as a **two-state classifier** (prior
stopped out vs prior won big) than as a continuous monotonic feature. A linear
quintile test partially misses this structure. A follow-up study should test
the two-state form explicitly: prior_pnl_r < 0 vs prior_pnl_r > 0.95 as the
informative states, with the middle as the baseline.

### E1 mid_day — correctly flagged ARITHMETIC_ONLY

WR spread 3.2% is below the 5% threshold. ExpR moves but WR barely does.
Correct verdict.

### E2 — flat everywhere, sensitivity confirms

E2 (recency-weighted) does not beat E1. All 3 sensitivity variants (1h, 2h,
4h half-life) are flat. The time-weighting adds nothing. The simplest encoding
(E1) carries all the accessible information. Occam confirmed.

### E3 — dead, consistent with W2e

Direction conditioning adds nothing in continuous form either. The W2e finding
(veto-pair dead) extends to continuous direction-aware carry: rho is negative
or near-zero everywhere. Direction is not the right axis for carry.

### Early_day — honestly sparse

16% coverage. No analysis possible. Expected and correctly handled.

### Honest overall verdict

**E1 on late_day sessions (COMEX_SETTLE) shows a real WR-moving signal.**
Prior-session outcome genuinely predicts target-session win probability, with
the strongest effect at the extremes (prior stopped out → bad; prior won big →
good). This is the independent information that binary gating could not access.

**But:** the signal is concentrated in the extreme bins, not smoothly monotonic.
The 3-quintile "rho=1.000" overstates the monotonicity. A more honest
characterisation is "two-state with noisy middle" rather than "smooth
continuous gradient."

**E2 and E3 add nothing beyond E1.** Time-weighting and direction conditioning
are noise on this shelf.

**Mid_day is ARITHMETIC_ONLY for E1.** The signal may be weaker or different
in structure for EUROPE_FLOW/SINGAPORE_OPEN.

**Early_day is untestable** due to sparse priors.

### What this means for next steps

1. **E1 on late_day is worth a dedicated follow-up** — but as a two-state
   feature (prior stopped vs prior won big), not as a continuous quintile
   feature. Pre-register a binary split at the natural boundary (prior_pnl_r
   = 0) with the specific hypothesis: "target WR and ExpR improve when the
   most recent prior session won."
2. **E2 and E3 can be parked.** No follow-up needed.
3. **The R7 confluence path is still the right implementation class** — carry
   enters as a binary or categorical state input to a score, not as a
   continuous linear feature.
4. **This is a COMEX_SETTLE finding at this point.** Do not generalise to other
   sessions without separate evidence.
