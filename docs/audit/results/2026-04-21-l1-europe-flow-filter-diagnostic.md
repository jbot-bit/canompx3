# L1 EUROPE_FLOW ORB_G5 — ATR-normalized replacement diagnostic

**Date:** 2026-04-21
**Branch:** `research/l1-europe-flow-filter-diagnostic`
**Script:** `research/audit_l1_europe_flow_filter_diagnostic.py`
**Parent:** PR #57 (corrected 6-lane baseline — L1 classified FILTER_CORRELATES_WITH_EDGE)

---

## Question

PR #57 showed L1 `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5`:
- Unfilt IS t=+1.61 (marginal), filt IS t=+2.28, Welch fire-vs-non p=0.001
- Filter fires 57%(2019) → 100%(2026) — selection vanished at current scale

Can an ATR-normalized replacement — `orb_size / atr_20 ≥ threshold` — restore
pre-2022 selectivity AND retain the Welch edge?

**Verdict: NO.** A Pathway-B pre-reg for ATR-normalized ORB_G5 replacement
is **NOT worth writing** with the features as they stand. Details below.

---

## Results

### Step 1 — ORB size distribution by year

| Year | N | orb μ | orb p50 | orb/atr μ | orb/atr p50 | Legacy ORB_G5 fire |
|------|---|-------|---------|-----------|-------------|--------------------|
| 2019 | 171 | 6.1 | 5.2 | 0.054 | 0.047 | 57.3% |
| 2020 | 256 | 14.9 | 12.2 | 0.064 | 0.056 | 93.4% |
| 2021 | 259 | 12.8 | 10.5 | 0.064 | 0.053 | 92.7% |
| 2022 | 258 | 20.8 | 18.6 | 0.061 | 0.054 | 99.6% |
| 2023 | 258 | 12.3 | 11.2 | 0.057 | 0.051 | 94.6% |
| 2024 | 259 | 15.6 | 13.0 | 0.058 | 0.050 | 96.5% |
| 2025 | 257 | 22.0 | 17.8 | 0.060 | 0.051 | 99.2% |
| 2026 | 72 | 22.4 | 20.9 | 0.050 | 0.049 | 100.0% |

**Key insight:** `orb_size/atr_20` ratio is **stationary** across years
(median 0.047–0.056 range, tight band). The legacy ORB_G5 fire-rate
drift (57→100%) is NOT due to volatility regime change — it's purely
price inflation making 5 absolute points small relative to MNQ's
nominal level. ATR-20 scales with price proportionally, so the ratio
stays flat.

This is the correct scale-normalization; but …

### Step 2 — Candidate ATR-normalized thresholds (per-year fire %)

| Year | Legacy G5 | ≥0.035 | ≥0.045 | ≥0.05 | ≥0.055 | ≥0.065 | ≥0.08 | ≥0.10 |
|------|-----------|--------|--------|-------|--------|--------|-------|-------|
| 2019 | 57.3% | 73.7% | 53.2% | 44.4% | 36.3% | 26.9% | 17.5% | 8.2% |
| 2020 | 93.4% | 83.6% | 64.1% | 56.6% | 51.2% | 39.1% | 24.2% | 12.9% |
| 2021 | 92.7% | 78.4% | 63.7% | 54.1% | 46.3% | 35.1% | 23.2% | 12.7% |
| 2022 | 99.6% | 82.2% | 65.1% | 55.8% | 49.6% | 36.0% | 23.3% | 11.6% |
| 2023 | 94.6% | 78.7% | 60.9% | 51.2% | 46.1% | 33.3% | 15.9% | 6.6% |
| 2024 | 96.5% | 76.4% | 56.8% | 49.8% | 43.6% | 32.8% | 18.5% | 7.7% |
| 2025 | 99.2% | 76.3% | 58.0% | 51.0% | 44.7% | 33.5% | 21.0% | 11.7% |
| 2026 | 100.0% | 72.2% | 51.4% | 37.5% | 30.6% | 23.6% | 9.7% | 0.0% |

**A threshold of ≥0.045 recovers 51–65% fire rate year-over-year** —
close to the pre-2022 legacy 57% selectivity. Scale stability is
achieved. But …

### Step 3 — IS Welch fire-vs-non-fire for each candidate (n=1718)

| Threshold | Fire% | ExpR_fire | ExpR_nonf | Δ | Welch t | Welch p |
|-----------|-------|-----------|-----------|---|---------|---------|
| **Legacy G5** | **92.1%** | **+0.064** | **−0.204** | **+0.269** | **+3.29** | **0.001** |
| ≥0.035 | 78.7% | +0.086 | −0.114 | +0.199 | +3.25 | 0.001 |
| ≥0.045 | 60.6% | +0.070 | +0.002 | +0.068 | +1.27 | 0.206 |
| ≥0.05 | 52.2% | +0.080 | +0.003 | +0.077 | +1.45 | 0.148 |
| ≥0.055 | 45.9% | +0.097 | −0.002 | +0.099 | +1.83 | 0.068 |
| ≥0.065 | 34.2% | +0.102 | +0.013 | +0.089 | +1.54 | 0.124 |
| ≥0.08 | 20.7% | +0.060 | +0.039 | +0.021 | +0.31 | 0.759 |
| ≥0.10 | 10.3% | +0.035 | +0.044 | −0.010 | −0.10 | 0.919 |

**Critical finding:** at ≥0.045 (the candidate that best recovers
pre-2022 fire rate), Welch fire-vs-non-fire **p=0.21** — the filter
does not discriminate at p<0.05. Δ=+0.068R is substantively smaller
than the legacy filter's +0.269R Δ at 92% fire.

Why? Legacy ORB_G5 at 92% fire isolates the smallest-orb-size days
(the 8% non-fire subset) which are genuinely the worst days to trade
breakouts. It's a "exclude the dead-zone" filter, not a "select the
hot days" filter. The ATR-normalized equivalent at 60% fire splits
the universe more evenly between fire and non-fire, so it lands
between two populations with less extreme contrast.

**The closest-to-significant candidate is ≥0.055 (fire 46%, p=0.068,
Δ=+0.099R).** Borderline but not a clear win. Would need to survive
BH-FDR against whatever family it was discovered in, which this
diagnostic did not pre-register (6 candidates scanned without
correction).

### Step 4 — 2026 OOS fire rate + ExpR (monitoring only)

At n=72 OOS, any conclusion is underpowered (95% CI ±0.21R).

| Threshold | Fire% | ExpR_fire | ExpR_nonf | Δ |
|-----------|-------|-----------|-----------|---|
| Legacy G5 | 100.0% | +0.293 | NaN | NaN |
| ≥0.035 | 72.2% | +0.271 | +0.349 | −0.078 |
| ≥0.045 | 51.4% | +0.348 | +0.235 | +0.113 |
| ≥0.05 | 37.5% | +0.503 | +0.167 | +0.336 |
| ≥0.055 | 30.6% | +0.415 | +0.239 | +0.177 |
| ≥0.065 | 23.6% | +0.270 | +0.300 | −0.030 |
| ≥0.08 | 9.7% | +0.375 | +0.284 | +0.091 |
| ≥0.10 | 0.0% | NaN | +0.293 | NaN |

2026 is noisy and underpowered. ≥0.05 looks best on the surface but
at n=27 fire trades this is not inferential.

### Step 5 — Per-year Welch for ≥0.045 (the ~57% fire recovery candidate)

| Year | N | Fire% | ExpR_f | ExpR_n | Δ | Welch t | p |
|------|---|-------|--------|--------|---|---------|---|
| 2019 | 171 | 53.2% | −0.242 | −0.065 | −0.177 | −1.20 | 0.232 |
| 2020 | 256 | 64.1% | −0.014 | −0.032 | +0.018 | +0.13 | 0.899 |
| 2021 | 259 | 63.7% | +0.079 | −0.080 | +0.159 | +1.15 | 0.251 |
| 2022 | 258 | 65.1% | +0.133 | +0.033 | +0.100 | +0.67 | 0.500 |
| 2023 | 258 | 60.9% | +0.111 | +0.115 | −0.004 | −0.03 | 0.979 |
| 2024 | 259 | 56.8% | +0.135 | −0.088 | +0.223 | +1.61 | 0.109 |
| 2025 | 257 | 58.0% | +0.165 | +0.114 | +0.051 | +0.35 | 0.723 |
| 2026 | 72 | 51.4% | +0.348 | +0.235 | +0.113 | +0.41 | 0.685 |

**Zero years individually significant at p<0.05.** Two years show
reverse sign (2019, 2023). Δ ranges from −0.18 to +0.22 year-over-year.
This is not a stable filter.

---

## Why legacy ORB_G5 "worked" historically

The legacy ORB_G5's Welch t=+3.29 on L1 IS is driven by the fact that
the filter fires 92% of the time. The 8% of days it excludes are
genuinely the **worst 8%** — days with orb_size < 5 points absolute,
which in most years are the low-volatility no-opportunity days where
breakouts fail most often.

This is an "exclude the dead-zone" selection, not a "select the
winners" selection. And at 100% fire (post-2022) there's no
exclusion happening anymore — the edge survives only because the
unfiltered baseline is already marginally positive (t=+1.61, p=0.11).

**The ATR-normalized ratio cannot replicate this dead-zone exclusion
because the low end of the ratio distribution is not specifically
unusually unprofitable** — the filter that ≥0.035 would test does
discriminate (p=0.001) but at 79% fire rate it's close to the legacy's
92% pass-through behavior and doesn't materially improve selection.

---

## Operational conclusion

**Do NOT write a Pathway-B pre-reg for ATR-normalized ORB_G5
replacement on L1 with current features.** Reasons:

1. No candidate threshold between 0.045 and 0.08 achieves p<0.05 on IS.
2. Per-year stability is poor — reverse-sign years exist, no year
   individually significant.
3. The closest candidate (≥0.055) is borderline and would need
   multiple-testing correction against ~6 scanned candidates (BH-FDR
   at K=6, q=0.05 would require p<0.008, not achieved).
4. 2026 OOS is underpowered and noisy.

**Implication for L1 posture:** filter is effectively pass-through
in 2026 but the lane's unfiltered baseline is marginally positive
(t=+1.61) and 2026 OOS is +0.293R per trade. Lane continues to work
via geometry, not via filter selection. Same conclusion as PR #57:
leave in place, don't urgently replace.

---

## Alternative research directions (not pursued here)

If the user still wants to improve L1's selection, try:

1. **Break-quality filters** — `orb_EUROPE_FLOW_break_bar_continues`,
   `orb_EUROPE_FLOW_break_delay_min`, `orb_EUROPE_FLOW_break_bar_volume`.
   These are trade-time-knowable and session-specific (not scale-dependent).
2. **Pre-break context** — `rel_vol_EUROPE_FLOW`, `orb_EUROPE_FLOW_pre_velocity`,
   `orb_EUROPE_FLOW_compression_tier`.
3. **Vol-regime overlay** — `garch_forecast_vol_pct`, `atr_vel_regime`
   as filters that discriminate on days the geometry is likely to work.

None of these is "replace ORB_G5" — they are fresh filter research on
L1's unfiltered lane universe. Would need a pre-reg under the Pathway
that fits the hypothesis family.

---

## Provenance

- Canonical data: `orb_outcomes`, `daily_features` (triple-joined).
- Canonical parser: `trading_app.eligibility.builder.parse_strategy_id`.
- Filter: `ORB_G5` via `research.filter_utils.filter_signal`.
- Holdout: 2026-01-01 (Mode A sacred).
- Read-only. No production code touched. No pre-reg created.
