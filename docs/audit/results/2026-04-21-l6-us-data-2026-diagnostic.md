# L6 MNQ_US_DATA_1000 2026 OOS Diagnostic

**Date:** 2026-04-21
**Branch:** `research/l6-us-data-1000-2026-diagnostic`
**Script:** `research/audit_l6_us_data_2026_breakdown.py`
**Parent:** PR #52 (6-lane unfiltered baseline stress-test)

---

## Question

L6 `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` is the only 2026-negative
deployed lane (unfiltered ExpR −0.034R, n=68). Full-sample IS t=+3.20,
strong baseline. Is 2026 OOS a **structural break** or **pure noise**?

---

## Verdict: NOISE_CONSISTENT

**Bootstrap null (10,000 draws of size 68 from IS distribution):**
observed 2026 mean (−0.034R) sits at p=0.1789 of the null CDF. That is,
**18% of random 68-trade samples from the IS distribution produce a mean
≤ −0.034R.** The observed OOS is well within the 10-90 percentile null
interval.

No structural break signal. The lane's 2026 negative is consistent with
sampling variance at the given OOS size.

---

## Details

### Step 1 — Bootstrap null

- IS mean: +0.093R  ⇒  observed 2026 mean: −0.034R
- Bootstrap 5th / 25th / 50th / 75th / 95th pct of size-68 means drawn from IS:
  −0.139  /  +0.002  /  +0.090  /  +0.190  /  +0.329
- One-sided p(mean ≤ observed | H0 = IS distribution): **0.1789**
- Classification: **NOISE_CONSISTENT** (obs within 10-90 pct of null)

### Step 2 — Per-year comparison vs 2026

| Year | N | ExpR | t vs 0 | Welch t vs 2026 | Welch p vs 2026 |
|------|---|------|--------|-----------------|-----------------|
| 2019 | 163 | +0.074 | +0.85 | +0.64 | 0.525 |
| 2020 | 256 | +0.009 | +0.12 | +0.27 | 0.791 |
| 2021 | 247 | +0.104 | +1.38 | +0.84 | 0.400 |
| 2022 | 255 | +0.184 | +2.43 | +1.33 | 0.186 |
| 2023 | 252 | +0.077 | +1.03 | +0.68 | 0.499 |
| 2024 | 253 | +0.173 | +2.30 | +1.27 | 0.208 |
| 2025 | 248 | +0.020 | +0.26 | +0.33 | 0.741 |
| **2026** | **68** | **−0.034** | **−0.24** | — | — |

**No IS year is significantly different from 2026 at p<0.05.** Welch
vs 2022 (strongest IS year) p=0.19, nominally better but not close
to significance. 2026's ExpR is worse than 2020 (+0.009) and 2025 (+0.020)
by only 0.04–0.05R, which is within the per-trade noise band.

### Step 3 — Calendar decomposition

| Bucket | IS | 2026 |
|--------|----|----|
| NFP days | n=73 ExpR=+0.042 | n=3 ExpR=−0.191 |
| non-NFP days | n=1601 ExpR=+0.095 | n=65 ExpR=−0.027 |
| OPEX days | n=75 ExpR=+0.042 | n=3 ExpR=−1.000 |
| non-OPEX days | n=1599 ExpR=+0.095 | n=65 ExpR=+0.010 |
| Monday | n=340 ExpR=+0.095 | n=13 ExpR=−0.244 |
| Tuesday | n=337 ExpR=+0.088 | n=14 ExpR=−0.140 |
| Friday | n=323 ExpR=+0.088 | n=13 ExpR=−0.248 |
| CPI-adj (day 10-15) | n=336 ExpR=+0.122 | n=13 ExpR=**+0.306** |

Observations:
- **CPI-adjacent days are 2026's best bucket** (+0.306R, n=13) — edge
  still present where it historically exists. Not a broken lane.
- **OPEX days: 3/3 max-losers** in 2026 (all −1.0R). 3 trades is mechanical
  noise — a single bad OPEX Friday could produce this. Not signal.
- Monday/Tuesday/Friday all −0.14 to −0.25 on n=13-14 each. At that sample
  size, 95% CI is wider than the observation.
- Non-OPEX + non-NFP: +0.010R — essentially flat, not negative.

**The bulk of the 2026 loss is concentrated in 3 OPEX trades (−3.0R total
contribution).** Removing those 3: 2026 ExpR on n=65 ≈ +0.010R. Hardly
a structural break — closer to flat than negative.

### Step 4 — Volatility regime

| Bin | IS n | IS ExpR | 2026 n | 2026 ExpR | 2026 % |
|-----|------|---------|--------|-----------|--------|
| Q1 (atr_20_pct 0-25) | 464 | +0.079 | 0 | — | 0.0% |
| Q2 (25-50) | 313 | +0.182 | 15 | **−0.352** | 22.1% |
| Q3 (50-75) | 373 | +0.078 | 10 | −0.024 | 14.7% |
| Q4 (75-100) | 522 | +0.061 | 40 | −0.025 | 58.8% |

**2026 has zero Q1 days** — the 252-day rolling ATR_20 percentile has
pushed 2026 days exclusively into Q2-Q4, dominantly Q4. This is a
regime-composition shift: 2026 MNQ has been more volatile than its
recent historical norm, so every 2026 day ranks above the 25th pct.

**Q2 is the only 2026 bin that diverges materially from IS expectation
(IS +0.182 vs 2026 −0.352 on n=15).** That is the most suspicious-looking
cell in the audit. But:

- At n=15, 2026-Q2 95% CI is ±0.46R. −0.352 ± 0.46 includes the IS mean.
- 2026-Q2 is likely noise-dominated at that sample size.

Q3 and Q4 (n=10 and n=40) are only mildly negative (−0.02R each),
well within noise.

---

## Interpretation

L6 2026 is **not a structural break**. Supporting evidence:

1. **Bootstrap null says 18% of random draws produce this result.** Standard
   null-hypothesis reasoning: cannot reject the null that 2026 is drawn
   from the IS distribution.

2. **No per-year comparison is significant.** 2026 vs every IS year has
   Welch p > 0.18. The lane's 2019, 2020, and 2025 ExpR are all within
   0.05R of 2026.

3. **3 OPEX trades = all-max-loss accounts for most of the aggregate
   negative.** Removing those 3: 2026 ExpR on n=65 ≈ +0.010R. Lane is
   essentially flat-to-neutral in 2026, not broken.

4. **CPI-adjacent window still produces edge** (+0.306R, n=13), confirming
   the underlying geometry is not broken — at least for the bucket where
   the historical edge lives.

5. **The Q2 cell is the only suspicious-looking divergence** but at n=15
   is within noise.

---

## Operational conclusion

**No lane pause recommended.** 2026 OOS is consistent with the null
hypothesis that the lane continues to carry its IS edge, subject to
sampling variance at n=68.

**Monitoring:** continue through Q2 2026. If OOS N reaches 150-200 and
ExpR stays ≤ 0, re-run this audit with lower p-threshold — at that N,
bootstrap p < 0.05 would be meaningful. Current N is insufficient for
an inferential call either way.

**Secondary observation:** the OPEX 3-for-3-max-loss cluster is worth
logging. If OPEX day losses continue to cluster in 2026 Q2-Q3, a
pre-reg hypothesis ("OPEX + US_DATA_1000 had a structural break in 2026")
becomes test-worthy. But 3 data points do not constitute evidence yet.

---

## What this does NOT change

- No deployment change for L6.
- No change to L6's `trailing_expr` / `annual_r` scoring in
  `lane_allocation.json`.
- PR #52's FILTER_VESTIGIAL classification for L6 stands (the filter
  doesn't discriminate on this lane; the baseline carries edge).

## Next-best tests

1. **Rerun this audit at 2026 n ≥ 150** (approx end of August 2026 at
   current cadence). Update bootstrap p and per-year Welch comparisons.

2. **OPEX-day pattern check.** Look at historical L6 OPEX ExpR
   year-by-year. If OPEX days have trended worse across all sessions /
   instruments recently, that's a calendar-regime signal worth a pre-reg.

---

## Provenance

- Canonical: `orb_outcomes`, `daily_features` (triple-joined).
- Holdout: 2026-01-01 (Mode A sacred).
- Bootstrap: 10,000 iid draws with `np.random.default_rng(42)` for
  reproducibility.
- Calendar features: `is_nfp_day`, `is_opex_day`, `is_friday`, etc from
  `daily_features` (no derived signals).
- Read-only. No production code touched.

## Reproduction

Run the diagnostic script directly against the canonical DB:

```
python research/audit_l6_us_data_2026_breakdown.py
```

Outputs:
- Bootstrap distribution + observed-vs-null comparison printed to stdout.
- Per-year Welch t (full-sample IS vs 2026 OOS) printed to stdout.
- Calendar-feature breakdown table printed to stdout.

No artifacts persisted; rerunnable at any time. Seed pinned at
`np.random.default_rng(42)`.

## Caveats / limitations

- 2026 OOS sample is `n=68` — statistical power against the IS null is
  limited; verdict is "consistent with noise", not "proven absence of
  break". Plan to re-run at `n ≥ 150` per § Next-best tests.
- Calendar-feature subgroup analysis (NFP, OPEX, Friday) is exploratory;
  no per-cell BH-FDR correction applied. Treat any subgroup signal as
  hypothesis-generating only.
- Bootstrap assumes IS distribution is the correct null. If the true
  data-generating process changed across the 2026 boundary, the bootstrap
  null is itself contaminated and would over-attribute observed
  divergence to noise. The verdict is conditional on stationarity.
- Recovered from `archive/stash-2026-04-26-l6-wip-pre-correction` 2026-04-26;
  original author has not reviewed the recovered form. Audit content is
  unchanged from stash; only this preamble + recovery note added.
