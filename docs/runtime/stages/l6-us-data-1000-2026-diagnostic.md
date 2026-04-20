---
slug: l6-us-data-1000-2026-diagnostic
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: L6 MNQ_US_DATA_1000 2026 breakdown — why is the lane negative in 2026?
---

# Stage: L6 US_DATA_1000 2026 diagnostic

## Task

PR #52 flagged L6 `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` as the only
2026-negative lane (unfiltered 2026 ExpR −0.034R, filtered −0.034R — filter
is vestigial). Full-sample IS t=+3.20 but 2026 OOS n=68 is net-negative.

Question: **is this a structural break, or noise in a small OOS sample?**

Possibilities to test:
1. **Calendar regime shift** — 10:00 ET (Brisbane 01:00) corresponds to
   US economic data releases. If release cadence, volatility, or
   scheduling pattern changed in 2025-2026, the lane's geometry may
   be broken.
2. **Release-type composition** — L6 may work on NFP but fail on CPI
   (or vice versa). If the 2026 OOS is heavy on one release type,
   average breaks down.
3. **Pure noise** — n=68 with per-trade std ≈ 0.9 gives 95% CI ±0.22R,
   so observed −0.034R is well within noise for a lane with +0.088R
   full-sample ExpR.
4. **Volatility-regime shift** — high-vol releases produce the edge;
   2026 Q1 may be a low-vol cluster.

## Scope Lock

- `research/audit_l6_us_data_2026_breakdown.py` (new)
- `docs/audit/results/2026-04-21-l6-us-data-2026-diagnostic.md` (new)

## Blast Radius

- Read-only. Zero production-code touch.
- Canonical data only: `orb_outcomes`, `daily_features`.
- No new filters registered. No config changes.

## Approach

1. Load L6 canonical universe.
2. **Noise check:** bootstrap 10,000 samples of size 68 from the
   pre-2026 distribution; compute fraction below −0.034R. If >10%,
   the observed 2026 result is well within noise.
3. **Calendar decomposition:** use `daily_features.is_nfp_day` +
   day-of-week + day-of-month to bucket 2026 trades by likely
   release type. Compare 2026 per-bucket ExpR to IS per-bucket.
4. **Volatility regime split:** split 2026 trades by
   `atr_20_pct` quartile; compare to IS per-quartile.
5. **Year-by-year t-tests of IS years vs 2026** to see if 2026 is
   worse than any prior year or just noise.
6. **Final classification:**
   - NOISE: bootstrap says >10% chance of observed result under null
   - STRUCTURAL_BREAK: bootstrap <1% AND calendar/vol regime
     decomposition shows consistent pattern
   - AMBIGUOUS: between those thresholds, flag for monitoring

## Acceptance criteria

1. Script runs without exceptions.
2. MD contains bootstrap null-hypothesis result, per-year t comparisons,
   calendar decomp, vol-regime decomp.
3. MD classifies the 2026 signal as NOISE / STRUCTURAL_BREAK / AMBIGUOUS
   with reasoning.
4. `python pipeline/check_drift.py` passes.

## Non-goals

- Not proposing lane pause (deploy-change decisions need a separate turn).
- Not amending the allocator.
- Not re-running PR #52's decomposition.
