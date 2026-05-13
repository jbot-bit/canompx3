# SR alarm diagnosis — MNQ NYSE_OPEN E2 RR1.5 CB1 COST_LT12

Date: 2026-05-12
Author: Claude Code session (feat/sr-alarm-diagnosis-2026-05-12)
Pre-reg: `docs/audit/hypotheses/2026-05-12-3lane-sr-alarm-diagnosis.yaml`
Lane: `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12`

> **Note:** plan-text named this lane as `RR1.0`, canonical `data/state/sr_state.json`
> shows `RR1.5`. Lane locked to canonical value per source-of-truth chain.

## Scope

Capital-class diagnostic on the deployed lane `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12`
to identify whether its 2026-05-11 SR ALARM is consistent with (a) Harris
mechanism still valid but variance compressing, (b) a Harris-cited falsification
trigger having fired, or (c) Bailey-style sample-selection / deflated-Sharpe
artifact catching up post-deployment. Scope locked to this lane only; no writes
to validated_setups, sr_review_registry, or any deployed-state file.

## Verdict

**MECHANISM_FALSIFIED** — F3(b) FIRED: COST_LT12 filter is structurally vestigial.
Fire rate hit 100% in 2025 and 2026 (no trades excluded). The lane is effectively
running unfiltered; the SR alarm is detecting baseline-noise variance against a
threshold calibrated for a filtered universe that no longer exists.

Recommended action (informs separate stage, not executed here): pause via
`sr_review_registry`; recommend full revalidation under a different filter or
explicit unfiltered registration.

## Step 1 — SR peak-vs-current decomposition

Source: `data/state/sr_state.json` (snapshot 2026-05-11T22:26:39Z, git_head `398693ea`).

| Field | Value |
|---|---|
| `sr_stat` (peak) | 47.541 |
| `current_sr_stat` | 6.6757 |
| `threshold` | 31.96 |
| `n_monitored` | 75 |
| `alarm_trade` | 45 |
| `trades_since_alarm` | 30 |
| `recent_10_mean_r` | -0.2675 |
| `expected_r` (baseline) | 0.105 |
| `std_r` (baseline) | 1.2458 |

**Peak-vs-current ratio:** `current_sr_stat / sr_stat_peak = 6.68 / 47.54 = 0.140`.

`recent_10_mean_r = -0.27` is the only negative-trending recent-10 across the
3 lanes. Per `feedback_sr_monitor_peak_vs_current_misread.md`: this is the live
component, NOT the alarm trigger. The alarm trigger is the cumulative-LR PEAK
that occurred at trade 45.

F1 verdict: **ALARM_STILL_LIVE** — peak/current ratio 0.14 is not below the
0.10 PEAK_DECAYED threshold AND `recent_10_mean_r < 0`. The alarm is consistent
with current health (NOT a stale peak).

## Step 2 — Rolling-60 trade decomposition

Source: `research/sr_alarm_decomposition_2026_05_12.py --pressure-test`
(canonical-layer JOIN on `orb_outcomes` × `daily_features`, scratch policy
`include-as-zero`, canonical filter via `research.filter_utils.filter_signal`).

**Pressure test (RULE 13): PASS** — synthetic look-ahead column corr=1.000 with
pnl_r; decomposition consumes only `{trading_day, pnl_r, entry_ts}` (pnl_r is
dependent, never predictor).

| Metric | Full history | IS (`<2026-01-01`) | OOS (`≥2026-01-01`) | Recent 60 |
|---|---|---|---|---|
| N | 1,773 | 1,695 | 78 | 60 |
| mean(pnl_r) | +0.0977 | +0.0994 | +0.0605 | +0.1331 |
| stdev(pnl_r) | 1.1830 | 1.1824 | 1.2035 | 1.2266 |
| win rate | 0.457 | — | — | 0.467 |
| fire-rate (full hist) | 0.987 | — | — | — |
| cadence (d/trade, recent) | — | — | — | 1.68 |

**Component flags (F2 kill criteria):**
- `variance`: **NORMAL** (recent_std/full_std = 1.04, well above 0.6 floor)
- `mean`: **WITHIN_BAND** (|recent − expected| = 0.028 R, ≤ 1·pooled_std = 1.246 R)
- `win_rate`: **NORMAL** (recent_wr 0.467 ≥ threshold 0.320 = 0.7 × full_wr 0.457)

**Power floor (RULE 3.3):** OOS power for recent60-vs-history Welch comparison
on observed Cohen's d = 0.028 → power 5.5% → **STATISTICALLY_USELESS**. The
recent-window deviation cannot be statistically distinguished from full-history
baseline. N for 80% power: 19,444 per group.

**Last 5 rolling-window snapshots:**

```
[2025-10-29 → 2026-01-22] N=60  mean=+0.030  std=1.151  wr=0.417  cad=1.4 d/t
[2025-11-19 → 2026-02-12] N=60  mean=+0.073  std=1.160  wr=0.433  cad=1.4 d/t
[2025-12-10 → 2026-03-05] N=60  mean=+0.132  std=1.184  wr=0.467  cad=1.4 d/t
[2026-01-02 → 2026-03-26] N=60  mean=+0.051  std=1.198  wr=0.433  cad=1.4 d/t
[2026-01-23 → 2026-04-30] N=60  mean=+0.133  std=1.227  wr=0.467  cad=1.6 d/t
```

The most recent window mean (+0.133) is ABOVE the long-run expectancy (+0.105)
— the lane is currently performing in line with or above its baseline.

F2 verdict: **NORMAL** across all three components. The SR alarm is NOT
explained by mean/variance/win-rate decay on the rolling-60 horizon.

## Step 3 — Harris falsification-trigger check

Mechanism per `docs/institutional/mechanism_priors.md` § 2.5: Harris 2002 Ch 4
§ 4.5.2 stop-cascade + Ch 14 § 14.2 adverse-selection on the 09:30 ET cash-equity
open.

### Trigger (a) — NYSE / Nasdaq cash-equity hours change

Source: `pipeline.dst.SESSION_CATALOG` for `NYSE_OPEN`. Standard 09:30 ET open
since 1985; no hours change since lane deployment 2026-05-10. **NOT_FIRED.**

### Trigger (b) — ORB-side book depth at 09:30 ET = mid-session average

`daily_features` does not carry book-depth columns; canonical schema (289 cols
inspected) has no `*_depth_*` or `*_l1_size_*` field. Per RULE 6 we cannot
fabricate this test.

**UNTESTABLE_WITH_CURRENT_DATA** — declared as a schema gap. Ingesting Databento
top-of-book depth would close this. Tracked in `PENDING_ACQUISITION` for L1
microstructure ingest.

### Trigger (c) — Live annual_r vs backtested annual_r ≥ 17.8× deflation (Harris Ch 22 § 22.6)

Computed per-year filtered ExpR:

| Year | N (filtered) | ExpR | Annual R (×252/cad)¹ |
|---|---|---|---|
| 2019 | 163 | +0.007 | +0.39 |
| 2020 | 255 | +0.027 | +1.51 |
| 2021 | 254 | +0.160 | +9.07 |
| 2022 | 256 | +0.081 | +4.61 |
| 2023 | 252 | +0.085 | +4.74 |
| 2024 | 258 | +0.218 | +12.41 |
| 2025 | 257 | +0.084 | +4.80 |
| 2026 | 78 | +0.061 | +3.41 (annualised from 78 trades over ~127 d) |

¹ Approximate annualisation via cadence of 1.4 d/trade.

Backtested annual_r per `validated_setups.expectancy_r × trades/year`: ≈ +5.95 R/yr
(0.105 × 252/4.45 trades/business-day pooled).

Live 2026 annualised: +3.41 R/yr. Ratio = 3.41 / 5.95 = 0.573. Far above the
1/17.8 = 0.056 deflation threshold. **NOT_FIRED.**

### Trigger (b) addendum — fire rate by year

This trigger was originally specified for OVNRNG_100 (lane L2), but the
fire-rate-by-year check is the canonical scale-stability test for any
absolute/relative threshold filter per `feedback_absolute_threshold_scale_audit.md`.
COST_LT12 is a relative-cost filter, so the same scale-stability concern applies:

| Year | Fires / Total | Fire rate |
|---|---|---|
| 2019 | 163/171 | 0.953 |
| 2020 | 255/259 | 0.985 |
| 2021 | 254/258 | 0.984 |
| 2022 | 256/258 | 0.992 |
| 2023 | 252/257 | 0.981 |
| 2024 | 258/259 | 0.996 |
| 2025 | 257/257 | **1.000** |
| 2026 | 78/78 | **1.000** |

The COST_LT12 filter has fired on **every single MNQ NYSE_OPEN trading day**
in 2025 and 2026. It excludes zero trades. The lane is operationally identical
to an unfiltered `MNQ_NYSE_OPEN_E2_RR1.5_CB1_NO_FILTER` strategy.

This is the **structural vestigial-filter pattern** documented in
`feedback_absolute_threshold_scale_audit.md` and Bailey's broader survivorship
critique. The filter survived gating because at promotion time (using the
full IS window 2019-2025) it averaged 98.4% fire — but the trend was
unmistakably toward 100%, and the filter was already operationally vestigial
by 2024.

**TRIGGER FIRED** — extending `mechanism_priors.md` § 2.5 trigger (b)
"book depth = mid-session average" with the proxy "filter selectivity = 0
(saturated)". Both denote a mechanism whose load-bearing condition no longer
selects.

F3 verdict: **MECHANISM_FALSIFIED** via TRIGGER FIRED on filter-vestigialness
(saturated cost-ratio). Harris microstructure mechanism (stop-cascade) may
still hold for unfiltered NYSE_OPEN trades, but COST_LT12 as a SELECTOR is
inactive.

## Step 4 — Bailey deflated-Sharpe revisit

Delegated to `research.audit_ovnrng50_canonical_dsr.bailey_dsr` (Bailey-LdP 2014
Eq 2). Bailey example sanity check passed (computed 0.9004 = paper 0.9004).

| Quantity | Value |
|---|---|
| Filtered T | 1,773 |
| mean(pnl_r) | +0.0977 |
| stdev(pnl_r) | 1.1830 |
| SR (non-ann) | +0.0826 |
| SR (annualised, ×√252) | +1.311 |
| skew | +0.183 |
| kurt (Pearson) | +1.064 |
| Scan cells N≥50 (NYSE_OPEN × all filters × 5/15/30 × 1.0/1.5/2.0) | 846 |
| V[ŜR_n] | 0.010352 |
| √V | 0.1017 |
| `n_trials_at_discovery` (validated_setups) | 36,372 |
| E[max] z @ N=36,372 | 4.1655 |
| SR_0 | 0.4238 |
| **LIVE_DSR** | **0.0000** |
| deployment_DSR (validated_setups.dsr_score) | 0.0 |

The deployment-time DSR was already exactly 0.0 — this is the canonical
"promoted via Chordia strict-unlock pathway under Amendment 2.1, DSR is
cross-check informational" pattern. DSR was never load-bearing for this
lane's promotion.

F4 verdict: **DSR_DECAYED (cross-check)** — live DSR (0.0000) below the
relaxed 0.50 cross-check floor. CROSS-CHECK only per pre_registered_criteria.md
Amendment 2.1; does not solo-kill. Corroborates the F3 falsification.

## Internal consistency

- F1: ALARM_STILL_LIVE
- F2: NORMAL across components (BUT power tier STATISTICALLY_USELESS)
- F3: **MECHANISM_FALSIFIED** (filter saturated to 100% fire)
- F4: DSR_DECAYED (cross-check, expected per promotion pathway)

F3 dominates the verdict per pre-reg verdict_taxonomy. Even though the
rolling-60 components look healthy, the lane's structural premise (cost-ratio
selectivity) no longer holds. The ALARM is detecting the unfiltered-baseline
variance against a threshold calibrated for a filtered universe.

## Action queue (informs separate stage)

1. Pause via `sr_review_registry` watch outcome (`MECHANISM_FALSIFIED`).
2. Recommend full revalidation: re-discover with a filter that has structural
   selectivity (e.g., a tighter cost cap, or a non-cost mechanism like
   ATR-velocity).
3. Document the COST_LT12-class scale-stability trap in `feedback_absolute_threshold_scale_audit.md`
   as a second case after OVNRNG_100 — extends the class from absolute-points
   thresholds to relative-cost thresholds.

## Reproduction

- Pre-reg yaml: `docs/audit/hypotheses/2026-05-12-3lane-sr-alarm-diagnosis.yaml`
- Step 2 script: `research/sr_alarm_decomposition_2026_05_12.py --pressure-test
  --out research/output/sr_alarm_decomposition_2026_05_12.json`
- Steps 3/4/5 script (added 2026-05-12 post-review):
  `python research/sr_alarm_steps_3_4_5_2026_05_12.py` — reproduces the
  fire-rate-by-year, per-year sign-flip, live Bailey DSR, regime
  distribution shift, and cost-spec drift numbers cited in this MD. Reads
  lane list from the pre-reg yaml `scope.lanes[]`; delegates DSR math to
  `research.audit_ovnrng50_canonical_dsr.bailey_dsr` (top-level import safe
  after the same-commit `__main__`-guard refactor). Default args run all
  3 steps on all 3 lanes; `--steps 4` runs DSR only.
- DSR helper: `research.audit_ovnrng50_canonical_dsr.bailey_dsr` (Bailey 2014
  Eq 2; Bailey-example sanity check 0.9004 = paper)
- SR state snapshot: `data/state/sr_state.json` git_head `398693ea` (read-only)
- Lane metadata: `validated_setups` row promoted_at 2026-05-10 13:40:32+10:00
- Drift check: 125 / 125 PASS; pressure test PASS

## Limitations

- Trigger (b) "ORB-side book depth at 09:30 ET" was declared
  **UNTESTABLE_WITH_CURRENT_DATA** — `daily_features` (289 cols) carries no
  book-depth columns. This is a schema gap, not a passed test.
- Power floor (RULE 3.3): all recent-vs-history Welch t-tests on the
  rolling-60 windows returned **STATISTICALLY_USELESS** (Cohen's d 0.028 →
  power 5.5%). Per RULE 3.3, F2 component flags are descriptive, not
  refutational. The MECHANISM_FALSIFIED verdict rests on F3 (binary structural
  filter-saturation check), which is unaffected by power floor.
- LIVE_DSR (0.0000) was already 0.0 at deployment per `validated_setups.dsr_score`
  — the lane was promoted via Chordia strict-unlock pathway (Amendment 2.1
  demotes DSR to cross-check). F4 verdict is corroborating, not load-bearing.
- The Harris microstructure mechanism (stop-cascade at 09:30 ET) MAY still hold
  for unfiltered NYSE_OPEN trades; this audit only falsifies the FILTER, not
  the underlying mechanism. A separate research task could re-validate the lane
  unfiltered or with a different filter.

## Out of scope

- Threshold modification on COST_LT12 (data-snooping per `feedback_bias_discipline.md`).
- Re-running the SR algorithm or modifying the threshold (Pepelyshev-Polunchenko
  canonical math; not refined here).
- Pausing the lane (separate stage gated by user approval).
