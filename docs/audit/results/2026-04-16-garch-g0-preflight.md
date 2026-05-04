# Garch G0 Preflight

**Date:** 2026-04-16 11:16 AEST
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-g0-preflight.yaml`
**Purpose:** lock the garch research object before further sizing / additive-value / classifier work.

## Coverage

| Check | Status | Detail |
|---|---|---|
| python-executable | PASS | /mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python |
| required-imports | PASS | duckdb=ok, numpy=ok, pandas=ok, scipy=ok, arch=ok |
| required-columns | PASS | missing=none |
| canonical-freshness | PASS | orb latest=2026-04-14 00:00:00 daily_features latest=2026-04-15 00:00:00 |
| active-scope-gpct-coverage | PASS | MNQ/MES/MGC gpct_nonnull={'MES': 5135.0, 'MGC': 2396.0, 'MNQ': 5135.0}; zero elsewhere=['GC', 'M2K', 'M6E', 'MBT', 'SIL'] |
| prior-rank-functional | PASS | _prior_rank_pct sample returned 100.0; expected 100.0 from prior-only window |
| garch-forecast-callable | PASS | compute_garch_forecast(sample_series) -> 0.015796 |
| static-line-audit | PASS | build_daily_features.py lines 544-578, 581-615, 1208-1219, 1256-1268 remain prior-only |
| validated-count-in-scope | PASS | actual=45 expected=45 |
| validated-test-count | PASS | actual=429 expected=429 |
| validated-anchor-rebuild | PASS | MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5=ok, MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100=ok, MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12=ok |
| broad-row-count | PASS | actual=430 expected=430 |
| broad-cell-count | PASS | actual=431 expected=431 |
| broad-session-anchor-counts | PASS | COMEX_SETTLE=52, EUROPE_FLOW=42, TOKYO_OPEN=32, NYSE_OPEN=60 |
| holdout-boundary-consistency | PASS | holdout_policy=2026-01-01 broad=2026-01-01 validated=2026-01-01 |
| shuffle-null-degrades-real-effect | PASS | real_high=0.705 shuf_med=0.498; real_low=0.673 shuf_med=0.506 |
| shifted-garch-degrades-real-effect | PASS | real=(0.705,0.673) shifted=(0.492,0.485) |
| placebo-feature-degrades-real-effect | PASS | real=(0.705,0.673) placebo=(0.480,0.455) |

## Environment

- Python: `3.13.12`
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- Executable: `/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python`
- Imports: `duckdb=ok, numpy=ok, pandas=ok, scipy=ok, arch=ok`

## Canonical freshness and schema

### orb_outcomes

| Symbol | Max trading_day | Rows |
|---|---|---|
| GC | 2026-04-05 00:00:00 | 1295064 |
| M2K | 2026-03-06 00:00:00 | 1544508 |
| M6E | 2026-02-19 00:00:00 | 324870 |
| MBT | 2026-01-30 00:00:00 | 530124 |
| MES | 2026-04-14 00:00:00 | 1907856 |
| MGC | 2026-04-14 00:00:00 | 918396 |
| MNQ | 2026-04-14 00:00:00 | 2109564 |
| SIL | 2026-02-16 00:00:00 | 86520 |

### daily_features

| Symbol | Max trading_day | Rows |
|---|---|---|
| GC | 2026-04-06 00:00:00 | 4605 |
| M2K | 2026-03-06 00:00:00 | 4419 |
| M6E | 2026-02-20 00:00:00 | 4389 |
| MBT | 2026-01-30 00:00:00 | 4389 |
| MES | 2026-04-15 00:00:00 | 6093 |
| MGC | 2026-04-15 00:00:00 | 3354 |
| MNQ | 2026-04-15 00:00:00 | 6093 |
| SIL | 2026-02-16 00:00:00 | 1749 |

### garch column coverage

| Symbol | Rows | garch_nonnull | gpct_nonnull | first_garch_day | first_gpct_day |
|---|---|---|---|---|---|
| GC | 4605 | 4353 | 0 | 2011-04-19 00:00:00 | NaT |
| M2K | 4419 | 3663 | 0 | 2021-12-23 00:00:00 | NaT |
| M6E | 4389 | 2422 | 0 | 2021-12-23 00:00:00 | NaT |
| MBT | 4389 | 3633 | 0 | 2021-12-02 00:00:00 | NaT |
| MES | 6093 | 5315 | 5135 | 2020-03-18 00:00:00 | 2020-05-28 00:00:00 |
| MGC | 3354 | 2576 | 2396 | 2023-04-27 00:00:00 | 2023-07-06 00:00:00 |
| MNQ | 6093 | 5315 | 5135 | 2020-03-18 00:00:00 | 2020-05-28 00:00:00 |
| SIL | 1749 | 662 | 0 | 2024-12-20 00:00:00 | NaT |

Missing required columns: `none`

## Feature timing audit

- `_prior_rank_pct` sample check is expected to return `100.0` if the current row is NOT included in its own ranking window.
- Sample `_prior_rank_pct` result: `100.0`
- Sample `compute_garch_forecast` result on synthetic closes: `0.015796`
- Static line audit:
  - `pipeline/build_daily_features.py:544-578` prior-only rolling rank helper
  - `pipeline/build_daily_features.py:581-615` GARCH forecast function
  - `pipeline/build_daily_features.py:1208-1219` `garch_forecast_vol_pct` prior-only rank
  - `pipeline/build_daily_features.py:1256-1268` `garch_forecast_vol` from prior closes only

## Exact validated rebuild

- Validated strategies in scope: **45**
- Primary tests reproduced: **429**

| Anchor | Matched |
|---|---|
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 long high@70 | Y |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 short high@70 | Y |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 long low@40 | Y |

## Exact broad rebuild

- Broad in-scope rows: **430**
- Broad exact cells: **431** expected / **431** reference

| Session | Expected cells | Actual cells |
|---|---|---|
| COMEX_SETTLE | 52 | 52 |
| EUROPE_FLOW | 42 | 42 |
| TOKYO_OPEN | 32 | 32 |
| NYSE_OPEN | 60 | 60 |

## Holdout boundary

- `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`: `2026-01-01`
- `research/garch_broad_exact_role_exhaustion.py IS_END`: `2026-01-01`
- `research/garch_validated_role_exhaustion.py IS_END`: `2026-01-01`

## Destruction controls

- Real asymmetry: HIGH positive **304/431** (0.705), LOW negative **290/431** (0.673)
- Shuffle-null median fractions: HIGH **0.498**, LOW **0.506**
- Shifted-label fractions (lag 21): HIGH **0.492**, LOW **0.485**
- Placebo date-hash fractions: HIGH **0.480**, LOW **0.455**

## Verdict

SURVIVED SCRUTINY:
- python-executable: /mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python
- required-imports: duckdb=ok, numpy=ok, pandas=ok, scipy=ok, arch=ok
- required-columns: missing=none
- canonical-freshness: orb latest=2026-04-14 00:00:00 daily_features latest=2026-04-15 00:00:00
- active-scope-gpct-coverage: MNQ/MES/MGC gpct_nonnull={'MES': 5135.0, 'MGC': 2396.0, 'MNQ': 5135.0}; zero elsewhere=['GC', 'M2K', 'M6E', 'MBT', 'SIL']
- prior-rank-functional: _prior_rank_pct sample returned 100.0; expected 100.0 from prior-only window
- garch-forecast-callable: compute_garch_forecast(sample_series) -> 0.015796
- static-line-audit: build_daily_features.py lines 544-578, 581-615, 1208-1219, 1256-1268 remain prior-only
- validated-count-in-scope: actual=45 expected=45
- validated-test-count: actual=429 expected=429
- validated-anchor-rebuild: MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5=ok, MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100=ok, MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12=ok
- broad-row-count: actual=430 expected=430
- broad-cell-count: actual=431 expected=431
- broad-session-anchor-counts: COMEX_SETTLE=52, EUROPE_FLOW=42, TOKYO_OPEN=32, NYSE_OPEN=60
- holdout-boundary-consistency: holdout_policy=2026-01-01 broad=2026-01-01 validated=2026-01-01
- shuffle-null-degrades-real-effect: real_high=0.705 shuf_med=0.498; real_low=0.673 shuf_med=0.506
- shifted-garch-degrades-real-effect: real=(0.705,0.673) shifted=(0.492,0.485)
- placebo-feature-degrades-real-effect: real=(0.705,0.673) placebo=(0.480,0.455)

DID NOT SURVIVE:
- none

CAVEATS:
- This preflight does not prove profitability; it only verifies the object, plumbing, and falsification layer before more economics.
- The deterministic date-hash placebo is a synthetic null control by design; it is not a tradable feature.

NEXT STEPS:
- If this report is fully PASS, proceed to normalized sizing (`R3`) before classifier work.
- If any rebuild or holdout check fails, stop and repair the object definition before further research.
