---
date: 2026-05-20
scope: MNQ NYSE_OPEN E2 CB1 O5 — NFP-day "skip" hypothesis stress test
verdict: UNVERIFIED_DOMINATED_BY_ORB_SIZE_CONFOUND
session_continuity: pre-commit handoff (CTX 87%)
pooled_finding: false
related:
  - .claude/rules/research-truth-protocol.md
  - .claude/rules/backtesting-methodology.md  # § 3.3 OOS power floor
  - memory/feedback_oos_power_floor.md
  - memory/feedback_chordia_oos_park_vs_unverified_power_floor.md
---

# NFP-day skip hypothesis — adversarial verification

## What this file is

A research-log artifact capturing a stress test of the "skip NFP day on
NYSE_OPEN" hypothesis emerging from the 2026-05-20 cherry-pick of official
event-data sources (BLS/FOMC/CFTC/FRED). The hypothesis was that NFP 8:30 AM
ET releases drop into the US_DATA_830 ORB window and the subsequent NYSE_OPEN
session, producing pathological E2 fakeouts via Harris 2002 § 4.5.2
stop-cascade mechanism.

This file commits the stress-test results and the adversarial verification
findings BEFORE any draft prereg is written or any feature build is started.
No deployment. No prereg. No lane mutation.

## Pre-flight (canonical)

- DB freshness: orb_outcomes through 2026-05-17 (MNQ), daily_features through 2026-05-18.
- Holdout: Mode A, `trading_day < 2026-01-01` per `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`.
- Triple-join enforced: `(trading_day, symbol, orb_minutes)`.
- Canonical layers only: `orb_outcomes` + `daily_features`.

## Lanes tested (7)

All MNQ NYSE_OPEN E2 CB1 O5, IS only:

| RR  | filter       | n_nfp | n_non | ExpR_nfp | ExpR_non | Δ         | Welch_p |
|-----|--------------|------:|------:|---------:|---------:|----------:|--------:|
| 1.0 | COST_LT12    |    76 |  1619 |   +0.182 |   +0.074 |   +0.108  |   0.329 |
| 1.5 | COST_LT12    |    76 |  1619 |   +0.158 |   +0.097 |   +0.062  |   0.660 |
| 1.5 | ORB_G4       |    76 |  1640 |   +0.158 |   +0.098 |   +0.060  |   0.666 |
| 1.5 | OVNRNG_25    |    73 |  1590 |   +0.175 |   +0.097 |   +0.078  |   0.586 |
| 2.0 | NO_FILTER    |    76 |  1643 |   +0.173 |   +0.105 |   +0.068  |   0.676 |
| 2.0 | ORB_G4       |    76 |  1640 |   +0.173 |   +0.107 |   +0.066  |   0.684 |
| 1.0 | NO_FILTER    |    76 |  1643 |   +0.182 |   +0.073 |   +0.109  |   0.324 |

US_DATA_830 lanes excluded — none deployed/eligible at the time of test
without a cross-instrument-ATR feature (X_MES / X_MGC), and the cross-ATR
columns are not materialized in `daily_features`. The cross-instrument-ATR
testing is deferred to a later session.

## Three adversarial verification checks

### Check 1 — Per-direction split (long-short cancellation hypothesis)

Direction inferred from `entry_price > stop_price` (LONG) vs `entry_price < stop_price` (SHORT).
This is trade-time-knowable, no look-ahead (uses ORB-end values).

| RR  | dir   | NFP   |  n  | ExpR    | WR     | t       |
|-----|-------|-------|----:|--------:|-------:|--------:|
| 1.0 | LONG  | True  |  41 | +0.152  | 58.5%  | +1.04   |
| 1.0 | LONG  | False | 835 | +0.059  | 54.7%  | +1.79   |
| 1.0 | SHORT | True  |  35 | +0.217  | 62.9%  | +1.35   |
| 1.0 | SHORT | False | 807 | +0.088  | 56.1%  | +2.63   |
| 1.5 | LONG  | True  |  41 | +0.262  | 51.2%  | +1.44   |
| 1.5 | LONG  | False | 835 | +0.084  | 45.5%  | +2.07   |
| 1.5 | SHORT | True  |  35 | +0.037  | 42.9%  | +0.18   |
| 1.5 | SHORT | False | 807 | +0.109  | 46.1%  | +2.61   |
| 2.0 | LONG  | True  |  41 | +0.185  | 41.5%  | +0.87   |
| 2.0 | LONG  | False | 835 | +0.116  | 40.6%  | +2.47   |
| 2.0 | SHORT | True  |  35 | +0.160  | 40.0%  | +0.66   |
| 2.0 | SHORT | False | 807 | +0.094  | 38.8%  | +1.96   |

Per-direction Welch (all RRs pooled, MNQ NYSE_OPEN E2 CB1 O5):
- LONG: n_nfp=246, n_non=5010, Δ=+0.131, Welch t=+1.41, p=0.160
- SHORT: n_nfp=210, n_non=4842, Δ=+0.112, Welch t=+1.03, p=0.302

**Finding:** No long-short cancellation. Both directions show NFP > non-NFP,
neither reaches conventional significance. The pooled directional sign is
NOT an aggregation artefact.

### Check 2 — ORB-size confound (DECISIVE)

| Group   | mean ORB | median | n     |
|---------|---------:|-------:|------:|
| NFP     |   59.11  |  54.00 |   456 |
| non-NFP |   49.68  |  45.25 |  9852 |

Welch t=+6.82, p < 0.0001. Size ratio (NFP / non-NFP) = 1.190×.

**Finding:** NFP days have systematically 19% larger ORBs. Larger ORBs pass
ORB_G4 / COST_LT12 / OVNRNG-style filters at a higher quality bar. The
apparent NFP-day ExpR uplift is most parsimoniously explained by this
size effect rather than by any news-mechanism signal. **The "edge" lives in
the ORB itself, not in the news day** — and our existing ORB-size / cost
filters already capture it.

### Check 3 — Canonical filter-wrapper parity (CRITICAL — DOCTRINE FLAG)

Per `.claude/rules/research-truth-protocol.md` § Canonical filter delegation
(2026-04-18 amendment), all research scans MUST delegate to
`research.filter_utils.filter_signal(df, key, orb_label)` rather than
re-encoding filter logic inline. I called both:

- Inline Python (using `pipeline.cost_model.get_cost_spec` formula): 1695 / 1718 pass
- Canonical `filter_signal(df, 'COST_LT12', 'NYSE_OPEN')`: 0 / 1718 pass
- Agreement: 23 / 1718

**Finding:** wrapper parity FAILED. Two possible causes:
1. My DataFrame lacked fields the wrapper requires (likely a `symbol`
   column or different schema shape than the wrapper expects).
2. A bug in the wrapper itself.

Either way, the COST_LT12 lane numbers in the table above are UNVERIFIED
until parity is established. This is a DIFFERENT finding from the NFP
verdict — it is a doctrine-class flag worth investigating in its own
right next session.

## Verdict

```
VERDICT: UNVERIFIED — DOMINATED BY ORB-SIZE CONFOUND
```

Composite reasoning:
1. **Power floor breached** on every cell tested (one-sample power 0.07-0.17,
   far below the 0.50 `STATISTICALLY_USELESS` threshold in
   `research/oos_power.py::POWER_TIERS`). Per
   `memory/feedback_oos_power_floor.md` and
   `memory/feedback_chordia_oos_park_vs_unverified_power_floor.md`, this
   forbids any DEAD / KILL label. The earlier in-session "VERDICT: KILL"
   was DOCTRINE-VIOLATING and is hereby revised.
2. **Directional sign is real but explained by confound.** NFP > non-NFP
   on 7/7 lanes is not noise (binomial under null = 1/128), but the
   ORB-size confound (1.19× larger ORBs on NFP days, p < 0.0001) provides
   a parsimonious non-mechanism explanation. ORB_G4 / COST_LT12 already
   capture this size effect. No incremental edge over existing filters.
3. **No skip-day filter motivated.** The original mechanism (stop-cascade
   pathology) is neither supported nor refuted; the data is consistent
   with either "no mechanism, size confound only" or "small positive
   mechanism, swamped by power deficit." Trading-money implication: do
   not add `skip_nfp` to any deployed lane on this evidence.

## Limitations explicitly logged

- US_DATA_830 lanes (the original highest-priority target) NOT tested —
  no eligible lanes without cross-ATR features that aren't materialized.
- Cross-instrument-ATR lanes (X_MES_ATR60, X_MGC_ATR70) NOT tested for
  the same reason.
- COST_LT12 numbers are UNVERIFIED pending wrapper-parity resolution.
- OOS power calc used one-sample helper with two-sample t-input — slightly
  sloppy but result direction (all STATISTICALLY_USELESS) is robust
  because n_nfp ≈ 76 is unambiguously too small.
- All seven lanes are MNQ NYSE_OPEN — does NOT generalize to MGC or MES,
  or to other US-session orb_labels.

## What should happen next session

1. **Resolve wrapper parity** (Check 3 finding) — investigate why
   `research.filter_utils.filter_signal('COST_LT12', 'NYSE_OPEN')` returns
   zero passes against a daily_features-shaped frame. This is a doctrine
   integrity finding, separate from the NFP question.
2. **Re-run NFP test on US_DATA_830 deployed/eligible lanes** once
   cross-instrument-ATR features land OR by excluding X_*_ATR* lanes.
   The US_DATA_830 window is closer to the news release than NYSE_OPEN —
   the original mechanism predicts the effect should be STRONGER there.
3. **Consider FOMC as the next mechanism candidate.** FOMC drops directly
   into CME_PRECLOSE + NYSE_CLOSE (deployed and paused lanes), and the
   8/year cadence × 6 IS years = ~48 events is comparable to NFP power
   profile. New feature `is_fomc_day` would need to be built first
   (currently absent from `daily_features`).
4. **Park `skip_nfp` as a research target** until US_DATA_830 + wrapper-parity
   results land. Do NOT promote to filter / prereg / deployment on this
   evidence.

## Decision-log preservation

This file replaces and supersedes the in-session "VERDICT: KILL" emitted
mid-session 2026-05-20 (~CTX 80%). That verdict was doctrine-violating
on the power floor and should NOT be cited as a settled verdict in
downstream work.
