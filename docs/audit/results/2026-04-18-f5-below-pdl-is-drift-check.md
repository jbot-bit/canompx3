# F5_BELOW_PDL IS Drift Check — MNQ US_DATA_1000 O5 E2 CB1 RR1.0 long

**Date:** 2026-04-18
**Purpose:** Lock the IS baseline (pre-2026-01-01) for F5_BELOW_PDL so the
one-shot OOS validator can compute eff_ratio and direction-match gates
without re-deriving IS on the fly. This document is a PRE-REG COMPANION,
not an independent finding.

## Canonical scope

| Dimension | Value | Canonical source |
|-----------|-------|------------------|
| instrument | MNQ | `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` |
| session | US_DATA_1000 | `pipeline.dst.SESSION_CATALOG` |
| orb_minutes | 5 | `orb_outcomes.orb_minutes` |
| entry_model | E2 | `orb_outcomes.entry_model` |
| confirm_bars | 1 | only value present in DB for this cell |
| rr_target | 1.0 | `orb_outcomes.rr_target` |
| direction | long | `orb_US_DATA_1000_break_dir='long'` |
| filter | `F5_BELOW_PDL` = `(orb_mid < prev_day_low)` | research-side feature (not in `ALL_FILTERS` yet) |
| IS window | `trading_day >= 2019-05-07 AND trading_day < 2026-01-01` | `HOLDOUT_SACRED_FROM = 2026-01-01` from `trading_app.holdout_policy` |

`orb_mid = (orb_US_DATA_1000_high + orb_US_DATA_1000_low) / 2.0` — known at
ORB end (5 min after session open). `prev_day_low` is prior session's close —
available at start of current session. Both strictly before E2 CB1 entry.
Per `.claude/rules/backtesting-methodology.md` Rule 6.1, this is **safe
(trade-time-knowable)**.

## Locked IS numbers

| Metric | Value |
|--------|------:|
| Total long entries IS | 881 |
| N_on (F5_BELOW_PDL = 1) | 136 |
| N_off (F5_BELOW_PDL = 0) | 745 |
| ExpR_on | +0.3258 |
| ExpR_off | -0.0112 |
| Δ_IS (on - off) | +0.3370 |
| Welch t | +4.0176 |
| Welch p | 0.000084 |
| SD_on (sample) | 0.8909 |
| WR_on | 0.6912 |
| IS min trading_day | 2019-05-07 |
| IS max trading_day | 2025-12-31 |

Matches the 2026-04-15 mega scan (`docs/audit/results/2026-04-15-mega-deployed-sessions-only.md:26`)
to reported precision:
- mega scan: N_on=136, ExpR_on=+0.326, Δ_IS=+0.337, t=+4.02, p=0.0001
- re-verified above: N_on=136, ExpR_on=+0.3258, Δ_IS=+0.3370, t=+4.0176, p=0.000084

## Feature integrity

- `F5_BELOW_PDL` is computed identically in `research/mega_deployed_sessions_only.py:132` and
  `research/prior_day_features_orb.py:170`: `(mid < pdl).astype(int)`.
- `prev_day_low` is provenance-safe: prior-day canonical, never re-indexes into current day.
- No post-entry data used (outcome, pnl_r, MAE/MFE, double_break).
- Look-ahead category per Rule 6.1: **SAFE**.

## Not re-tunable

- Threshold is binary (`mid < pdl`). No parameterisation.
- Direction is locked to long to match upstream mega-scan survivor cell.
- RR is locked to 1.0 (RR1.5 and RR2.0 exist as upstream secondary survivors but are out of scope for this pre-reg per user scope rule).

## Companion artifact for

`docs/audit/hypotheses/2026-04-18-f5-below-pdl-mnq-us-data-1000-o5-rr1-one-shot.yaml`
