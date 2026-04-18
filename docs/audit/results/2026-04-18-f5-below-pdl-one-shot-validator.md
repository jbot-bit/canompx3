# F5_BELOW_PDL LONG MNQ US_DATA_1000 O5 RR1.0 — One-Shot Validator

**Pre-reg:** `docs/audit/hypotheses/2026-04-18-f5-below-pdl-mnq-us-data-1000-o5-rr1-one-shot.yaml`
**Pre-reg sha:** `311f79979251cadfce5c82d7413260ca1b069e80`
**Lock date:** 2026-04-18
**IS window:** [2019-05-07, 2026-01-01)
**OOS window:** [2026-01-01, 2026-04-18)
**Mode A sacred boundary:** 2026-01-01 (from trading_app.holdout_policy)

## VERDICT: **KILL**

Kill reasons: K1 ExpR_on_OOS<0, K2 eff_ratio<0.4, K3 sign flip

## IS drift check vs pre-reg locked

| Metric | Pre-reg locked | Validator re-computed | Drift |
|---|---:|---:|---:|
| N_on | 136 | 136 | +0 |
| ExpR_on | +0.3258 | +0.3258 | +0.000046 |
| Δ (on-off) | +0.3370 | +0.3370 | +0.000010 |

IS drift within tolerance: **True**

## Aggregate metrics

| Metric | IS (locked) | OOS (one-shot) |
|---|---:|---:|
| N_total_long | — | 33 |
| N_on (F5_BELOW_PDL=1) | 136 | 8 |
| N_off (F5_BELOW_PDL=0) | 745 | 25 |
| ExpR_on | +0.3258 | -0.0243 |
| ExpR_off | -0.0112 | -0.1398 |
| Δ (on-off) | +0.3370 | +0.1155 |
| SD_on | 0.8909 | 1.0431 |
| WR_on | 0.6912 | 0.5000 |
| Welch t (on vs off) | +4.0176 | +nan |
| Welch p | 0.000084 | nan |

## Gate evaluation

| Gate | Rule | Value | Result |
|---|---|---:|:---:|
| Primary: OOS ExpR >= 0 | ExpR_on_OOS >= 0 | -0.0243 | FAIL |
| Primary: eff_ratio >= 0.40 | ExpR_on_OOS / ExpR_on_IS | -0.0745 | FAIL |
| Primary: direction match | sign(OOS) == sign(IS=+) | - | FAIL |
| Primary: N_on_OOS >= 5 | N_on_OOS | 8 | PASS |
| Secondary: bootstrap p < 0.10 | moving-block B=10000, block=5 | nan | N/A |

## Per-fire log — OOS on-signal (F5_BELOW_PDL=1, long)

| trading_day | entry_price | stop_price | target_price | pnl_r | outcome | orb_mid | prev_day_low |
|---|---:|---:|---:|---:|---|---:|---:|
| 2026-01-08 | 25615.25 | 25569.50 | 25661.00 | +0.9381 | win | 25592.25 | 25727.00 |
| 2026-01-14 | 25589.75 | 25533.00 | 25646.50 | +0.9498 | win | 25561.25 | 25803.25 |
| 2026-01-19 | 25336.25 | 25311.75 | 25360.75 | -1.0000 | loss | 25323.88 | 25590.25 |
| 2026-02-04 | 25197.25 | 25107.75 | 25286.75 | +0.9679 | win | 25152.38 | 25218.00 |
| 2026-02-17 | 24566.75 | 24461.25 | 24672.25 | -1.0000 | loss | 24513.88 | 24670.50 |
| 2026-03-20 | 24313.25 | 24260.00 | 24366.50 | -1.0000 | loss | 24286.50 | 24324.25 |
| 2026-03-27 | 23520.25 | 23444.00 | 23596.50 | -1.0000 | loss | 23482.00 | 23762.00 |
| 2026-04-02 | 23921.75 | 23864.75 | 23978.75 | +0.9501 | win | 23893.12 | 23943.00 |

## Compliance

- [x] IS window respected: trading_day < 2026-01-01
- [x] OOS window respected: 2026-01-01 <= trading_day < 2026-04-18
- [x] No threshold tuning (binary predicate, nothing to tune)
- [x] Feature is trade-time-knowable (backtesting-methodology.md Rule 6.1)
- [x] Triple-join on (trading_day, symbol, orb_minutes)
- [x] Script refuses re-run if output md exists
- [x] Pre-reg sha pinned: 311f79979251cadfce5c82d7413260ca1b069e80

## Decision

- KILL: K1 ExpR_on_OOS<0, K2 eff_ratio<0.4, K3 sign flip. Declare F5_BELOW_PDL on this exact lane DEAD. Postmortem required. Do not wire into ALL_FILTERS.
