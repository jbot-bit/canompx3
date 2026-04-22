# MNQ geometry shortcut — v1

**Scope:** `MNQ / US_DATA_1000 / O5 / E2 / RR1.0 / CB1 / long`
**Truth layers:** `orb_outcomes` + `daily_features` only
**Feature:** `clearance_r = (prev_day_high - orb_high) / (orb_high - orb_low)`
**Holdout split:** `2026-01-01`

## IS bucket stats

| bucket | N | WinRate | ExpR | Total_R |
|---|---:|---:|---:|---:|
| co_located_break | 273 | 57.5% | +0.0813 | +22.2 |
| choked | 86 | 48.8% | -0.0691 | -5.9 |
| mid_clearance | 95 | 50.5% | -0.0326 | -3.1 |
| open_air | 427 | 55.5% | +0.0535 | +22.9 |
| baseline_all | 881 | 54.9% | +0.0409 | +36.0 |

## OOS bucket stats

| bucket | N | WinRate | ExpR | Total_R |
|---|---:|---:|---:|---:|
| co_located_break | 7 | 57.1% | +0.1064 | +0.7 |
| choked | 4 | 100.0% | +0.9410 | +3.8 |
| mid_clearance | 5 | 40.0% | -0.2121 | -1.1 |
| open_air | 19 | 36.8% | -0.2794 | -5.3 |
| baseline_all | 35 | 48.6% | -0.0532 | -1.9 |

## IS bucket-vs-rest Welch checks

| bucket | delta_vs_rest | t_stat | p_value |
|---|---:|---:|---:|
| co_located_break | +0.0585 | +0.856 | 0.3924 |
| choked | -0.1219 | -1.121 | 0.2649 |
| open_air | +0.0246 | +0.386 | 0.6999 |

## Notes

- `co_located_break`: `clearance_r <= 0.0`
- `choked`: `0.0 < clearance_r <= 1.0`
- `mid_clearance`: `1.0 < clearance_r <= 2.0`
- `open_air`: `clearance_r > 2.0`
- Rows with `orb_risk <= 0` are excluded fail-closed.
- Row-level audit CSV: `docs/audit/results/2026-04-22-mnq-geometry-shortcut-v1-rows.csv`

SURVIVED SCRUTINY:
- read-only canonical query only
- fixed holdout split
- fixed geometry bins; no threshold tuning

DID NOT SURVIVE:
- nothing adjudicated here beyond the bounded geometry question

CAVEATS:
- single-lane shortcut only
- Welch checks are descriptive; full family decision lives in the pre-reg framing

NEXT STEPS:
- if choked is materially weaker and co-located/open-air carry the lane, promote this into the first MNQ geometry family runner
