# H1 Retrospective One-Shot — MES LONDON_METALS O30 RR1.5 long overnight_range_pct>=80

**Date committed:** 2026-04-18
**Window:** 2026-01-01 to 2026-04-18 (exclusive upper bound)
**Governing pre-reg:** `docs/audit/hypotheses/2026-04-18-h1-mes-london-metals-signal-only-shadow.yaml`
**Mode A status:** this is part of the single-shot OOS consumption. Not re-runnable.

## Retrospective aggregate

| Metric | IS (pre-2026, from drift check) | OOS retrospective (2026-01-01 to 2026-04-18) |
|---|---:|---:|
| N (long entries total) | — | 30 |
| N (on-signal, overnight_range_pct>=80) | 189 | 11 |
| ExpR on-signal | +0.2158 | +0.3097 |
| SD on-signal | 1.1629 | 1.2540 |
| WR on-signal | 0.5243 | 0.5455 |
| Wins / Losses on-signal | — | 6 / 5 |
| ExpR off-signal | -0.1069 (from drift check) | -0.3802 |

## Retrospective gate status (INDICATIVE ONLY — final verdict is combined 2026-01-01 to 2026-12-15 at review_date)

- **Primary: OOS ExpR >= 0**: +0.3097 — PASS
- **Primary: eff_ratio >= 0.40**: +1.4352 — PASS
- **Primary: direction match (sign +)**: PASS
- **Secondary: N_OOS >= 30**: 11 — FAIL (underpowered at retrospective; forward shadow may close gap by review_date)

**Interpretation:** this retrospective read is PART of the one-shot OOS consumption. It does NOT constitute a gate evaluation. Combined gate evaluation occurs ONCE at 2026-12-15 review date on the full 2026-01-01 to 2026-12-15 universe per the pre-reg. Retrospective PASSes here do not guarantee final PASS; retrospective FAILs here do not auto-kill (forward shadow may shift the combined result).

## Per-fire log (retrospective window)

| trading_day | ovn_range_pct | entry_price | stop_price | target_price | pnl_r | outcome |
|---|---:|---:|---:|---:|---:|---|
| 2026-01-26 | 91.67 | 6949.50 | 6940.75 | 6962.62 | -1.0000 | loss |
| 2026-01-30 | 93.33 | 6943.50 | 6926.00 | 6969.75 | -1.0000 | loss |
| 2026-02-02 | 100.00 | 6911.00 | 6895.00 | 6935.00 | +1.3832 | win |
| 2026-02-06 | 95.00 | 6818.00 | 6796.75 | 6849.88 | +1.4110 | win |
| 2026-02-13 | 81.67 | 6850.25 | 6827.50 | 6884.38 | -1.0000 | loss |
| 2026-03-02 | 96.67 | 6790.00 | 6768.50 | 6822.25 | +1.4120 | win |
| 2026-03-04 | 88.33 | 6813.25 | 6795.25 | 6840.25 | +1.3957 | win |
| 2026-03-09 | 96.61 | 6657.25 | 6633.50 | 6692.88 | +1.4201 | win |
| 2026-03-23 | 91.23 | 6516.50 | 6497.25 | 6545.38 | -1.0000 | loss |
| 2026-03-31 | 100.00 | 6437.25 | 6421.00 | 6461.62 | +1.3849 | win |
| 2026-04-02 | 100.00 | 6537.75 | 6528.00 | 6552.38 | -1.0000 | loss |

## CSV shadow log initialized

Shadow log csv at `docs/audit/results/h1-mes-london-metals-shadow-log.csv` initialized
with 11 retrospective on-signal fire(s) + headers. Daily forward shadow appends from
2026-04-18 through 2026-12-15.

## Compliance checklist

- [x] Window upper bound: `trading_day < 2026-04-18` enforced in SQL — no data >= 2026-04-18 queried.
- [x] Window lower bound: `trading_day >= 2026-01-01` enforced — Mode A sacred boundary respected.
- [x] Output written once; script refuses re-run if output md exists.
- [x] No threshold tuning; `overnight_range_pct >= 80` locked from pre-reg.
- [x] No gate thresholds overridden; gate values inherited from pre-reg verbatim.
