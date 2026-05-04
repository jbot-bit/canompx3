# Mode A audit of retired validated_setups — 2026-04-19

**Generated:** 2026-04-18T22:31:40+00:00
**Script:** `research/mode_a_audit_retired_setups.py`
**IS boundary:** `trading_day < 2026-01-01` (Mode A)

## Motivation

Phase 3 of the 2026-04-19 overnight session re-validated the 38 ACTIVE lanes under Mode A. The session's adversarial self-audit (Correction 2 of 9) identified that 23 RETIRED lanes were NOT similarly re-validated. If any was retired on Mode B numbers where Mode A shows a viable edge, that's survivorship bias in the re-validation work itself.

This audit recomputes Mode A IS stats for every retired lane and flags any that passes ALL of:

- Mode A N >= 100 (deployable sample)
- Mode A ExpR > 0.05 (modest positive edge)
- Mode A |t| >= 3.00 (Chordia with-theory bar)
- Mode A years_positive >= 4 (per-year stability, absolute count, N>=10 per year)

Flagged lanes are REVIVE CANDIDATES for committee review, NOT automatic reinstatements.

## Summary

- Total retired lanes audited: **23**
- Revive candidates (pass all 4 criteria): **0**
- Stay-retired (fails at least one criterion): **23**

## Per-lane Mode A audit

| Instr | Session | Om | RR | Filter | Dir | Retired | Mode-A N | Mode-A ExpR | Mode-A Sharpe | Mode-A t | Yrs+ | Verdict |
|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| GC | EUROPE_FLOW | 5 | 1.0 | OVNRNG_50 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | NYSE_OPEN | 5 | 1.0 | OVNRNG_50 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | NYSE_OPEN | 5 | 1.0 | ATR_P50 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | NYSE_OPEN | 5 | 1.0 | ATR_P70 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | NYSE_OPEN | 5 | 1.0 | OVNRNG_10 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | NYSE_OPEN | 5 | 1.5 | ATR_P50 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | NYSE_OPEN | 5 | 1.5 | ATR_P70 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | NYSE_OPEN | 5 | 1.5 | OVNRNG_10 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | NYSE_OPEN | 5 | 2.0 | ATR_P50 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | NYSE_OPEN | 5 | 2.0 | ATR_P70 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | NYSE_OPEN | 5 | 2.0 | OVNRNG_10 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | US_DATA_1000 | 5 | 1.0 | PDR_R080 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | US_DATA_1000 | 5 | 1.0 | ATR_P50 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | US_DATA_1000 | 5 | 1.0 | ATR_P70 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | US_DATA_1000 | 5 | 1.0 | OVNRNG_10 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | US_DATA_1000 | 5 | 1.0 | ORB_G5 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| GC | US_DATA_830 | 5 | 1.0 | OVNRNG_10 | long | 2026-04-11 | 0 | — | — | — | 0/0 | stay-retired |
| MNQ | EUROPE_FLOW | 5 | 1.0 | ORB_G5_NOFRI | long | 2026-04-11 | 617 | 0.0456 | 0.48 | 1.27 | 5/7 | stay-retired |
| MNQ | US_DATA_1000 | 5 | 1.0 | ORB_G5_NOFRI | long | 2026-04-11 | 701 | 0.0655 | 0.70 | 1.84 | 6/7 | stay-retired |
| MNQ | COMEX_SETTLE | 5 | 1.0 | ORB_G5_NOFRI | long | 2026-04-11 | 686 | 0.0932 | 1.03 | 2.72 | 6/7 | stay-retired |
| MNQ | TOKYO_OPEN | 5 | 1.0 | GAP_R015 | long | 2026-04-11 | 147 | 0.0420 | 0.22 | 0.59 | 5/7 | stay-retired |
| MNQ | TOKYO_OPEN | 5 | 1.5 | GAP_R015 | long | 2026-04-11 | 147 | 0.1253 | 0.52 | 1.38 | 5/7 | stay-retired |
| MNQ | CME_PRECLOSE | 5 | 1.0 | GAP_R015 | long | 2026-04-11 | 132 | 0.1156 | 0.56 | 1.49 | 5/7 | stay-retired |

## Stay-retired — summary of stay reasons

| Primary reason to stay retired | Count |
|---|---:|
| instrument not active | 17 |
| Mode A |t|<3.00 | 4 |
| Mode A ExpR <= 0.05 | 2 |

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mode_a_audit_retired_setups.py
```

Read-only. No writes to validated_setups or experimental_strategies.

