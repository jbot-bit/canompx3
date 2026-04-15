# Garch Findings — Consolidated Stress Test (All 32 Families)

**Date:** 2026-04-15
**Trigger:** User demanded stress test on ALL garch findings, not just NYSE_OPEN. Answers: which families survive the stress battery vs which are within shuffle-envelope noise.

**Tests per family:**
- A. Shuffle control (50 shuffles) — is observed positive fraction outside shuffle envelope?
- B. Per-year consistency — does the directional pattern hold year over year?
- C. Long-vs-short split — direction-symmetric or asymmetric?
- D. Continuous regression — linear slope of pnl_r on garch_pct.

**Classifications:**
- SURVIVES_POSITIVE_BOTH_DIRS: clean positive signal both directions (deploy sizer candidate)
- SURVIVES_POSITIVE_LONG/SHORT_ONLY: positive only one direction
- POSITIVE_INCONSISTENT: outside envelope but year/direction mixed
- SURVIVES_INVERSE_BOTH_DIRS: clean inverse both directions (deploy SKIP)
- SURVIVES_INVERSE_LONG/SHORT_ONLY: inverse only one direction
- INVERSE_INCONSISTENT: outside envelope but mixed
- NOISE_WITHIN_ENVELOPE: indistinguishable from shuffle noise

---

## Classification summary

| Classification | Count |
|---|---|
| NOISE_WITHIN_ENVELOPE | 12 |
| SURVIVES_POSITIVE_BOTH_DIRS | 11 |
| SURVIVES_POSITIVE_LONG_ONLY | 4 |
| SURVIVES_POSITIVE_SHORT_ONLY | 3 |
| SURVIVES_INVERSE_BOTH_DIRS | 1 |
| SURVIVES_INVERSE_SHORT_ONLY | 1 |

---

## Per-family full stress-test grid

| Inst | Session | Cells | Pos% | Long Pos% | Short Pos% | Shuf envelope | Envelope | Regr long p | Regr short p | Yr consistency | Class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MES | CME_REOPEN | 54 | 50.0% | 0.0% | 100.0% | [0.33, 0.65] | WITHIN | 0.000 | 0.004 | 3/5 | NOISE_WITHIN_ENVELOPE |
| MES | COMEX_SETTLE | 54 | 59.3% | 33.3% | 85.2% | [0.33, 0.65] | WITHIN | 0.272 | 0.007 | 2/6 | NOISE_WITHIN_ENVELOPE |
| MES | NYSE_CLOSE | 41 | 43.9% | 35.0% | 52.4% | [0.29, 0.72] | WITHIN | 0.015 | 0.278 | 1/5 | NOISE_WITHIN_ENVELOPE |
| MES | NYSE_OPEN | 54 | 61.1% | 29.6% | 92.6% | [0.35, 0.69] | WITHIN | 0.832 | 0.011 | 2/6 | NOISE_WITHIN_ENVELOPE |
| MGC | NYSE_OPEN | 54 | 50.0% | 55.6% | 44.4% | [0.35, 0.69] | WITHIN | 0.328 | 0.901 | 1/3 | NOISE_WITHIN_ENVELOPE |
| MGC | US_DATA_830 | 54 | 64.8% | 29.6% | 100.0% | [0.35, 0.65] | WITHIN | 0.062 | 0.000 | 2/3 | NOISE_WITHIN_ENVELOPE |
| MNQ | CME_REOPEN | 54 | 57.4% | 14.8% | 100.0% | [0.32, 0.63] | WITHIN | 0.006 | 0.000 | 4/6 | NOISE_WITHIN_ENVELOPE |
| MNQ | COMEX_SETTLE | 54 | 50.0% | 37.0% | 63.0% | [0.34, 0.62] | WITHIN | 0.388 | 0.433 | 4/6 | NOISE_WITHIN_ENVELOPE |
| MNQ | LONDON_METALS | 54 | 38.9% | 33.3% | 44.4% | [0.34, 0.61] | WITHIN | 0.309 | 0.632 | 3/6 | NOISE_WITHIN_ENVELOPE |
| MNQ | NYSE_CLOSE | 42 | 50.0% | 4.8% | 95.2% | [0.32, 0.70] | WITHIN | 0.088 | 0.026 | 3/6 | NOISE_WITHIN_ENVELOPE |
| MNQ | US_DATA_1000 | 54 | 57.4% | 48.1% | 66.7% | [0.31, 0.60] | WITHIN | 0.117 | 0.598 | 2/6 | NOISE_WITHIN_ENVELOPE |
| MNQ | US_DATA_830 | 54 | 40.7% | 11.1% | 70.4% | [0.32, 0.61] | WITHIN | 0.012 | 0.449 | 2/6 | NOISE_WITHIN_ENVELOPE |
| MNQ | NYSE_OPEN | 54 | 14.8% | 0.0% | 29.6% | [0.34, 0.63] | BELOW_NEG | 0.000 | 0.638 | 1/6 | SURVIVES_INVERSE_BOTH_DIRS |
| MGC | US_DATA_1000 | 54 | 33.3% | 37.0% | 29.6% | [0.34, 0.70] | BELOW_NEG | 0.014 | 0.540 | 1/3 | SURVIVES_INVERSE_SHORT_ONLY |
| MES | CME_PRECLOSE | 24 | 95.8% | 100.0% | 91.7% | [0.29, 0.67] | ABOVE_POS | 0.024 | 0.000 | 4/5 | SURVIVES_POSITIVE_BOTH_DIRS |
| MES | LONDON_METALS | 54 | 85.2% | 70.4% | 100.0% | [0.33, 0.70] | ABOVE_POS | 0.088 | 0.000 | 5/6 | SURVIVES_POSITIVE_BOTH_DIRS |
| MES | SINGAPORE_OPEN | 54 | 83.3% | 92.6% | 74.1% | [0.33, 0.65] | ABOVE_POS | 0.000 | 0.789 | 5/6 | SURVIVES_POSITIVE_BOTH_DIRS |
| MES | TOKYO_OPEN | 54 | 94.4% | 100.0% | 88.9% | [0.30, 0.64] | ABOVE_POS | 0.000 | 0.147 | 4/6 | SURVIVES_POSITIVE_BOTH_DIRS |
| MES | US_DATA_830 | 54 | 83.3% | 92.6% | 74.1% | [0.34, 0.63] | ABOVE_POS | 0.024 | 0.033 | 5/6 | SURVIVES_POSITIVE_BOTH_DIRS |
| MGC | CME_REOPEN | 54 | 85.2% | 88.9% | 81.5% | [0.37, 0.62] | ABOVE_POS | 0.001 | 0.003 | 2/3 | SURVIVES_POSITIVE_BOTH_DIRS |
| MGC | LONDON_METALS | 54 | 85.2% | 70.4% | 100.0% | [0.35, 0.69] | ABOVE_POS | 0.123 | 0.000 | 3/3 | SURVIVES_POSITIVE_BOTH_DIRS |
| MGC | TOKYO_OPEN | 54 | 88.9% | 81.5% | 96.3% | [0.32, 0.64] | ABOVE_POS | 0.537 | 0.003 | 3/3 | SURVIVES_POSITIVE_BOTH_DIRS |
| MNQ | BRISBANE_1025 | 54 | 88.9% | 100.0% | 77.8% | [0.37, 0.67] | ABOVE_POS | 0.000 | 0.041 | 5/6 | SURVIVES_POSITIVE_BOTH_DIRS |
| MNQ | CME_PRECLOSE | 33 | 100.0% | 100.0% | 100.0% | [0.30, 0.70] | ABOVE_POS | 0.004 | 0.018 | 4/6 | SURVIVES_POSITIVE_BOTH_DIRS |
| MNQ | TOKYO_OPEN | 54 | 94.4% | 88.9% | 100.0% | [0.35, 0.66] | ABOVE_POS | 0.000 | 0.000 | 4/6 | SURVIVES_POSITIVE_BOTH_DIRS |
| MES | US_DATA_1000 | 54 | 66.7% | 77.8% | 55.6% | [0.29, 0.64] | ABOVE_POS | 0.134 | 0.691 | 4/6 | SURVIVES_POSITIVE_LONG_ONLY |
| MGC | COMEX_SETTLE | 54 | 74.1% | 88.9% | 59.3% | [0.37, 0.67] | ABOVE_POS | 0.000 | 0.437 | 3/3 | SURVIVES_POSITIVE_LONG_ONLY |
| MGC | EUROPE_FLOW | 54 | 77.8% | 96.3% | 59.3% | [0.31, 0.70] | ABOVE_POS | 0.000 | 0.017 | 2/3 | SURVIVES_POSITIVE_LONG_ONLY |
| MNQ | SINGAPORE_OPEN | 54 | 83.3% | 100.0% | 66.7% | [0.38, 0.67] | ABOVE_POS | 0.000 | 0.153 | 3/6 | SURVIVES_POSITIVE_LONG_ONLY |
| MES | EUROPE_FLOW | 54 | 72.2% | 63.0% | 81.5% | [0.32, 0.66] | ABOVE_POS | 0.000 | 0.000 | 3/6 | SURVIVES_POSITIVE_SHORT_ONLY |
| MGC | SINGAPORE_OPEN | 54 | 74.1% | 48.1% | 100.0% | [0.34, 0.66] | ABOVE_POS | 0.527 | 0.000 | 3/3 | SURVIVES_POSITIVE_SHORT_ONLY |
| MNQ | EUROPE_FLOW | 54 | 83.3% | 66.7% | 100.0% | [0.33, 0.65] | ABOVE_POS | 0.345 | 0.010 | 4/6 | SURVIVES_POSITIVE_SHORT_ONLY |

---

## Deployable findings (only classifications that survive)

| Inst | Session | Class | Long Pos% | Short Pos% | Action |
|---|---|---|---|---|---|
| MNQ | TOKYO_OPEN | SURVIVES_POSITIVE_BOTH_DIRS | 88.9% | 100.0% | SIZER family — pre-reg both directions |
| MNQ | SINGAPORE_OPEN | SURVIVES_POSITIVE_LONG_ONLY | 100.0% | 66.7% | SIZER long-only |
| MNQ | EUROPE_FLOW | SURVIVES_POSITIVE_SHORT_ONLY | 66.7% | 100.0% | SIZER short-only |
| MNQ | NYSE_OPEN | SURVIVES_INVERSE_BOTH_DIRS | 0.0% | 29.6% | SKIP family — both directions |
| MNQ | CME_PRECLOSE | SURVIVES_POSITIVE_BOTH_DIRS | 100.0% | 100.0% | SIZER family — pre-reg both directions |
| MNQ | BRISBANE_1025 | SURVIVES_POSITIVE_BOTH_DIRS | 100.0% | 77.8% | SIZER family — pre-reg both directions |
| MES | TOKYO_OPEN | SURVIVES_POSITIVE_BOTH_DIRS | 100.0% | 88.9% | SIZER family — pre-reg both directions |
| MES | SINGAPORE_OPEN | SURVIVES_POSITIVE_BOTH_DIRS | 92.6% | 74.1% | SIZER family — pre-reg both directions |
| MES | LONDON_METALS | SURVIVES_POSITIVE_BOTH_DIRS | 70.4% | 100.0% | SIZER family — pre-reg both directions |
| MES | EUROPE_FLOW | SURVIVES_POSITIVE_SHORT_ONLY | 63.0% | 81.5% | SIZER short-only |
| MES | US_DATA_830 | SURVIVES_POSITIVE_BOTH_DIRS | 92.6% | 74.1% | SIZER family — pre-reg both directions |
| MES | US_DATA_1000 | SURVIVES_POSITIVE_LONG_ONLY | 77.8% | 55.6% | SIZER long-only |
| MES | CME_PRECLOSE | SURVIVES_POSITIVE_BOTH_DIRS | 100.0% | 91.7% | SIZER family — pre-reg both directions |
| MGC | CME_REOPEN | SURVIVES_POSITIVE_BOTH_DIRS | 88.9% | 81.5% | SIZER family — pre-reg both directions |
| MGC | TOKYO_OPEN | SURVIVES_POSITIVE_BOTH_DIRS | 81.5% | 96.3% | SIZER family — pre-reg both directions |
| MGC | SINGAPORE_OPEN | SURVIVES_POSITIVE_SHORT_ONLY | 48.1% | 100.0% | SIZER short-only |
| MGC | LONDON_METALS | SURVIVES_POSITIVE_BOTH_DIRS | 70.4% | 100.0% | SIZER family — pre-reg both directions |
| MGC | EUROPE_FLOW | SURVIVES_POSITIVE_LONG_ONLY | 96.3% | 59.3% | SIZER long-only |
| MGC | US_DATA_1000 | SURVIVES_INVERSE_SHORT_ONLY | 37.0% | 29.6% | SKIP short-only |
| MGC | COMEX_SETTLE | SURVIVES_POSITIVE_LONG_ONLY | 88.9% | 59.3% | SIZER long-only |