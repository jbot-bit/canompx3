---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Sync Audit — 2026-03-30

## DB Ground Truth (queried fresh)

### Validated Setups by (instrument, session, entry_model)

| Inst | Session | EM | N_strat | BestExpR | BestShp | TotTrd |
|------|---------|---:|--------:|---------:|--------:|-------:|
| MES | CME_PRECLOSE | E2 | 1 | 0.1426 | 1.841 | 854 |
| MES | NYSE_OPEN | E2 | 10 | 0.0995 | 1.284 | 13988 |
| MES | US_DATA_1000 | E2 | 3 | 0.0696 | 0.944 | 3527 |
| MGC | TOKYO_OPEN | E2 | 3 | 0.2832 | 1.369 | 375 |
| MGC | US_DATA_1000 | E2 | 1 | 0.1317 | 0.758 | 274 |
| MNQ | BRISBANE_1025 | E2 | 8 | 0.1056 | 0.923 | 11590 |
| MNQ | CME_PRECLOSE | E2 | 48 | 0.2840 | 2.954 | 41609 |
| MNQ | COMEX_SETTLE | E2 | 59 | 0.2214 | 2.035 | 78076 |
| MNQ | EUROPE_FLOW | E2 | 16 | 0.1274 | 1.724 | 25937 |
| MNQ | LONDON_METALS | E2 | 48 | 0.1778 | 1.354 | 70567 |
| MNQ | NYSE_CLOSE | E2 | 17 | 0.2629 | 1.683 | 7894 |
| MNQ | NYSE_OPEN | E2 | 48 | 0.1675 | 1.562 | 88242 |
| MNQ | SINGAPORE_OPEN | E2 | 124 | 0.2120 | 1.683 | 212094 |
| MNQ | TOKYO_OPEN | E2 | 37 | 0.1089 | 1.597 | 53343 |
| MNQ | US_DATA_1000 | E2 | 58 | 0.1500 | 1.798 | 102110 |
| MNQ | US_DATA_830 | E2 | 7 | 0.0727 | 0.993 | 10359 |

**Total: 488 validated strategies across 16 session-instrument groups.**
**All 12 sessions have at least 1 validated strategy.**
**CME_REOPEN is the only session with 0 validated strategies.**

### Edge Families by (instrument, session)

| Inst | Session | Fam | CORE | REG | FIT | WATCH | DECAY | SINGL |
|------|---------|----:|-----:|----:|----:|------:|------:|------:|
| MES | CME_PRECLOSE | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| MES | NYSE_OPEN | 5 | 5 | 0 | 0 | 0 | 0 | 1 |
| MES | US_DATA_1000 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| MGC | TOKYO_OPEN | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| MGC | US_DATA_1000 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| MNQ | BRISBANE_1025 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| MNQ | CME_PRECLOSE | 18 | 18 | 0 | 0 | 0 | 0 | 3 |
| MNQ | COMEX_SETTLE | 12 | 12 | 0 | 0 | 0 | 0 | 1 |
| MNQ | EUROPE_FLOW | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| MNQ | LONDON_METALS | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| MNQ | NYSE_CLOSE | 15 | 15 | 0 | 0 | 0 | 0 | 11 |
| MNQ | NYSE_OPEN | 19 | 19 | 0 | 0 | 0 | 0 | 1 |
| MNQ | SINGAPORE_OPEN | 18 | 18 | 0 | 0 | 0 | 0 | 0 |
| MNQ | TOKYO_OPEN | 19 | 19 | 0 | 0 | 0 | 0 | 4 |
| MNQ | US_DATA_1000 | 25 | 25 | 0 | 0 | 0 | 0 | 3 |
| MNQ | US_DATA_830 | 5 | 5 | 0 | 0 | 0 | 0 | 0 |

## Contradictions Found

### 1. BRISBANE_1025 marked "noise-gated, currently inactive" (TRADING_RULES.md:646)
**DB truth:** 8 validated MNQ strategies, 6 CORE edge families, best ExpR=0.1056
**Fix:** Remove "noise-gated — currently inactive" from the Brisbane time table.

### 2. RR4.0 listed as DEAD in STRATEGY_BLUEPRINT.md NO-GO (line 261)
**DB truth:** 18 FDR-significant strategies at RR4.0 (SINGAPORE_OPEN, BRISBANE_1025)
**SINGAPORE_OPEN RR4.0 is deployed in Apex Lane 2.**
**Action:** FLAG — do not remove from NO-GO without user approval. The NO-GO may predate
the current validation run, or the context may be "RR4.0 unfiltered is dead" while
filtered variants survive.

### 3. ARCHITECTURE.md instrument table — checked, correct
Active: MGC, MNQ, MES. Dead: M2K, MCL, SIL, M6E, MBT. Matches DB.
No contradictions.

### 4. CME_REOPEN — 0 validated strategies but listed in session table
Not a contradiction — it's a defined session with 0 survivors. No status label to fix.
