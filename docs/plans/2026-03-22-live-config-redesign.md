# Live Config Redesign — From Ground Truth

**Date:** 2026-03-22
**Status:** DESIGN (not implemented)
**Input:** 832 validated strategies, 255 RR locks, 248 edge families

## Problem

LIVE_PORTFOLIO has 46 specs designed around a prior validated inventory that no longer exists. Only 8/46 resolve to strategies. 38 specs are dead. Meanwhile, 150 resolvable (instrument, session, filter) combos exist with no live spec targeting them — including MGC's strongest edge (TOKYO_OPEN ORB_G4, 0.256R) and high-value MNQ strategies.

## Current State (8 specs that resolve)

| Spec | Instruments | Best ExpR | Notes |
|------|------------|-----------|-------|
| CME_PRECLOSE_E2_VOL_RV12_N20 | MNQ | 0.374 | 4 strats |
| CME_PRECLOSE_E2_ATR70_VOL | MES, MNQ | 0.346 | 4 strats |
| CME_PRECLOSE_E2_X_MES_ATR60 | MNQ | 0.360 | 3 strats |
| CME_PRECLOSE_E2_X_MGC_ATR70 | MNQ | 0.231 | 2 strats |
| CME_PRECLOSE_E2_ORB_G5 | MES | 0.238 | 1 strat |
| COMEX_SETTLE_E2_ATR70_VOL | MNQ | 0.360 | 2 strats |
| SINGAPORE_OPEN_E2_ATR70_VOL | MNQ | 0.221 | 1 strat |
| NYSE_CLOSE_E2_ATR70_VOL | MNQ | 0.303 | 2 strats |

## Top Untapped Inventory (no live spec)

| Instrument | Session | Filter | Best ExpR | N strats | Family |
|-----------|---------|--------|-----------|----------|--------|
| MNQ | CME_PRECLOSE | ORB_G8 | 0.286 | 1 | PURGED |
| MNQ | CME_PRECLOSE | X_MES_ATR70 | 0.268 | 2 | WHITELISTED |
| MGC | TOKYO_OPEN | ORB_G4 | 0.256 | 1 | ROBUST |
| MGC | TOKYO_OPEN | ORB_G4_CONT | 0.255 | 1 | ROBUST |
| MNQ | CME_REOPEN | ATR70_VOL | 0.303 | 2 | WHITELISTED |
| MES | CME_PRECLOSE | ORB_G4 | 0.231 | 1 | SINGLETON |
| MNQ | BRISBANE_1025 | ATR70_VOL | 0.229 | 1 | WHITELISTED |
| MES | NYSE_CLOSE | ORB_G6 | 0.227 | 1 | SINGLETON |
| MNQ | EUROPE_FLOW | ATR70_VOL | 0.210 | 2 | — |
| MNQ | COMEX_SETTLE | X_MES_ATR60 | 0.193 | 2 | — |

## Design Principles for New Specs

1. **Build from inventory, not from wishes.** Every spec must have at least one resolvable strategy in validated_setups + family_rr_locks.
2. **Prefer ROBUST/WHITELISTED families over PURGED/SINGLETON.** PURGED families can be included but should be flagged.
3. **Don't over-specify.** Each spec should be (session, entry_model, filter_type). Instrument resolution is automatic via the JOIN.
4. **Include MGC.** MGC TOKYO_OPEN ORB_G4 (2 ROBUST families) is MGC's only viable edge. It MUST be in the spec list.
5. **Tier by evidence.** CORE = N≥100 + ROBUST/WHITELISTED. REGIME = N<100 or PURGED/SINGLETON.

## Proposed Spec List (from ground truth)

### CORE tier (always-on)
- CME_PRECLOSE E2 ATR70_VOL (MES+MNQ, 0.29-0.35R, ROBUST/WHITELISTED)
- CME_PRECLOSE E2 VOL_RV12_N20 (MNQ, 0.37R, PURGED but strongest edge)
- CME_PRECLOSE E2 X_MES_ATR60 (MNQ, 0.36R, PURGED)
- CME_PRECLOSE E2 X_MES_ATR70 (MNQ, 0.27R, WHITELISTED)
- CME_PRECLOSE E2 X_MGC_ATR70 (MNQ, 0.23R, ROBUST)
- CME_PRECLOSE E2 ORB_G5 (MES, 0.24R, SINGLETON)
- CME_PRECLOSE E2 ORB_G8 (MNQ, 0.29R, PURGED)
- CME_PRECLOSE E2 ORB_G4 (MES, 0.23R, SINGLETON)
- COMEX_SETTLE E2 ATR70_VOL (MNQ, 0.36R, ROBUST)
- CME_REOPEN E2 ATR70_VOL (MNQ, 0.30R, WHITELISTED)
- NYSE_CLOSE E2 ATR70_VOL (MNQ, 0.30R, PURGED)
- TOKYO_OPEN E2 ORB_G4 (MGC, 0.26R, ROBUST) — **MGC's primary edge**
- TOKYO_OPEN E2 ORB_G4_CONT (MGC, 0.26R, ROBUST) — same family variant
- SINGAPORE_OPEN E2 ATR70_VOL (MNQ, 0.22R, PURGED)
- BRISBANE_1025 E2 ATR70_VOL (MNQ, 0.23R, WHITELISTED)
- EUROPE_FLOW E2 ATR70_VOL (MNQ, 0.21R, —)

### Potentially add (ExpR 0.15-0.22R, large N)
- NYSE_OPEN E2 VOL_RV12_N20 (MES+MNQ, 0.11-0.16R, large N)
- NYSE_OPEN E2 X_MGC_ATR70 (MNQ, 0.19R)
- US_DATA_1000 E2 X_MES_ATR70 (MNQ, 0.19R)
- COMEX_SETTLE E2 VOL_RV12_N20 (MNQ, 0.18R)
- MES NYSE_CLOSE ORB_G6 (0.23R, SINGLETON)

## Decisions Needed Before Implementation

1. **Include PURGED families in CORE tier?** Several top strategies (CME_PRECLOSE VOL_RV12_N20 at 0.37R) are in PURGED families. They have strong individual metrics but failed family robustness checks.
2. **Minimum ExpR for inclusion?** 0.22 is the current LIVE_MIN_EXPECTANCY_R. Many strong strategies (MES NYSE_OPEN at 0.08-0.11R) would be excluded.
3. **How many specs?** Prior had 46 (mostly dead). New set could be 16-30 (all alive).
4. **exclude_instruments:** Several prior exclusions (BH FDR per instrument) were based on the old validation. Need to re-evaluate.
