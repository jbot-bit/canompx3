# Volume-Driven Session Coverage Audit

**Date:** 2026-02-26
**Status:** Approved, pending implementation

## Problem

The `enabled_sessions` config in `pipeline/asset_configs.py` was last validated during the Feb 14 volume spike analysis (PASS2), before the DST overhaul, M2K onboarding, and event-based session rename. A fresh analysis against current data revealed 5 instrument x session combinations with strong structural volume spikes (2x+ baseline) that have no outcomes built.

## Volume Spike Analysis

### Method

- Binned all `bars_1m` volume into 30-minute Brisbane-time buckets per instrument
- Computed median volume per bucket
- Established per-instrument baseline as the median of all bucket medians
- Resolved each session's Brisbane time for both US winter (EST) and summer (EDT)
- Compared session-time volume to baseline

### Decision Matrix (avg of winter + summer volume multiplier)

| Session | Bris Time (W/S) | MGC | MNQ | MES | M2K |
|---------|-----------------|-----|-----|-----|-----|
| US_DATA_830 | 23:30/22:30 | 3.6x ON | **7.5x OFF** | 8.3x ON | 13.8x ON |
| NYSE_OPEN | 00:30/23:30 | **4.2x OFF** | 13.1x ON | 15.1x ON | 22.8x ON |
| US_DATA_1000 | 01:00/00:00 | 3.8x ON | 9.9x ON | 12.2x ON | 16.1x ON |
| COMEX_SETTLE | 04:30/03:30 | 1.1x ON | **5.6x OFF** | **7.0x OFF** | **8.1x OFF** |
| CME_PRECLOSE | 06:30/05:30 | 0.7x OFF | 4.5x ON | 6.6x ON | 9.1x ON |
| NYSE_CLOSE | 07:00/06:00 | 0.4x OFF | 2.8x ON | 4.7x ON | 7.2x ON |
| CME_REOPEN | 09:00/08:00 | 0.3x ON | 0.6x ON | 0.5x ON | 0.6x ON |
| TOKYO_OPEN | 10:00 | 0.6x ON | 0.7x ON | 0.8x ON | 0.6x ON |
| SINGAPORE_OPEN | 11:00 | 1.1x ON | 0.6x ON | 0.6x ON | 0.4x ON |
| LONDON_METALS | 18:00/17:00 | 1.4x ON | 1.2x ON | 1.2x ON | 1.1x ON |

### Root Cause of MNQ x US_DATA_830 Gap

Forensic git history trace:
1. Feb 14 volume analysis removed fixed "2300" from both MNQ and MES
2. MES later received "US_DATA_OPEN" (DST-aware equivalent) which became US_DATA_830
3. MNQ was never given US_DATA_OPEN — propagation error during incremental session additions
4. Gap carried through the Feb 25 event-based rename unchanged

## Decision

**Option A: Add all 5 missing high-volume sessions, keep all existing sessions as-is.**

Rationale: Volume measures liquidity, not edge. Low-volume sessions (CME_REOPEN 0.3x, TOKYO_OPEN 0.6x) host the majority of validated strategies. Removing sessions based on volume narratives would constitute Storytelling Bias (Sin #3 per backtesting literature). The validation pipeline is the sole arbiter of edge — sessions survive or are purged by math, not by narrative.

## Changes

### Config (`pipeline/asset_configs.py`)

| Instrument | Add Session | Volume Multiplier |
|-----------|-------------|-------------------|
| MNQ | US_DATA_830 | 7.5x |
| MGC | NYSE_OPEN | 4.2x |
| MNQ | COMEX_SETTLE | 5.6x |
| MES | COMEX_SETTLE | 7.0x |
| M2K | COMEX_SETTLE | 8.1x |

All existing sessions remain untouched.

### Prerequisites

Daily features data already exists for all 5 combos (verified: 1,245-2,581 break days each). `orb_COMEX_SETTLE_*` and `orb_NYSE_OPEN_*` columns are populated for all instruments. No schema changes, no new resolvers, no code changes beyond config.

### Rebuild Chain (per affected instrument)

1. `outcome_builder.py --instrument X` — E1/E2/E3 outcomes for new sessions
2. `strategy_discovery.py --instrument X` — grid search including new combos
3. `strategy_validator.py --instrument X` — validate candidates
4. `build_edge_families.py --instrument X` — cluster into families

Instruments affected: MGC (1 new session), MNQ (2 new sessions), MES (1 new session), M2K (1 new session).
