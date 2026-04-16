# Design: Discovery DELETE+INSERT Fix

**Date:** 2026-04-10
**Problem:** Discovery does DELETE FROM experimental_strategies WHERE instrument+orb_minutes before INSERT OR REPLACE. This wipes ALL prior hypothesis files' strategies. Same instrument can't have results from multiple files.
**Impact:** GC proxy exploration lost 2/3 of results. MNQ multi-RR lost 3 RR1.0 strategies.

## Root Cause

Line 1593-1601 in strategy_discovery.py. The DELETE exists to prevent "zombie" strategies from grid changes (filter removed, entry model disabled). This is valid for LEGACY mode (no hypothesis file, grid from config.py). Invalid for HYPOTHESIS mode (each file defines its own scope, cross-file strategies are intentional).

## Fix

**strategy_discovery.py lines 1586-1601:**
- When hypothesis_sha IS NOT NULL: SKIP the DELETE. INSERT OR REPLACE handles same-ID overwrites. Different files' strategies accumulate.
- When hypothesis_sha IS NULL (legacy mode): keep current DELETE for zombie cleanup.
- Single-use SHA enforcement already prevents accidental re-runs of the same file.

**config.py line 2864 _pdr_validated:**
- Add GC sessions: (GC, LONDON_METALS), (GC, EUROPE_FLOW), (GC, NYSE_OPEN), (GC, US_DATA_1000)
- Add GC to GAP gate at line 2873: (GC, CME_REOPEN)

## What this does NOT change

- No schema changes
- No change to how metrics are computed
- No change to validator behavior
- No change to legacy (no hypothesis file) behavior
- No lookahead bias risk — purely data retention mechanics

## Verification

1. Run GC discovery with file 1, count experimental_strategies
2. Run GC discovery with file 3, count — should be file 1 + file 3, not just file 3
3. Run validator — should process ALL accumulated strategies
4. Drift check passes
