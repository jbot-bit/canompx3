# NON-ML PIPELINE AUDIT REPORT

**Audit Date:** 2026-03-21
**Database:** `C:\Users\joshd\canompx3\gold.db`
**Instruments Audited:** MGC, MES, MNQ
**Auditor:** Fresh agent, zero context, no bias

---

## A. INVENTORY TRUTH

**validated_setups:** 11 total rows. ALL are MNQ. ALL are `status='active'`.

| Instrument | Rows | Sessions | Filters | Entry Model |
|---|---|---|---|---|
| MGC | 0 | - | - | - |
| MES | 0 | - | - | - |
| MNQ | 11 | CME_PRECLOSE (7), COMEX_SETTLE (3), BRISBANE_1025 (1) | ATR70_VOL (8), VOL_RV12_N20 (1), X_MES_ATR60 (1), none-standard stop (1) | E2 only |

**Pre-Mar-19 state** (from `_vs_backup`): 530 active strategies -- MGC=144, MES=138, MNQ=217, M2K=31. Entry models: E0-dominant for MNQ (217 E0), E1-dominant for MGC (94 E1, 40 E0, 10 E3), E0-dominant for MES (106 E0, 27 E1, 5 E3).

**Net change:** 530 -> 11 active strategies (-98%). MGC/MES inventory destroyed by E0 purge + Mar 19 rediscovery under stricter validation (E1/E2 only, trade-count walk-forward for MGC).

**edge_families:** 5 families, all MNQ. 1 ROBUST (5 members), 1 WHITELISTED (3 members), 2 SINGLETON, 1 PURGED. Family heads all resolve to real validated_setups rows.

**VERDICT: CRITICAL.** MGC and MES have zero validated strategies. 98% inventory collapse from pre-Mar-19 state.

---

## B. LIVE SPEC RESOLUTION

**LIVE_PORTFOLIO** defines 46 specs (32 core, 2 regime, 14 vol-regime).

**Resolution results:**
- **MGC:** 0 strategies loaded from 45 applicable specs. 43 specs produce "no variant found", 1 seasonal-gated, 1 BH FDR excluded. **EMPTY PORTFOLIO.**
- **MES:** 0 strategies loaded from 42 applicable specs. 40 "no variant found", 2 excluded. **EMPTY PORTFOLIO.**
- **MNQ:** 4 strategies loaded from 42 applicable specs. 37 "no variant found", 4 excluded, 1 seasonal. **9.5% resolution rate.**

**Root cause:** `_load_best_regime_variant()` joins `validated_setups` to `family_rr_locks` on instrument+session+filter+entry+aperture+cb+rr. With MGC/MES having 0 rows in validated_setups, all 46 specs fail. For MNQ, only 5 of 361 family_rr_locks match the 11 validated_setups rows.

**LIVE_PORTFOLIO was designed for an inventory that no longer exists.** The 46 specs reference ORB_G4/G5/G6/G8, VOL_RV12_N20, DIR_LONG -- filter types from the old E0/E1 era that produced hundreds of validators per instrument. The current Mar 19 rebuild produced only ATR70_VOL/X_MES_ATR60 survivors.

**VERDICT: CRITICAL.** 91% of live specs are broken. Live portfolio functionally empty for MGC/MES, 9.5% for MNQ.

---

## C. DISCOVERY COVERAGE

Discovery ran for all 3 instruments on 2026-03-19:
- **MGC:** 22,464 rows. 9 sessions. E1+E2 only (no E0, no E3). 24 filter types.
- **MES:** 25,918 rows. 11 sessions. E1+E2 only. 26 filter types.
- **MNQ:** 31,104 rows. 12 sessions. E1+E2 only. 27 filter types.

**Session gaps in discovery vs live specs:**
- MGC missing: `CME_PRECLOSE`, `NYSE_CLOSE`, `BRISBANE_1025` -- excluded by `asset_configs.get_enabled_sessions('MGC')` but LIVE_PORTFOLIO specs reference them without instrument-specific exclusion.
- MES missing: `BRISBANE_1025` (same reason).
- MNQ: complete.

**No rebuild_manifest entry exists for the Mar 19 discovery.** This session was run outside the standard rebuild path. No `validation_run_log` entry exists from Mar 19 either.

**VERDICT: PASS.** Discovery is complete for all instruments (within their enabled sessions).

---

## D. VALIDATION FUNNEL

| Stage | MGC | MES | MNQ |
|---|---|---|---|
| Discovery total | 22,464 | 25,918 | 31,104 |
| Positive ExpR | 7,567 (33.7%) | 2,652 (10.2%) | 5,499 (17.7%) |
| E2 above noise floor (0.32) | 473 | 21 | (not checked separately) |
| Phase 1 pass (sample>=50) | ~3,365 (E2) | limited | - |
| PASSED | **0** | **0** | **11** |
| REJECTED | 22,176 | 25,918 | 25,717 |
| SKIPPED (alias) | 288 | 0 | 5,376 |

**MGC zero survivors explained:**
1. E0 purged (was 40 of 144 backup strategies).
2. WF_START_OVERRIDE = 2022-01-01 restricts walk-forward to post-2022 data.
3. WF_TRADE_COUNT_OVERRIDE = 30 (MGC uses trade-count mode, requiring 75 outcomes post-anchor).
4. Combined effect: ORB_G filters on MGC produce sparse trades (many sessions have <75 filtered outcomes after 2022), killing all strategies at Phase 4b with "Insufficient valid windows: 0 < 3".
5. This is honest filtering, not a bug. MGC's ORB edge depends on high-ATR regimes and the post-2022 WF anchor means not enough data under strict filters.

**MES zero survivors explained:**
1. E0 purged (was 106 of 138 backup strategies).
2. No WF_START_OVERRIDE for MES, but strategies still fail Phase 1 (small samples) and Phase 4b (insufficient WF windows).
3. MES has only 21 E2 strategies above 0.32 noise floor, and those fail Phase 3 (yearly robustness) or Phase 4b.

**MNQ 11 survivors:**
- All E2, all ATR70_VOL or cross-asset ATR filters.
- 3 sessions: CME_PRECLOSE (7), COMEX_SETTLE (3), BRISBANE_1025 (1).
- All promoted at 2026-03-19 10:04 Brisbane time.

**VERDICT: CRITICAL.** MGC depletion is structural (WF window problem for filtered strategies with N<100). MES depletion is dominated by negative expectancy (85.6%).

---

## E. DATA FRESHNESS

| Stage | MGC | MES | MNQ |
|---|---|---|---|
| bars_1m | 2026-03-07 | 2026-03-07 | **2026-03-21** |
| bars_5m | 2026-03-07 | 2026-03-07 | **2026-03-21** |
| daily_features | 2026-03-06 | 2026-03-06 | **2026-03-20** |
| orb_outcomes | 2026-03-06 | 2026-03-06 | **2026-03-20** |
| experimental_strategies | **2026-03-19** | **2026-03-19** | **2026-03-19** |
| validated_setups | None | None | **2026-03-19** |
| edge_families | None | None | **2026-03-21** |
| family_rr_locks | 2026-03-16 | 2026-03-16 | 2026-03-16 |

**MGC and MES bars are 14 calendar days stale relative to MNQ.**

**Rebuild manifest:**
- MGC: Last COMPLETED rebuild 2026-03-18 14:14. Last 4 attempts FAILED at `outcome_builder_O5` (Mar 18 14:24 - 15:57).
- MES: Last completed run through all steps: 2026-03-16 17:54 (failed at health_check only). 2 stuck RUNNING entries from Mar 16.
- MNQ: Only 2 manifest entries ever. Both FAILED (one at health_check Mar 16, one at outcome_builder_O5 Mar 20).

**VERDICT: CRITICAL.** MGC/MES bars 14 days behind MNQ. Multiple failed rebuild attempts.

---

## F. FAMILY INTEGRITY

**edge_families:** 5 MNQ families. All heads resolve to existing validated_setups rows. Member counts match. Created 2026-03-21 09:59.

**family_rr_locks:** 692 rows total (MGC=175, MES=107, MNQ=361, M2K=49). **687 of 692 are orphaned** (point to no matching validated_setups row). Only 5 MNQ locks match current validated_setups.

**strategy_trade_days:** 1,735 strategies tracked. **1,734 are orphaned** (not in current validated_setups). Only 1 current strategy has trade days recorded. The other 10 current validated strategies have 0 trade days.

**VERDICT: CRITICAL.** 99.3% of family_rr_locks and 99.9% of strategy_trade_days are orphaned.

---

## G. SESSION-NAME / CHAIN CONSISTENCY

**CRITICAL MISMATCH CONFIRMED:**
- `regime_strategies` (183,702 rows) uses **numeric session names**: `0030`, `0900`, `1000`, `1100`, `1800`, `2300`
- `regime_validated` (4,185 rows, all MNQ) uses **numeric session names**: `0900`, `1000`
- All other tables (`experimental_strategies`, `validated_setups`, `edge_families`, `family_rr_locks`, `live_config specs`) use **event-based names**: `CME_PRECLOSE`, `COMEX_SETTLE`, etc.

**Consequence:** `rolling_portfolio.load_rolling_validated_strategies()` returns strategies with `orb_label='0900'` (numeric). `live_config.build_live_portfolio()` tries to match against specs with `orb_label='CME_PRECLOSE'` (event-based). **These never match.** The CORE tier rolling lookup always falls through to the baseline path.

The rolling portfolio is **completely disconnected** from the live resolution chain.

**VERDICT: CRITICAL.** Rolling portfolio invisible to live portfolio builder due to session name mismatch.

---

## H. CONFIG CONSISTENCY

**NOISE_EXPR_FLOOR:** Currently `{'E1': 0, 'E2': 0}` -- "TEMPORARILY ZEROED" per comment at config.py:94. The Mar 19 validation ran with these zeroed floors. All 11 MNQ survivors have ExpR > 0.32 and would pass even with E2=0.32 applied.

**WF_TRADE_COUNT_OVERRIDE:** `{'MGC': 30}` -- only MGC uses trade-count mode. MES/MNQ use calendar mode. The trade-count mode combined with WF_START_OVERRIDE=2022-01-01 is the primary killer for MGC strategies.

**Rerunning validation now would NOT produce a materially different funnel** because the noise floor is still zeroed. The only difference would come from updated bars (which are stale for MGC/MES anyway).

**VERDICT: PASS with warning.** Noise floor is zeroed and documented. Would not change current results but creates risk for future rebuilds.

---

## I. PORTFOLIO BUILDER / PAPER TRADER INTEGRITY

The paper trader with `--raw-baseline` flag works correctly because it reads directly from `orb_outcomes` and constructs 1 strategy per session (NO_FILTER), bypassing `validated_setups`, `edge_families`, and `family_rr_locks` entirely.

The standard `build_portfolio()` path reads from `validated_setups` and returns an empty portfolio for MGC/MES (0 strategies).

**"Paper trader works" (via --raw-baseline) is being confused with "inventory is healthy."** The raw baseline path is a research convenience that has no relationship to the validated pipeline.

**VERDICT: CRITICAL.** Paper trader functionality masks inventory depletion.

---

## J. REBUILD GAPS

See Final Truth Sheet below.

---

## TOP 5 INVENTORY KILL-SHOT FINDINGS

1. **MGC and MES have ZERO validated strategies.** The E0 purge + Mar 19 rediscovery eliminated the entire pre-existing inventory (144 MGC, 138 MES strategies). 530 -> 11 active strategies (-98% collapse).

2. **strategy_trade_days is 99.9% orphaned.** 1,734 of 1,735 entries reference strategies that no longer exist. 10 of 11 current validated strategies have 0 trade days recorded.

3. **family_rr_locks is 99.3% orphaned.** 687 of 692 locks point to non-existent validated strategies. MGC (175), MES (107) locks are entirely disconnected.

4. **MNQ inventory concentrated in 3 sessions with a single filter family.** No session diversification from pre-rebuild era (12 sessions) survived. Only CME_PRECLOSE, COMEX_SETTLE, BRISBANE_1025 with ATR70_VOL family.

5. **Pre-Mar-19 backup (_vs_backup) is unusable.** 530 rows but E0-dominant for MNQ (217 E0, purged), mixed E0/E3 for others (also purged/retired).

## TOP 5 LIVE-CHAIN KILL-SHOT FINDINGS

1. **LIVE_PORTFOLIO was designed for an inventory that no longer exists.** 46 specs reference filter types from the old E0/E1 era. Only ATR70_VOL/X_MES_ATR60 survive. Resolution: MGC 0%, MES 0%, MNQ 9.5%.

2. **Rolling portfolio completely disconnected.** `regime_validated` uses numeric session names (`0900`), live specs use event-based names (`CME_PRECLOSE`). Zero matches ever.

3. **MGC discovery missing 3 sessions that LIVE_PORTFOLIO references.** CME_PRECLOSE, NYSE_CLOSE, BRISBANE_1025 excluded by `get_enabled_sessions('MGC')` but live specs reference them.

4. **All 4 loaded MNQ strategies resolve via baseline fallback, not rolling.** Rolling evaluation system contributes nothing to live portfolio.

5. **LIVE_MIN_EXPECTANCY_R (0.22) is a no-op.** All validated survivors already exceed 0.32.

## TOP 5 REBUILD-INTEGRITY KILL-SHOT FINDINGS

1. **MGC/MES bars 14 days stale.** Latest bars_1m: 2026-03-07 for both vs 2026-03-21 for MNQ. Discovery ran on stale data.

2. **Mar 19 discovery+validation ran outside standard rebuild pipeline.** No rebuild_manifest entry, no validation_run_log entry. Telemetry infrastructure bypassed.

3. **MES has 2 stuck RUNNING rebuild entries from Mar 16.** Never cleaned up.

4. **MGC's last 4 rebuild attempts all FAILED at outcome_builder_O5.** Persistent issue since Mar 18.

5. **Noise floor "TEMPORARILY ZEROED" with no restoration date.** Creates risk for future rebuilds promoting noise-floor-inferior strategies.

---

## FINAL TRUTH SHEET

| Instrument | Stage | Status | Row Count | Latest Timestamp | Blocker | Evidence |
|---|---|---|---|---|---|---|
| MGC | ingest | STALE | 3,566,524 | 2026-03-07 | No DBN files since Mar 7 | MAX(ts_utc) FROM bars_1m |
| MGC | bars_5m | STALE | 714,894 | 2026-03-07 | Blocked by stale ingest | MAX(ts_utc) FROM bars_5m |
| MGC | daily_features | STALE | 8,832 | 2026-03-06 | Blocked by stale bars | MAX(trading_day) FROM daily_features |
| MGC | orb_outcomes | STALE | 2,429,460 | 2026-03-06 | Blocked by stale features | MAX(trading_day) FROM orb_outcomes |
| MGC | discovery | COMPLETE | 22,464 | 2026-03-19 | Ran on stale outcomes; missing 3 sessions | experimental_strategies |
| MGC | validation | EMPTY | 0 | N/A | Honest: WF Phase 4b kills all (trade-count mode + 2022 anchor) | validated_setups |
| MGC | edge_families | EMPTY | 0 | N/A | No validated strategies | edge_families |
| MGC | family_rr_locks | STALE | 175 | 2026-03-16 | All orphaned | Orphan check: 175/175 |
| MGC | live_resolution | DISCONNECTED | 0 | N/A | 0 validated, 45 specs unresolvable | build_live_portfolio() |
| MGC | rolling_linkage | DISCONNECTED | 0 | N/A | Numeric session names | regime_validated uses 0900 |
| MES | ingest | STALE | 2,489,284 | 2026-03-07 | No DBN files since Mar 7 | MAX(ts_utc) FROM bars_1m |
| MES | bars_5m | STALE | 498,277 | 2026-03-07 | Blocked by stale ingest | MAX(ts_utc) FROM bars_5m |
| MES | daily_features | STALE | 6,189 | 2026-03-06 | Blocked by stale bars | MAX(trading_day) FROM daily_features |
| MES | orb_outcomes | STALE | 1,947,816 | 2026-03-06 | Blocked by stale features | MAX(trading_day) FROM orb_outcomes |
| MES | discovery | COMPLETE | 25,918 | 2026-03-19 | Ran on stale outcomes | experimental_strategies |
| MES | validation | EMPTY | 0 | N/A | Honest: Phase 1/3/4b kills all | validated_setups |
| MES | edge_families | EMPTY | 0 | N/A | No validated strategies | edge_families |
| MES | family_rr_locks | STALE | 107 | 2026-03-16 | All orphaned | Orphan check: 107/107 |
| MES | live_resolution | DISCONNECTED | 0 | N/A | 0 validated, 42 specs unresolvable | build_live_portfolio() |
| MES | rolling_linkage | DISCONNECTED | 0 | N/A | Numeric session names | regime_validated uses numeric |
| MNQ | ingest | COMPLETE | 1,812,784 | 2026-03-21 | None | MAX(ts_utc) FROM bars_1m |
| MNQ | bars_5m | COMPLETE | 362,568 | 2026-03-21 | None | MAX(ts_utc) FROM bars_5m |
| MNQ | daily_features | COMPLETE | 4,491 | 2026-03-20 | None | MAX(trading_day) FROM daily_features |
| MNQ | orb_outcomes | COMPLETE | 1,552,284 | 2026-03-20 | None | MAX(trading_day) FROM orb_outcomes |
| MNQ | discovery | COMPLETE | 31,104 | 2026-03-19 | None | experimental_strategies |
| MNQ | validation | COMPLETE | 11 | 2026-03-19 | 11 survivors (3 sessions, ATR70_VOL family) | validated_setups |
| MNQ | edge_families | COMPLETE | 5 | 2026-03-21 | 1 ROBUST, 1 WHITELISTED, 2 SINGLETON, 1 PURGED | edge_families |
| MNQ | family_rr_locks | STALE | 361 | 2026-03-16 | 356/361 orphaned | Orphan check |
| MNQ | live_resolution | PARTIAL | 4 | N/A | Only 4 of 42 specs resolve; all via baseline | build_live_portfolio() |
| MNQ | rolling_linkage | DISCONNECTED | 0 | N/A | Numeric session names in regime_validated | load_rolling_validated_strategies() |

---

## FINAL DIAGNOSTIC

**State: REBUILD-INCOMPLETE + LIVE-MAPPING-BROKEN**

The pipeline machinery is statistically honest. The depletion is real and caused by:
1. E0 purge eliminating the dominant entry model from pre-Mar-19 inventory
2. MGC's structural WF limitation (trade-count mode + 2022 anchor + sparse filtered trades)
3. MES's fundamental negative expectancy at E1/E2
4. Live_config specs designed for an inventory that no longer exists
5. Rolling portfolio disconnected by session name migration

The system correctly identifies that very few strategies survive honest testing. But the downstream infrastructure (live specs, family locks, rolling linkage) has not been reconciled with this reality.
