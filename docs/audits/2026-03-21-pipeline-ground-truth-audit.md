# GROUND-TRUTH PIPELINE & INVENTORY AUDIT REPORT

**Date:** 2026-03-21
**Database:** `C:\Users\joshd\canompx3\gold.db`
**Active Instruments:** MES, MGC, MNQ

---

## A. INVENTORY TRUTH

**QUESTION:** What is the exact state of validated_setups, by instrument?

**EVIDENCE:**

| Instrument | validated_setups rows | Status | Sessions | Filters | Entry Models |
|---|---|---|---|---|---|
| **MNQ** | 11 | all `active` | BRISBANE_1025 (1), CME_PRECLOSE (6), COMEX_SETTLE (3) | ATR70_VOL (9), VOL_RV12_N20 (1), X_MES_ATR60 (1) | E2 only |
| **MGC** | 0 | -- | -- | -- | -- |
| **MES** | 0 | -- | -- | -- | -- |
| **M2K** | 0 | (not active instrument) | -- | -- | -- |

- `edge_families`: 5 rows, all MNQ. 1 ROBUST, 1 WHITELISTED, 2 SINGLETON, 1 PURGED.
- `_vs_backup` (pre-Mar-19 snapshot): M2K=31, MES=138, MGC=144, MNQ=217 (all E0 for MNQ, mixed E0/E1/E3 for others).
- The _vs_backup is a relic from before E0 purge (promoted_at = 2026-02-25). It is unusable.

**VERDICT: CRITICAL.** MGC and MES have zero validated strategies. The live portfolio for these instruments is empty. M2K has 98 PASSED in experimental_strategies but is no longer an active instrument and has zero rows in validated_setups.

---

## B. LIVE SPEC RESOLUTION

**QUESTION:** Do live_config specs resolve to real validated rows?

**EVIDENCE:** (from `build_live_portfolio()` execution)

| Instrument | Specs Tested | Resolved | Not Resolved | Excluded |
|---|---|---|---|---|
| MNQ | 46 | 4 | 38 | 4 |
| MGC | 46 | 0 | 45 | 1 |
| MES | 46 | 0 | 42 | 4 |

MNQ resolves only 4 of 46 specs. The 4 resolved specs are all from the new vol-regime filters: `CME_PRECLOSE_E2_VOL_RV12_N20`, `CME_PRECLOSE_E2_ATR70_VOL`, `CME_PRECLOSE_E2_X_MES_ATR60`, `COMEX_SETTLE_E2_ATR70_VOL`.

The remaining 38 MNQ specs (including all ORB_G4/G5/G6/G8 specs, all DIR_LONG, all TOKYO_OPEN, all SINGAPORE_OPEN, all NYSE_OPEN, etc.) return "no variant found."

**Root cause:** The Mar 19 discovery+validation replaced the entire validated_setups table. The old ORB_G* strategies that live_config specs reference no longer exist in validated_setups. The 46 live_config specs were designed for a 530-row validated inventory (from _vs_backup era) but now point at an 11-row inventory.

**VERDICT: CRITICAL.** 91% of live specs are broken. The live portfolio is functionally empty for 2 of 3 instruments and 92% empty for MNQ.

---

## C. DISCOVERY COVERAGE

**QUESTION:** Did discovery run fully for all instruments?

**EVIDENCE:**

| Instrument | Canonical Rows | Total Rows | Sessions | Created At | Filters |
|---|---|---|---|---|---|
| MNQ | 25,728 | 31,104 | 12 | 2026-03-19 | 48+ filter types |
| MGC | 22,176 | 22,464 | 9 | 2026-03-19 | 48+ filter types |
| MES | 25,918 | 25,918 | 11 | 2026-03-19 | All canonical |
| M2K | 30,096 | 30,600 | 13 | 2026-03-12 | Stale, not refreshed |

All three active instruments had full discovery on Mar 19. Coverage is broad (9-12 sessions, 48+ filter types including new ATR70_VOL/X_MES_ATR*/DIR_LONG filters).

**VERDICT: PASS.** Discovery is complete and current for active instruments.

---

## D. VALIDATION DEPLETION AUDIT

**QUESTION:** What is the exact funnel from discovery to validated?

**Evidence (all instruments, canonical rows only, from validation_notes):**

### MGC (22,176 canonical)
| Phase | Rejected | Cumulative Rejected | Description |
|---|---|---|---|
| Phase 1 (sample size) | 2,484 | 2,484 | N < 30 |
| Phase 2 (ExpR <= 0) | 14,136 | 16,620 | No positive expectancy |
| Phase 2b (noise floor) | 4,663 | 21,283 | ExpR <= E1:0.22/E2:0.32 |
| Phase 3 (yearly robustness) | 439 | 21,722 | <75% years positive |
| Phase 4b (walkforward) | 452 | 22,174 | Insufficient WF windows (<3) |
| Phase FDR (BH, global K) | 2 | 22,176 | Adjusted p >= 0.05 |
| **PASSED** | **0** | | |

### MES (25,918 canonical)
| Phase | Rejected | Cumulative | Description |
|---|---|---|---|
| Phase 1 | 1,277 | 1,277 | |
| Phase 2 | 22,183 | 23,460 | 85.6% killed at ExpR |
| Phase 2b | 2,431 | 25,891 | |
| Phase 3 | 12 | 25,903 | |
| Phase 4b | 15 | 25,918 | |
| **PASSED** | **0** | | |

### MNQ (25,728 canonical)
| Phase | Rejected | Cumulative | Description |
|---|---|---|---|
| Phase 1 | 1,368 | 1,368 | |
| Phase 2 | 19,868 | 21,236 | |
| Phase 2b | 4,475 | 25,711 | |
| Phase 3 | 0 | 25,711 | |
| Phase 4b | 6 | 25,717 | |
| Phase FDR | 0 | 25,717 | |
| **PASSED** | **11** | | |

**Key finding:** Phase 2b (noise floor gate E1=0.22, E2=0.32) kills 4,663 MGC and 4,475 MNQ strategies. But `config.py` currently has noise floors **ZEROED** (E1=0, E2=0). The validation ran at ~10:04am Mar 19 with active noise floors (0.22/0.32); the floors were zeroed at 10:44am Mar 19. If validation re-ran with current zeroed config, the funnel would widen significantly.

MGC's terminal problem is Phase 4b: 452 strategies pass P1-P3 but fail walkforward ("Insufficient valid windows: 0 < 3"). These are filtered strategies with small N (50-89) that cannot generate 3 valid WF windows.

**VERDICT: CRITICAL.** MGC depletion is a structural WF window problem for filtered strategies with N<100. MES depletion is dominated by negative expectancy (85.6%). The noise floor gate is inconsistent -- code says 0 but the DB was validated with 0.22/0.32.

---

## E. FILTER DEPENDENCE

**QUESTION:** Do MGC/MES survive only with specific filters?

**EVIDENCE:** MGC 452 strategies that pass P1-P3 but fail P4b are predominantly:
- CME_REOPEN session with ORB_G5/G6/G8/FAST10/CONT filters, E2, N=50-89
- TOKYO_OPEN session with ORB_G4/G5 FAST variants

These are REGIME-tier strategies (N<100) that cannot generate 3 WF windows. The walkforward minimum window requirement is the bottleneck, not the filter itself.

MES has only 27 strategies surviving to Phase 3+, all failing at P3 or P4b.

**VERDICT: CRITICAL.** MGC strategies exist (452 pass P1-P3) but the WF gate kills all of them. This is a structural limitation of MGC's REGIME-tier sample sizes combined with strict WF window requirements.

---

## F. DATA FRESHNESS

**EVIDENCE:**

| Stage | MGC | MES | MNQ |
|---|---|---|---|
| bars_1m | 2026-03-07 | 2026-03-07 | **2026-03-21** |
| bars_5m | 2026-03-07 | 2026-03-07 | **2026-03-21** |
| daily_features | 2026-03-06 | 2026-03-06 | **2026-03-20** |
| orb_outcomes | 2026-03-06 | 2026-03-06 | **2026-03-20** |
| experimental_strategies | **2026-03-19** | **2026-03-19** | **2026-03-19** |
| validated_setups | None | None | **2026-03-19** |
| edge_families | None | None | **2026-03-21** |

**VERDICT: CRITICAL.** MGC and MES bar data is 14 days stale (Mar 7). Only MNQ has current data. MGC/MES discovery ran on Mar 19 data ending Mar 6, so discovery is aligned with available bar data, but that data is 2 weeks old.

---

## G. FAMILY INTEGRITY

**EVIDENCE:**
- 5 edge_families, all MNQ.
- All 5 family heads exist in validated_setups.
- 2 SINGLETON families, 1 ROBUST (5 members), 1 WHITELISTED (3 members), 1 PURGED (1 member).
- 692 family_rr_locks across all instruments (M2K=49, MES=107, MGC=175, MNQ=361).
- family_rr_locks are from Mar 16 (stale -- predate the Mar 19 rebuild).
- The PURGED family (`MNQ_15m_0c33ad3aa6f646e1c35965e57af298ab`, X_MES_ATR60) is still in validated_setups and edge_families.

**VERDICT: PASS with warnings.** Family integrity is internally consistent for the 5 MNQ families. But family_rr_locks are stale (Mar 16) while validated_setups are from Mar 19. The locks still contain entries for M2K (49), MES (107), MGC (175), and MNQ (361) but validated_setups only has 11 MNQ rows.

---

## H. PORTFOLIO BUILDER INTEGRITY

**QUESTION:** Do live_config, portfolio builder, and paper_trader use the same universe?

**EVIDENCE:**
1. `live_config.py` has 46 specs (44 core, 2 regime).
2. `build_live_portfolio()` queries `validated_setups` with `LOWER(status) = 'active'` and joins `family_rr_locks`.
3. Rolling portfolio (`regime_strategies`, `regime_validated`) uses **OLD numeric session names** (0900, 1000, etc.) while validated_setups and live_config use **event-based names** (CME_PRECLOSE, COMEX_SETTLE). The rolling portfolio returns 13 strategies for MNQ but **none match** any live_config spec due to session name mismatch.
4. Paper trader has `--raw-baseline` mode that bypasses validated_setups entirely, using `orb_outcomes` directly.

**VERDICT: CRITICAL.** The rolling portfolio is completely disconnected from the live pipeline due to session name migration (numeric -> event-based). All rolling resolution silently fails, and the portfolio builder falls through to direct `validated_setups` lookup, which returns almost nothing. The `--raw-baseline` mode is the only functional paper trading path.

---

## I. DISCOVERY / VALIDATION STATISTICAL HONESTY

**EVIDENCE:**

1. **Total experimental_strategies:** 116,224 rows across 7 instruments.
2. **Canonical rows:** 108,106 (is_canonical=True).
3. **Canonical with p_value:** 105,612 (the "Global K" for BH FDR).
4. **FDR gate uses Global K across ALL instruments:** Confirmed at line 1116-1122 of `strategy_validator.py`. Query: `SELECT strategy_id, p_value FROM experimental_strategies WHERE is_canonical = TRUE AND p_value IS NOT NULL`.
5. **BH alpha = 0.05.**
6. **FDR rejection note records K=115,646** (from the Mar 16 run), but current Global K is 105,612. The K shifted because Mar 19 discovery replaced strategies with different p_value availability.
7. **is_canonical meaning:** Marks the "head" strategy within each trade-day-hash group. Non-canonical strategies are aliases that produce the same trade days. Only canonical strategies are counted for multiple-testing correction. Implemented at `_mark_canonical()` in `strategy_discovery.py` line 196-212.
8. **n_trials_at_discovery:** Set to `total_combos` per instrument (all RR/CB/filter combos for that instrument's discovery run). For MNQ: 6,048 combos.
9. **No confusion between N (trade count) and K (multiple-testing family size):** K is the global canonical count used for BH FDR; N is per-strategy sample_size. These are distinct columns and used correctly.

**VERDICT: PASS.** Statistical machinery is honestly implemented. BH FDR uses global K across all instruments (conservative). Canonical marking correctly deduplicates by trade-day hash. The K-value discrepancy (115,646 vs 105,612) is minor and would make current correction slightly more conservative.

---

## J. ML FAMILY-SIZE HONESTY

**EVIDENCE:**

1. **ML sweep:** 36 configs tested (12 sessions x 3 apertures) at RR2.0 per-aperture, plus 3 flat-mode runs (RR2.0, RR1.5, RR1.0) = **at least 39 independent tests.**
2. **5 models survived** sweep quality gates (AUC >= 0.52, CPCV >= 0.50, OOS positive).
3. **Bootstrap at 200 permutations:** 5/7 configs passed (p=0.005, which is the 1/200 resolution floor). **This is a resolution artifact, not true significance.**
4. **Bootstrap at 5000 permutations (5K overnight run):**
   - NYSE_OPEN O30: p=0.0016 **PASS**
   - US_DATA_1000 O30: p=0.0176 **PASS**
   - US_DATA_1000 O15: p=0.0930 (marginal)
   - US_DATA_830 O30: p=0.0512 (marginal)
   - NYSE_OPEN flat: p=0.0546 (marginal)
   - CME_PRECLOSE flat: p=0.1190 **FAIL**
   - CME_PRECLOSE RR1.5: (not shown, but crash on remaining tests)
5. **No family-wide BH FDR correction applied across 39 ML tests.** Each p-value is interpreted per-config, not corrected for the family of 39 tests.
6. **Selection uplift warning in sweep log:** "Selection uplift (+49.6R) is >50% of full delta" -- the code itself flagged this as cherry-picked (White 2000).
7. **ML methodology audit (Mar 21):** 4 FAILs documented -- sample size (EPV=2.4, need 10+), negative baselines, bootstrap resolution, selection bias.

**If BH FDR were applied across K=39 ML tests at alpha=0.05:**
- The strongest survivor (NYSE_OPEN O30, p=0.0016) would need adjusted p < 0.05.
- With BH correction: rank 1 of 39, threshold = 0.05/39 = 0.00128. p=0.0016 > 0.00128. **FAILS.**
- Even the rank 2 (US_DATA_1000 O30, p=0.0176) fails trivially.

**VERDICT: CRITICAL -- STATISTICAL HONESTY FAILURE.** No ML survivor survives BH FDR correction at the correct family size (K=39). All bootstrap p-values were interpreted per-config without family correction. The claim "O30 RR2.0 ML works" is uncorrected for multiple testing across the 39-config sweep.

---

## K. REBUILD GAPS

| Instrument | Stage | Status | Evidence |
|---|---|---|---|
| **MNQ** | ingest through edge_families | **COMPLETE** | All stages current (Mar 19-21) |
| **MGC** | ingest | STALE (Mar 7) | bars_1m latest = 2026-03-07 |
| **MGC** | bars/features/outcomes | STALE (Mar 6) | daily_features latest = 2026-03-06 |
| **MGC** | discovery | COMPLETE | Mar 19 |
| **MGC** | validation | COMPLETE but 0 PASSED | Noise floor + WF window kills all |
| **MGC** | families | EMPTY | No validated strategies to cluster |
| **MGC** | live_resolution | BROKEN | 0 rows in validated_setups |
| **MES** | ingest | STALE (Mar 7) | Same as MGC |
| **MES** | bars/features/outcomes | STALE (Mar 6) | Same as MGC |
| **MES** | discovery | COMPLETE | Mar 19 |
| **MES** | validation | COMPLETE but 0 PASSED | Phase 2 kills 85.6% |
| **MES** | families | EMPTY | No validated strategies |
| **MES** | live_resolution | BROKEN | 0 rows in validated_setups |
| **M2K** | all stages | STALE (Mar 12) | Not in active instruments |

**VERDICT: CRITICAL.** MGC and MES have complete discovery but zero validation survivors. The rebuild chain is technically complete but produces an empty result. The noise floor config inconsistency (zeroed in code but used non-zero in the validation that produced current DB state) is a rebuild integrity issue.

---

## DIAGNOSTIC TABLE

| Dimension | MNQ | MGC | MES |
|---|---|---|---|
| bars_1m current | YES (Mar 21) | NO (Mar 7) | NO (Mar 7) |
| discovery complete | YES | YES | YES |
| validated_setups | 11 rows | 0 rows | 0 rows |
| edge_families | 5 | 0 | 0 |
| live specs resolving | 4/46 (9%) | 0/46 (0%) | 0/46 (0%) |
| rolling portfolio match | 0/13 (session name mismatch) | N/A | N/A |
| paper trader functional | YES (--raw-baseline) | YES (--raw-baseline) | YES (--raw-baseline) |

---

## TOP 5 INVENTORY KILL-SHOT FINDINGS

1. **MGC and MES have ZERO validated strategies.** The Mar 19 rebuild wiped the previous 573 MGC / 261 MES validated strategies and replaced them with zero. Before Mar 19, MGC had 573 validated strategies (confirmed by pipeline_audit_log rows_before=573 on Mar 18).

2. **91% of live_config specs are broken.** 42 of 46 specs return "no variant found" for MNQ. 100% of specs fail for MGC and MES. The 46-spec live_config was designed for a ~530-row inventory that no longer exists.

3. **The rolling portfolio is completely disconnected** from the live pipeline. `regime_strategies` and `regime_validated` use old numeric session names (0900, 1000) while the rest of the pipeline uses event-based names (CME_PRECLOSE, COMEX_SETTLE). Zero rolling strategies match any live_config spec.

4. **M2K has 98 PASSED strategies in experimental_strategies but 0 in validated_setups** and is not in active instruments. These strategies are orphaned.

5. **The _vs_backup (530 rows) is entirely E0 for MNQ** and mixed E0/E1/E3 for others. Since E0 is purged, this backup is unusable as a recovery source.

## TOP 5 STATISTICAL HONESTY KILL-SHOT FINDINGS

1. **ML bootstrap p-values are not family-corrected.** The sweep tested K=39 configs; 5 survived; none survive BH FDR at K=39, alpha=0.05. The best survivor (NYSE_OPEN O30, p=0.0016) fails at BH threshold 0.00128.

2. **Noise floor config is zeroed but DB was validated with non-zero floors.** `NOISE_EXPR_FLOOR` in `config.py` is `{"E1": 0, "E2": 0}` but the validation_notes show floors of 0.22/0.32 were active during the Mar 19 validation. If validation re-runs, Phase 2b will pass everything, dramatically changing the funnel.

3. **The BH FDR global K has drifted.** The FDR rejection notes record K=115,646 (from Mar 16) but current canonical count is 105,612. The FDR correction applied to the 11 MNQ survivors was computed against a K that no longer exists. This is conservative (larger K = stricter correction), so it does not invalidate survivors, but it is an integrity issue.

4. **Selection uplift warning in ML sweep:** The code itself flagged "+49.6R selection uplift is >50% of full delta" (White 2000 cherry-picking). The "honest_delta" reported for the ML bundle is cherry-picked across 36 per-aperture configs.

5. **Bootstrap p=0.005 at 200 permutations is a resolution floor artifact.** At 200 permutations, p=0.005 = 1/200, meaning the real delta was the maximum observed. The 5000-perm re-run showed only 2 of 7 configs survive at p<0.05, and even those fail family correction.

## TOP 5 REBUILD-CHAIN KILL-SHOT FINDINGS

1. **MGC/MES bar data is 14 days behind** (latest Mar 7). Ingestion has not run since Mar 7 for these instruments. MNQ is current (Mar 21).

2. **MGC's 452 strategies that pass P1-P3 all fail Phase 4b** (walkforward) with "Insufficient valid windows: 0 < 3". This is a structural limitation: filtered MGC strategies have N=50-89 (REGIME tier) which cannot generate 3 WF windows. This was not the case before Mar 19 when MGC had 573 validated strategies.

3. **The Mar 19 discovery refresh replaced ALL experimental_strategies** for MGC/MES/MNQ. The new discovery included ATR70_VOL/X_MES_ATR*/X_MGC_ATR* filters (vol-regime research). But validation was only logged in `validation_run_log` for runs up to Mar 16. The Mar 19 validation wrote to the DB (evidenced by validation_notes) but did NOT log a new run in `validation_run_log`.

4. **family_rr_locks are stale** (last updated Mar 16, 19:51). The validated_setups they reference were replaced on Mar 19. The locks still contain entries for M2K (49), MES (107), MGC (175), and MNQ (361) but validated_setups only has 11 MNQ rows.

5. **rebuild_manifest shows repeated MGC FAILED runs** on Mar 18 and Mar 20 (outcome_builder_O5 failures). The last COMPLETED MGC rebuild was on Mar 18 at 14:14. No COMPLETED rebuild exists for MNQ after the Mar 19 discovery refresh -- the only MNQ rebuild attempt on Mar 20 FAILED at outcome_builder_O5.

---

## FINAL DIAGNOSTIC

This project is in a **REBUILD-INCOMPLETE + LIVE-MAPPING-BROKEN** state, not a "statistically contaminated" state. The specific combination:

1. **Statistically honest** -- BH FDR, WF gates, noise floors, and validation phases are correctly implemented. The machinery is sound.

2. **Inventory-depleted by structural walkforward limitation** -- MGC and MES have genuine positive-expectancy strategies (MGC: 452 pass P1-P3) but the WF window minimum (3 windows) is impossible to satisfy for REGIME-tier sample sizes.

3. **Live-mapping catastrophically broken** -- 46 live_config specs target an inventory that was wiped on Mar 19. The rolling portfolio is disconnected by session name migration. The result is a functionally empty live portfolio.

4. **Noise floor config inconsistency creates rebuild instability** -- The next validation run will use zeroed noise floors (from current config.py), producing a radically different funnel than the current DB state.

**Immediate blockers before any ML work:**
- Re-run validation for MGC/MES/MNQ with current (zeroed) noise floors to establish honest baseline.
- Fix or rebuild rolling portfolio with event-based session names.
- Align live_config specs with actual validated_setups inventory.
- Apply BH FDR across the 39-config ML sweep before claiming any ML survivor.
