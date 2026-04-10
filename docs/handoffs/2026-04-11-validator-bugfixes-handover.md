# Handover — Validator Bug Fixes + Strategy Recovery + Cross-Validation

**Date:** 2026-04-11
**Branch:** `discovery-wave4-lit-grounded` (pushed to origin)
**Session focus:** Fix 2 validator silent-failure bugs, recover orphaned strategies, run GC→MGC cross-validation
**Parallel terminal:** Wave 4 discovery running concurrently — be careful of DB writes

---

## What shipped

### 1. Validator Phase C DELETE bug (commit `c0f3c2b4`)
**Bug:** Phase C batch write used `DELETE WHERE instrument=? AND orb_minutes IN (?)` — nuked ALL validated strategies for that instrument, not just the ones being re-validated.

**Impact:** MNQ RR1.0 strategies (3) were silently wiped when RR1.5/2.0 strategies were validated for the same instrument.

**Fix:** Scope DELETE to `strategy_id IN (...)` from `serial_results` (only strategies being processed).

**Test coverage:** 2 regression tests in `TestPhaseC_DeleteScope`.

### 2. Validator FDR pool bugs (commit `cb65dcff`)
**Bug A:** FDR pool hardcoded to `ACTIVE_ORB_INSTRUMENTS` at line 1947. Running `--instrument GC` under Pathway A silently skipped FDR because GC was never in the pool.

**Bug B:** Silent pass-through at line 1988 — strategies not in `fdr_results` dict were silently promoted without FDR evaluation.

**Fix A:** Include current run's instrument in the pool if not already in `ACTIVE_ORB_INSTRUMENTS`.

**Fix B:** Fail-closed when strategy has non-NULL `p_value` but no FDR result (indicates pool build error). NULL-p strategies still pass through (legitimately can't be FDR-evaluated — they weren't discovered with a p-value).

**Verification:** End-to-end GC family-mode run produced total K=277 across 12 sessions (previously would have excluded GC). 1 GC strategy promoted via FDR, 1 correctly rejected via FDR.

**Test coverage:** 1 regression test in `TestFdrPoolIncludesCurrentInstrument`.

### 3. GC → MGC cross-validation script (commit `322dcf18`)
**Purpose:** Confirm GC proxy edges transfer to MGC micro data before deployment.

**Script:** `scripts/research/gc_to_mgc_cross_validation.py`

**Result:** **9/10 GC strategies FAIL on MGC micro.** Only 1 borderline CONFIRMED (GC_US_DATA_1000 ATR_P70 → MGC ExpR=0.001, not deployable).

**Implication:** GC proxy is valid for research but NOT for MGC deployment. Price correlation r=0.99999 is necessary but not sufficient — microstructure differences matter at 1-minute ORB level. **MGC proxy deployment path is DEAD.**

**Memory topic:** `gc_mgc_cross_validation_results.md`

### 4. Orphaned strategy recovery
**Problem:** 7 strategies with `validation_status='PASSED'` in `experimental_strategies` but missing from `validated_setups`. Two root causes:
- **DELETE bug (5 strategies):** Wiped by Phase C DELETE when other strategies were validated
- **SHA drift (4 hypothesis files):** Files were timestamp-bumped post-discovery, orphaning the SHA references (drift check #94 was flagging this)

**Fix:** Mapped orphaned SHAs to current files via git history:
- `2ef6e1a61d81` → `gc-proxy-wave2-regime-robust.yaml`
- `d059f87d69c9` → `gc-proxy-stable-sessions.yaml`
- `0afa682a9183` → `gc-proxy-broad-sweep.yaml`
- `d244de974ee8` → `gc-proxy-wave3-expansion.yaml`

Updated `hypothesis_file_sha` in `experimental_strategies` to current file versions, reset `validation_status=NULL`, re-ran validator.

**Result:** Recovered strategies across instruments. Final state (post my work, pre other-terminal wave4 adds):
- MNQ: 8 validated (3 RR1.0 recovered)
- GC: 10 validated (5 recovered, 5 pre-existing)
- MES: 1 validated (CME_PRECLOSE G8 RR1.0 recovered)

### 5. Era stability audit
Verified criterion 9 is working correctly across instruments:
- **MES 2023:** Genuine regime failure (ExpR -0.07 to -0.09, N=196-232). Rate hike compression — not a methodology error.
- **GC 2015-2019:** Pre-gold-bull era. Structural rejection, correct.
- **MGC:** Too few trades (50-73) to trigger criterion 9. Data shortage, not filter error.

**MES CME_PRECLOSE G8 survives 2023 on a technicality:** N=24 in 2023 (below criterion 9's N>=50 threshold → exempt). The G8 filter is so restrictive that few days qualify in the low-vol 2023 regime. The filter IS the strategy — acceptable but worth noting.

**Memory topic:** No new file needed — added note to validator fix commit message.

### 6. family_rr_locks populated
After recovery, `family_rr_locks` was out of sync (drift check #61 failing). Ran `scripts/tools/select_family_rr.py`: 46 strategies across 27 families locked.

### 7. Hypothesis SHA drift checks cleared
All orphan SHAs resolved. Drift check #96 (Phase 4 SHA integrity) now green.

---

## Known issues / golden nuggets

### FDR K counting uses observed, not declared
The validator counts `K` from `experimental_strategies` rows for the session, not from the hypothesis file's `total_expected_trials`. Example: MES comprehensive declared K=8, but only 1 survived discovery to `experimental_strategies`, so observed K=1. BH FDR at K=1 is trivially easy (adj_p = raw_p).

**Current impact:** NONE — the MES strategy has p=0.0009, which passes FDR even at K=8. No current strategy is victimized.

**Future fix:** Store `declared_k` in `experimental_strategies` at discovery time, then validator uses `max(observed_k, declared_k)` as the denominator. Deferred — not blocking any current work.

### Validator runs one testing_mode per invocation, but hypothesis files can mix modes
When recovering the 12 GC orphans, I had to split them manually by hypothesis file's testing_mode:
- 9 from `gc-proxy-broad-sweep.yaml` (family) → ran `--testing-mode family`
- 3 from `gc-proxy-wave3-expansion.yaml` (individual) → ran `--testing-mode individual`

Used a temporary `validation_status='PENDING_INDIVIDUAL'` marker to split the queue.

**Future improvement:** Validator should read `testing_mode` from the hypothesis file per-strategy, not from a CLI arg. Out of scope for this session.

### Parallel terminal work
The `discovery-wave4-lit-grounded` branch has commits from a parallel terminal running wave 4 discovery:
- `51387a6e` 14 wave-4 hypothesis files
- `4928b840` wave 4 results (27 positive BH FDR on real micro)
- `014d352f` MGC proxy hypothesis files use GC instrument
- `6dc34af1`, `91efdc26`, `c8649473`, `6a4f4d2d` drift fixes + plan

**Post my work, the DB state changed:** MNQ 8→27, GC 10→17, MES 1→2. The other terminal added strategies during my session. Coordinate before any further DB writes.

**Unstaged file on branch (NOT mine):** `trading_app/strategy_discovery.py` — likely wave4 in-progress work. Do not commit.

---

## Verification

- **Tests:** 126/126 pass in `test_strategy_validator.py` (2 new Phase C tests + 1 new FDR pool test)
- **Drift:** 95/0/7 — zero failures (was 3: SHA orphans #96, family_rr_locks #61)
- **End-to-end:** GC family-mode validator run with total K=277 across 12 sessions, FDR gate firing correctly

---

## Next steps (for next session)

1. **Merge this branch to main** after wave 4 discovery work is complete. The validator fixes should NOT wait for wave 4 — they're independent bug fixes. Consider cherry-picking `c0f3c2b4` and `cb65dcff` to main directly if wave 4 needs more time.

2. **Resolve the unstaged `strategy_discovery.py`** — coordinate with parallel terminal to commit or discard.

3. **Consider fixing declared-K FDR** as a standalone task. Low priority since no current strategy is affected, but a correctness improvement.

4. **MGC diversification path:** Since GC proxy does NOT transfer to MGC, the only path to MGC diversification is discovery on native MGC micro data (3.8 years). This is a hard problem. Options:
   - Accept MNQ-dominated portfolio until more MGC data accumulates
   - Relax criterion 9 / sample size for MGC specifically (requires criteria amendment)
   - Wait for more MGC history

5. **MES portfolio value:** r=0.626 to MNQ. 1 strategy recovered (CME_PRECLOSE G8). Don't overweight — it's lightly diversifying but not a game-changer.

---

## Files changed this session

- `trading_app/strategy_validator.py` — 2 bug fixes
- `tests/test_trading_app/test_strategy_validator.py` — 3 new regression tests
- `scripts/research/gc_to_mgc_cross_validation.py` — new research script
- `docs/runtime/stages/validator-delete-scope.md` — created, deleted
- `docs/runtime/stages/validator-fdr-pool-fix.md` — created, deleted
- `C:\Users\joshd\.claude\projects\C--users-joshd-canompx3\memory\gc_mgc_cross_validation_results.md` — new memory topic
- `C:\Users\joshd\.claude\projects\C--users-joshd-canompx3\memory\MEMORY.md` — updated index
- `C:\Users\joshd\.claude\plans\dapper-crunching-plum.md` — plan file (approved)
