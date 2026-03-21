# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

---

## Current Session
- **Tool:** Claude Code (ML Audit + Fix Planning Terminal)
- **Date:** 2026-03-21 (night)
- **Branch:** `main`
- **Status:** ML audit complete. Fix plan designed. Implementation NOT started (paused for pipeline rebuild in other terminal).

### What was done this session

#### 1. Bootstrap 5K Results Updated (memory + docs)
- 3/7 PASS, 3 MARGINAL, 1 FAIL (from `logs/ml_bootstrap_5k_overnight.log`)
- Memory files corrected to match actual log output

#### 2. Zero-Context ML Audit (28 questions, 5 kill shots)
- `docs/plans/2026-03-21-ml-zero-context-audit.md`
- Key findings verified against code/logs:
  - BH FDR at family=7 → only 1 survivor (NYSE_OPEN O30 p=0.0016). At family=68 → 0 survivors.
  - 10/12 sessions at O30 RR2.0 have Sharpe=nan (negative baselines)
  - EUROPE_FLOW has winter lookahead from LONDON_METALS (~42% of rows)
  - No FDR code exists in bootstrap script
  - Constant-column drop uses full data, not train-only

#### 3. ML Fix Execution Plan (6 bugs, 7 phases)
- `docs/plans/2026-03-21-ml-fix-execution-plan.md`
- `docs/plans/2026-03-21-ml-methodology-fix-design.md` (V2)
- Execution order: B→A→C→E→F→D (lookahead first, then determinism, then methodology)
- Implementation NOT started — only the V2 version gate (`ML_METHODOLOGY_VERSION=2`) is deployed

#### 4. Upstream Discovery: Live Portfolio is EMPTY
- `build_live_portfolio()` returns 0 strategies, 47 warnings
- 42 specs say "no variant found" — live_config references filters not in validated_setups
- Raw baseline (`build_raw_baseline_portfolio()`) works fine — 11 strategies
- MGC/MES data stale (Mar 6). MNQ current (Mar 20).

#### 5. Version Gate Deployed
- Commit `9853817`: `ML_METHODOLOGY_VERSION=2` in config.py
- Old V1 model rejected at inference → fail-open → Layer 1 raw baseline runs clean

### Truth State (verified from code/logs/DB, not memory)
- **Raw baseline = tradeable.** 11 MNQ strategies, works with `--raw-baseline`.
- **Filtered portfolio = BROKEN.** 0 strategies. live_config→validated_setups mismatch.
- **ML = FROZEN.** V2 gate rejects old models. 6 bugs identified, 0 fixed in code yet.
- **MGC/MES = STALE** (Mar 6). Other terminal doing rebuild.
- **MNQ = CURRENT** (Mar 20).

### Pipeline Status
- bars_1m: 15M rows
- daily_features: 34K rows
- orb_outcomes: 8.4M rows
- validated_setups: 11 rows (all MNQ E2)
- edge_families: 5 rows (1 ROBUST, 1 WHITELISTED, 2 SINGLETON, 1 PURGED)

### Tasks Pending (in task system)
1. Phase 1: Fix B — EF/LM DST lookahead (drop cross-session + level features)
2. Phase 2: Fix A — deterministic config tiebreaker (4 ORDER BY clauses)
3. Phase 3: Fix C — train-only constant-column drop
4. Phase 3.1: Fix E — positive baseline gate
5. Phase 4.1: Fix F — feature reduction to 5
6. Phase 5.1: Fix D — BH/FDR in bootstrap
7. Phase 6: Retrain + bootstrap 5K + BH FDR
8. Phase 7: Update docs with real numbers

### Next Steps (for incoming session)
1. **Wait for pipeline rebuild** (other terminal) — MGC/MES need fresh data
2. **Audit live_config** — 42 "no variant found" specs need investigation. Either rebuild validated_setups or strip live_config to match reality.
3. **Then implement ML fixes** — 6 bugs, phased, one at a time
4. **Then retrain on full pre-registered universe** — accept 0 survivors if honest result

---

## Prior Session
- **Tool:** Claude Code
- **Date:** 2026-03-21 (earlier)
- **Summary:** Multi-RR portfolio built. ML audit found 4 FAILs. Bootstrap 5K code committed. Confluence design started. Session crashed mid-brainstorm.
