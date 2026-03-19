# HANDOFF.md — Cross-Tool Session Baton

The outgoing tool updates this before the user switches. The incoming tool reads it first.

**Rule:** If you made decisions, changed files, or left work half-done — update this file.
If nothing changed, leave it as-is.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code. If another tool made commits since your last read, your mental model is wrong.

---

## Current Session
- **Tool:** Claude Code
- **Date:** 2026-03-19
- **Branch:** `pipeline-fixes`
- **Status:** Post-rebuild audit complete. MNQ null test running overnight (~100 seeds, sigma=5.0).

### What was done this session

#### 1. Full Post-Rebuild Audit
Pipeline rebuilt today (MGC/MES/MNQ × O5/O15/O30). Results:
- **MGC: 0 validated** (was 573). 64% negative ExpR, remainder killed by noise floor + WF + yearly robustness.
- **MES: 0 validated** (was 261). 86% negative ExpR.
- **MNQ: 11 validated** (was 1,443). All E2, all ATR70_VOL/VOL filtered, CME afternoon sessions.
- **M2K: 95 stale** (not rebuilt, from Mar 12).

#### 2. Root Cause Analysis — MGC Wipeout
- **Structural impossibility:** Noise floor requires tight G-filters (G4+/G5+) for high ExpR. Tight filters produce only 100-130 trades over 10 years. WF needs 135+ trades minimum (45 train + 30 test × 3 windows). Noise floor + WF = mathematically incompatible for MGC.
- NOT a bug — the gates are working correctly. The edge doesn't survive honest gating.

#### 3. Noise Floor Calibration Audit (Literature-Grounded)
- **NOISE_EXPR_FLOOR zeroed in config.py** for MNQ null test calibration. MUST restore after.
- **MGC null (sigma=1.2 vs actual 0.70):** 1.71x too volatile = CONSERVATIVE (safe).
- **MNQ null (sigma=5.0 vs actual 5.95, ATR70 days 6.6-8.3):** 0.84x too quiet = LENIENT (dangerous).
- **Literature:** Aronson/Masters MCP (permutation of actual returns) is gold standard — automatically preserves all distributional properties. Our parametric Gaussian approach requires correct sigma. Bailey & López de Prado FST: V[SR] must match actual data for threshold to be valid.
- **Memory file:** `null_test_methodology.md` has full comparison with citations.

#### 4. Code Audit Findings (from 3 exploration agents)
- **Noise floor gate:** HOT tier in live_config has NO noise floor gate (violates fail-closed)
- **rolling_portfolio.MIN_EXPECTANCY_R = 0.10:** Below both noise floors, allows sub-floor families
- **Ambiguous bars:** Hardcoded to loss — conservative, ~0.2% of trades
- **ATR_20_PCT:** bisect_left percentile off by ~1-2% (minor)
- **MAE/MFE friction:** Inconsistent with actual P&L friction treatment
- **Drift check:** `python pipeline/check_drift.py` broken (ModuleNotFoundError). Use `python -m pipeline.check_drift`.
- **15 drift violations** detected including zeroed noise floors.

#### 5. Academic Literature Review
Read actual PDFs from `resources/`:
- `deflated-sharpe.pdf` (Bailey & López de Prado 2014)
- `false-strategy-lopez.pdf` (Bailey & López de Prado 2018)
- `man_overfitting_2015.pdf` (Man AHL Advisory Board 2015)
- `Evidence_Based_Technical_Analysis_Aronson.pdf` Ch.5 pp.238-258 (Monte Carlo Permutation)

### MNQ Null Test Status
- Location: `scripts/tests/null_seeds/mnq/`
- Config: `instrument=MNQ, sigma=5.0, noise_floors={E1:0, E2:0}`
- **Sigma=5.0 is LENIENT** — actual MNQ sigma=5.95 (overall), 6.6-8.3 (ATR70 days the strategies trade on)
- Results establish a LOWER BOUND on noise ceiling only
- ETA: ~6hrs from start (~early morning AEST Mar 20)

### Decision Tree (blocked on null test)
- **If MNQ E2 ceiling > 0.37 at sigma=5.0:** All 11 dead (even easy null kills them) — DEFINITIVE
- **If MNQ E2 ceiling ≤ 0.32 at sigma=5.0:** Inconclusive — need sigma=5.95 and 7.5 runs
- **If MNQ E2 ceiling 0.32-0.37 at sigma=5.0:** Some might survive easy null but die at honest null — INCONCLUSIVE

### Next Steps (for incoming session)
1. **Check null test results** — read `scripts/tests/null_seeds/mnq/` output
2. **Restore noise floors** in config.py (E1=0.22, E2=0.32) if null test is done
3. **If inconclusive:** queue sigma=5.95 and sigma=7.5 runs
4. **Fix drift check** ModuleNotFoundError (path resolution in check_drift.py)
5. **Fix HOT tier** noise floor gap in live_config.py
6. **Rebuild M2K** with current gates to see what survives

### Files modified (uncommitted from prior Codex session)
- `scripts/infra/windows_agent_launch.py`
- `scripts/tools/worktree_manager.py`
- `tests/test_tools/test_worktree_manager.py`
- `tests/test_tools/test_windows_agent_launch.py`

### Files created this session
- `GOLD_DB_AUDIT_2026-03-19.txt` — snapshot (can delete)

---

## Prior Session
- **Tool:** Claude Code (earlier today)
- **Date:** 2026-03-19 (morning)
- **Summary:** 100-seed MGC null test completed. Multi-aperture rebuild. Adversarial review framework. Zero-context audit. MNQ null test started.

## Session Before That
- **Tool:** Codex
- **Date:** 2026-03-18
- **Summary:** Workstream lifecycle + microstructure pilot (uncommitted on this branch)
