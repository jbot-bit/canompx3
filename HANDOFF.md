# HANDOFF.md — Cross-Tool Session Baton

The outgoing tool updates this before the user switches. The incoming tool reads it first.

**Rule:** If you made decisions, changed files, or left work half-done — update this file.
If nothing changed, leave it as-is.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code. If another tool made commits since your last read, your mental model is wrong.

---

## Current Session
- **Tool:** Claude Code (Paper Trading Terminal)
- **Date:** 2026-03-21
- **Branch:** `main`
- **Status:** Major session — paper trading built, ML verified, blueprint + skill system upgraded.

### What was done this session (Mar 20-21)

#### 1. Raw Baseline Paper Trading (BUILT + VERIFIED)
- `build_raw_baseline_portfolio()` in `trading_app/portfolio.py`
- `--raw-baseline` CLI on paper_trader.py and run_live_session.py
- 2025 replay: 2,606 trades, 59.4% WR, +0.105R/trade, +272.56R total
- Lookahead audit: CLEAN across all 6 execution paths

#### 2. ML Exhaustive Sweep (COMPLETE)
- Tested RR2.0, RR1.5, RR1.0 flat + RR2.0 per-aperture
- Results: RR2.0 per-aperture has 5/36 models passing 4-gate quality check
- NYSE_OPEN O30 AUC=0.658, US_DATA_1000 O30 AUC=0.612
- Logs: `logs/ml_sweep_*.log`

#### 3. Bootstrap Permutation Test (5/7 PASS)
- 200 permutations per survivor
- NYSE_OPEN O30: p=0.005, US_DATA_1000 O30: p=0.005
- US_DATA_1000 O15: p=0.020, US_DATA_830 O30: p=0.020
- NYSE_OPEN flat: p=0.005
- CME_PRECLOSE: p=0.095 (marginal), p=0.145 (fail)
- Logs: `logs/ml_bootstrap_results.log`, `logs/ml_bootstrap_remaining.log`

#### 4. ML Replay Validation (COMPLETE)
- RR2.0 O30: ML adds +12.20R, reduces DD by 12.46R vs raw baseline
- Model on disk: `models/ml/meta_label_MNQ_hybrid.joblib` (per-aperture, RR2.0)

#### 5. Strategy Research Blueprint (NEW DOC)
- `docs/STRATEGY_BLUEPRINT.md` — 11 sections, methodology + current state
- 3 audit passes, verified against live code/data
- Referenced from CLAUDE.md document authority table

#### 6. Skill System Upgrade (15 touchpoints)
- Shared rule: `strategy-awareness.md` (blueprint routing for all skills)
- Upgraded: brainstorm, 4t, 4tp, orient, discover, regime-check, trade-book, bloomey-review, verify-done, code-review (11 skills)
- New: research, ml-verify (2 skills)
- Multi-take deliberation baked into planning skills
- Next→ routing for skill chaining
- Effort frontmatter on deep-think skills

### Next Steps (for incoming session)
1. **Multi-RR portfolio design** — RR1.0 O5 raw (11 sessions) + RR2.0 O30 ML-filtered (4 sessions)
2. **Paper trade raw baseline** — deploy signal-only mode
3. **April 2026: 2026 holdout test** — 3 pre-registered strategies, N≥100 per session
4. **Edge families rebuild** — 0 rows currently, needed for fitness tracking
5. **Simple regime filter** — ATR>50pct as ML-free alternative (deferred)

---

## Prior Session
- **Tool:** Claude Code (earlier today)
- **Date:** 2026-03-19 (morning)
- **Summary:** 100-seed MGC null test completed. Multi-aperture rebuild. Adversarial review framework. Zero-context audit. MNQ null test started.

## Session Before That
- **Tool:** Codex
- **Date:** 2026-03-18
- **Summary:** Workstream lifecycle + microstructure pilot (uncommitted on this branch)
