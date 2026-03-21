# HANDOFF.md — Cross-Tool Session Baton

The outgoing tool updates this before the user switches. The incoming tool reads it first.

**Rule:** If you made decisions, changed files, or left work half-done — update this file.
If nothing changed, leave it as-is.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code. If another tool made commits since your last read, your mental model is wrong.

---

## Current Session
- **Tool:** Claude Code
- **Date:** 2026-03-21 (evening)
- **Branch:** `main`
- **Status:** Session continuity — updating stale docs after mid-session crash. Auditing ChatGPT assessment.

### What was done this session (Mar 21 continued)

#### 1. Bootstrap 5K — COMPLETE (Phipson & Smyth corrected)
- 5000 permutations per survivor, Phipson & Smyth (2010) p-value correction
- **3/7 PASS:** NYSE_OPEN O30 p=0.0016, US_DATA_1000 O30 p=0.0176, US_DATA_830 O30 p=0.0376
- **3 MARGINAL:** US_DATA_1000 O15, NYSE_OPEN flat, CME_PRECLOSE flat (all 0.05-0.09)
- **1 FAIL:** CME_PRECLOSE RR1.5
- Code: commit `9252930` (200→5K), `4cee702` (Phipson-Smyth fix, deterministic seed)
- ML audit item #6 (bootstrap resolution) is now FIXED. 3 open FAILs remain.

#### 2. Confluence Features Design — DRAFTED
- `docs/plans/2026-03-21-confluence-features-design.md`
- Commits: `4326abd` (initial), `3b5b850` (corrected ML confidence)
- Key: univariate scan FIRST on positive baselines, ML integration LAST (after 3 FAILs fixed)
- 3 genuinely new candidates: `vwap_in_prior_orb`, `pdh_pdl_proximity_R`, `pre_session_sweep`

#### 3. Memory/Doc Updates
- All memory files updated to reflect 5K results and 3 (not 4) open ML FAILs
- MEMORY.md strategic direction and action queue refreshed

### Truth State (no bias)
- **Raw baseline = tradeable path.** Layer 1 (O5 RR1.0) p<1e-9, survives Bonferroni. BUILT.
- **ML = research-only.** 3 open FAILs (EPV, negative baselines, selection bias). NOT production.
- **2026 forward test is binding.** 3 pre-registered strategies. No early peeking.
- **Edge comes from ORB/session structure,** not ML. The rules/playbook IS the edge.

### Next Steps (for incoming session)
1. **Paper trade raw baseline** — deploy signal-only mode, collect 2026 forward data
2. **Confluence univariate scan** — test existing ML features as standalone signals on positive baselines
3. **Fix EPV** — reduce ML features to ≤5 strongest (cheapest remaining audit fix)
4. **April 2026: CME_PRECLOSE on sacred 2026 holdout** — 3 pre-registered, N≥100
5. **Simple regime filter** — ATR>50pct as ML-free alternative (deferred)
6. **CUSUM fitness** — faster regime break detection (deferred)

---

## Prior Session
- **Tool:** Claude Code (Paper Trading Terminal)
- **Date:** 2026-03-21 (earlier)
- **Summary:** Multi-RR portfolio built (commit 3e2557c). ML audit found 4 FAILs. Bootstrap 5K code committed. Confluence design started. Session crashed mid-brainstorm.

## Session Before That
- **Tool:** Claude Code (earlier today)
- **Date:** 2026-03-19 (morning)
- **Summary:** 100-seed MGC null test completed. Multi-aperture rebuild. Adversarial review framework. Zero-context audit. MNQ null test started.
