# HANDOFF.md — Cross-Tool Session Baton

The outgoing tool updates this before the user switches. The incoming tool reads it first.

**Rule:** If you made decisions, changed files, or left work half-done — update this file.
If nothing changed, leave it as-is.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code. If another tool made commits since your last read, your mental model is wrong.

---

## Current Uncommitted Session
- **Tool:** Claude Code
- **Date:** 2026-03-19
- **Branch:** `pipeline-fixes`
- **Status:** Adversarial audit in progress. MNQ null test running overnight (21/100 seeds).

### What was done this session
1. **Full system architecture audit** — 3 exploration agents mapped every gate, filter, threshold, and bias risk
2. **Adversarial audit framework designed** — 6-module plan in `docs/plans/2026-03-19-adversarial-audit-design.md`
3. **RED-flag checks:**
   - 4A (WF orb_minutes join): **CLEARED** — no sample inflation
   - 4E (noise floor timing): **CLEARED** — MNQ 11 strategies validated with E2=0.32 floor active
4. **Raw baseline extracted (Module 1):**
   - MGC: ALL sessions significantly negative unfiltered. G4 converts CME_REOPEN to +0.39R on N=129.
   - MNQ: 23 positive BH FDR survivors across 7 sessions UNFILTERED. Real raw signal.
   - MES: 1 positive survivor (CME_PRECLOSE RR1.0, barely positive). Effectively zero edge.
5. **Zero-context audit data package** — `docs/plans/2026-03-19-zero-context-audit-data.md` — raw tables for cold evaluation by a fresh session
6. **Multi-signal research** — academic literature search for non-ORB signals. Top candidate: Intraday Momentum (Gao et al. 2018 JFE). Full findings in session transcript.
7. **Null test audit script** — `scripts/tools/zero_context_audit.py`

### Files created (untracked)
- `docs/plans/2026-03-19-adversarial-audit-design.md` — audit framework
- `docs/plans/2026-03-19-zero-context-audit-data.md` — raw data for cold review
- `scripts/tools/zero_context_audit.py` — generates the data package

### MNQ Null Test Status
- Location: `scripts/tests/null_seeds/mnq/`
- 21/100 seeds complete, 4 workers active
- Config: `instrument=MNQ, sigma=5.0, noise_floors={E1:0, E2:0}`
- Early results: 12/15 seeds = 0 survivors, 3 seeds = ~430 survivors (outliers)
- **Noise floors ZEROED in config.py:97-100** for calibration. MUST restore after test.
- ETA: ~15hrs from 12:35 AEST Mar 19

### Decision Tree (blocked on null test)
- **If MNQ E2 ceiling ≤ 0.32:** Strategies survive → restore floors → rebuild → paper trade
- **If MNQ E2 ceiling > 0.37:** All dead → accept null → pivot to intraday momentum research or lean into ML meta-labeling V4
- **The 11 MNQ survivors have ExpR 0.320-0.374** — razor-thin margin above MGC-derived floor

### Key Insight from Raw Data
MNQ has **real unfiltered signal** (23 BH FDR survivors, p < 0.05). The ATR70_VOL composite filter on the 11 validated strategies may be ADDING to a real base signal, not MANUFACTURING edge from noise. This is the best-case interpretation. The null test determines if the filtered ExpR (0.32-0.37) is above or below the noise ceiling for MNQ specifically.

### Prior Codex session (uncommitted, from Mar 18)
Workstream lifecycle upgrade + microstructure pilot still uncommitted on this branch.
Files: `scripts/infra/windows_agent_launch.py`, `scripts/tools/worktree_manager.py`, `tests/test_tools/test_worktree_manager.py`, `tests/test_tools/test_windows_agent_launch.py`

---

## Last Session
- **Tool:** Codex
- **Date:** 2026-03-18
- **Summary:** Workstream lifecycle + microstructure pilot (uncommitted)
