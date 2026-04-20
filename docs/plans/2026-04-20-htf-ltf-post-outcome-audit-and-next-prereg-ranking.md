# HTF/LTF Post-Outcome Audit And Next Pre-Reg Ranking

**Date:** 2026-04-20
**Status:** DESIGN DECISION
**Scope:** audit the completed HTF/LTF-related outcomes already on disk, mark what is actually proven, and rank the next highest-EV pre-registration paths without reopening dead branches.

## Executive decision

The broader HTF->LTF brief is **not dead**, but the repo has already killed several narrow routes that must not be confused with the whole class.

What is closed:

- bounded ORB `E_RETEST` continuation as a production-worthy ORB entry-model extension on the locked pilot scope
- simple weekly/monthly break-aligned ORB filters
- narrow PDH/PDL level pass/fail event study v1
- narrow PDH/PDL sweep/reclaim event study v1

What remains alive:

- ORB-compatible context overlays on already-profitable lanes
- standalone non-ORB event families whose geometry is genuinely different from the null families above

Immediate highest-EV path is **not** a new broad exploratory scan. It is:

1. keep the already-surviving `H04` confluence on its narrow shadow / deployment-shape path
2. make the next new discovery pre-reg a **low-blast ORB context-overlay family**
3. only then spend discovery budget on a new standalone family if the objective is true trade-class expansion rather than incremental ORB improvement

## Truth check

### 1. Bounded ORB retest pilot

Source:

- [docs/audit/results/2026-04-20-orb-retest-entry-pilot-v1.md](/mnt/c/Users/joshd/canompx3/docs/audit/results/2026-04-20-orb-retest-entry-pilot-v1.md)
- [research/orb_retest_entry_pilot_v1.py](/mnt/c/Users/joshd/canompx3/research/orb_retest_entry_pilot_v1.py)

Exact question tested:

- on retest-eligible days, does `ORB break -> first retest of ORB boundary -> continuation entry` beat canonical `E2` on the same cell/day universe?

Result:

- family `K = 54`
- `1` paired-delta survivor
- `0` trading-relevant survivors
- most adequately sampled cells negative on paired delta

Truth mark:

- **DEAD** for the bounded ORB execution-variant route on the locked scope

Why:

- the one survivor still had negative absolute IS expectancy
- this was a valid test of a specific ORB execution role, not of the whole HTF/LTF class

### 2. Simple HTF weekly/monthly break filters

Source:

- `HANDOFF.md` HTF branch closeout section
- `docs/plans/2026-04-20-htf-branch-closeout.md`

Exact question tested:

- does aligning the ORB break with a simple break of `prev_week_*` / `prev_month_*` improve the trade?

Result:

- prev-week v1 family killed
- prev-month v1 family killed

Truth mark:

- **DEAD** for simple break-through weekly/monthly HTF filters

### 3. Narrow level pass/fail event study

Source:

- [docs/audit/results/2026-04-19-level-pass-fail-v1.md](/mnt/c/Users/joshd/canompx3/docs/audit/results/2026-04-19-level-pass-fail-v1.md)
- [research/research_level_pass_fail_v1.py](/mnt/c/Users/joshd/canompx3/research/research_level_pass_fail_v1.py)

Exact question tested:

- in the first 30 minutes of key sessions, do first `close_through` or `wick_fail` interactions at `prev_day_high/low` predict positive next-2-bar signed return?

Result:

- family `K = 72`
- `0` primary survivors

Truth mark:

- **DEAD** for this narrow PDH/PDL pass/fail event family

### 4. Narrow sweep/reclaim event study

Source:

- [docs/audit/results/2026-04-19-sweep-reclaim-v1.md](/mnt/c/Users/joshd/canompx3/docs/audit/results/2026-04-19-sweep-reclaim-v1.md)

Exact question tested:

- do swept close-through events at PDH/PDL that reclaim within 2 bars predict positive next-2-bar signed return?

Result:

- family `K = 36`
- `0` primary survivors

Truth mark:

- **DEAD** for this narrow trapped-side re-test

### 5. Exact deployed-lane context overlays

Source:

- [docs/audit/results/2026-04-20-mnq-live-context-overlays-v1.md](/mnt/c/Users/joshd/canompx3/docs/audit/results/2026-04-20-mnq-live-context-overlays-v1.md)
- [docs/audit/hypotheses/2026-04-20-h04-cmx-short-relvol-q3-f6-shadow-v1.yaml](/mnt/c/Users/joshd/canompx3/docs/audit/hypotheses/2026-04-20-h04-cmx-short-relvol-q3-f6-shadow-v1.yaml)

Exact question tested:

- do a few exact live-detectable overlays improve two exact active MNQ lanes without broadening scope?

Result:

- `H04_CMX_SHORT_RELVOL_Q3_AND_F6` = `CONTINUE`
- `H01` and `H03` = `PARK`
- `H02` and `H05` = `KILL`

Truth mark:

- **CONDITIONAL** for the narrow H04 overlay

Why:

- IS evidence is strong and family-corrected
- OOS is still thin, so this is shadow / deployment-shape only, not capital-ready truth

## De-tunnel

Three distinct roles exist, and the repo has now tested examples of all three:

1. **ORB execution variant**
   - tested: yes
   - verdict: dead on current bounded retest geometry

2. **ORB context overlay**
   - tested: yes, narrowly
   - verdict: one live candidate survives (`H04`), others mixed

3. **Standalone event family**
   - tested: only in narrow PDH/PDL pass/fail and sweep/reclaim forms
   - verdict: those narrow forms are dead
   - untested: broader standalone classes with different geometry, especially open-displacement reversal and non-ORB breakout/retest

What was implicitly ignored until now:

- that a broader HTF/LTF brief can be alive even when one ORB execution variant is dead
- that a standalone family needs different trade geometry and cannot be judged by event-study-only warm cells
- that the current repo already contains a real surviving overlay candidate, so the highest-EV next move is not necessarily another discovery sweep

## Where edge exists

### Immediate, evidence-backed

Edge currently exists in the existing ORB book and, within this branch, in one narrow overlay candidate:

- `H04_CMX_SHORT_RELVOL_Q3_AND_F6`
  - role: ORB-compatible context / sizing / deployment-shape candidate
  - base lane: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
  - correct use today: shadow / deployment-shape only

### Not proven

These are not edge yet:

- ORB retest continuation
- simple HTF break filters
- level pass/fail
- sweep/reclaim
- broad standalone HTF/LTF rhetoric

## Biggest mistake to avoid

The main mistake is category error:

- treating "HTF/LTF" as one branch
- then using a dead ORB retest pilot or a null event study to kill the whole research class

The second mistake is wasted-motion bias:

- widening dead families by a little and calling it fresh work

The next branch must therefore be **structurally different**, not just numerically adjacent.

## Next pre-reg ranking

### Rank 1 — Keep H04 on the narrow shadow path

Status:

- already pre-registered
- already routed correctly
- already tied to a live lane

Why this is rank 1:

- strongest evidence already on disk
- lowest infrastructure cost
- directly monetizable if OOS accumulates
- no need to spend new discovery budget to justify its existence

Action:

- do not broaden it
- do not combine it with new features
- let the existing narrow shadow / deployment-shape contract run

### Rank 2 — New ORB context-overlay discovery family

This is the next **new** pre-reg with the best EV / cost ratio.

**Working title:** `prior-day-direction-split-orb-overlays-v1`

Purpose:

- test whether prior-day positional context helps existing ORB trades **as context overlays**, not as standalone trades and not as execution rewrites

Why this is the right next branch:

- uses existing canonical `orb_outcomes` + `daily_features`
- no pipeline schema work
- directly addresses the symmetry blindspot documented in [2026-04-15-htf-sr-untested-axes-roadmap.md](/mnt/c/Users/joshd/canompx3/docs/audit/hypotheses/2026-04-15-htf-sr-untested-axes-roadmap.md)
- stays inside ORB where the repo already has real edge

Recommended locked scope:

- instrument: `MNQ` only
- sessions: `NYSE_OPEN`, `COMEX_SETTLE`, `US_DATA_1000`
- aperture: `O5`
- entry model: `E2`
- confirm bars: `CB1`
- RR targets: `1.0`, `1.5`
- features:
  - `F1_NEAR_PDH_15`
  - `F5_BELOW_PDL`
  - `F6_INSIDE_PDR`
- directions: `long`, `short`
- honest family size: `3 features × 3 sessions × 2 RR × 2 directions = K 36`

Canonical feature definitions:

- `F1_NEAR_PDH_15`: `abs(orb_mid - prev_day_high) / atr_20 < 0.15`
- `F5_BELOW_PDL`: `orb_mid < prev_day_low`
- `F6_INSIDE_PDR`: `orb_mid > prev_day_low AND orb_mid < prev_day_high`

Primary claim:

- at least one locked cell shows positive delta between on-signal and off-signal ExpR in IS, survives BH-FDR at family `K = 36`, has `N_on_IS >= 100`, and preserves direction in OOS when `N_on_OOS >= 30`

Why this scope is honest:

- it does not reopen the full 96-cell phase-1 family
- it does not pretend cross-instrument universality when canonical active-book opportunity is MNQ-dominant
- it keeps the next question on the profitable ORB surface instead of on null event-study families

Kill criteria:

- `0` survivors at family `K = 36`
- any apparent survivor that is positive only because off-signal rows are catastrophically negative while on-signal remains non-positive
- any result that vanishes when direction is fixed but only exists in pooled long+short framing

Secondary rule:

- if a survivor appears, `MES` replication can be run as a separate confirmatory follow-up, not inside the discovery family

### Rank 3 — Standalone open-displacement reversal family

This is the highest-EV **standalone** branch, but it ranks below the ORB overlay family because the geometry is new and the deployment path is longer.

**Working title:** `open-displacement-reclaim-reversal-v1`

Why this branch is different enough to justify running later:

- it is not ORB retest
- it is not PDH/PDL pass/fail
- it is not PDH/PDL sweep/reclaim
- it matches external literature on large opening-move reversals more closely than the repo's existing null families

Grounding:

- Grant, Wolf, Yu (2005): large opening price changes in US stock-index futures show significant intraday reversals, though costs matter
- Osler (2000): support/resistance levels can predict intraday trend interruptions, but heterogeneity is real
- Chan (2013 Ch 7): stop-triggered breakouts and intraday momentum are real, which makes reversal-after-overreaction a meaningful contrast family

Recommended pre-reg shape:

- instruments: `MNQ`, `MES`
- session: `NYSE_OPEN` only
- event family: one mechanical open-displacement reversal definition only
- no threshold sweep
- one of:
  - fixed gap threshold from literature, or
  - ATR-scaled opening displacement threshold
- not both in the same family

Why this is rank 3 instead of rank 2:

- new trade geometry and exit logic required
- higher risk of accidental ontology sprawl
- lower immediate monetization than the ORB overlay path

## Branches to keep closed

Do not reopen these without a structurally new pre-reg:

- bounded ORB `E_RETEST`
- simple `prev_week` / `prev_month` break-aligned filters
- level pass/fail v1 widened a little
- sweep/reclaim v1 widened a little

Those would be wheel-spinning, not new research.

## Institutional verdict

**Verdict:** `CONDITIONAL`

Where edge exists:

- in the existing ORB book
- and conditionally in one narrow overlay candidate: `H04`

Biggest mistake:

- over-reading null sub-branches as a verdict on the whole HTF/LTF brief

Best next action:

1. keep `H04` on its existing narrow shadow / deployment-shape path
2. pre-register the narrow `MNQ` direction-split prior-day ORB overlay family at `K = 36`
3. only after that, spend discovery budget on a standalone open-displacement reversal family if the goal is a genuinely new trade class

## External sources

- Grant, Wolf, Yu, *Intraday price reversals in the US stock index futures market: A 15-year study*  
  https://doi.org/10.1016/j.jbankfin.2004.04.006
- Osler, *Support for Resistance: Technical Analysis and Intraday Exchange Rates*  
  https://ssrn.com/abstract=888805
- Wang and Wilmott, *Support and Resistance Levels in Financial Markets*  
  https://ssrn.com/abstract=7484
- Baltussen, Da, Lammers, Martens, *Hedging Demand and Market Intraday Momentum*  
  https://ssrn.com/abstract=3760365
