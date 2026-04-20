# Session Handoff — Phase 2-9 X_MES_ATR60 Re-audit (2026-04-20)

**Branch:** `research/phase-2-9-xmes-block-bootstrap-reaudit` (pushed to origin, 5 commits)
**Status:** PARKED. Re-audit complete; not to be merged to main without rebase.

---

## What this session accomplished

Ran a block-bootstrap null re-audit of 4 phase-2-9 framing-audit MNQ COMEX_SETTLE lanes:

| Lane | p_boot | Verdict |
|---|---|---|
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | 0.019 | CONFIRM |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | 0.046 | CONFIRM (borderline) |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | 0.007 | REFERENCE_PASS |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | 0.006 | REFERENCE_PASS |

Pre-reg, script (canonical contract load after band-aid refactor), result doc with self-review — all on branch. `check_drift.py` 104/104 with canonical DB path.

---

## The tunnel-vision finding — DO NOT LOSE THIS

**The 4 lanes I re-audited ARE NOT IN THE LIVE ALLOCATOR.** Queried `docs/runtime/lane_allocation.json` (2026-04-18) at end of session:

Live 6 DEPLOY lanes on `topstep_50k_mnq_auto` use filter keys `ORB_G5` / `ATR_P50_O15` / `COST_LT12`. **Zero** `OVNRNG_100`, **zero** `X_MES_ATR60`. The 2026-04-11 research-provisional promotion was overridden by the 2026-04-18 allocator pick that chose ORB_G5 on COMEX_SETTLE instead.

Implication: I spent a session validating the null-floor for a lane set the portfolio already rejected. The re-audit is a methodologically correct confirmation of an academic question — not a portfolio-actionable result. The phase-2-9 framing audit's CONTINUE verdict **stands** in the sense that the lanes aren't retired; but they aren't trading either.

**Process lesson:** before re-auditing a "promoted" or "validated" lane set, always query `docs/runtime/lane_allocation.json` and confirm the lanes are actually in the live allocator's DEPLOY list. Research-provisional status does not imply deployment. Added to `.claude/rules/backtesting-methodology.md` historical failure log as 2026-04-20 entry.

---

## Next-session research mandates — ranked by EV

### MANDATE #1 (highest EV) — verify `bull_short_avoidance` signal against canonical data
- **Memory claim:** p=0.0007, positive 14 of 17 years; "activate when NYSE_OPEN lanes deploy"
- **Current state:** `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` IS DEPLOYED (allocator 2026-04-18, N=262, ExpR=+0.12). Lane takes both long and short. If the signal is real, current lane is bleeding edge on short trades after bull days.
- **Scope:** fresh session. Pre-reg (Pathway B individual, K=1, theory-citation mandatory), canonical query against `orb_outcomes` + `daily_features.prev_day_direction`, block-bootstrap null, OOS check on 69 days of 2026 OOS data.
- **Action if verified:** direct filter edit on a deployed lane (half-size or skip shorts after bull days).
- **Action if stale/overfit:** remove from memory as "queued for activation"; update NO-GO registry.
- **Why fresh session:** requires proper pre-reg + adversarial audit. Not a tack-on test.

### MANDATE #2 — ORB_G5 cross-session fire-day overlap on live 3 ORB_G5 lanes
- **Concern:** 3 of 6 DEPLOY lanes use ORB_G5 (EUROPE_FLOW, COMEX_SETTLE, US_DATA_1000_O15). If their per-day fire masks have Jaccard > 0.5 pairwise, the portfolio is concentrated in one low-vol-day regime under the guise of 3 independent sessions.
- **Method:** canonical overlap-decomposition per `research/rel_vol_cross_scan_overlap_decomposition.py`. Jaccard + Nyholt Meff + non-overlap-subset edge decomposition per 2026-04-19 failure-log standard.
- **Action if Jaccard > 0.5:** recommend dropping 1 of the 3 and freeing an allocator slot.
- **Action if Jaccard < 0.3:** portfolio diversification story is defensible.

### MANDATE #3 — Re-audit the 38 allocator-evaluated lanes as a set
- **Observation:** allocator picked 6 DEPLOY + 2 PAUSED out of 38 scored lanes (2026-04-18). The tail 30 includes ~20 "validated but not deployed" — are any of them research-provisional or formerly deployed? Is the allocator's ranking leaving EV on the table?
- **Method:** pull `docs/runtime/lane_allocation.json` full `all_scores_count: 38` list + join with `validated_setups`. Inspect gap between #6 and #7 lanes.
- **Action:** either confirm allocator cutoff is well-calibrated, or surface unpromoted-yet-viable lanes to user.

### MANDATE #4 — lower priority: 2026 OOS behavior per live lane
- 69 days of OOS exist (2026-01-02 → 2026-04-16). Per-lane OOS sign-match vs IS has not been systematically audited on the current 6 DEPLOY set. Allocator uses trailing-12mo ExpR (which includes OOS) but that's not the same as a dir_match check per lane.

### MANDATE #5 (lowest, academic) — seed-robustness distribution on XMES RR1.5
- RR1.5 p_boot=0.0464 is seed-borderline. Run 20 seeds to characterize the tail. Only matters if someone decides to deploy this lane.

### DO NOT RE-OPEN
- vol-regime-confluence sprint result (`research/vol-regime-confluence-2026-04-20`) — closed, documented, COEXISTS_BOTH verdict
- X_MES_ATR60 as a confluence variant on ORB_G5 — killed by vol-regime Stage H
- X_MES_ATR60 standalone block-bootstrap — closed by this re-audit (CONFIRM)

---

## This branch's final state

- 5 commits: prereg → script → band-aid fix → canonical refactor → result doc
- `pipeline/check_drift.py` 104/104 with `DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db`
- Parent branch: `research/vol-regime-confluence-2026-04-20` (which itself has cross-thread scope vs origin/main — do not fast-forward merge)
- **Merge path to main:** would require rebase onto current `origin/main` to pick up MGC merges that predate the vol-regime base. Worth doing only if the re-audit result is cited externally. Otherwise leave on branch as an audit artifact.
- **Recommended action on this branch:** leave pushed to origin. No PR. Future sessions can cite it by commit hash or branch name.

## Resume prompt for next session

> Picking up from 2026-04-20 phase-2-9 X_MES_ATR60 re-audit session. Re-audit branch `research/phase-2-9-xmes-block-bootstrap-reaudit` is parked on origin (5 commits, CONFIRM verdict). Handoff at `docs/handoffs/2026-04-20-xmes-reaudit-session-handoff.md`. Next mandate is **#1: verify `bull_short_avoidance` signal (p=0.0007 memory-claimed) against canonical `orb_outcomes` + `daily_features.prev_day_direction` on MNQ NYSE_OPEN lane that's currently deployed as COST_LT12 RR1.0**. Fresh pre-reg required (Pathway B, K=1, theory citation mandatory).
