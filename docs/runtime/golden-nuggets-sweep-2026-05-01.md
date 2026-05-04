# Golden Nuggets Sweep — 2026-05-01

**Mandate:** "what other golden nugs we got? ensuring honest and validated before actioning."

**Honesty discipline:** every nugget tagged HONEST (verified) or UNVERIFIED (needs gate before action). No claim makes it to actionable column without a verifiable artifact path.

**Method:** swept `docs/runtime/parked-cells.yaml`, `docs/runtime/action-queue.yaml`, `docs/runtime/decision-ledger.md`, open PRs, recent_findings memory.

---

## TIER 1 — Live capital gating (highest immediate value)

### N1. Allocator re-run on canonical worktree (PR #189 aftermath)
**Status:** HONEST — direct quote from PR #196 (open closeout): "PR #189 fixed the structural bug but `lane_allocation.json` is still on PR #188's interim O15 pause."
**What it unlocks:** O15 lanes (MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15, MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15) currently paused with the doctrine reason "allocator hardcodes orb_minutes=5; this O15 lane was scored against O5 data." That hardcode was structurally fixed by PR #189 (merged today). The paused lanes have not been reassessed yet.
**Why it's a nugget:** real, measurable lanes sitting paused on a now-resolved blocker. Re-run takes minutes; impact is up to 2 additional deployable lanes. Capital-class.
**Verification source:** `docs/runtime/decisions/2026-05-01-allocator-orb-fix-lane-composition.md` + `lane_allocation.json` paused entries.
**Gate before action:** none. The structural fix landed; this is the operational follow-through. **However, requires a `gold.db` write (rebalance), which conflicts with concurrent Phase D append on other terminal.** Coordinate with the user before running.
**Action shape:** `python scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto` once Phase D writes complete.

### N2. PR #177 (cryptography-pin Phase 2 advisory leak fix)
**Status:** HONEST — bug verified via REPL with mocked date in PR description. Without merge, on/after **2026-10-29** every commit BLOCKS with "ADVISORY: ... non-blocking" message (false-positive blocker).
**Why it's a nugget:** time-bomb. Costs nothing now; costs every commit on the project after 2026-10-29 if unmerged.
**Verification source:** PR #177 body — REPL trace + 4 regression tests added.
**Gate before action:** standard PR review.
**Action shape:** review + merge.

---

## TIER 2 — Research-validated, capital-pending (medium-high value, needs OOS)

### N3. D2 cell — MES EUROPE_FLOW E2 RR1.5 + ovn_range_pct > 80
**Status:** HONEST verdict (PARK_PENDING_OOS_POWER); UNVERIFIED for direct deployment.
**Why it's a nugget:** parked-cells.yaml says "D2 PASS_ADDITIVITY (worst_rho +0.32, N=387)" against the deployed-lane portfolio — meaning if/when OOS power floor clears, D2 is statistically separable from the existing book. This is rare; D4 by contrast FAIL_ADDITIVITY (worst_rho +0.81 — too correlated to be truly additive).
**Eligibility timeline:** parked-cells.yaml entry says "OOS power-floor accrual ETA per addendum 2 (efeaa2fc): D2 reaches N_OOS_on=50 around 2026-10-09 at current 0.177 trades/cal-day rate."
**Re-open trigger:** N_OOS_on >= 50 (automatic per `parked_pathway_b_additivity_triage` action-queue exit criterion).
**Verification source:** `docs/audit/results/2026-04-29-parked-pathway-b-additivity-triage.md` + `docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml`.
**Gate before action:** OOS power floor clears (no manual trigger; data-bound). NOT a session-1 action.
**Action shape:** monitor; reactivate Phase 1 carrier-wiring stage spec when N_OOS_on hits 50.

### N4. D6 GARCH overlay — already wired Phase 1 shadow on COMEX_SETTLE
**Status:** HONEST — `parked-cells.yaml` cell `d6-mnq-comex-settle-garch70-overlay` verdict PARK_CONDITIONAL_DEPLOY_RETAINED. Phase 1 shadow wiring already implemented per `pr48-mgc-shadow-overlay-phase1-implemented` precedent (decision-ledger entry confirms).
**Why it's a nugget:** OOS lift +0.6934R, dir_match TRUE, OOS_lift>=0.40*IS_lift. Currently shadow-only (no execution-engine size mutation). The next legitimate state IS Phase 2 (live size-multiplier), gated on a SEPARATE Phase 2 pre-reg.
**Re-open trigger:** Phase 2 pre-reg authored; SR alarm on parent lane has NOT fired (60-day window from lock 2026-04-29 → expires ~2026-07-28).
**Verification source:** `docs/audit/results/2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1-result.md`.
**Gate before action:** Phase 2 pre-reg with size-multiplier action (NOT yet authored). The pre-reg #4 in this session's batch (`2026-05-01-mnq-garch-p70-cross-session-companion`) is the cross-session generalisation companion, not the Phase 2 size-multiplier action.
**Action shape:** **distinct from this session's batch.** Phase 2 pre-reg is a separate write — design proposal needed first.

### N5. PR48 MGC:cont_exec — Phase 1 shadow observation in progress
**Status:** HONEST observation phase per `pr48-mgc-shadow-overlay-phase1-implemented` decision-ledger entry. Action-queue item `pr48_mgc_shadow_observation` is `waiting_observation`.
**Why it's a nugget:** distinct from D6. This is the MGC continuous-execution overlay riding on prior-day rel-vol-quintile sizing. PARENT research showed `MGC` is the only research-level deploy candidate from the 2026-04-23 frozen-rule replay; `MES` improved but remains `SIZER_ALIVE_NOT_READY`.
**Verification source:** decision-ledger entries `pr48-sizer-split` + `pr48-mgc-shadow-overlay-phase1-implemented`.
**Gate before action:** observation period — operator records dashboard/live-log behaviour as PASS/FAIL/redesign.
**Action shape:** check operator log for any recorded observation result. If empty → flag to user that observation has not been collected.

---

## TIER 3 — Verified KILLs (NO-GO; preserve so we don't redo)

These are verified DEAD-FOR-ORB; preserve as NO-GOs so future sessions don't re-test:

### N6. D5 conditional half-size — KILL
**Status:** HONEST KILL via locked decision rule. Two KILL criteria fired: KILL_PAIRED_P (paired t=-1.45, p=0.148) + KILL_C9 (2019 ExpR_D5=-0.108 below floor).
**Discipline lesson:** lock-before-run prevented post-hoc rationalisation. Headline numbers passed (+0.2705 abs SR_ann, +21.85% rel uplift, OOS DIR_MATCH) but the locked rule killed it correctly.
**Verification source:** `docs/runtime/parked-cells.yaml:d5-mnq-comex-settle-garch70-conditional-half-size`.

### N7. D-0 v2 GARCH percentile sizing-buckets — PARK_ABSOLUTE_FLOOR_FAIL
**Status:** HONEST. Sharpe uplift +16.48% (>=15% relative gate PASS) but absolute Sharpe diff +0.0283 < 0.05 floor and p=0.50 (underpowered).
**Operational consequence:** "2026-05-15 daily shadow-append gate eval has no clean PASS basis."
**Verification source:** `docs/runtime/parked-cells.yaml:phase-d-d0-v2-garch-clean-rederivation`.

### N8. PR48 conditional shadow buckets MNQ — KILL at K=6
**Status:** HONEST KILL on the exact bounded `mnq_parent_structure_shadow_buckets_v1` family. No test cleared BH-FDR. Apparent subset-quality lift was wrong role shape: `selected_trade_mean_r` improved, but `policy_ev_per_opportunity_r` fell on both exact-parent branches.
**Discipline lesson:** subset-quality vs policy-EV distinction matters; "selected trades did better" ≠ "the policy's expected value improved."
**Verification source:** decision-ledger entry `mnq-parent-structure-shadow-buckets-kill`.

---

## TIER 4 — Eval-pending tooling (NOT capital-class)

### N9. PR #191 (mcp-server-git) — GO-WITH-MITIGATION
**Status:** HONEST — verdict on the safety analysis. Q1 + Q2 verified BYPASSED via direct upstream source quote (`mcp-server-git/server.py:128-130` uses `repo.index.commit(message)` — NOT shelling out to `git commit`).
**Why it's a nugget:** known security gap if adopted unmitigated. Q3 mitigation (PostToolUse `mcp__git__.*` matcher) is VIABLE but not yet implemented.
**Gate before action:** mitigation hook authored AND `.mcp.json` does not include the server until mitigation lands.
**Disposition:** keep PR draft; do not adopt.

### N10. PR #193 (seq-thinking) — pending live eval
**Status:** UNVERIFIED — verdict scaffolding done (COMPLEMENT), but live eval not run. Three real incidents named for replay.
**Gate before action:** live eval session in worktree with `.mcp.json` patch applied; replay 3 incidents; collect 6 metric placeholders.
**Disposition:** keep PR draft; queue eval session.

### N11. PR #99 (MNQ geometry transfer COMEX family — DRAFT 9 days old)
**Status:** HONEST result (PD_CLEAR_LONG on MNQ COMEX_SETTLE: IS ExpR_on +0.1841, OOS ExpR_on +0.1321 N=15) but **stale**. The decision-ledger then recorded multiple post-2026-04-23 closure audits (`prior-day-bridge-execution-triage`, `prior-day-geometry-routing-2026-04-23`, `prior-day-geometry-execution-translation-2026-04-23`) that **closed this thread as `ARCHITECTURE_CHANGE_REQUIRED` not deployable**.
**Disposition:** PR #99 is superseded by closure audits; should be CLOSED with reference to the closure-audit chain in the PR body. Do NOT merge — the framing was correct at the time but the routing claim has been retracted.

---

## TIER 5 — UNVERIFIED until measured (named claim, no artifact yet)

### N12. MES q45_exec bridge — blocked
**Status:** UNVERIFIED — action-queue item `mes_q45_exec_bridge` says "alive MES q45_exec research branch" but no result-doc lookup confirms current "alive" status. Marked blocked by `pr48_mgc_shadow_observation`.
**Gate before action:** PR48 MGC observation completes AND a fresh result-doc verifies MES q45_exec is still alive.
**Disposition:** do not act on memory of "alive" status without re-verification.

### N13. Track D Gate 0 microstructure — DESIGN_ONLY
**Status:** HONEST design pre-reg locked at `docs/audit/hypotheses/2026-04-23-mnq-comex-settle-gate0-microstructure-v1.yaml`, but not executable until Databento top-of-book tables and bounded runner exist.
**Gate before action:** Databento subscription delivers top-of-book data + new pipeline tables built.
**Disposition:** parked, dependency-bound. NOT a near-term nugget.

### N14. Cross-asset chronology spec frozen
**Status:** HONEST. Spec exists at `docs/plans/2026-04-25-cross-asset-session-chronology-spec.md` (278 lines, PR #101 commit 87ebc885). Action-queue item closed.
**Gate before action:** future cross-asset prereg consumes it. NOT itself an actionable nugget; supports a future hypothesis if/when one is authored.

---

## TIER 6 — Honesty-gate gaps surfaced by MotherDuck eval (Q3/Q4)

### N15. `experimental_strategies.validation_pathway` is NULL across all 45,532 rows
**Status:** HONEST data-quality gap — surfaced by PR #192 Q3 (DEGENERATE result).
**Why it's a nugget:** any analysis that filters/groups by `validation_pathway` from `experimental_strategies` produces noise. Multiple research scripts may be doing this implicitly.
**Gate before action:** sweep research/ for `validation_pathway` filters; either backfill the column or document the gap explicitly.
**Disposition:** flag for next research session; not session-1.

### N16. `c8_oos_status` is NULL across all 82 deployed_validated_setups rows
**Status:** HONEST data-quality gap — surfaced by PR #192 Q4 (uncomputed snapshot-wide, not pathway-conditional).
**Why it's a nugget:** Criterion 8 (OOS gate) verdict is being read from a column that's never been written to. This is a known gap; the snapshot-wide-NULL framing makes it concrete.
**Gate before action:** locate the canonical writer for `c8_oos_status` (or document that it's intentionally uncomputed), and decide whether the gap blocks any current decision.
**Disposition:** capital-class data-integrity question. Surface to user.

---

## What the sweep does NOT promote

Following `feedback_two_track_decision_rule.md`, every "is this still alive?" question must compare vs the highest-EV item:

- **TIER 1 dominates.** N1 (allocator re-run) and N2 (PR #177) are concrete, well-defined, near-zero-risk operations with measurable next-session value.
- **TIER 2 (D2/D6/PR48 MGC) gates on time, not on a decision** — they're observation-bound, so no further "decision" to make this session.
- **TIER 3 (KILLs) is preservation work**, not actionable.
- **TIER 4 (tooling)** is non-capital-path; defer.
- **TIER 5 (unverified)** explicitly fails the validation gate; do not touch without re-verification.
- **TIER 6 (data-quality)** is real but session-2+ work.

---

## Recommended next-session entry order (gated on user GO)

1. **N1 — allocator re-run** (5-10 min after Phase D writes complete; user coordination)
2. **N2 — PR #177 review/merge** (review takes ~15 min; merge is one-click)
3. Run pre-reg #1 from this session's batch (Chordia revalidation; pure read-only)
4. Decide on N4 D6 Phase 2 size-multiplier pre-reg authoring vs N15/N16 data-quality sweep
5. Park everything else

Total session time for items 1-3: under one hour, all HONEST verified.
