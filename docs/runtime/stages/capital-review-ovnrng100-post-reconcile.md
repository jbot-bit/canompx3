---
task: Capital review on deployed lane MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 following PR #228 OOS reconciliation. Four converging concerns flagged by evidence-auditor 2026-05-05 require a /capital-review pass before the next allocator rebalance. NO writes to validated_setups, lane_allocation.json, or chordia_audit_log.yaml until this stage's pre-reg locks. THIS STAGE FILE IS A SCOPE PLACEHOLDER ONLY — actual work to begin in a fresh session per context-budget discipline.
mode: DESIGN
---

## Scope Lock

- docs/audit/hypotheses/<TBD>-capital-review-ovnrng100.yaml (to be created)
- docs/audit/results/<TBD>-capital-review-ovnrng100.md (to be created)
- research/<TBD>-capital-review-ovnrng100.py (to be created if scan needed)
- docs/runtime/stages/capital-review-ovnrng100-post-reconcile.md (this file)

## Blast Radius

- New audit files only. Zero edits to pipeline/, trading_app/, or scripts/ in this stage.
- Reads canonical layers (orb_outcomes, daily_features, validated_setups) via read-only DuckDB.
- Reads docs/runtime/lane_allocation.json for current DEPLOY status confirmation.
- Reads docs/audit/results/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.md for prior context.
- Reads docs/audit/results/2026-05-05-oos-reconcile-ovnrng100.md for the four findings.
- Reads docs/audit/results/2026-05-01-target-b-6lane-vestigialness-fresh-audit.md for the filter-vestigialness pattern.
- No writes to canonical state. No allocator changes. No deployment toggles.
- /capital-review skill route is the parent workflow; this stage hosts the artifacts.

## Why this stage is paused (DESIGN mode)

Started 2026-05-05 in a session at ~26% remaining context after evidence-auditor flagged
four concerns on the merged PR #228 reconciliation. Capital-class research at low
context-budget is exactly the failure mode `feedback_meta_tooling_n1_tunnel_2026_05_01.md`
and `adversarial-audit-gate.md` warn against. Stage filed in DESIGN mode to preserve
findings; IMPLEMENTATION to be opened in a fresh session.

## Findings to address (from evidence-auditor 2026-05-05)

PR #228 (commit 120882f1) reconciled OOS_ExpR for the deployed lane to canonical +0.1658
(was LEGACY +0.2029 in validated_setups). The reconciliation itself is sound, but four
concerns surfaced that the result doc explicitly routes to /capital-review:

1. **OOS fire rate 91.67% (66 of 72 holdout bars).** OVNRNG_100 filter is nearly vacuous
   in 2026 OOS — the +0.1658 OOS_ExpR is essentially an unfiltered base-rate measurement,
   not evidence of filter edge. Same vestigialness class as the 6 lanes audited in
   `2026-05-01-target-b-6lane-vestigialness-fresh-audit.md` (≥75% fire rate,
   zero-to-negative lift). Question to answer: is OVNRNG_100 doing any work for this lane
   in 2026, or has the threshold drifted with price (per `feedback_absolute_threshold_scale_audit.md`)?

2. **Runner exit code 1 (FAIL) vs result doc "RECONCILED" header.** Pre-reg criterion 4
   was `|IS_ExpR_runner − validated_setups.expectancy_r| ≤ 0.001` and the runner's
   measured delta was 0.0020 → FAIL. Result doc reframes this post-hoc as a spec bug
   (scratch-inclusive vs scratch-exclusive accounting). The explanation is plausible
   but contradicts `pre_registered_criteria.md` no-post-hoc-rescue. Capital review must
   assess: was the deployment decision in 2026-04 conditioned on the LEGACY +0.2029 inflated
   OOS, and does the corrected canonical value change the deployment math?

3. **No runner stdout artifact committed in PR #228.** The "+0.0000 delta" claim was
   unverifiable at merge time without re-execution. Auditor independently re-ran the
   script and confirmed the number. Going forward, lane-touching audits should commit
   stdout (per `feedback_token_efficient_audit_loop.md`).

4. **Stacked capital risks on the deployed lane:**
   - DSR = 0.1746951317604647 (MinBTL violation — n_trials_at_discovery=35,616 from
     pre-Phase-0 brute-force regime per Bailey et al 2013).
   - OOS underpowered: t=+1.150, p_two=0.250, N=66 (per
     `feedback_oos_power_floor.md` this is UNVERIFIED, not DEAD).
   - Static OOS in validated_setups still reads +0.2029 (canonical is +0.1658).
   - Live trailing_expr=+0.2412 N=150 in 2026-05-03 rebalance (post-deployment, independent measurement).

   Joint interpretation: three priors point to overstated edge; trailing performance
   is the only positive signal, and it's small-N.

## Pre-reg requirements (when fresh session starts)

Per `.claude/rules/research-truth-protocol.md` § Phase 0 + `prereg-writer-prompt.md`:

- testing_mode: "individual" (single-lane diligence)
- pathway: "B" (theory-driven K=1 confirmatory; not new family search)
- theory_citation required: Carver 2015 Ch 9-10 (capital allocation under uncertainty),
  Bailey-Lopez de Prado 2014 p.7-9 (DSR threshold), Harvey-Liu 2015 (OOS power floor).
- Numeric kill criteria. No "investigate" or "reconsider" verbs.
- Mode A holdout (2026-01-01 sacred); no tuning against OOS.
- No upstream-K framing; this is K=1 not search-family.
- Lock before run.

## Forbidden in this stage

- Writes to `validated_setups`, `experimental_strategies`, `chordia_audit_log.yaml`, `lane_allocation.json`.
- Deployment toggles (lane stays as currently allocated until next rebalance).
- Updating `validated_setups.oos_exp_r=+0.2029` to canonical +0.1658 — that requires
  the strategy_validator path with its own pre-reg, not this audit.
- Threshold modification on OVNRNG_100 based on what the audit observes (would be
  data-snooping per `feedback_bias_discipline.md`).

## Acceptance criteria for this stage's done state

1. Pre-reg yaml committed at `docs/audit/hypotheses/<date>-capital-review-ovnrng100.yaml`
   passing the `prereg-writer-prompt.md` § FORBIDDEN gate.
2. Result doc at `docs/audit/results/<date>-capital-review-ovnrng100.md` with verbatim
   stdout from any runner that executes.
3. Decision verdict: REMAIN_DEPLOY | DOWNSIZE | UNDEPLOY | UNVERIFIED, each with explicit
   triggers traced to pre-reg numeric criteria.
4. evidence-auditor independent-context pass (per `adversarial-audit-gate.md`).
5. If verdict ≠ REMAIN_DEPLOY: separate follow-up stage opened for the actual allocator
   change. This stage does not push allocator state.

## Deliberately out of scope

- RR1.0 sibling lane diligence (open separate stage if needed).
- Other 4 paused MES_CME_PRECLOSE / MES_COMEX_SETTLE / MES_SINGAPORE_OPEN / MES_US_DATA_830
  lanes in current allocation (separate /capital-review).
- New filter discovery to replace OVNRNG_100 (would be a new pathway-A pre-reg).
- DSR re-derivation methodology (Bailey-Lopez de Prado standalone audit; out-of-scope
  per PR #228 result doc).

## Provenance

- evidence-auditor verdict: `PARTIALLY_GROUNDED` on PR #228, recommended /capital-review
  as next-best step.
- PR #228 merge commit: `120882f1` (squash of `528ce0c3`).
- Trailing live state: `docs/runtime/lane_allocation.json` rebalance_date=2026-05-03 —
  re-read at start of fresh session before any analysis.
- Memory anchors:
  - `MEMORY.md` § Validated signals (provenance of PR #228)
  - `feedback_oos_power_floor.md`
  - `feedback_pooled_not_lane_specific.md`
  - `feedback_d4_aistudio_audit_lessons.md`
  - `feedback_audit_thread_dead_end_mine_canonical.md`
