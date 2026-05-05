---
pooled_finding: false
audit_target: "Self-correction addendum to 2026-05-05-capital-review-ovnrng100.md (PR #236)"
auditor_context: opus-4-7-fresh-self-audit
canonical_layers: [orb_outcomes, daily_features, validated_setups]
verdict: "CONDITIONAL_REMAIN_DEPLOY pending filter-reparametrisation pre-reg"
parent_claims:
  - docs/audit/results/2026-05-05-capital-review-ovnrng100.md
  - docs/audit/results/2026-05-05-oos-reconcile-ovnrng100.md
supersedes_claim: "DOWNSIZE verdict in 2026-05-05-capital-review-ovnrng100.md (PR #236)"
---

# Self-correction addendum — capital-review OVNRNG_100

**Parent doc:** `docs/audit/results/2026-05-05-capital-review-ovnrng100.md` (PR #236, commit `58e92894`)
**Pre-reg of parent:** `docs/audit/hypotheses/2026-05-05-capital-review-ovnrng100.yaml`
**Date:** 2026-05-05 (same day as parent — caught immediately on user-prompted post-merge audit)

## Scope

This is a self-correction addendum to PR #236. The user prompted a post-merge tunnel-vision audit ("Stop. Prove this is true before using it."). That audit surfaced a framing error in the parent doc that survived both my self-review and the independent fresh-context evidence-auditor pass. No allocator action was taken on the parent verdict (audit was documentary), so the cost of correction is zero.

This addendum does NOT re-run any data — it reframes the verdict on the existing canonical-layer evidence.

## What was wrong with the parent verdict

The parent verdict was **DOWNSIZE**, traced to F1 VESTIGIAL OR F3 DEPLOYMENT_MATH_BROKEN, AND F4 not outright zero/negative. The trace itself is procedurally correct against the pre-reg's locked taxonomy.

**The error is in the FRAMING the pre-reg locked**, not in the trace through it. The frame conflated three structurally distinct questions:

1. **Promotion-math integrity** — is the historical promotion's MT correction defensible? (F3: NO. MinBTL violated by 1,272x. OOS power useless.)
2. **Filter parameterisation drift** — is the OVNRNG_100 absolute-points threshold still expressing what it was promoted to express? (F1: NO. Fire rate drifted 4.3% → 91.7%.)
3. **Forward EV** — is the lane plausibly profitable from here? (NOT TESTED CLEANLY in parent doc.)

The parent doc treated (1) + (2) as evidence FOR DOWNSIZE. But (1) is a sunk-cost diagnostic about historical promotion confidence — it does not refute future EV. (2) is a parameterisation issue that motivates re-tuning, not de-allocating. The actually-load-bearing question for capital is (3) — and the parent never tested it cleanly.

## What the canonical-layer evidence actually says about (3) — forward EV

Reading the F1 per-year table in the parent doc, the 2026 OOS row carries information the parent dismissed:

| Year | N_universe | Universe ExpR | N_filter | Filter ExpR | **Lift (filter − universe)** |
|---|---:|---:|---:|---:|---:|
| 2026 (Jan-Apr) | 72 | +0.0687 | 66 | +0.1658 | **+0.0971R** |

Universe is positive in OOS. Filter still adds ~+0.10R lift in OOS. The parent characterised the +0.166R OOS_ExpR as "essentially the unfiltered base rate" — but the unfiltered base rate is +0.069R, and the filter ExpR is **2.4x the unfiltered ExpR**. That is not vacuous. The filter still discriminates; it just admits more days than at promotion.

Live trailing N=153 ExpR=+0.24R t=2.52 is the strongest forward-EV evidence available. RULE 3.3 power tier is DIRECTIONAL_ONLY (63% power) — it is informational, not confirmatory, but **the direction is positive** with the highest sample size among the three forward-evidence streams (OOS=66, live=153, Q1-2026=58).

## Three structurally-different framings the parent doc never weighed

### Framing A: "Is the lane's edge alive (filter-agnostic)?"

The parent treated F1 as "filter is dead". The honest read is "filter parameterisation is mis-tuned for current scale, but a +0.10R OOS lift over a positive universe ExpR means the EDGE the filter targets may still be alive." This is a re-parameterisation question, not a de-allocation question.

### Framing B: "Standalone vs filter vs allocator — which layer is broken?"

The lane is three layers:
- Signal layer (E2 break entry on COMEX_SETTLE) — universe ExpR positive 2026, no evidence of signal death
- Filter layer (OVNRNG_100 absolute-points) — F1 confirms this is mis-tuned
- Allocator layer (session-regime HOT gate) — operating as designed; not the issue

The parent verdict (DOWNSIZE) implicitly attacks all three layers when only the filter layer is broken. The right action is filter REPARAMETRISATION (relative-vol threshold instead of absolute-points), not lane-DOWNSIZE.

### Framing C: "Single-lane vs portfolio class"

`docs/audit/results/2026-05-01-target-b-6lane-vestigialness-fresh-audit.md` confirms the **same vestigial-filter pattern across 5 of 6 deployed lanes** (≥75% 2026 fire rate). Two of the three currently active lanes (OVNRNG_100 and the other deployed-portfolio members) share this class.

Single-lane DOWNSIZE leaves portfolio risk concentrated on equally-vestigial sister lanes. The honest portfolio-class action is either uniform sizing-down across all vestigial lanes OR a portfolio-class filter-reparametrisation pre-reg covering all 3 active lanes simultaneously.

The parent doc cited Carver Ch 11 (portfolios) but never did the cross-lane comparison the citation implies.

## Verdict

**CONDITIONAL_REMAIN_DEPLOY pending filter-reparametrisation pre-reg.**

The forward EV is plausibly alive at observed exposure:
- 2026 OOS universe ExpR positive (+0.069R N=72, descriptive)
- 2026 OOS filter lift positive (+0.097R over universe)
- Live trailing N=153 +0.24R t=2.52 (DIRECTIONAL_ONLY tier, positive)

The promotion-math diagnostics (F3 BINDING triggers: MinBTL 1,272x, OOS power 0.07/0.32) are sunk-cost confidence statements about the LEGACY promotion path — they do not refute forward EV, they only refute "we can be 95%+ sure this lane has edge based on backtests".

The filter-parameterisation drift (F1) motivates a **relative-vol-threshold reparametrisation pre-reg** (covering this lane and the 2 sister vestigial lanes), not a lane-level DOWNSIZE.

A DOWNSIZE that only touches this lane — while leaving the other 2 active lanes at full exposure with the same vestigial-filter class — is the wrong shape: it does not improve portfolio risk, it just shifts concentration.

## What this addendum changes

1. **Parent verdict DOWNSIZE → CONDITIONAL_REMAIN_DEPLOY pending filter-reparametrisation pre-reg.** Decision-ledger entry `capital-review-ovnrng100-downsize-2026-05-05` superseded by this addendum's verdict.
2. **No allocator action.** Lane stays at currently-allocated exposure until next rebalance (the parent verdict had the same forbidden_actions clause — nothing changes operationally).
3. **The follow-up stage queue is REORDERED.** Instead of "allocator-side downsize implementation", the new highest-EV follow-up is **portfolio-class filter-reparametrisation pre-reg** covering OVNRNG_100, VWAP_MID_ALIGNED, and COST_LT12 (the 3 active lanes' filters). The "per-lane decay tripwire" and "generalised absolute-threshold scale-drift sweep" stages remain queued.

## What this addendum does NOT change

- F2 NOT_A_VIOLATION verdict (PR #228 reframe is still a documented spec bug, not a no-rescue rescue) — unchanged.
- F1 fire-rate measurement (4.3% → 91.7%) — measurement is correct; only the IMPLICATION drawn from it changes (re-parameterise vs DOWNSIZE).
- F3 numeric BINDING triggers (MinBTL 1,272x, OOS power 0.07/0.32) — measurements unchanged; only their relevance to FORWARD verdict downgraded (these are sunk-cost diagnostics).
- F4 quarterly trend table — unchanged.
- The Amendment 2.1 DSR-as-cross-check correction landed in the parent doc — still correct.
- All canonical-layer queries — re-verified by the post-merge tunnel-vision audit; numbers reproduce exactly.

## Why this is not a "post-hoc rescue"

The pre-reg yaml's verdict_taxonomy locked four options (REMAIN_DEPLOY | DOWNSIZE | UNDEPLOY | UNVERIFIED). The user-prompted post-merge audit identified that **the taxonomy itself was misframed** — REMAIN_DEPLOY's trigger ("F1 NOT VESTIGIAL AND F3 DSR ≥ 0.50 AND F4 NOT LIVE_DECAY") locks the verdict to be conditional on backward-looking diagnostics rather than forward-EV evidence. The taxonomy's REMAIN_DEPLOY trigger requires F3 DSR ≥ 0.50 — but Amendment 2.1 explicitly demoted DSR to cross-check, making this trigger structurally unreachable.

The pre-reg locked a frame that REMAIN_DEPLOY could not satisfy. That is a pre-reg specification bug. Per the no-rescue rule, the pre-reg yaml stays locked as written; this addendum documents the framing correction in the result doc, exactly as PR #228 documented its own pre-reg specification bug (criterion 4 scratch-policy mismatch).

The parallel is intentional: the discipline used to handle PR #228's spec bug is the same discipline applied here.

## Reproduction

This addendum runs no new SQL. It re-reads the parent doc's per-year F1 table and the parent doc's live-trailing N=153 row, both of which were re-verified from canonical layers in the post-merge tunnel-vision audit (same session as this addendum). All numbers cited here are extracted from the parent doc, not re-computed.

To re-verify the load-bearing numbers in this addendum, run the four SQL blocks documented in the parent doc's reproduction section against `pipeline.paths.GOLD_DB_PATH`. The 2026 universe ExpR (+0.069 N=72), filter ExpR (+0.166 N=66), and live trailing N=153 (ExpR +0.24 t=2.52) reproduced exactly in the post-merge audit run on 2026-05-05.

## Caveats and limitations

- **Same author as parent doc.** This addendum is a self-correction by the same author who wrote the parent verdict. The framing-error catch came from a user-prompted tunnel-vision audit, not from independent fresh-context review. Bias is acknowledged. An independent fresh-context re-audit on the addendum is queued (follow-up #2).
- **Forward EV is INFERRED, not MEASURED.** "Plausibly alive at observed exposure" rests on three pieces of evidence (2026 OOS universe positive, 2026 OOS filter lift positive, live trailing positive) — all DIRECTIONAL_ONLY tier per RULE 3.3 power floor. None individually clear confirmatory thresholds. The case is "best directional read on available evidence", not statistical proof.
- **The CONDITIONAL_REMAIN_DEPLOY verdict is NOT in the parent pre-reg taxonomy.** Pre-reg locked four options (REMAIN_DEPLOY | DOWNSIZE | UNDEPLOY | UNVERIFIED). This addendum effectively introduces a fifth ("CONDITIONAL_REMAIN_DEPLOY pending re-parametrisation") which is closer to "REMAIN_DEPLOY with a follow-up condition" than to any of the four originally locked. The pre-reg yaml is left as-locked per no-rescue rule; the addendum documents the framing correction without modifying the yaml. Future capital-review pre-regs should include a CONDITIONAL_* taxonomy class.
- **Portfolio-class extrapolation is INFERRED.** Claim that "single-lane DOWNSIZE leaves portfolio risk concentrated on equally-vestigial sister lanes" is based on the 2026-05-01 6-lane vestigialness audit, not on a fresh portfolio-EV run. The cross-lane comparison the parent doc's Carver-Ch-11 citation implied was not done in either parent or addendum.
- **Operational status unchanged.** Lane stays at currently-allocated exposure (`status=DEPLOY`, regime HOT) until next rebalance. This addendum does not push allocator state any more than the parent did.
- **Filter-reparametrisation pre-reg has not been written.** It is queued as the highest-EV follow-up but not started. If the user wants to act on the revised verdict, that pre-reg is the next step.

## Disconfirming Checks

- **Could the 2026 OOS filter lift (+0.097R) be noise?** Yes — N=66 with sd≈1.17 gives SE≈0.144, so 95% CI on the lift overlaps zero. The lift is descriptive evidence, not confirmatory. RULE 3.3 power tier on this OOS slice is STATISTICALLY_USELESS — the same finding the parent F3 used to argue DEPLOYMENT_MATH_BROKEN. Honest framing: the OOS slice cannot refute either positive or zero edge; the +0.097R lift is the best directional read available, not statistical proof.
- **Could live trailing N=153 +0.24R be a streak that's about to reverse?** Possibly. The parent doc's F4 quarterly trend table shows monotonic decay through Q1 2026 (+0.08R) — the +0.24R 12-month figure is heavily weighted by Q1-Q3 2025 contribution. Q2 2026 partial (+0.79R N=8) is too small to read either way. The forward direction is not settled.
- **Could the universe ExpR be positive in 2026 only because of regime luck?** N=72 t≈0.50 — yes, this could be regime-conditional noise. The +0.069R is descriptive, not statistically distinguishable from zero.
- **What if F1 is the structural truth and forward edge IS dead?** The data shows: universe positive (small sample), filter lift positive (small sample), live positive (DIRECTIONAL_ONLY power). None of these individually rule out "edge is dead and we're seeing noise". The CONDITIONAL_REMAIN_DEPLOY verdict explicitly acknowledges this — it is "plausibly alive at observed exposure" pending the filter-reparametrisation pre-reg, not "alive, full-confidence".
- **Could DOWNSIZE be the right answer after all?** Yes, if the filter-reparametrisation pre-reg comes back with no relative-vol threshold that beats the 2026 OOS lift, then the lane should be DOWNSIZED — the absolute-points threshold is dead and there's no replacement. This addendum does not preclude future DOWNSIZE; it just says "DOWNSIZE NOW is premature; gather the re-parametrisation evidence first."

## Provenance

- Parent PR: #236 commit `58e92894` (merged 2026-05-05)
- Tunnel-vision audit prompt: user message 2026-05-05 ("Stop. Prove this is true before using it.")
- Re-verification: every load-bearing number in the parent doc reproduced from canonical layers in the same session
- Author bias: same author as parent (acknowledged limitation; an independent re-audit is queued)

## Follow-up stages (REORDERED — supersedes parent doc § "Recommended downstream artifacts")

1. **Portfolio-class filter-reparametrisation pre-reg** — covers OVNRNG_100, VWAP_MID_ALIGNED, COST_LT12. Tests relative-vol thresholds (e.g., `overnight_range >= prev_atr_20 * X`) against the same lane universes. Single pre-reg, K=3 family, theory citation: scale-stability per `feedback_absolute_threshold_scale_audit.md` + Carver Ch 9-10 vol-targeting framework.
2. **Independent re-audit of THIS addendum** — the parent + addendum were both written by me. Run an independent fresh-context evidence-auditor pass on the addendum to verify the framing correction holds before treating CONDITIONAL_REMAIN_DEPLOY as the operative verdict.
3. **Per-lane decay tripwire** — unchanged from parent doc.
4. **Generalised absolute-threshold scale-drift sweep** — unchanged from parent doc.

## Memory anchors

- `MEMORY.md` § Validated signals (PR #228 + PR #236 provenance)
- `feedback_oos_power_floor.md`
- `feedback_absolute_threshold_scale_audit.md`
- `feedback_audit_thread_dead_end_mine_canonical.md` — applies here too: parent verdict was a tunnel-vision narrowing
- New (to be added if user authorizes): `feedback_pre_reg_taxonomy_can_be_misframed.md` — pre-reg verdict_taxonomy locks the answer space; if the locked options don't include the right answer, the trace through the taxonomy is procedurally correct but substantively wrong
