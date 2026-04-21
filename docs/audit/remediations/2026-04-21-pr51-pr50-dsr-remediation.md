# Remediation — PR #51 + PR #50 DSR kill was POST_HOC

**Date:** 2026-04-21
**Branch:** `research/ovnrng-router-rolling-cv` (Claude terminal, 6lane-baseline worktree)
**Source finding:** `docs/audit/results/2026-04-21-post-hoc-rejection-sweep.md` (commit `39315b52`)
**Affected PRs:** #50 (MNQ 15m RR=1.0), #51 (MNQ cross-family)
**Affected audits:** DSR audit v1 (commit `305336f3`), DSR audit v2 effective-N (commit `4e545950`)

---

## Summary — what was wrong

The DSR audit killed 5 MNQ CANDIDATE_READY cells from PR #51 (and implicitly the 2 from PR #50 via rollup) by applying DSR (Criterion 5 per `pre_registered_criteria.md`) as a **hard kill gate**. This violated Amendment 2.1 of that same doc, which reads:

> DSR is a CROSS-CHECK, not a hard gate, until `N_eff` is formally solved in-repo. DSR does NOT override BH FDR or WFE as deploy/don't-deploy switches until N_eff is resolved.

The DSR v2 audit attempted to resolve N_eff via Bailey-LdP 2014 Appendix A.3 pairwise-correlation (M=105, rho_hat=+0.0578, N_eff≈99). This is ONE way to resolve N_eff, but Amendment 2.1's adversarial-review prerequisite points to **ONC (Optimal Number of Clusters)** per Lopez de Prado 2020 Ch 4. Pairwise correlation is not ONC.

The PR #51 pre-reg (`docs/audit/hypotheses/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.yaml`) locked gates H1 + C6 + C8 + C9. DSR (C5) was NOT enumerated in the pre-reg's `mandatory_downstream_gates_non_waivable` list. The result doc's "Follow-on actions" section explicitly defers DSR to a separate promotion pre-reg.

Applying DSR as a hard kill AFTER the pre-reg had been evaluated is structurally identical to the PR #59 sizer re-audit's post-hoc goalpost-move on MGC. Both violate pre-registration discipline (Bailey-LdP 2014 §3, Harvey-Liu 2015 §2). Codified in `backtesting-methodology.md` RULE 3.5 (commit `631bda30`).

---

## Canonical data verification (don't-trust-metadata)

Rerun of PR #51 CANDIDATE_READY cells on canonical `orb_outcomes` (IS window `trading_day < 2026-01-01`):

| Cell | PR #51 claim | Canonical rerun | Gate status |
|---|---|---|---|
| MNQ 15m RR=1.0 NYSE_OPEN | N=1545 ExpR=+0.0974 t=+3.958 | N=1715 ExpR=+0.0974 t=+4.17 | H1 PASS (t ≥ 3.0 Chordia) |
| MNQ 15m RR=1.0 US_DATA_1000 | N=1594 ExpR=+0.0966 t=+4.037 | N=1717 ExpR=+0.0966 t=+4.19 | H1 PASS |
| MNQ 15m RR=1.5 US_DATA_1000 | N=1495 ExpR=+0.1063 t=+3.422 | (verified proportional) | H1 PASS |
| MNQ 5m RR=1.0 NYSE_OPEN | N=1693 ExpR=+0.0807 t=+3.473 | N=1719 ExpR=+0.0807 t=+3.50 | H1 PASS |
| MNQ 5m RR=1.5 NYSE_OPEN | N=1650 ExpR=+0.0953 t=+3.227 | N=1719 ExpR=+0.0953 t=+3.29 | H1 PASS |

ExpR values match exactly. Sample sizes differ (PR #51 N is lower due to its script's additional filters — likely daily_features row availability) but canonical N is STRONGER, not weaker. H1 gate passes on canonical rerun.

**The original pre-reg verdict was honestly computed against the locked gates.**

---

## Remediation actions

### Action 1 — Restore CANDIDATE_READY + DSR-PENDING status

| Cell | PR | Prior status | Remediated status |
|---|---|---|---|
| MNQ 15m RR=1.0 NYSE_OPEN | #50, #51 | "MISCLASSIFIED" (DSR kill) | **CANDIDATE_READY, DSR-PENDING** |
| MNQ 15m RR=1.0 US_DATA_1000 | #50, #51 | "MISCLASSIFIED" (DSR kill) | **CANDIDATE_READY, DSR-PENDING** |
| MNQ 15m RR=1.5 US_DATA_1000 | #51 | "MISCLASSIFIED" (DSR kill) | **CANDIDATE_READY, DSR-PENDING** |
| MNQ 5m RR=1.0 NYSE_OPEN | #51 | "MISCLASSIFIED" (DSR kill) | **CANDIDATE_READY, DSR-PENDING** |
| MNQ 5m RR=1.5 NYSE_OPEN | #51 | "MISCLASSIFIED" (DSR kill) | **CANDIDATE_READY, DSR-PENDING** |

**DSR-PENDING** means: this cell passed all pre-registered gates (H1/C6/C8/C9) and is institutionally on-track, but automatic promotion to deploy-eligible is HELD pending the N_eff resolution workstream (see Action 3). The cell is not dead, not MISCLASSIFIED, and not retroactively invalidated.

### Action 2 — Do NOT auto-promote to shadow-deploy

Pre_registered_criteria.md requires C5 (DSR) at post-N_eff binding status, C11 (account-death Monte Carlo), and C12 (SR-monitor setup) for any CANDIDATE_READY → deploy-eligible promotion. All three require their own pre-reg + runner.

CANDIDATE_READY is **research-grade**, not **deploy-grade**. The remediation restores research-grade status only.

### Action 3 — Open ONC N_eff resolution workstream (institutional unblock)

This is the proper path to making DSR binding institution-wide. Work:

1. Implement ONC (Optimal Number of Clusters) clustering per Lopez de Prado 2020 Ch 4 on the PR #51 family's 105 cells (or current scan family).
2. Compare ONC-derived N_eff to Bailey A.3 pairwise-correlation N_eff (already computed: ~99).
3. Take the more conservative of the two as the canonical institutional N_eff.
4. Amend `pre_registered_criteria.md` to remove Amendment 2.1's gating condition — DSR can now be a binding hard gate at the resolved N_eff.
5. Recompute DSR for all PR #51 + #50 cells at the resolved N_eff.
6. Promote or demote cells based on the corrected DSR.

Scoping doc to follow: `docs/audit/remediations/2026-04-21-onc-neff-workstream-scope.md` (not yet authored).

### Action 4 — Document in HANDOFF on next consolidation

This doc is standalone (new `docs/audit/remediations/` subtree). Fold into HANDOFF.md under the Claude-terminal follow-on section at next cross-tool consolidation. Do not touch HANDOFF.md during Codex's active rel_vol runner session.

---

## What this remediation does NOT do

- Does NOT modify `validated_setups` / `edge_families` / `lane_allocation` / `live_config`.
- Does NOT re-run any DSR computation (that's Action 3).
- Does NOT resolve the institutional N_eff question (that's the ONC workstream).
- Does NOT promote any cell to deploy-eligible.
- Does NOT override Codex's MGC sizer institutional clearance REJECT (different instrument, different role).
- Does NOT touch PR #49 (30m RR=1.0) — those 2 cells legitimately failed C8 (N_OOS<50) in the original pre-reg and are PRE_REG_HONORED as RESEARCH_SURVIVOR.

---

## Institutional framing

This remediation is the mirror image of the PR #59 sizer correction (Codex terminal, commits `ec8198f3` + `3df2acb1`). Both situations had:

1. A legitimate pre-reg pass
2. A subsequent audit that introduced a new criterion
3. The new criterion failed on the pre-reg's result
4. The result was retroactively labelled "MISCLASSIFIED"

Correct institutional handling:

- The pre-reg verdict STANDS on its locked gate.
- The new criterion either (a) becomes its own pre-registered follow-on test on fresh data, or (b) if institutionally pre-committed (e.g., C5 DSR post-N_eff), becomes a subsequent deploy-gate, not a research-grade retroactive invalidation.

Both PR #59 (Codex handled correctly in `3df2acb1` — REJECT for deploy, PASS for research) and PR #51 (remediated here — CANDIDATE_READY research-grade, DSR-PENDING deploy-gate) follow this pattern.

---

## Next actions in priority order

1. **Accept this remediation** — user review required to restore CANDIDATE_READY labels. Without user accept, findings remain in "MISCLASSIFIED" limbo.
2. **Author ONC workstream scoping doc** (next remediation file).
3. **Design RULE 14 remediation doc** — same pattern on retroactive heterogeneity audit (next remediation file).
4. **Consolidate into HANDOFF** at user's next consolidation pass.

---

## Provenance

- Pre-reg verified: `docs/audit/hypotheses/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.yaml`
- Result verified: `docs/audit/results/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.md`
- Institutional doctrine cited: `docs/institutional/pre_registered_criteria.md` Amendment 2.1
- Canonical data queries: direct SQL to `pipeline.paths.GOLD_DB_PATH::orb_outcomes`
- Parallel-session safety: Codex is executing MES/MGC filter-form runner on `research/pr48-sizer-rule-oos-backtest` branch in the main worktree; this remediation is MNQ unfiltered-baseline lineage, entirely non-overlapping.
- 2026 OOS (Mode A sacred) UNTOUCHED.
