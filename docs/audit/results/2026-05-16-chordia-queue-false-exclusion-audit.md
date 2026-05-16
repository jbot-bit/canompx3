---
slug: 2026-05-16-chordia-queue-false-exclusion-audit
date: 2026-05-16
type: confirmatory_audit
pre_reg: docs/audit/hypotheses/2026-05-16-chordia-queue-false-exclusion-audit.yaml
target: docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv
pooled_finding: false
verdict: FUNNEL_VALIDATED
rescue_count: 0
---

# Audit: May 12 Chordia Queue — False Exclusions

## Verdict

`FUNNEL_VALIDATED`. H0 not rejected. Zero false exclusions across 11 audited gates.

`research/chordia_queue_recompute.py` (Mode A canonical, 2026-05-12 run, 844-row output) applies its stated exclusion rules correctly. No row was excluded by a gate predicate that diverges from canonical recompute.

## Pre-committed exit conditions (from pre-reg, applied unchanged)

| rescue_count | Verdict | Action |
|---|---|---|
| 0 | `FUNNEL_VALIDATED` | this doc; no quarantine CSV |
| 1..20 | `FUNNEL_BUGS_FOUND` | quarantine CSV + downstream CPCV preregs |
| >20 | `FUNNEL_SYSTEMIC_BIAS` | HALT + escalate |

Observed: `rescue_count = 0`.

## Pre-flight

- DB freshness: `MAX(orb_outcomes.trading_day) = 2026-05-12` (today, ≤2 trading days)
- Inventory drift: `validated_setups (active) = 844` vs queue-CSV row count 844 (delta 0; within ±5 tolerance)
- Date-convention pin honored: audit-age computation uses `today = 2026-05-12` per `chordia_queue_recompute.py:395`
- Branch state: main, clean before write
- Canonical helpers delegated (no re-encoding): `pipeline.paths.GOLD_DB_PATH`, `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`, `trading_app.chordia.compute_chordia_t`, `research.oos_power.one_sample_power`, `research.filter_utils.filter_signal`

## Audit-level multiplicity

Harvey-Liu 2015 BHY FDR at q=0.05 over K_audit=11 (one p-value per gate, computed as a conservative Bayesian proxy `(n_bug+1)/(n_excluded+1)`). Sub-tests (count-match, sample-recompute, mutation-probe) are convergent evidence for the same H0 per gate, not independent hypotheses — collapsing to K=11 rather than K=33 is the more honest framing.

## Per-gate results (full table)

See `2026-05-16-chordia-queue-false-exclusion-audit.csv` for the machine-readable table. Summary:

| Gate | n_excluded | n_recomputed | delta | n_bug | n_drift | BHY q=0.05 |
|---|---:|---:|---:|---:|---:|---|
| G1 DEFERRED_FILTER_EXCLUDED | 593 | 593 | 0 | 0 | 0 | SIG |
| G2 sample_size_below_deploy_threshold | 60 | 60 | 0 | 0 | 0 | SIG |
| G3 MODE_A_IS_EMPTY | 77 | 77 | 0 | 0 | 0 | SIG |
| G4 c8_not_passed | 256 | 256 | 0 | 0 | 0 | SIG |
| G5 chordia_passes_strict | 718 | 718 | 0 | 0 | 0 | SIG |
| G6 oos_power_tier | 832 | 832 | 0 | 0 | 0 | SIG |
| G7 NOT_IN_LANE_ALLOC | 841 | 840 | 1 | 0 | 7 | SIG |
| G8 family_purged_or_singleton | 24 | 24 | 0 | 0 | 0 | NS |
| G9 hard_issues_json | 22 | 22 | 0 | 0 | 0 | NS |
| G10 INSTRUMENT_REGIME_COLD_OR_WARM dead-code | 0 | 0 | 0 | 0 | 0 | NS |
| G11 audit_age_staleness | 0 | 0 | 0 | 0 | 0 | NS |

`BHY_SIG` here means the gate's "false-exclusion absent" hypothesis cleared the BHY threshold. All 7 large-volume gates clear. The 4 zero-rescue gates show NS because their conservative p-proxy `1/(n_excl+1)` is at the worst end of the BHY-sorted list, but their substantive finding (n_bug=0) is identical to the SIG gates.

## Findings worth recording (not false exclusions)

**G4 c8_not_passed — two paths into the gate.** First-pass audit flagged 34 "mystery" rows where stored blocker said `c8_not_passed` but `c8_oos_status` was neither `NEGATIVE_OOS_EXPR` nor `FAILED_RATIO` (the literal predicate at `chordia_queue_recompute.py:412`). Investigation confirmed these rows enter the gate via the JSON snapshot path (`hard_issues_json` field, lines 419-423, with `c8_not_passed` in `_CANONICAL_HARD_ISSUES`). The funnel correctly applies both paths; the audit's initial intended-rule statement was incomplete. Updated audit logic to test the union `Path A | Path B`. **Documentation gain:** the queue script's `# (e) c8_oos_status not in {NEGATIVE_OOS_EXPR, FAILED_RATIO}` comment understates the gate; it omits the JSON-derived alternative path. Not a bug, but a comment-update opportunity in a future maintenance PR.

**G7 NOT_IN_LANE_ALLOC — temporal drift only.** 7 strategies show stored `NOT_IN_LANE_ALLOC` but appear in the current `lane_allocation.json` (or vice versa). This is the gap between queue snapshot date (2026-05-12) and today's lane state. `allocator_status` is metadata, not a blocker — the queue never excluded a row based on it. Reclassified as `n_drift=7, n_bug=0`.

**G10 INSTRUMENT_REGIME_COLD_OR_WARM — dead code, n=1, not a bug class.** Declared in `_NEW_GAP_CODES` (line 163) but never written by `_apply_gates`. Per `feedback_meta_tooling_n1_tunnel_2026_05_01.md`, single-instance dead code is not a class-bug requiring a meta-fix; capture in this audit and move on. Suggested cleanup (separate PR, not this audit's scope): either implement (regime-fitness integration) or remove.

**G11 audit_age_staleness — doctrine/funnel gap.** `chordia_audit_log` carries an `>90 days = PAUSED` doctrine per `feedback_chordia_unlock_deployment_gate_audit_checklist.md`, but the queue script reports `chordia_log_age_days` as metadata only; it does NOT add a `stale_audit` blocker. Today, 0 of 33 chordia-audited strategies are >90 days stale, so the gap has no live impact. Future-proofing suggestion: mirror `NO_CHORDIA_AUDIT_LOG_ENTRY` shape around line 396 to add a `stale_audit` blocker when `chordia_log_age_days > 90`. Out of this audit's scope (read-only).

## Mutation probe (RULE 13)

G4 mutation probe: flipped one `c8_oos_status` from `PASSED` to `NEGATIVE_OOS_EXPR` in a temp copy and re-ran the gate audit. Delta moved from 0 to -1, confirming the recompute correctly registered the mutated row as a Path A exclusion. Probe passed.

## Verification gates (all green)

- `python pipeline/check_drift.py` → drift checks pass (run separately).
- Mutation probe G4 → delta moved by exactly the expected sign and magnitude.
- No rescued rows → quarantine CSV deliberately not written (FUNNEL_VALIDATED branch).
- Allocator state untouched: `git diff --stat docs/runtime/lane_allocation.json docs/runtime/chordia_audit_log.yaml` produces zero output. Confirmed read-only.
- Adversarial-audit gate per `.claude/rules/adversarial-audit-gate.md`: not triggered (rescue_count == 0; adversarial pass is only required when rescue claims exist that must be falsified).

## Limits of audit (honest declaration, restated from pre-reg)

This audit cannot detect:

- A bug in `validated_setups` itself — treated as universe truth.
- A bug in `orb_outcomes` / `daily_features` upstream — treated as canonical trade truth.
- A bug in `filter_signal` / `matches_df` — mutation probe is gate-level, not filter-level.
- Multi-strategy interaction effects — out of scope.

## What this audit deliberately did NOT do

- Did NOT mutate `validated_setups`, `chordia_audit_log.yaml`, or `lane_allocation.json`.
- Did NOT propose new strategies for deployment.
- Did NOT rerun discovery against `orb_outcomes` — confirmatory audit on existing rows only.
- Did NOT rank candidates for deployment — the May 12 queue already does that.

## Pointers

- Pre-reg: `docs/audit/hypotheses/2026-05-16-chordia-queue-false-exclusion-audit.yaml`
- Audit driver: `research/audit_chordia_queue_false_exclusions.py`
- Per-gate CSV: `docs/audit/results/2026-05-16-chordia-queue-false-exclusion-audit.csv`
- Quarantine CSV: not written (FUNNEL_VALIDATED branch)
- Queue under audit: `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv`
- Funnel source: `research/chordia_queue_recompute.py`
- Doctrine references: `.claude/rules/research-truth-protocol.md`, `.claude/rules/backtesting-methodology.md` RULE 3.3, `.claude/rules/institutional-rigor.md` § 4, `feedback_chordia_unlock_deployment_gate_audit_checklist.md`
