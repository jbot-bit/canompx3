# Plan — Auto Deployment Review Stage (deferred)

**Status:** DEFERRED (saved 2026-05-21 per user instruction "save b plan for later")
**Estimated effort:** 4-6 hours, multi-stage (full IMPLEMENTATION staging required)
**Adversarial-audit gate:** REQUIRED (touches lane_allocation.json write path or operator-decision routing)

---

## Purpose

Close the operator-bottleneck gap surfaced by the 2026-05-21 fast-lane funnel diagnosis:

- `docs/runtime/fast_lane_status.yaml` shows 38 MNQ entries at stage `HEAVYWEIGHT_COMPLETE` with `next_action_token=operator_deployment_decision`.
- The fast-lane / cherry-pick / Chordia-bridge automation runs the conveyor belt up to the deploy gate, but the gate itself is a human operator with no decision-support automation.
- Result: cleared edges sit idle for days-to-weeks while waiting for manual review.

This stage closes the gap with a `scripts/tools/auto_deployment_review.py` that produces an institutional-grade ranked verdict sheet (semi-auto) and, in a later sub-stage, optionally writes to `lane_allocation.json` (full-auto) behind an opt-in flag.

## Scope (proposed split into 3 sub-stages)

### Sub-stage A — Verdict ranker (read-only)
- Input: `fast_lane_status.yaml`, `chordia_audit_log.yaml`, `lane_allocation.json`, `validated_setups` (DB).
- For each entry at stage `HEAVYWEIGHT_COMPLETE` or `ENRICHED`:
  - Look up Chordia verdict (PASS_CHORDIA / FAIL_CHORDIA / PARK).
  - Look up C8 OOS status from validated_setups.
  - Compute fresh correlation via `trading_app.lane_correlation.check_candidate_correlation` against current deployed set (per profile).
  - Compute portfolio-EV delta if rotation executed (replace lowest-EV incumbent if displacing).
  - Emit per-row: DEPLOY / DISPLACE-WITH-RATIONALE / PARK / KILL verdict + structured rationale.
- Output: `docs/runtime/auto_deployment_review_<date>.md` (operator-decision sheet) and a side-channel JSON for sub-stage B.
- Acceptance: every entry classified; rationale cites canonical sources (no inlined numbers).

### Sub-stage B — Mutation gate (semi-auto)
- Reads sub-stage A JSON.
- Requires explicit `--apply` flag and operator-confirmation prompt before any `lane_allocation.json` mutation.
- Writes audit-trail row to `docs/runtime/auto_deployment_review_log.yaml` (append-only).
- Refuses to apply if drift check fails or correlation gate fails on the candidate.
- Acceptance: dry-run mode default; --apply requires confirm; full reversibility via `git checkout`.

### Sub-stage C — Full-auto mode (opt-in only)
- Behind `AUTO_DEPLOY_ENABLED=1` env var (defaults off).
- Adds Shiryaev-Roberts-style drift-monitor on PASS_CHORDIA deflation_headroom to refuse auto-deploys when the heavyweight-gate trial-budget is exhausted (Bailey-López de Prado 2014 MinBTL).
- Acceptance: opt-in only; full audit-trail; passes adversarial-audit gate.

## Inputs

| Input | Source |
|---|---|
| Fast-lane funnel state | `docs/runtime/fast_lane_status.yaml` |
| Heavyweight Chordia verdicts | `docs/runtime/chordia_audit_log.yaml` |
| Current allocation | `docs/runtime/lane_allocation.json` |
| Validated setups (C8, OOS) | `gold.db` → `validated_setups` |
| Correlation engine | `trading_app.lane_correlation` |
| Profile lane definitions | `trading_app.prop_profiles.get_profile_lane_definitions` |
| Holdout policy | `trading_app.holdout_policy` |

## Outputs

- `docs/runtime/auto_deployment_review_<date>.md` — operator decision sheet (always written).
- `docs/runtime/auto_deployment_review_<date>.json` — side-channel for sub-stage B.
- `docs/runtime/auto_deployment_review_log.yaml` — append-only audit trail of applied rotations.
- `lane_allocation.json` — mutated only by sub-stage B/C with --apply.

## Risks

- **Capital-class:** any path that writes `lane_allocation.json` is real-capital-routing code → adversarial-audit gate required → full IMPLEMENTATION staging, never TRIVIAL.
- **Selection-bias regression:** auto-ranking off `expectancy_r` would resurrect the very bug `feedback_high_r_inventory_comes_from_chordia_not_raw_expr.md` codifies → must rank off PASS_CHORDIA + C8 + correlation, never raw IS ExpR.
- **MinBTL trial-budget exhaustion:** auto-deploying without checking deflation_headroom risks deploying selection artifacts → must read `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` extract and gate on remaining headroom.
- **Manual-append ordering trap:** auto-mode must refuse to deploy if `chordia_audit_log.yaml` was modified after the last rebalance without operator acknowledgement → `feedback_chordia_audit_log_manual_append_ordering_n1_2026_05_21.md`.

## Doctrine grounding

- `feedback_high_r_inventory_comes_from_chordia_not_raw_expr.md` — the rule the ranker must follow.
- `feedback_allocator_gate_class_pattern_fail_open.md` — every new gate needs a paired drift check.
- `feedback_chordia_audit_log_manual_append_ordering_n1_2026_05_21.md` — ordering trap to defend against.
- `feedback_canonical_inline_copy_parity_bug_class.md` — never inline thresholds; parse canonical doctrine doc at runtime.
- `docs/institutional/pre_registered_criteria.md` — the criteria the ranker must enumerate.
- `.claude/rules/institutional-rigor.md` § 4 (delegate to canonical) and § 6 (no silent failures).
- `.claude/rules/adversarial-audit-gate.md` — gate this stage must pass.

## Acceptance (whole stage)

1. Sub-stage A runs end-to-end on current canonical state and produces a decision sheet citing each entry's canonical lookup (Chordia verdict line, C8 status, fresh correlation report).
2. Drift check counts (1) sub-stage A's verdict file freshness vs canonical source mtimes, and (2) sub-stage B's audit log append-monotonicity.
3. Sub-stage B refuses --apply on any candidate where the canonical correlation engine reports `gate_pass=False` (delegated, not re-encoded).
4. Sub-stage C disabled by default; integration test confirms the env-var gate.
5. Adversarial-audit gate dispatched after sub-stage B lands, before sub-stage C is enabled.

## Why this is parked, not built now

User explicitly chose (a) the one-shot displaced-rotation analyzer for tonight. The auto-deployment-review stage is the structural answer to "how do we get them through faster?" but it's a multi-session institutional-rigor build requiring:
- Adversarial-audit gate dispatch
- Capital-class scope_lock approval
- ≥3 paired drift checks
- Sub-stage decomposition

Pick this up when (a) the one-shot analyzer has surfaced ≥3 actionable rotations and confirmed the diagnostic shape, OR (b) the operator-deployment-decision backlog exceeds 50 entries.
