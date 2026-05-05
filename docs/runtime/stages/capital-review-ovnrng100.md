---
task: Execute the paused capital-review on deployed lane MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100. Resolves the four findings filed by the placeholder doc capital-review-ovnrng100-post-reconcile.md (merged main as PR #234). Produces a verdict (REMAIN_DEPLOY | DOWNSIZE | UNDEPLOY | UNVERIFIED) with explicit triggers traced to numeric criteria. NO writes to validated_setups, lane_allocation.json, chordia_audit_log.yaml, or trading_app/.
mode: IMPLEMENTATION
---

## Scope Lock

- docs/audit/hypotheses/2026-05-05-capital-review-ovnrng100.yaml (new)
- docs/audit/results/2026-05-05-capital-review-ovnrng100.md (new)
- docs/runtime/stages/capital-review-ovnrng100.md (this file, new)

## Blast Radius

- New audit artifact files only. Zero edits to pipeline/, trading_app/, or scripts/.
- Reads canonical layers (orb_outcomes, daily_features, validated_setups) read-only via DuckDB at GOLD_DB_PATH.
- Reads docs/runtime/lane_allocation.json for current DEPLOY status confirmation.
- Reads docs/audit/results/2026-05-05-oos-reconcile-ovnrng100.md for the canonical OOS recompute that triggered this stage.
- Reads docs/audit/results/2026-05-01-target-b-6lane-vestigialness-fresh-audit.md for the filter-vestigialness pattern.
- Reads docs/institutional/literature/{bailey_et_al_2013,bailey_lopez_de_prado_2014,harvey_liu_2015,chan_2013_ch1}.md for grounding.
- Reads docs/institutional/pre_registered_criteria.md for locked thresholds (Criterion 2 MinBTL, Criterion 5 DSR, Criterion 7 sample size, Criterion 8 OOS).
- No writes to canonical state. No allocator changes. No deployment toggles. No row updates.
- The verdict is documentary; implementation of any allocator action requires a SEPARATE follow-up stage with its own scope_lock.

## Why this stage exists

PR #234 (merge `d4860bd0`, 2026-05-05) merged a placeholder DESIGN-mode doc that paused the capital-review on this deployed lane after evidence-auditor flagged four concerns on the merged PR #228 OOS reconciliation. The placeholder explicitly routed implementation to "a fresh Opus session". This stage is that session.

## Pre-reg requirement

Per `.claude/rules/research-truth-protocol.md` § Phase 0 + `prereg-writer-prompt.md`:

- testing_mode: "individual"
- pathway: "B" (theory-driven K=1 confirmatory diligence; not new family search)
- theory citations: Bailey et al 2013 Theorem 1 (MinBTL), Bailey-LdP 2014 Equation 2 (DSR), Harvey-Liu 2015 p.17 (OOS power-floor), Chan 2013 Ch 1 p.4 (canonical-machinery delegation), Carver 2015 Ch 9-10 (capital allocation under uncertainty)
- Numeric kill criteria. No "investigate" or "reconsider" verbs.
- Mode A holdout (2026-01-01 sacred); no tuning against OOS.
- No upstream-K framing; this is K=1 not search-family.
- Lock before run.

## Forbidden in this stage

- Writes to `validated_setups`, `experimental_strategies`, `chordia_audit_log.yaml`, `lane_allocation.json`.
- Deployment toggles (lane stays as currently allocated until next rebalance, which evaluates the verdict).
- Updating `validated_setups.oos_exp_r=+0.2029` to canonical +0.1658 — that requires the `strategy_validator` path with its own pre-reg, not this audit.
- Threshold modification on OVNRNG_100 based on what the audit observes (data-snooping per `feedback_bias_discipline.md`).
- "Just ship it" recommendations — verdict must be one of the four locked options, traced to numeric criteria.

## Acceptance criteria for done state

1. Pre-reg yaml committed at `docs/audit/hypotheses/2026-05-05-capital-review-ovnrng100.yaml` passing the `prereg-writer-prompt.md` § FORBIDDEN gate (numeric kill criteria, theory citations, no waiver wording).
2. Result doc at `docs/audit/results/2026-05-05-capital-review-ovnrng100.md` with verbatim canonical-layer query output for each of the four findings.
3. Decision verdict: REMAIN_DEPLOY | DOWNSIZE | UNDEPLOY | UNVERIFIED, each with explicit triggers traced to numeric criteria.
4. If verdict ≠ REMAIN_DEPLOY: separate follow-up stage opened for the actual allocator change. THIS stage does not push allocator state.
5. `python pipeline/check_drift.py` passes.

## Provenance

- Parent placeholder doc: `docs/runtime/stages/capital-review-ovnrng100-post-reconcile.md` (merged in PR #234, commit `d4860bd0`).
- Canonical OOS recompute: `docs/audit/results/2026-05-05-oos-reconcile-ovnrng100.md` (PR #228, commit `120882f1`).
- Strict-unlock prior: `docs/audit/results/2026-05-02-mnq-comex-ovnrng100-rr15-chordia-unlock-v1.md`.
- Vestigialness pattern reference: `docs/audit/results/2026-05-01-target-b-6lane-vestigialness-fresh-audit.md`.
- Live allocator state: `docs/runtime/lane_allocation.json` rebalance_date=2026-05-03.
- Memory anchors:
  - `MEMORY.md` § Validated signals (PR #228 provenance)
  - `feedback_oos_power_floor.md`
  - `feedback_absolute_threshold_scale_audit.md`
  - `feedback_audit_thread_dead_end_mine_canonical.md`
  - `feedback_d4_aistudio_audit_lessons.md`
