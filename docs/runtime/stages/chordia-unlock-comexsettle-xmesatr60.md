---
task: chordia-unlock-comexsettle-xmesatr60
mode: IMPLEMENTATION
phase: 1/4
spec: docs/runtime/action-queue.yaml#chordia_audit_unlock_pass_chordia_strategies
created: 2026-05-04
agent: claude
scope_lock:
  - docs/audit/hypotheses/2026-05-04-mnq-comexsettle-xmesatr60-chordia-unlock-v1.yaml
  - docs/audit/results/2026-05-04-mnq-comexsettle-xmesatr60-chordia-unlock-v1.md
  - docs/audit/results/2026-05-04-mnq-comexsettle-xmesatr60-chordia-unlock-v1.csv
  - docs/runtime/chordia_audit_log.yaml
  - docs/runtime/action-queue.yaml
acceptance:
  - prereg_locked: hypothesis YAML exists, mirrors PRECLOSE precedent, total_expected_trials=1, no theory grant claimed
  - replay_run: research/chordia_strict_unlock_v1.py executed against the prereg; result md+csv produced
  - audit_row_appended: chordia_audit_log.yaml has a row with strategy_id MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60, verdict from measured outcome (not assumed)
  - drift_clean: pipeline/check_drift.py exit 0 (Check #134 still passes)
  - queue_closed: action-queue.yaml chordia_audit_unlock_pass_chordia_strategies status=closed with override_note citing all 8 originally identified strategies' final dispositions
---

# Stage: Chordia unlock — COMEX_SETTLE X_MES_ATR60 sibling

## Why this stage exists

P1 action queue item `chordia_audit_unlock_pass_chordia_strategies` lists 8
originally identified PASS_CHORDIA-without-audit strategies. As of 2026-05-02,
6/8 are resolved in `chordia_audit_log.yaml` and 1/8 (`MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075`)
has been dropped from `validated_setups` by a pipeline rebuild — implicit close.

One real gap remains: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60`
(deployable, N=673, t_chordia=4.322 vs threshold 3.79). Allocator currently
fails-closed to MISSING (safe) but doctrine ledger is incomplete.

The PRECLOSE sibling (`MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60`) was
audited on 2026-05-02 with verdict PARK due to OOS sign-flip. Its prereg
explicitly excluded the COMEX_SETTLE sibling in `out_of_scope` — deliberate
deferral, not omission.

## Approach

Bounded conditional-role audit using the existing route-contract runner.
No new code; only one new prereg YAML and one new audit log row.

1. **Phase 1** — Draft prereg `docs/audit/hypotheses/2026-05-04-mnq-comexsettle-xmesatr60-chordia-unlock-v1.yaml`
   mirroring the PRECLOSE precedent. Swap session, strategy_id, holdout
   boundary unchanged at 2026-01-01.
2. **Phase 2** — Run `research/chordia_strict_unlock_v1.py --hypothesis-file <prereg>`.
   Runner is bounded: writes only to `docs/audit/results/<stem>.{md,csv}`.
3. **Phase 3** — Append audit row to `chordia_audit_log.yaml` mirroring PRECLOSE
   format. Verdict comes from the measured Phase-2 result, NOT assumed by
   symmetry. Run `python pipeline/check_drift.py` to confirm Check #134 passes.
4. **Phase 4** — Close action queue P1 with override_note documenting all
   8 dispositions. Update `memory/recent_findings.md`. Commit.

## Blast radius

Reads: `gold.db` (read-only via runner), canonical literature/precedent docs.
Writes: 1 new prereg YAML, 1 new result md, 1 new result csv, 1 new row in
`chordia_audit_log.yaml`, status flip in `action-queue.yaml`. NO writes to
`validated_setups`, `experimental_strategies`, allocator config, or any
production code. Allocator behavior changes only as a downstream effect of
the new audit row at next rebalance — same mechanism as the 2026-05-02 6-strategy
batch.

## Anti-tunnel guard

Per user: "no tunnel pigeon-holing". This stage is bounded; if Phase 2
produces an unexpected verdict (e.g. PASS_CHORDIA where PRECLOSE got PARK)
that's a finding, not a problem — log it and proceed. After Phase 4 closes,
the next /next is the higher-EV thread (prior-day-context theory-grant
feasibility per `docs/audit/results/2026-05-02-deployable-pool-edge-survey.md`),
NOT mechanical queue-grinding.

## Halt conditions

- Phase 2 runner exit non-zero with clear infrastructure error → STOP, surface to user.
- Phase 2 produces an audit-row template inconsistent with the 6 precedent rows
  (e.g. missing required fields per Check #134) → STOP, audit the runner before
  committing the new row.
- Drift check fails post-append → STOP, do NOT push around the failure.
