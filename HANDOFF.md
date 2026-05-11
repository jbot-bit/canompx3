# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-11
- **Branch:** stage4/family-singleton-conditional (worktree
  `.worktrees/stage4-family-singleton-conditional`)
- **Base:** origin/main @ 62732518 (post-PR #257)
- **Summary:** Stage 4 IMPLEMENTED. trading_app/deployability.py now
  conditionally downgrades `family_singleton` to a `warning` (routing
  to `CONTROLLED_LIVE_PILOT_CANDIDATE`) when the row clears the locked
  binding criteria from pre_registered_criteria.md (C3 + C4 banded +
  C6 + C7 + C9 + C10), with C5 dsr_score required computed-and-reported
  (cross-check per Amendment 2.1, NOT gating). 33 tests passing in
  test_deployability.py (+7 new SINGLETON fixtures).
- **Empirical regression on real gold.db:** 33 / 276 active SINGLETONs
  clear the floor (all 33 are MNQ; 0 MES); 243 stay hard-blocked
  (Chordia C4 is the dominant blocker: 242 of 243). All 5 original MES
  Stage-2 candidates remain HARD-BLOCKED (fail C4 Chordia). Capital
  impact NONE — lane allocator reads validated_setups.status not the
  deployability verdict.
- **Canonical delegation:** C4 delegates to
  trading_app.chordia.chordia_verdict_label; C10 delegates to
  ALL_FILTERS[filter_type].requires_micro_data + pipeline.data_era.is_micro.
  No parallel logic.

## Next Steps — Active

1. **STAGE 4 — IMPLEMENTED** on branch `stage4/family-singleton-conditional`.
   Pre-merge **adversarial-audit gate pending** per
   `.claude/rules/adversarial-audit-gate.md` (truth-layer + judgment-class
   change requires independent-context evidence-auditor pass before PR
   merge). After audit returns PASS, open PR.
2. **5 MES Stage-2 candidates outcome:** still HARD-BLOCKED under
   Stage 4 because they fail C4 Chordia (t ≈ 2.2-2.6, BAND C). The
   original Stage 2 expectation that Disposition C "unlocks at most
   5 after C5 + C8 resolved" is empirically refuted under the
   literature-grounded floor — 0 of 5 clear.
3. **33 MNQ CONTROLLED_LIVE_PILOT_CANDIDATE rows:** the new conditional-
   warning surface. All MNQ; sample includes CME_PRECLOSE-cluster
   sibling RR variants of deployed lanes. Lane-correlation gates
   downstream would collapse most of them before any real deployment.
   No allocator change in Stage 4.
4. **Stage 1 PR #258** awaiting merge.
5. **Stage 2 / Stage 3 / Stage 3.5 branches** awaiting PR bundling
   decision.
6. **Doctrine fix for `pre_registered_criteria.md:290-303`** (Stage 3
   § 5) — separate workstream, MEDIUM severity, NOT blocking.
7. Track D MNQ COMEX_SETTLE Gate 0 runner design — carried forward.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
