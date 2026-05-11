# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Stage 2/3/3.5 docs bundle (2026-05-11, Claude Opus, branch `docs/family-singleton-doctrine-bundle`)

Docs-only PR bundling the audit trail from three sibling stage worktrees that
drove Stage 4 (PR #260, merged `d8925815`) end-to-end. 6 new docs files; 0
production code; 0 DB writes.

- Stage 2: Disposition C lock + 3 self-audit passes (analysis doc + stage doc).
  Empirical landscape on 276 active SINGLETONs. Third-pass correction:
  `pre_registered_criteria.md` Amendment 2.1 already downgraded C5 to
  CROSS-CHECK; the "non-operational drift" framing was wrong for two prior
  passes.
- Stage 3: Floor spec + 3 user decisions (analysis doc + stage doc). BINDING
  criteria = C3+C4+C6+C7+C9+C10. Flags intra-doc inconsistency in
  `pre_registered_criteria.md:290-303` vs line 480-494 (separate workstream,
  MEDIUM severity).
- Stage 3.5: shelf-wide C8 OOS backfill APPLIED to gold.db (844 rows flipped
  via `scripts/tools/backfill_deployability_evidence.py --write --evidence
  c8_oos --instrument ALL`). Bloomberg-grade audit trail with reproducibility
  commands. Capital impact zero (all deployed lanes + siblings PASS).

Capital safety: lane allocator reads `validated_setups.status`, not the
deployability verdict string. Grep-verified during Stage 4 audit.

## Stage 1 — Routine-TBBO slippage registry refactor (2026-05-11, Claude Opus, branch `stage1/generalize-tbbo-slippage-inference`)

Landed: `refactor(deployability): registry-driven routine-TBBO slippage inference`.
- Replaced MNQ-only `_mnq_routine_tbbo_slippage_applies()` (deployability.py:349)
  with a registry-driven dispatcher; added `RoutineTbboPilot` dataclass +
  `ROUTINE_TBBO_SLIPPAGE_REGISTRY` populated from MNQ + MES pilot v1 evidence.
- New BLOCKING drift check `check_routine_tbbo_slippage_registry_coverage`
  parses `## Verdict: **PASS**` lines on `*slippage*pilot*v1*.md` and fails
  closed on under/over-coverage.
- Empirical: 2 MES COMEX_SETTLE rows drop the `slippage_missing` hard issue
  (verdict still `BLOCKED_FAMILY_FRAGILE` pending Stage 2 family_singleton
  policy decision).
- 124 drift checks pass, 33 deployability + 6 drift-check tests pass, 198 in
  Stage 1 scope, 4551 broader tests pass (1 pre-existing WSL-doctor failure
  unrelated to this diff).
- Capital safety verified: lane allocator keys off `s.status`, not the
  deployable flag — no MES capital deploys without Stage 3 profile +
  lane_allocation.json edit.
- Stage doc: `docs/runtime/stages/stage1-generalize-tbbo-slippage-inference.md`.
- Survey doc landed as evidence provenance:
  `docs/audit/results/2026-05-11-mes-profile-feasibility-readonly-survey.md`.

Next: Stage 2 (separate worktree from `origin/main`) — doctrine decision on
`family_singleton` policy. Without Stage 2, no MES verdict actually flips
to deployable.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-11
- **Commit:** 82ab4c06 — prereg: lock 3 LLM-drafted hypotheses for chordia-MISSING audit unlock
- **Files changed:** 4 files
  - `HANDOFF.md`
  - `docs/audit/hypotheses/2026-05-11-llm-cme-preclose-atr-p30-o15.yaml`
  - `docs/audit/hypotheses/2026-05-11-llm-cme-preclose-orb-vol-16k-o15.yaml`
  - `docs/audit/hypotheses/2026-05-11-llm-tokyo-open-atr-vel-ge105.yaml`

## Next Steps — Active

1. **STAGE 4 — IMPLEMENTED + AUDIT-PASS** on branch
   `stage4/family-singleton-conditional`. Adversarial-audit gate
   returned CONDITIONAL with 1 finding (C5 docstring/code contradiction:
   helper was gating on NULL dsr_score contrary to Amendment 2.1). Fix
   committed as follow-up: helper now returns 3-tuple `(passes, failed,
   dsr_reported)`; C5 NULL surfaces as audit-trail flag, not blocker;
   2 new tests added (NULL DSR + binding pass → CONTROLLED_LIVE_PILOT;
   NULL DSR + failing Chordia → still hard). 50 tests pass; 123 drift
   checks pass; empirical regression unchanged (33 MNQ pass, 0 MES).
   Lane allocator independence VERIFIED (grep, not just inferred).
   **Ready to open PR.**
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
