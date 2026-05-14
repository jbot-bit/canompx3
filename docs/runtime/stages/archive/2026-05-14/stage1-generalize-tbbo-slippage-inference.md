---
task: Generalize routine-TBBO slippage inference from MNQ-hardcoded helper to instrument-keyed registry driven by committed pilot evidence
mode: IMPLEMENTATION
scope_lock:
  - trading_app/deployability.py
  - pipeline/check_drift.py
  - tests/test_trading_app/test_deployability.py
  - tests/test_pipeline/test_check_drift.py
  - docs/audit/results/2026-05-11-mes-profile-feasibility-readonly-survey.md
  - docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v1.md
  - docs/runtime/stages/stage1-generalize-tbbo-slippage-inference.md

## Scope expansion (mid-implementation, 2026-05-11)

Added `docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v1.md` to scope_lock.
Reason: the new drift check parses `## Verdict: **PASS|WARN|FAIL**` (MES pilot v1
convention). The existing MNQ pilot v1 doc uses `## Executive verdict` + body text
+ `**Status:** COMPLETE` instead, which is not machine-parseable by a non-magical
regex. Per institutional-rigor §5 (no parser hacks / no future divergence bait),
the right fix is to add a canonical `## Verdict: **PASS**` metadata line to the
MNQ doc — pure metadata, body unchanged, recording what the existing executive
verdict already says ("CONSERVATIVE vs measured" + "Status: COMPLETE"). Two-line
edit, no semantic change.
---

## Blast Radius

- `trading_app/deployability.py` — replaces `_mnq_routine_tbbo_slippage_applies()` at line 349 with registry-driven `_routine_tbbo_slippage_applies()`. Adds `RoutineTbboPilot` dataclass + `ROUTINE_TBBO_SLIPPAGE_REGISTRY` populated from MNQ + MES v1 pilots. Updates single call site at line 543. Updates `_controlled_slippage_event_tail_detail` at line 357 to read pilot.sessions/basis from registry not MNQ-only frozenset.
- `pipeline/check_drift.py` — adds `check_routine_tbbo_slippage_registry_coverage()` that globs `docs/audit/results/*slippage*pilot*v1*.md`, parses `## Verdict: **PASS**` heading, fails closed if any PASS pilot is absent from the registry.
- `tests/test_trading_app/test_deployability.py` — keeps existing MNQ tests passing unchanged (backward-compat proof). Adds parametric MES cases covering each pilot v1 session + a non-pilot session that must still flag `slippage_missing`.
- `tests/test_pipeline/test_check_drift.py` — adds fixture pair: registry-covered pilot doc passes, registry-missing pilot doc fails closed.
- Reads: `gold.db` read-only (existing build_deployability_audit behavior unchanged); committed pilot result MDs.
- Writes: none. No DB mutation, no `lane_allocation.json` mutation, no profile changes.
- Downstream impact: `slippage_missing` hard-issue count drops by exactly the count of MES rows whose `entry_model=='E2'` and `orb_label` in MES pilot v1 sessions {CME_PRECLOSE, COMEX_SETTLE, SINGAPORE_OPEN, US_DATA_830}. Two known candidates expected to flip: `MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10` and `MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075`. `MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15` does NOT flip (US_DATA_1000 is not in MES v1 sessions). Both flippers retain `family_singleton` hard issue → still BLOCKED until Stage 2 doctrine decision.
- Capital safety: the lane allocator (`scripts/tools/rebalance_lanes.py:109`) keys off `s.status in ("DEPLOY","RESUME","PROVISIONAL")` from a separate surface, NOT off `deployability.deployable` or the verdict string. Verdict flips are reporting-only until Stage 3 explicit profile + lane_allocation.json edit. Verified by grep this turn.
- Codex conflict surface: the 3 recent commits on `deployability.py` (474094fa, c87c344f, 069ec333) are already on `origin/main`; this worktree branches from `origin/main`, so no conflict.

## Verification commands

```bash
# In this worktree (.worktrees/stage1-tbbo-slippage-registry)
python pipeline/check_drift.py
pytest tests/test_trading_app/test_deployability.py -q
pytest tests/test_pipeline/test_check_drift.py -q

# Empirical hard-issue regression check on the 3 closest-to-deployable MES rows
python -c "
from trading_app.deployability import build_deployability_audit
r = build_deployability_audit(scope='all-active', instruments={'MES'}, strict=True)
target_ids = {
    'MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10',
    'MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075',
    'MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15',
}
for s in r['strategies']:
    if s['strategy_id'] in target_ids:
        hard = [i['id'] for i in s['issues'] if i['severity']=='hard']
        print(s['strategy_id'], '->', hard)
# Expected:
#   MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10 -> ['family_singleton']
#   MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075 -> ['family_singleton']
#   MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15 -> ['family_singleton','slippage_missing']  (unchanged — pilot v1 didn't cover US_DATA_1000)
"
```

## Done criteria

1. `python pipeline/check_drift.py` — full pass.
2. `pytest tests/test_trading_app/test_deployability.py tests/test_pipeline/test_check_drift.py -q` — full pass.
3. Empirical regression command above prints the expected hard-issue lists.
4. Self-review: re-read modified `deployability.py` block + new drift check + new tests. Confirm no parallel re-implementation of session lists; everything reads from `ROUTINE_TBBO_SLIPPAGE_REGISTRY`. Confirm new drift check is fail-closed (raises / returns failure on registry-missing pilot, not silent skip).
5. Survey doc `docs/audit/results/2026-05-11-mes-profile-feasibility-readonly-survey.md` rides into the Stage 1 commit as evidence provenance.
