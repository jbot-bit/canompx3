---
task: TopStep canonical compliance fixes (Stages 2-8 of plan)
mode: IMPLEMENTATION
slug: topstep-canonical-fixes
scope_lock: [pipeline/cost_model.py, trading_app/prop_profiles.py, trading_app/prop_firm_policies.py, trading_app/pre_session_check.py, trading_app/risk_manager.py, trading_app/topstep_scaling_plan.py, trading_app/live/copy_order_router.py, trading_app/live/session_orchestrator.py, trading_app/live/broker_base.py, pipeline/check_drift.py]
blast_radius: All backtests (cost model change F-4) + risk_manager.can_enter() callers + session_orchestrator HWM init path + CopyOrderRouter (all live execution) + pre_session_check (startup gate)
agent: claude-code-opus
created: 2026-04-08T03:00:00Z
updated: 2026-04-08T03:00:00Z
plan: docs/plans/2026-04-08-topstep-canonical-fixes.md
audit: docs/audit/2026-04-08-topstep-canonical-audit.md
---

## Purpose

Fix the 6 active TopStep canonical compliance findings (F-4, F-6, F-5, F-2b, F-2, F-1)
plus add a drift check for canonical-source annotations (Stage 8). All grounded in the
20-source canonical corpus at `docs/research-input/topstep/`. Plan and audit are
already committed (see `plan:` and `audit:` frontmatter fields).

## Blast Radius

- **F-4 commissions** → all `pipeline/cost_model.py` consumers: backtests via
  `pipeline/outcome_builder.py`, `validated_setups` populated by
  `trading_app/strategy_discovery.py`, allocator in `trading_app/lane_allocator.py`,
  risk_manager.py R-multiple computations.
- **F-6 5-XFA cap** → `pre_session_check.py` startup path; ACCOUNT_PROFILES sum.
- **F-5 HWM XFA freeze** → `prop_profiles.py` AccountProfile dataclass schema +
  `session_orchestrator.py` HWM tracker init at lines 407-416.
- **F-2b Shadow asymmetry** → `copy_order_router.py` submit/cancel paths +
  `session_orchestrator.py` reconcile loop.
- **F-2 Hedging guard** → `risk_manager.py` `can_enter()` + all callers.
- **F-1 Scaling Plan enforcer** → new module `topstep_scaling_plan.py` + RiskManager
  + session_orchestrator wiring (HWM tracker → risk manager balance read).
- **Stage 8 drift check** → `pipeline/check_drift.py` total check count +1.

## Stages (one commit each, verified between)

| # | Stage | Status |
|---|---|---|
| 1 | F-9 + F-10 + F-11 annotations | DONE (commit 923c5ce) |
| 2 | F-4 MNQ/MES commissions | IN PROGRESS |
| 3 | F-6 5-XFA aggregate cap | PENDING |
| 4 | F-5 HWM XFA freeze formula | PENDING |
| 5 | F-2b CopyOrderRouter shadow asymmetry | PENDING |
| 6 | F-2 Hedging guard | PENDING |
| 7 | F-1 Scaling Plan enforcer | PENDING |
| 8 | Drift check for canonical annotations | PENDING |

## Stage gate per stage

1. Read all files in scope_lock that the stage touches
2. Articulate exact change (already done in plan doc)
3. Edit with @canonical-source citations
4. Run `python -m pipeline.check_drift` — must pass with same count or +1
5. Run targeted tests for touched module
6. `git diff` self-review against canonical citation
7. Commit with `fix(...): F-X title [stage N/8]`
8. Update Status table above

## Acceptance for whole task

- All 8 stages committed
- All findings F-1 through F-8 marked FIXED in `docs/audit/2026-04-08-topstep-canonical-audit.md`
- `python -m pipeline.check_drift` passes with N+1 checks
- All targeted tests green
- No regression in existing test suite
- Stage file deleted after acceptance (per stage-gate protocol)
