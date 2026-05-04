# Spec — `paper_trades` scratch handling policy

**Status:** ACTIVE. Adopted 2026-04-28 as part of Stage 7 of `docs/runtime/stages/scratch-eod-mtm-canonical-fix.md`.
**Stage 7 acceptance:** met via this policy + Stage 5b rebuild + paper_trade_logger re-run.
**Companion result file:** `docs/audit/results/2026-04-27-paper-trades-scratch-parity.md`.

## Scope

This spec governs how `paper_trades.pnl_r` is populated for `exit_reason='scratch'` rows. It applies after the canonical Stage 5 fix to `trading_app/outcome_builder.py` lands and `orb_outcomes` is rebuilt with realized-EOD MTM.

## Architectural fact

`trading_app/paper_trade_logger.py:120-154` reads `orb_outcomes.pnl_r` directly and inserts it verbatim into `paper_trades`. There is no independent computation of P&L in the paper-trade pipeline. Therefore:

- **`paper_trades.pnl_r` correctness is downstream of `orb_outcomes.pnl_r` correctness.** No code change to `paper_trade_logger.py` is required by this spec.
- **The class bug fix in Stage 5 (canonical scratch-EOD-MTM in `outcome_builder.py`) automatically propagates to `paper_trades` via the next idempotent backfill.**

## Policy

| Population class | Policy | Required state |
|---|---|---|
| Currently-deployed lanes (per `load_allocation_lanes(profile_id)` for any active profile) | All `outcome='scratch'` rows MUST have non-NULL `pnl_r` per Criterion 13 `realized-eod`. | ≥99% population (drift check `check_orb_outcomes_scratch_pnl` enforces). |
| Retired-strategy historical rows (rows whose `strategy_id` is no longer in any active profile) | NULL `pnl_r` is acceptable when the rows were inserted under the pre-fix bug AND the strategy is no longer deployed. The historical record is preserved AS-WAS. | No threshold; documented in this policy. |

**Currently-deployed parity is the binding constraint.** As of 2026-04-28 post-Stage-5b rebuild + paper_trade_logger re-run for `topstep_50k_mnq_auto`:
- DEPLOYED: 14 scratches / 14 populated = 100% (target: ≥99%)
- RETIRED: 22 scratches / 0 populated = 0% (no target — historical record).

## Rationale

1. **No allocator decision rests on retired-strategy historical paper_trades data.** The allocator (`load_allocation_lanes`, `lane_allocation.json`, `prop_profiles.ACCOUNT_PROFILES`) consumes only active strategies. Retired strategies' historical `paper_trades.pnl_r` does not feed any live decision.

2. **Re-deriving retired-strategy rows would require artificial reactivation.** The canonical writer `paper_trade_logger.py` enumerates only active deployed lanes per the chosen profile. To populate retired rows, a separate one-off script would need to enumerate retired strategies — adding maintenance burden and ambiguity (which retired strategies? at what date cutoffs?). Per `institutional-rigor.md` § 5 (no dead code), retired-strategy backfill scaffolding violates the maintenance principle.

3. **Audit trail honesty.** Per Backtesting Rule 11, prior published results are not retro-edited. The retired-strategy paper_trades rows reflect the historical paper-trade simulation that ran with the pre-fix bug. Mutating them post-hoc would erase the audit trail of the bug's empirical scope.

4. **Future deployments self-heal.** If a retired strategy is ever re-deployed, the next `paper_trade_logger.py` run with the strategy in scope automatically wipes-and-reinserts its rows from rebuilt `orb_outcomes` (idempotent DELETE+INSERT, line 237-247). The historical NULLs are gone in the same operation.

## What this spec does NOT do

- Does not change `trading_app/paper_trade_logger.py`. The module is already correct given upstream `orb_outcomes` is correct.
- Does not change `trading_app/paper_trader.py` (live paper-trader writer for streaming-mode runs). That writer is out of scope for the Stage 5 class-bug fix because it operates on live execution events, not historical replay. Live execution already forces flat at session end via `trading_app/risk_manager.py::F-1`. If a live scratch is ever logged with NULL `pnl_r` post-fix, that would be a separate live-pipeline bug and warrants its own audit.
- Does not delete the 22 retired-strategy NULL paper_trades rows. They remain as historical record.

## Drift check

`pipeline/check_drift.py::check_orb_outcomes_scratch_pnl` (added Stage 5) enforces ≥99% population of `outcome='scratch'` rows in `orb_outcomes`. By the architectural fact above, paper_trades parity for currently-deployed lanes is implicit. No separate paper_trades drift check is needed.

## Re-deployment runbook

If a future allocator change adds back a previously-retired strategy:
1. Update `prop_profiles.ACCOUNT_PROFILES` and/or `lane_allocation.json` to include the strategy.
2. Re-run `python -m trading_app.paper_trade_logger --profile <profile_id>`.
3. The idempotent DELETE+INSERT will wipe the legacy NULL rows for that strategy and re-insert them with realized-EOD MTM `pnl_r`.

## Cross-references

- `trading_app/outcome_builder.py` — canonical writer that produced the upstream NULL bug; fixed in Stage 5.
- `trading_app/paper_trade_logger.py:120-154, :237-340` — paper-trades writer that mirrors `orb_outcomes`.
- `pipeline/check_drift.py::check_orb_outcomes_scratch_pnl` — enforcement (advisory until Stage 5b rebuild completes; PASSING as of 2026-04-28).
- `docs/institutional/pre_registered_criteria.md` § Criterion 13 — Scratch treatment policy.
- `docs/audit/results/2026-04-27-paper-trades-scratch-parity.md` — empirical state record.
