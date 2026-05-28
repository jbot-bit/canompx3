---
task: |
  NQ-mini Stage 2 of 3 — wire `resolve_execution_symbol` /
  `resolve_execution_order` into the live order-routing path
  (`SessionOrchestrator` + `webhook_server`) with fail-closed integer
  qty-divisor enforcement. Plumbing gap from the 2026-05-16 design note
  is fixed: `Portfolio` now carries an `account_profile` reference, so
  the orchestrator can resolve broker contract + qty without touching
  unrelated wiring.

  Stage 2 is DORMANT-ONLY: no `ACCOUNT_PROFILES` row populates
  `execution_symbol_map` / `execution_qty_divisor`. The wiring is
  inert until Stage 3 (explicit profile-activation decision) lands.

  Closes action-queue item `nq_mini_stage2_wiring_2026_05_15`
  (status flips open -> parked-on-Stage-3-decision; this commit ships
  the Stage 2 work the item was tracking). The item now waits for the
  Stage 3 profile-activation decision and is no longer
  `/next`-eligible auto-work.

  Driver: memory/mini_vs_micro_commission_fix.md (~77% commission
  reduction = $26K -> $52K/yr per contract on activation).

mode: IMPLEMENTATION
updated: 2026-05-29T00:00Z
agent: claude (opus 4.7)
supersedes: docs/plans/2026-05-16-stage2-nq-mini-plumbing-gap-finding.md

scope_lock:
  - trading_app/portfolio.py
  - trading_app/prop_profiles.py
  - trading_app/live/session_orchestrator.py
  - trading_app/live/webhook_server.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_webhook_server.py
  - docs/runtime/action-queue.yaml
  - docs/plans/2026-05-16-stage2-nq-mini-plumbing-gap-finding.md
  - docs/runtime/fast_lane_graveyard_digest.yaml
  - HANDOFF.md

## Blast Radius
- `trading_app/portfolio.py` -- adds `Portfolio.account_profile: AccountProfile | None`. Default `None` preserves existing call sites; `build_profile_portfolio()` wires the source profile through. Read-only field outside the orchestrator.
- `trading_app/prop_profiles.py` -- adds `resolve_execution_order(profile, strategy_symbol, strategy_qty) -> (broker_symbol, broker_qty, qty_divisor)`. Wraps `resolve_execution_symbol`; raises `ValueError` on non-integer division (fail-closed). No existing call site changes.
- `trading_app/live/session_orchestrator.py` -- order-build path now resolves execution contract + qty via the profile map when `Portfolio.account_profile` is set. Adds `execution_instrument`, `execution_qty_divisor`, `execution_contract_symbol` state; `_resolve_execution_order(strategy_qty)`; `_reject_execution_qty(...)` records an `EXECUTION_QTY_REJECTED` signal and notifies the operator. Falls back to identity (no substitution) when `account_profile is None`.
- `trading_app/live/webhook_server.py` -- order-emit path mirrors the orchestrator: substitutes contract via `_get_account_profile().execution_symbol_map` and divides qty via `execution_qty_divisor`; rejects fractional qty.
- Tests: 2 new webhook test cases (`test_profile_execution_map_resolves_contract_and_divides_qty`, `test_profile_execution_map_rejects_fractional_webhook_qty`); orchestrator slice updated for the new state. 259/259 pass (1 deselected pre-existing failure unrelated to this stage -- see Acceptance section).
- `docs/runtime/action-queue.yaml` -- item `nq_mini_stage2_wiring_2026_05_15` title flips to "Stage 3 of 3", status open -> parked, next_action rewritten to describe the Stage 3 profile-activation decision.
- `docs/runtime/fast_lane_graveyard_digest.yaml` -- rebuild required: the status transition (open -> parked) changes the digest's structural hash for this slug. Resolves drift Check 178 (`MISSING_FROM_DIGEST` for `nq_mini_stage2_wiring_2026_05_15`).
- `docs/plans/2026-05-16-stage2-nq-mini-plumbing-gap-finding.md` -- self-updates status line to "STAGE 2 DORMANT WIRING IMPLEMENTED 2026-05-29".
- `HANDOFF.md` -- baton update.
- Reads: profile-id lookup, broker contract resolver (read-only). Writes: signal records in `_write_signal_record` on rejection. No DB writes anywhere; no broker/live state mutation.

## Acceptance (all required before deleting this stage file)
- Webhook + orchestrator tests pass (259/259, 1 deselected). Pre-existing failure `TestOperatorStateExport::test_publish_state_passes_operator_payloads` verified to fail on clean HEAD before Stage 2 changes -- unrelated cascade, surfaced separately, NOT introduced here.
- `python pipeline/check_drift.py` PASSES (167 checks; pre-stage state 165 PASSED + 2 violations; Check 73 healed via `refresh_data.py --instrument MGC`; Check 178 resolves on digest rebuild).
- Adversarial-audit gate on `session_orchestrator.py` (per `.claude/rules/adversarial-audit-gate.md`): commit is `[judgment]` + touches `trading_app/live/` => evidence-auditor gate REQUIRED. Verdict: <pending>.
- Self-review (institutional-rigor §§ 1, 2, 4, 5, 6, 8, 10): see Self-Review section below.

## Self-Review (institutional-rigor § 1)
- § 4 Canonical delegation: `resolve_execution_order` wraps `resolve_execution_symbol`; no re-encoded substitution logic. `Portfolio.account_profile` is a passthrough reference to the existing `AccountProfile` dataclass.
- § 5 No dead code: new field `Portfolio.account_profile` is read by the orchestrator's `_resolve_execution_order` path and by tests. Default `None` preserves call sites in synthesis/replay/research scripts that build Portfolios without a live profile.
- § 6 No silent failures: `resolve_execution_order` raises `ValueError` on non-integer division; the orchestrator catches and converts to an `EXECUTION_QTY_REJECTED` signal record + operator notification; the webhook returns HTTP 400 with the divisor reason.
- § 10 Canonical sources: `resolve_execution_symbol` from `prop_profiles` is the single source; no duplicate map lookups added.
- § 11 Verify-don't-trust: tests exercise the contract via `_run_trade` (real handler path) with a mock router that captures the submitted `symbol` / `qty` -- not docstring claims.
- Fail-closed: with `account_profile is None`, identity path runs (no substitution, no divisor). With substitution active + indivisible qty, signal rejection + notification fire; no half-sized order reaches the broker.

## NOT done by this stage (deferred / separately gated)
- Stage 3: explicit profile-activation decision (which profile, divisor, firm-policy review, demo/live rollout, rollback condition). Tracked by the parked action-queue item.
- The pre-existing failure `TestOperatorStateExport::test_publish_state_passes_operator_payloads` (mock patches `bot_state.write_state` at module path but `_publish_state` does a local import -- patch is ineffective). Recorded in HANDOFF carry-over; NOT this stage's scope.
- Any `ACCOUNT_PROFILES` row mutation.
- Drift check `check_nq_mini_substitution_wired_or_unused` becomes non-inert on activation (Stage 3); no change to the check itself in this stage.
