---
task: STAGE 2 of multi-account control plane — per-account (account-keyed) daily-loss kill belts in RiskManager. Each account halts independently on its own MODELED realized loss. copies=1 byte-identical. HARD PREREQUISITE for Stage 3 (simultaneous independent multi-account). Engine PnL math UNTOUCHED.
mode: IMPLEMENTATION
scope_lock:
  - trading_app/risk_manager.py
  - trading_app/live/session_orchestrator.py
  - trading_app/live/session_safety_state.py
  - tests/test_trading_app/test_risk_manager.py
  - pipeline/check_drift.py
blast_radius: "risk_manager.py: daily_pnl_dollars scalar → account-keyed dict + per-account _halted; on_trade_exit gains optional account_id; can_enter checks the entering account's belt. None/single-account path byte-identical (the dollar-cap spec's inert-when-None invariant preserved). session_orchestrator.py: configures RiskManager with the per-account contract map (from matched profile copies + contracts) at session start; crash-recovery restore adapts to the map. session_safety_state.py: persists the per-account map (parity with today's scalar). check_drift.py: confirm/extend the daily_loss_dollars < tier.max_dd guard. NO execution_engine.py edit — the engine's unchanged on_trade_exit(pnl_r, pnl_dollars) call charges all configured accounts internally via RiskManager. Reads: matched profile (read-only). Writes: data/state/session_safety_*.json (existing path)."
---

## Context

Stage 1 (single-account dashboard selector) shipped to origin/main. Stage 2 =
per-account daily-loss kill belts (stage doc `2026-06-10-dashboard-live-account-selection.md:22-23`).
The −$677 forensic that gated this is RESOLVED (bot exonerated).

## Grounding correction (verified against code, third pass)

A per-account DOLLAR daily-loss belt already ships end-to-end for the ONE live
account (`RiskLimits.max_daily_loss_dollars`, `RiskManager.daily_pnl_dollars`
scalar, `on_trade_exit` accrual at `risk_manager.py:401-420`, `can_enter` Check 1
at `:162-175`). Spec: `docs/specs/daily_loss_dollar_cap.md` (SHIPPED 2026-05-26).

The genuine gap: per-SHADOW independent loss tracking when `copies > 1`. Today
`daily_pnl_dollars` is ONE scalar — the primary only. All copies halt together
off the primary's PnL.

### Seam reality (THIRD-PASS correction — the plan's orchestrator-fan-out was wrong)

`RiskManager.on_trade_exit(pnl_r, pnl_dollars)` is called ONLY from
`execution_engine.py:632` (scratch path) and `:1566` (`_exit_trade`). The
orchestrator NEVER calls it. The engine computes ONE `pnl_dollars` from ONE
`ActiveTrade.contracts` and has NO reference to the copy router or per-account
contracts. So the modeled per-account fan-out CANNOT live in the orchestrator's
exit path (there is none) — it lives INSIDE `RiskManager`, configured once by the
orchestrator at session start with a per-account contract map. The engine's
`on_trade_exit` signature and PnL math are untouched (operator hard constraint).

## Approach (MODELED per-account belt — honest scope)

- `RiskManager` holds `daily_pnl_dollars: dict[int, float]` (account_id → realized $)
  and `_halted: dict[int, bool]`. A `None` / unconfigured account is the
  single-account path (scalar-equivalent, byte-identical).
- The orchestrator configures the per-account contract map: `{account_id: contracts}`
  for `all_account_ids`. When copies trade 1:1 (today's CopyOrderRouter mirror),
  every account charges the same modeled dollars and halts together — correct for
  copies=1/mirror. It diverges the moment Stage 3 gives accounts different contracts.
- `on_trade_exit(pnl_r, pnl_dollars, account_id=None)`: when an account map is
  configured, charge EACH configured account `pnl_dollars × (acct_contracts /
  primary_contracts)`; arm each account's belt independently. `account_id=None`
  with no map → today's scalar behaviour exactly.
- `can_enter` checks the entering account's belt (primary when no account given).

## Hard constraints (carried from the rules)

- Do NOT touch `execution_engine.py` PnL logic. The engine's `on_trade_exit` call
  is unchanged; RiskManager does the modeled fan-out internally.
- Fail-closed: an uncomputable per-account dollars must NOT silently skip accrual.
- Adversarial-audit gate MANDATORY (`risk_manager.py` + `trading_app/live/` + capital).
  `evidence-auditor` PASS before "done".

## Honest limitation (stated, not silenced)

This builds MODELED per-account PnL (primary `pnl_dollars` scaled by each
account's contract ratio). TRUE per-account realized PnL from broker fills is
deferred to Stage 3 with fill ingestion — it would require touching engine PnL,
which is forbidden here. When copies trade identically the belt halts them
together (correct for copies=1/mirror); independent halt does real work once
Stage 3 gives accounts different contracts/lanes.

## Drift guard — CONFIRMED, not extended

`check_daily_loss_dollars_below_mll` (`pipeline/check_drift.py:7487`) already
covers the per-account case: Stage 2 adds NO new declaration surface. Per-account
contracts derive from the existing `copies` field; the dollar cap is the same
per-profile `daily_loss_dollars` the check already validates `< tier.max_dd`. All
accounts in a profile share that one cap, so the existing per-profile invariant
binds every account. Adding a redundant per-account check would be dead code
(institutional-rigor §5). Decision: CONFIRM, do not extend. Drift 187/0.

## Flatten-on-halt — SEPARATE Stage 2b

Deferred per `daily_loss_dollar_cap.md:109-113`. Not folded in.

## Done criteria

Tests pass (show output) + dead code swept (`grep -r`) + `check_drift.py` passes
+ self-review + evidence-auditor gate PASS.
