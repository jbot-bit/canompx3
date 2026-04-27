---
slug: nq-mini-execution-stage1-account-profile
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 3
created: 2026-04-27
updated: 2026-04-27
task: Execution-layer NQ↔MNQ symbol-substitution contract on AccountProfile — Stage 1 of 3. Adds two optional frozen fields (`execution_symbol_map`, `execution_qty_divisor`) and a free resolver helper. Identity default; no consumer wiring this stage. Stage 2 wires session_orchestrator + webhook_server. Stage 3 reconciles risk_manager F-1 XFA math.
---

# Stage 1: AccountProfile NQ↔MNQ symbol-substitution contract

scope_lock:
  - trading_app/prop_profiles.py
  - tests/test_trading_app/test_prop_profiles.py
  - docs/runtime/stages/nq-mini-execution-stage1-account-profile.md

blast_radius:
  - trading_app/prop_profiles.py — additive: 2 frozen Optional fields on AccountProfile (execution_symbol_map, execution_qty_divisor), one __post_init__ validator, one module-level resolve_execution_symbol() helper. No existing call sites changed; no ACCOUNT_PROFILES rows mutated.
  - tests/test_trading_app/test_prop_profiles.py — additive: identity-default test, valid mapping test, fail-closed validation tests, resolver behaviour tests. New file or new test class in existing file.
  - Downstream consumers (informational, NOT touched): trading_app/live/session_orchestrator.py, trading_app/live/webhook_server.py, trading_app/risk_manager.py, trading_app/live/projectx/order_router.py — all construct/read AccountProfile by field access, all defaults preserve byte-equivalent behaviour.

## Why this stage exists

Per `memory/mini_vs_micro_commission_fix.md` + `memory/self_funded_realistic_assessment.md`,
self-funded futures economics depend on trading 1 NQ instead of 10 MNQ — same $20/pt
exposure, ~77% commission reduction, $26K → $52K/yr per contract on the same edge.

`pipeline/cost_model.py:135-141` already has a canonical `NQ` `CostSpec`. What's missing
is the execution-layer plumbing for a profile to declare "translate strategy symbol X
to broker symbol Y at order build time, and divide qty by the contract-multiplier ratio."

This stage is the data-model contract only. Identity default — zero behaviour change
on existing profiles.

## Exact question

Can `trading_app/prop_profiles.AccountProfile` carry a frozen, declarative
execution-symbol-substitution contract that:

- defaults to identity (no behaviour change for any current profile)
- expresses both the symbol map AND the qty divisor explicitly (no implicit
  division-by-cost-spec-ratio that drifts with cost-spec edits)
- is consumable by a downstream stage (session_orchestrator + webhook_server)
  without that stage needing to know prop_profiles internals
- fails closed on inconsistent declarations (symbol map says NQ but no divisor;
  divisor=10 but no symbol map)

## Required outputs (Stage 1 only)

1. Two new optional, frozen fields on `AccountProfile`:
   - `execution_symbol_map: Mapping[str, str] | None = None`
   - `execution_qty_divisor: Mapping[str, int] | None = None`
2. `__post_init__` validation:
   - either both None (identity) or both populated
   - keys in `execution_qty_divisor` ⊇ keys in `execution_symbol_map`
   - every divisor value ≥ 1, integer
   - every symbol-map source key is in `pipeline.asset_configs.ASSET_CONFIGS`
   - every symbol-map target key is in `pipeline.cost_model.COST_SPECS`
3. Module-level free function (NOT a method on AccountProfile to keep the
   dataclass frozen-pure): `resolve_execution_symbol(profile, strategy_symbol) -> tuple[str, int]`
   returning `(broker_symbol, qty_divisor)` — defaults to `(strategy_symbol, 1)`.
4. Tests covering: identity default, valid NQ↔MNQ mapping, every fail-closed
   path, resolver behaviour for both identity and mapped cases.

## Rules (Stage 1)

- Do not modify `session_orchestrator.py`, `webhook_server.py`, `risk_manager.py`,
  `projectx/order_router.py`. That is Stage 2.
- Do not change any existing `ACCOUNT_PROFILES` entry. Default identity
  preserves byte-equivalence.
- Do not edit `cost_model.py` (NQ already there).
- Do not pre-wire `self_funded_tradovate` to use the map — that profile change
  ships in a later stage with order-router translation in the same commit.
- Holdout 2026-01-01 untouched (this is execution-layer, not research).

## Blast radius

Allowed in Stage 1 (scope_lock above):

- `trading_app/prop_profiles.py` (additive: 2 fields, 1 free function, validation)
- `tests/test_trading_app/test_prop_profiles.py`

Not allowed in Stage 1 (separate stages):

- `trading_app/live/session_orchestrator.py` — Stage 2
- `trading_app/live/webhook_server.py` — Stage 2
- `trading_app/risk_manager.py` — Stage 3 (F-1 XFA risk math)
- `trading_app/live/projectx/order_router.py` — Stage 2 verification
- Any `ACCOUNT_PROFILES` row mutation — Stage 4

## Blast radius — downstream consumers (informational)

Already-known callers of `effective_daily_lanes(profile)` and direct `profile.*`
field access (per `allocator_wiring_apr13.md`): portfolio.py, derived_state.py,
account_survival.py, paper_trade_logger.py, bot_dashboard.py, multi_runner.py,
run_live_session.py. None of these construct AccountProfile, so adding optional
fields with `None` defaults does not break them.

## Verification

- `pytest tests/test_trading_app/test_prop_profiles.py -v` — new tests pass
- `python pipeline/check_drift.py` — clean
- `grep -r "execution_symbol_map\|execution_qty_divisor" trading_app/ pipeline/` —
  only the two field definitions, the resolver, and tests reference them
  (proves Stage 1 discipline)

## Final verdict options

- `IMPLEMENT` — selected
- `PARK`
- `REDESIGN`

## Non-goals

- No order-router change
- No risk-manager change
- No live profile mutation
- No cost-model edit
- No claim that this enables NQ-mini trading; this is Stage 1 of N
