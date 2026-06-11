task: "Stage 3a-i — per-account contract source: AccountProfile.account_contracts (account_id-keyed) replaces the {aid:1} hardcode at session_orchestrator.py:694, feeding RiskManager's existing per-account belts; per-account C11 survival via SizingContext reuse; new feasibility drift check. Clamp stays at 1 (belts diverge by lane, not live-size)."
mode: IMPLEMENTATION

## Scope Lock
- trading_app/prop_profiles.py
- trading_app/live/session_orchestrator.py
- trading_app/account_survival.py
- pipeline/check_drift.py
- docs/specs/per_account_contracts.md
- tests/test_trading_app/test_prop_profiles.py
- tests/test_trading_app/test_account_survival_per_account.py
- tests/test_pipeline/test_check_drift_account_contracts.py

## Blast Radius
- prop_profiles.py — NEW optional frozen field `account_contracts: tuple[tuple[int,int],...] = ()` + `resolve_account_contracts()` helper + `__post_init__` validation. Default `()` → uniform-1 → byte-identical to today. ~30 ACCOUNT_PROFILES entries unchanged (field defaults).
- session_orchestrator.py — REPLACE `{aid:1 for aid...}` hardcode at :694 with `account_profile.resolve_account_contracts(all_account_ids)`; None-profile guard keeps `{aid:1}`. configure_accounts() at :828 consumes unchanged. Capital path (live belt arming) but byte-identical at default.
- account_survival.py — NEW `evaluate_per_account_survival()` reusing `_scenarios_for_context`/`simulate_survival`/`_evaluate_gate` per distinct contract count. ZERO new sim/gate math (institutional-rigor §4). Reads gold.db read-only.
- check_drift.py — NEW `check_account_contracts_feasibility()`: shape + positivity + (contracts>1 → daily_loss_dollars/tier.max_dd must bound contracts×single-contract worst-day). Registered in check list. Known-violation injection test.
- docs/specs/per_account_contracts.md — NEW spec; @canonical-source target.
- Reads: gold.db (read-only, survival). Writes: none. No schema change. No execution_engine.py PnL change. DEPLOYED_MAX_CONTRACTS_CLAMP stays 1.
- Capital-path gate: touches risk_manager consumers + trading_app/live/ + account_survival (C11) → adversarial-audit gate (evidence-auditor PASS) MANDATORY before done.

## Notes
- C1: field keyed by broker int account_id, NOT position (preflight.py:966 reorders for --account-id).
- C2: per-account survival via SizingContext (:197), NOT vestigial contracts_per_trade_micro (:139).
- Stage 3b (broker-fill routing, on_trade_exit(account_id=)) + clamp lift = OUT OF SCOPE.
- Orchestrator wire (:694) is a 6-line resolver branch; its logic lives entirely in the
  unit-proven `resolve_account_contracts` (C1 reorder test + byte-identical empty-map test).
  SessionOrchestrator has no hermetic test harness (broker auth + DB) and no test instantiates
  it directly — a full-init harness for a thin branch is high-cost/low-value/fragile. The
  byte-identical guarantee (empty field → old `{aid:1}`) + the None-profile fallback are both
  covered by resolver unit tests; the wire is a direct substitution of that proven call.
- Belt-divergence (plan step 3) is ALREADY covered by Stage 2's
  test_risk_manager.py::test_independent_halt_different_contracts ({101:1, 202:3}); 3a adds the
  SOURCE of that map, not new belt math. No new risk_manager test needed.
