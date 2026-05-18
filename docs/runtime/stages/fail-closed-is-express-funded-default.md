---
task: Flip AccountProfile.is_express_funded default to False (fail-closed); add drift check requiring every ACCOUNT_PROFILES entry to declare the field explicitly
mode: IMPLEMENTATION
slug: fail-closed-is-express-funded-default
created: 2026-05-18
scope_lock:
  - trading_app/prop_profiles.py
  - pipeline/check_drift.py
  - tests/test_trading_app/test_prop_profiles_is_express_funded_explicit.py
---

## Blast Radius

- trading_app/prop_profiles.py — flip dataclass default from True to False. Every existing ACCOUNT_PROFILES entry already declares the field explicitly per inline confirmation (line 105-107 comment was "Default True because the active deployment topstep_50k_mnq_auto is XFA-shaped"), so flipping the default does NOT change behavior of any existing profile.
- pipeline/check_drift.py — add check_account_profiles_declare_is_express_funded that AST-parses prop_profiles.py to assert every AccountProfile(...) call in ACCOUNT_PROFILES literally includes is_express_funded=. Catches "forgot to think about it" at commit time before any new profile ships with default-inherited classification.
- tests — new test file confirming dataclass default is False AND every existing profile entry declares the field.
- Reads: trading_app.prop_profiles.ACCOUNT_PROFILES; gold.db NOT touched.
- Writes: none.
- Does NOT touch: telemetry_maturity.py, run_live_session.py, lane_allocation.json, validated_setups, broker code, session_orchestrator.

## Rationale

Follow-up to commit a2d5ea56 (telemetry-maturity FAIL→WARN demotion).
Today's logic at scripts/run_live_session.py::_check_telemetry_maturity
checks `prof.is_express_funded` to decide WARN vs FAIL. If a new profile
is added without setting the field, dataclass default of True silently
classified it as Express-Funded → WARN even if it were a real-capital
broker. Fail-closed default flips this: forgetting the field = treated
as real-capital = FAIL preserved. Drift check prevents implicit reliance
on the default by demanding explicit declaration.
