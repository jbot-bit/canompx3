# Stage 1: Profile-aware strict-DD budget resolver (single source of truth)

task: Replace the flat `STRICT_DD_BUDGET_FRACTION = 0.80` constant with a
  profile-aware resolver `effective_strict_dd_budget(profile, rules)` so the
  strict drawdown budget is sourced ONCE and is correct per account class:
  - express-funded (prop wrappers) → $1,800-equiv (fraction 0.90 of the $2,000 MLL),
    a deliberate operator risk-knob relaxation from 0.80 ($1,600).
  - self-funded (real capital) → risk-first / relaxed, sourced from the profile's
    OWN self-imposed DD limit, NEVER a prop fraction (self-funded-sizing-doctrine).
  Fail-CLOSED: any profile whose class can't be resolved gets the STRICTER
  (express) budget, never the relaxed one.
  This is the UPSTREAM single-source change. It does NOT touch lane selection
  (allocator) or arm C11 — Stage 2+ wire the allocator to read this resolver.
mode: IMPLEMENTATION

## Scope Lock
- trading_app/account_survival.py
- trading_app/prop_profiles.py
- tests/test_trading_app/test_account_survival.py
- scripts/tools/c11_clearance_scenarios.py

## Blast Radius
- account_survival.py:57 (constant) + :787 (call site) — the strict-gate verdict.
  Changing 0.80→resolver(0.90 express) RELAXES the express budget $1,600→$1,800.
  This is intentional (operator). topstep_50k_mnq_auto verdict at baseline
  ($2,038 DD) still FAILS even at $1,800 — so no profile silently flips to PASS
  from this stage alone. Verified: baseline DD $2,038 > $1,800.
- prop_profiles.py — add a self-funded risk-first budget source (read self-imposed
  DD limit; new helper or field) + marker for the leak drift-check.
- c11_clearance_scenarios.py:99,130 — research script, reads the same budget;
  point at resolver so research and gate agree.
- test_account_survival.py:243 — `== 1600.0` assert becomes resolver-driven
  (1800 express). The circular `==1600` assert flagged in c11-clearance-audit.
- Reads: gold.db (read-only via existing replay). Writes: none (no DB, no config).
- NOT capital-path at runtime: this changes a verdict THRESHOLD, not which trades
  the bot takes. Lane selection (the capital path) is Stage 2, adversarial-gated.

## Verification
- `python -m trading_app.account_survival --profile topstep_50k_mnq_auto`
  → strict budget reports $1,800; verdict still FAIL (DD $2,038 > $1,800).
- self_funded_tradovate resolves to its risk-first budget (~$3,000 self-imposed),
  NOT a prop fraction.
- `python pipeline/check_drift.py` passes (esp. self-funded leak check).
- targeted tests in test_account_survival.py pass (show output).

## HARD CONSTRAINTS
- Fail-closed: unresolved class → strict (express) budget.
- Self-funded budget NEVER sourced from a prop fraction (doctrine + drift check).
- C11 not armed; allocator untouched this stage.
