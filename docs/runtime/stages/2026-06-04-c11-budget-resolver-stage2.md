# Stage 2: Self-funded strict-DD budget source (close Stage-1 scope)

task: Finish the profile-aware strict-DD budget resolver by giving SELF-FUNDED
  profiles a risk-first budget SOURCE that does NOT derive from a prop-firm
  tier number. Stage 1 (preserved on branch `c11-budget-resolver-stage1`,
  commit d9689828, PUSHED to origin) added `effective_strict_dd_budget()` but
  its self-funded branch still returns `rules.dd_limit_dollars * 1.00`, and
  `dd_limit_dollars` for self_funded = `tier.max_dd` (account_survival.py:611) —
  a PROP-SHAPED figure. Per self-funded-sizing-doctrine.md that is the exact
  leak class the doctrine forbids: a prop number bounding personal-capital risk.
mode: IMPLEMENTATION

## Pre-flight (run FIRST on resume — Stage 1 lives on a branch, not main)
1. `git rev-parse --verify c11-budget-resolver-stage1` — confirm d9689828 reachable.
2. Decide base: cherry-pick Stage-1 onto a fresh branch off origin/main, OR
   continue in main's working tree if the dirty Stage-1 set is still present
   (`git diff origin/main --stat -- trading_app/account_survival.py` shows 36 lines).
   DO NOT double-apply. Verify which state you're in before editing.
3. `git fetch origin` + check no peer advanced these files
   (`git log origin/main..origin/main` and scan c11-* worktrees).

## Scope Lock
- trading_app/prop_profiles.py
- trading_app/account_survival.py
- tests/test_trading_app/test_account_survival.py
- pipeline/check_drift.py

## Blast Radius
- prop_profiles.py — add a self-funded self-imposed DD-limit SOURCE (a field on
  AccountProfile OR a helper reading `max_risk_per_trade` × a horizon multiple).
  This is the canonical source the resolver must read for self_funded. New field
  default must be None/explicit so existing prop profiles are unaffected.
- account_survival.py:88-90 — resolver self-funded branch reads the new profile
  source instead of `rules.dd_limit_dollars * 1.00`. Fail-CLOSED preserved:
  unresolved self-funded source → fall back to stricter express belt, never looser.
- test_account_survival.py — extend the existing
  `test_effective_strict_dd_budget_is_profile_aware_and_fails_closed` to assert
  the self-funded budget now comes from the profile source, NOT tier.max_dd, and
  the fail-closed path still resolves to express when the source is missing.
- check_drift.py:7318 `check_prop_caps_do_not_leak_into_self_funded` — UPGRADE
  from marker-only to STRUCTURAL: assert `effective_strict_dd_budget()` for a
  self_funded profile does not equal `tier.max_dd * express_fraction` (i.e. prove
  the self-funded path is genuinely de-coupled from the prop number). This closes
  the "honest floor, not ceiling" gap named in self-funded-sizing-doctrine.md.
- Reads: gold.db read-only (account_survival sim). Writes: none to gold.db.

## Done criteria (all four)
1. `pytest tests/test_trading_app/test_account_survival.py -p no:timeout` green,
   incl. a NEW assertion proving self-funded budget != prop-derived number.
2. `python pipeline/check_drift.py` passes (the upgraded structural leak check).
3. Dead code swept (`grep -r` the old flat-constant path is gone).
4. Self-review: confirm topstep_50k_mnq_auto verdict UNCHANGED ($2,039 still
   FAILS at $1,800 express) — this stage must NOT flip any profile to PASS.

## Explicit NON-goals (do NOT do in this stage)
- Do NOT arm C11 or change topstep verdict. C11 stays NO-GO.
- Do NOT wire the cap_x0.75 ORB cap (that is the SEPARATE real C11 fix, behind
  the still-OPEN adversarial-audit gate `9b3fc530` — see
  project_c11_fix_is_tighter_orb_cap_not_throttle_2026_06_04.md).
- Do NOT touch lane allocation / book-building (that is a later capital-path stage).

## Tier / gate
Tier B capital-path (touches account_survival verdict + a drift semantic).
Requires adversarial-audit (evidence-auditor) pass before any merge to main,
per institutional-rigor.md § 2. Stage-1 backup is safe on origin regardless.
