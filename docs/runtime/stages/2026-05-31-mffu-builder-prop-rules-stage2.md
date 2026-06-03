---
task: "Stage 2 — encode MFFU Builder + Flex specs, add firm_specific_rules schema field, fix Rapid 100k/150k sim-cap leak, add Rapid 25k tier"
mode: IMPLEMENTATION
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/prop_firm_policies.py
  - tests/test_prop_profiles_mffu.py
  - docs/audit/2026-05-31-mffu-forced-progression-live-cap-memo.md
---

## Scope decision (operator, 2026-05-31)

A + B as DATA this session; C as DESIGN MEMO only.
- A: prop_profiles.py — mffu_builder/_flex specs + tiers + Rapid 100k/150k cap fix + Rapid 25k.
- B: prop_firm_policies.py — mffu_builder/_flex PayoutPolicy (80/20, $2k cap, 5 sim payouts,
  0.50 consistency, forced-live trigger). Pure verbatim data into existing schema.
- C: docs/audit memo — forced eval→sim→live progression + Live max-1-account cap. NEW
  capital-path logic (no `max_live_accounts` field exists; no stage-transition logic). Gated
  for separate adversarial-audit sign-off. NOT implemented this commit.
  - Operator note: TopStep also forces progression. UNVERIFIED anecdote — operator's friend
    withdrew many times from one account, suggesting forced-live MAY be avoidable. Treat as
    UNOFFICIAL signal; the memo must verify against verbatim source before encoding any
    "avoidance" path. Do not encode the anecdote as fact.

## Blast Radius

- `trading_app/prop_profiles.py` — adds `firm_specific_rules: Mapping[str,object] | None = None`
  to frozen `PropFirmSpec` (optional default None → every existing spec unchanged, zero
  migration). Adds `mffu_builder` + `mffu_flex` specs. Adds Builder/Flex/Rapid-25k entries to
  `ACCOUNT_TIERS`. Fixes Rapid 100k (6/60→10/100) and 150k (8/80→15/150) to verbatim sim-funded
  caps.
- Consumers of `PROP_FIRM_SPECS`/`ACCOUNT_TIERS`: `prop_portfolio.build_book` (line 558,
  `contract_budget = tier.max_contracts_micro`), `account_survival._build_rules`,
  `pre_session_check`, lane allocator. New keys are ADDITIVE — no existing `ACCOUNT_PROFILES`
  profile uses `mffu` (verified: 0 profiles with firm="mffu"), so new tiers have zero live
  consumers.
- Rapid 100k/150k cap fix: the ONLY real consumer is `prop_portfolio.py:558` book-builder
  budget. `account_survival.py:564` hardcodes `contracts_per_trade_micro=1` and never reads the
  tier cap — so the survival sim is NOT affected. No active mffu profile → no live-capital
  effect today; this is a correctness/honesty fix.
- Reads: docs/research-input/mffu/*.md (verbatim source). Writes: none to gold.db.
- New test file: zero callers, asserts encoded values == verbatim manifest.

## Truth grounding (this session, verbatim help.myfundedfutures.com)

- Builder source: docs/research-input/mffu/mffu_builder_50k.md (article 14290805).
- Rapid sim-funded caps (CORRECTED vs prior memory): 25k=3/30, 50k=5/50 (repo ALREADY correct),
  100k=10/100 (repo had 6/60 LIVE ladder — BUG), 150k=15/150 (repo had 8/80 — BUG).
- Builder: EOD trailing, MLL locks +$100, 80/20, $2k payout cap, 5 sim payouts, 50% payout
  consistency, $1k soft-pause DLL, 21d post-breach cooldown, news unrestricted, 4mini/40micro,
  $50k-only, 2 MLL options (Default $2,000 / Add-On $1,500), $3k eval target.

## Done criteria

- tests pass (show output), dead code swept, `python pipeline/check_drift.py` passes,
  self-review passed. Diff keeps Builder/Rapid/Flex SEPARATED.
