---
task: validator-honesty-fix
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/criterion_ladder_check.py
  - scripts/tools/minbtl_retro_report.py
  - docs/audit/results/2026-05-04-mgc-chain-quarantine.md
  - trading_app/strategy_validator.py
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift.py
  - tests/test_trading_app/test_strategy_validator/test_pool_freshness.py
  - tests/test_trading_app/test_strategy_validator/test_mgc_stress_stamp.py
blast_radius: Stage 1 NEW-only (read-only tools + quarantine doc, no canonical edits). Stage 2 strategy_validator.py FDR summary block ~L2262 gets caveat + per-session K echo (pure string change, no behavior); pre-flight L2575-2583 gets new try/except calling _check_prereg_present (mirrors _check_mode_a_holdout_integrity); check_drift.py pure additive new function check_prereg_present_for_recent_runs in CHECKS; test_check_drift.py extended. Stage 3 check_drift.py pure additive check_validator_pool_freshness (advisory) + new test_pool_freshness.py idempotency. Stage 4 strategy_validator.py append-only Mode-A pre-flight MGC stress_test stamp (informational, mirrors DSR pattern); new test_mgc_stress_stamp.py. Companion tests for every edit. No call-site changes. No data-shape changes.
agent: claude
updated: 2026-05-04
---

# Validator Honesty Fix — MGC/MNQ post-audit

**Status:** IMPLEMENTATION (1/4)
**Date:** 2026-05-04
**Worktree:** `canompx3-validator-honesty` on `validator-honesty-fix`
**Owner:** Claude (this session)

## Why this stage exists

The 2026-05-04 MGC strategy_validator chain rerun produced "1 PASSED / 18,962
REJECTED" at stratified BH-FDR (per-session K range 1,440–3,408). The 1
survivor (MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4) survived a doctrinally-banned
brute-force discovery: realized N=19,875 vs locked operational cap N≤300
(~66x violation) vs strict Bailey E=1.0 bound N≤4 for 2.7yr MGC clean data
(~5,000x violation).

Five honest fixes ground in already-locked criteria:

1. Read-only ladder + MinBTL retro-report tools to identify the survivor and
   stamp it research-provisional.
2. Per-session K vector in the validator's headline log (the mechanism is
   already correct — line 2141 builds `effective_k_by_session` — only the
   reported summary at line 2262-2266 collapses to aggregate `total_k`).
3. Pre-reg pre-flight gate (Criterion 1 — `pre_registered_criteria.md:42-53`)
   that refuses to start NEW promotion runs without a matching
   `docs/audit/hypotheses/<date>-<instr>-*.yaml` file.
4. Pool-freshness drift check (advisory) that reports K-drift between a row's
   frozen `discovery_k` and the current per-session pool size.
5. MGC stress-test informational stamp (mirrors Amendment 2.1 DSR pattern —
   informational only, does not block promotion).

## Scope and blast radius

See frontmatter `scope_lock` and `blast_radius` for the canonical lists.

## Acceptance criteria

- [ ] Stage 1: criterion_ladder_check.py prints C1-C12 status for any MGC
      strategy_id; minbtl_retro_report.py reports selection-bias multipliers
      for the 2026-05-04 MGC run; quarantine doc lists survivor + grandfathered
      rows with `provenance.btl_status = research_provisional`.
- [ ] Stage 2: per-session K vector visible in validator headline log;
      pre-reg drift check FAILS on synthetic empty-prereg fixture, PASSES once
      yaml present; existing 67 post-Phase-0 rows do not trip the check
      (legacy carve-out for `validation_run_id IS NULL` or pre-existing rows).
- [ ] Stage 3: pool-freshness drift check runs in <5s; reports K-drift % for
      recent rows. Idempotency test asserts no row insertions/deletions on
      consecutive validator runs against unchanged DB.
- [ ] Stage 4: MGC validator runs invoke stress_test; stamp visible on
      validated_setups row. MNQ/MES unchanged.
- [ ] `python pipeline/check_drift.py` exits 0.
