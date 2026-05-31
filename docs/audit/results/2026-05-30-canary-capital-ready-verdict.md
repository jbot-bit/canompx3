# Canary harness - CAPITAL_READY verdict

**Date:** 2026-05-30  
**Verdict:** `CAPITAL_READY`  
**Canaries fired:** 10/10  
**Producer:** `scripts/tools/canary_report.py`

## Scope

Negative-control / guard-efficacy check: each canary injects a known
fake-edge contamination class and asserts the canonical guard meant to
catch it fires. The question - *can canompx3 reliably reject fake edge?*
`CAPITAL_READY` iff every guard caught its trap. Complements the null
harness (`scripts/tests/test_synthetic_null.py`), which owns the
noise-to-edge regime; this owns the guard-efficacy regime.

## Results

| Canary | Fired | Guard | Signature |
|--------|-------|-------|-----------|
| `randomized_entry_direction` | PASS | research.oos_power.one_sample_tstat/power_verdict + moving_block_bootstrap_p (Aronson Ch5 MCP) | |t|=0.89<3.0; tier=STATISTICALLY_USELESS; perm_p=0.351>0.05 |
| `shuffled_trading_days` | PASS | moving_block_bootstrap_p (Aronson Ch5 MCP) + one_sample_tstat | |t|=1.79<3.0; perm_p=0.061>0.05 |
| `permuted_session_labels` | PASS | pipeline.session_guard.is_feature_safe | is_feature_safe('orb_NYSE_CLOSE_size', 'TOKYO_OPEN')=False (expect False) |
| `lagged_vs_leaked_features` | PASS | pipeline.session_guard.is_feature_safe + daily-features-joins triple-join | lag_safe=True(T); leak_safe=False(F); t_inflation=1.734≈√3 |
| `synthetic_future_looking_feature` | PASS | trading_app.config E2 exclusions + pipeline.session_guard._NEVER_SAFE | is_e2_lookahead('VOL_RV70')=True; reason=set; is_feature_safe('daily_high')=False |
| `random_filter_real_sparsity` | PASS | moving_block_bootstrap_p (FST-aware null) + RULE 8.1 extreme-fire | fire_rate=0.32(not extreme); perm_p=0.298>0.05 |
| `post_entry_disguised_as_pre_entry` | PASS | research...t0_correlation (load-bearing) + session_guard (secondary, defeatable by rename) | T0(neutral)=1.00>0.7; T0(camouflage)=0.99>0.7; name_guard(neutral)=False(fail-closed); name_guard(camo)=True(DEFEATED→T0 load-bearing) |
| `holdout_2026_contamination` | PASS | trading_app.holdout_policy.enforce_holdout_date (Amendment 2.7) | enforce_holdout_date(2026-06-01)raised=True; override→2026-06-01(==2026-06-01) |
| `derived_table_only_claim` | PASS | layer discipline (RESEARCH_RULES) + meta-check (check_research_scans_call_guards) | derived_expr=+0.21 but canonical |t|=0.10<3.0; sharpe=-0.006 |
| `dsr_universe_gaming` | PASS | Amendment 3.5 universe-pin (V[SR] = pinned family, pre-2026, all siblings+failures) | V[SR]_full=0.2102 vs V[SR]_winners=0.0411; divergence=5.12>1.5 → DSR_UNIVERSE_UNPINNED |

## Verdict

**`CAPITAL_READY`** - all 10 guards fired; the pipeline rejected every synthetic fake-edge trap. Negative-control evidence that canompx3 can kill fake edge on demand (cf. Aronson 2007 Ch 8-9: a correct pipeline kills naively-flagged rules at scale).

## Reproduction

```bash
python scripts/tools/canary_report.py            # this verdict + MD
python scripts/tests/canary_suite.py             # the raw canary table
python pipeline/check_drift.py                   # Check 192 (blocking gate)
```

Outputs: this file; the blocking drift check `check_canary_suite_green`;
the Tier-1 suite `scripts/tests/canary_suite.py`.

## Limitations

- Tier-1 calls guards at their API boundary - it proves the guard FUNCTION
  fires, not that every scan ROUTES through it (the meta static-scanner
  `check_research_scans_call_guards` covers routing structurally; Tier-2
  end-to-end injection is deferred).
- Name-based guards are defeatable by renaming a post-entry field to an
  `_ALWAYS_SAFE` name; the value-based T0 tautology is the load-bearing catch
  (canary 7). A magnitude-only post-entry leak (e.g. `mae_r`) is out of scope.
- Canary 10 checks the DSR V[SR] universe is PINNED, not that N-hat uses ONC
  clustering (Amendment 3.5 ONC_PENDING).
