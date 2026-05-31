# Deliverable #3 — Research scans that bypass canonical guards (canary harness)

**Date:** 2026-05-30
**Producer:** `scripts/tools/canary_guard_coverage.py` (`scan_guard_coverage`)
**Drift check:** `pipeline/check_drift.py::check_research_scans_call_guards` (Check 147, ADVISORY)
**Companion prereg:** `docs/audit/hypotheses/drafts/2026-05-30-canary-contamination-harness.draft.yaml`

## Scope

The question: *which research scan scripts would INCORRECTLY PASS a fake edge
because they never invoke a canonical guard?* The Tier-1 canary suite
(`scripts/tests/canary_suite.py`) proves each guard FUNCTION fires; this list is
the structural complement — scans that **read a canonical layer**
(`orb_outcomes` / `daily_features`) **AND apply a filter or use the E2 entry
model** **BUT reference NO canonical guard** (`filter_signal` / `session_guard` /
`is_feature_safe` / `enforce_holdout_date` / `t0_correlation` /
`_valid_session_features`). Each such script could pass a fake edge because the
guard meant to catch its contamination class is never invoked. A guard that
works but is never called is still a hole — this is the bridge between "the
guard works" (Tier-1) and "the guard is used" (Tier-2, deferred).

## Reproduction

```bash
python scripts/tools/canary_guard_coverage.py            # the list (this file)
python scripts/tools/canary_guard_coverage.py --json     # machine-readable
python pipeline/check_drift.py                            # Check 147 (advisory)
```

Detection is AST-based (string-constant inspection for SQL + the
`entry_model='E2'` literal; source-text match for filter identifiers). The
`archive/` subtree is skipped. Opt-out marker: `# canary-guard-coverage: cleared`
(after manual verification).

## Outputs / The list

**66 scripts** flagged (51 under `research/`, 15 under `scripts/research/`):

- `research/audit_6lane_correlation_concentration.py`
- `research/audit_l1_orb_g5_arithmetic_only_check.py`
- `research/audit_l2_atr_p50_regime_vs_arithmetic.py`
- `research/audit_pre_break_context_cross_session.py`
- `research/audit_pre_velocity_vol_regime_interaction.py`
- `research/audit_sizing_substrate_diagnostic.py`
- `research/carry_encoding_exploration.py`
- `research/cme_fx_futures_orb_pilot.py`
- `research/cross_validate_strategies.py`
- `research/garch_a4b_binding_budget_replay.py`
- `research/garch_additive_sizing_audit.py`
- `research/garch_carry_collinearity_check.py`
- `research/garch_normalized_sizing_audit.py`
- `research/garch_partner_state_provenance_audit.py`
- `research/garch_profile_confluence_replay.py`
- `research/garch_profile_production_replay.py`
- `research/garch_proxy_native_sizing_audit.py`
- `research/garch_regime_family_audit.py`
- `research/garch_state_distinctness_audit.py`
- `research/garch_structural_decomposition.py`
- `research/garch_validated_scope_honest_test.py`
- `research/garch_w2_mechanism_pairing_audit.py`
- `research/garch_w2e_prior_session_carry_audit.py`
- `research/h2_exploitation_audit.py`
- `research/mnq_nyse_close_rr10_followup.py`
- `research/pr48_promotion_shortlist_v1.py`
- `research/prior_day_geometry_routing_audit.py`
- `research/r5_sizer_cross_lane_replication.py`
- `research/research_gc_mgc_translation_audit.py`
- `research/research_london_adjacent.py`
- `research/research_mae_stop_t6.py`
- `research/research_mes_compressed_spring.py`
- `research/research_mgc_e2_microstructure_pilot.py`
- `research/research_mgc_path_accurate_subr_v1.py`
- `research/research_mgc_payoff_compression_audit.py`
- `research/research_mgc_portfolio_diversifier_v1.py`
- `research/research_mnq_nyse_close_failure_mode_audit.py`
- `research/research_mnq_singapore_avoid.py`
- `research/research_portfolio_assembly.py`
- `research/research_portfolio_engine.py`
- `research/research_practical_playbook.py`
- `research/research_prop_firm_fit.py`
- `research/research_session_event_analysis.py`
- `research/research_signal_stack.py`
- `research/research_trade_book.py`
- `research/research_ultimate_portfolio.py`
- `research/research_vol_regime_filter.py`
- `research/research_vol_regime_switching.py`
- `research/research_vol_regime_wf.py`
- `research/research_wf_portfolio.py`
- `research/track_d_gate0_microstructure.py`
- `scripts/research/bull_short_adversarial.py`
- `scripts/research/bull_short_avoidance_audit.py`
- `scripts/research/cross_session_scan_v2.py`
- `scripts/research/exchange_range_t2t8.py`
- `scripts/research/gap_close_position_test.py`
- `scripts/research/gc_to_mgc_cross_validation.py`
- `scripts/research/per_session_bear_short_test.py`
- `scripts/research/prev_close_position_test.py`
- `scripts/research/scan_presession_features.py`
- `scripts/research/scan_presession_t2t8.py`
- `scripts/research/test_t80_oos.py`
- `scripts/research/wave4_feature_audit.py`
- `scripts/research/wave4_overnight_t2t8.py`
- `scripts/research/wave4_verify_survivors.py`
- `scripts/research/what_kills_edge.py`

## Verdict

ADVISORY at registration — not blocking. Like
`check_e2_lookahead_research_contamination` (advisory until its 18-script
registry was annotated), this surfaces the class without walling off commits.
The decision: leave advisory; promotion to BLOCKING follows annotation/refactor
of the 66-script pre-existing list. None of these scans is asserted contaminated
today — the flag means the guard is not *wired in*, which must be verified
script-by-script.

## Remediation

For each: route the filter/feature/holdout logic through the canonical guard
(`research.filter_utils.filter_signal`, `pipeline.session_guard.is_feature_safe`,
`trading_app.holdout_policy.enforce_holdout_date`, `...t0_correlation`), then
re-run the scanner. If a script is verified clean by other means, add the marker
`# canary-guard-coverage: cleared` with a one-line justification.

## Limitations

- A flag is NOT proof of contamination — it proves the guard is not *referenced*.
  A scan may be clean for other reasons (e.g. it only reads base rates with a
  filter mentioned in dead code). Each flagged script needs script-level review
  before any "contaminated" claim.
- Token-presence is a heuristic: a scan that imports a guard but does not
  actually call it on the right column would still pass this check. Tier-2
  (deferred) dynamic injection is what would catch a mis-wired call.
- The 66 count is a point-in-time snapshot; it shifts as scans land or are
  annotated. The live number is whatever the scanner prints, not this figure.
