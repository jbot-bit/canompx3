task: Close Fork #2 fingerprint-completeness — add survival-verdict fields to build_profile_fingerprint + a drift check that prevents future field drift
mode: IMPLEMENTATION

## Scope Lock
- trading_app/derived_state.py
- pipeline/check_drift.py
- tests/test_trading_app/test_account_survival.py
- tests/test_pipeline/test_check_drift_profile_fingerprint_coverage.py

## Blast Radius
- derived_state.py build_profile_fingerprint: adds self_imposed_dd_dollars + daily_loss_dollars to the hashed payload. Read by 6 callers (account_survival, conditional_overlays, lifecycle_state, opportunity_awareness, sr_monitor + the arm gate). Hash changes → every currently-cached survival/SR/overlay report registers "profile fingerprint mismatch" on next read → one-time forced re-derivation. Correct behavior (old caches predate the fix); NO verdict-math change.
- check_drift.py: new additive check_profile_fingerprint_field_coverage (static AST scan of account_survival.py for profile.<attr> + getattr(profile,"...") reads; asserts each survival-relevant field is in the fingerprint payload minus a reviewed label-only allowlist). Count self-reports via len(). Fail-closed: unparseable → violate.
- Tests: mutation-style anti-regression on the two fields; injected-violation test for the drift check.
- Reads: trading_app source (AST). Writes: none to gold.db. Capital-adjacent (account_survival.py truth layer) but changes ONLY cache-invalidation, not sizing/gate math.
