---
task: "Stage B2 — author research/vwap_mid_family_pooled_oos_v1.py runner"
mode: IMPLEMENTATION
scope_lock:
  - research/vwap_mid_family_pooled_oos_v1.py
---

## Blast Radius
- research/vwap_mid_family_pooled_oos_v1.py — new file, no callers.
- Reads (read-only): gold.db via duckdb (orb_outcomes ⨯ daily_features), trading_app.config.ALL_FILTERS['VWAP_MID_ALIGNED'], trading_app.holdout_policy.HOLDOUT_SACRED_FROM, trading_app.config.WF_START_OVERRIDE['MNQ'], research.filter_utils.filter_signal, research.oos_power, trading_app.chordia, trading_app.eligibility.builder.parse_strategy_id, statsmodels.regression.linear_model.OLS, the pre-reg yaml at docs/audit/hypotheses/2026-05-17-mnq-usdata1000-vwapmid-family-pooled-oos-v1.yaml, and docs/runtime/chordia_audit_log.yaml.
- Writes: NONE during Stage B2 (execution_gate.allowed_now: false → live writes refused; --dry-run prints WOULD WRITE lines only).
- Post-promotion writes (Stage B3+, not this stage): docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-family-pooled-oos-v1.md + .per-cell.csv + .by-year.csv.
- No edits to validated_setups, lane_allocation.json, chordia_audit_log.yaml, bot_state.json, live_config.json, pipeline/check_drift.py, or any prior result MD / pre-reg.
- Drift-check impact: zero (no production code edited; new research/ file).
