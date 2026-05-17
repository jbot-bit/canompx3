---
task: "Path A — VWAP_MID_ALIGNED short-only O15 RR1.5 K=1 with post-selection disclosure (Stages 1+2)"
mode: IMPLEMENTATION
scope_lock:
  - docs/audit/hypotheses/drafts/2026-05-17-mnq-usdata1000-vwapmid-short-only-k1-v1.draft.yaml
  - research/vwap_mid_short_only_o15_k1_v1.py
  - docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-short-only-k1-v1.md
  - docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-short-only-k1-v1.deployment.csv
---

## Blast Radius
- docs/audit/hypotheses/drafts/2026-05-17-mnq-usdata1000-vwapmid-short-only-k1-v1.draft.yaml — new file, quarantine path; reviewed by humans before promotion. No code reads it until Stage 2.
- research/vwap_mid_short_only_o15_k1_v1.py — new file, no callers. Reads (read-only): gold.db via duckdb (orb_outcomes JOIN daily_features), pipeline.paths.GOLD_DB_PATH, pipeline.cost_model.COST_SPECS['MNQ'], trading_app.config.ALL_FILTERS['VWAP_MID_ALIGNED'], trading_app.holdout_policy.HOLDOUT_SACRED_FROM, trading_app.config.WF_START_OVERRIDE['MNQ'], research.filter_utils.filter_signal, research.oos_power.one_sample_power + power_verdict, trading_app.chordia.compute_chordia_t, trading_app.eligibility.builder.parse_strategy_id, trading_app.lane_correlation (read-only pairwise correlation against docs/runtime/lane_allocation.json), statsmodels.regression.linear_model.OLS.
- docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-short-only-k1-v1.md — new result MD with deployment-gate evidence.
- docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-short-only-k1-v1.deployment.csv — new four-gate audit table (Chordia / Harvey-Liu / cost-aware / correlation).
- Writes: NONE to validated_setups, experimental_strategies, lane_allocation.json, chordia_audit_log.yaml, bot_state.json, live_config.json, prop_profiles.py, pipeline/check_drift.py, or any prior pre-reg / result MD.
- Drift-check impact: zero (no production code edited; new research/ + docs/audit/ files only).
