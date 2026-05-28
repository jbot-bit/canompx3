---
task: |
  Lane B Stage 4a — promote NYSE_PREOPEN MNQ E2 NFP-spillover v1 prereg out of
  drafts/ and author the bounded verdict runner + tests.

  Scope: promote-and-build ONLY. Do NOT execute the runner. Do NOT emit verdict
  MD/CSV. Stage 4b (separate, gated by explicit approval) will run the runner
  against canonical layers and emit the verdict artifact.

  Why split: per institutional-rigor § 1 (self-review before claim-of-done) and
  § 2 (after any fix, review the fix), authoring + executing the runner in one
  stage couples build mistakes to verdict mistakes. The 4a/4b split lets us
  audit the runner against synthetic seeded data before letting it touch the
  canonical decision.

  Stages 1-3B already shipped the canonical machinery (resolver in pipeline/dst.py
  6aed3f72, session_guard ordering 15149e60, holiday source in
  pipeline/market_calendar.py 296c54f2, orb_outcomes/daily_features materialized).
  Both execution_gate.blockers in the prereg are now satisfied:
    (a) NYSE_PREOPEN session resolver IS in DYNAMIC_ORB_RESOLVERS (Stage 1)
    (b) NYSE holiday exclusion source IS canonical (Stage 2 — is_nyse_holiday in
        pipeline/market_calendar.py wired fail-closed into compute_orb_range for
        NYSE_PREOPEN only)

  K-budget re-verified live this session: estimate_k_budget.py PASS — N=27 fits
  MNQ 6.65yr horizon (requires 6.59yr, 0.06yr headroom).

  Canonical data re-verified live: 1757-1758 trading days per (orb_minutes, RR)
  cell on MNQ E2 CB1; IS=1673, OOS=84 per cell uniformly.

mode: IMPLEMENTATION
updated: 2026-05-28T00:00Z
agent: claude (opus 4.7)
supersedes: none

scope_lock:
  - docs/audit/hypotheses/drafts/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.draft.yaml
  - docs/audit/hypotheses/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.yaml
  - research/mnq_nyse_preopen_e2_nfp_spillover_v1.py
  - tests/test_research/test_mnq_nyse_preopen_e2_nfp_spillover_v1.py

## Blast Radius
- docs/audit/hypotheses/drafts/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.draft.yaml — DELETED via git mv. File body unchanged so SHA stays stable.
- docs/audit/hypotheses/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.yaml — NEW (mv target). Now drift-included; Check 57 (check_hypothesis_minbtl_compliance) will re-evaluate at commit time. K-budget PASS verified pre-promotion.
- research/mnq_nyse_preopen_e2_nfp_spillover_v1.py — NEW bounded standalone runner. Reads canonical orb_outcomes + daily_features only. Imports: pipeline.paths.GOLD_DB_PATH, pipeline.calendar_filters.is_nfp_day, trading_app.holdout_policy.HOLDOUT_SACRED_FROM, trading_app.hypothesis_loader.load_hypothesis_metadata + compute_file_sha + find_hypothesis_file_by_sha, research.oos_power.one_sample_power + one_sample_tstat + power_verdict, statsmodels.stats.multitest.multipletests. ZERO writes to gold.db. ZERO writes to validated_setups. ZERO writes to experimental_strategies (Stage 4a contract). ZERO allocator/live-config changes. The --emit-verdict flag is deliberately NOT implemented in 4a; the runner has a --dry-run-cells flag that prints the enumeration without computing the verdict, and a --check-prereq flag that asserts canonical layers + holiday wiring are intact. The verdict-emission path is Stage 4b.
- tests/test_research/test_mnq_nyse_preopen_e2_nfp_spillover_v1.py — NEW unit tests against an in-memory synthetic seed: K=27 enumeration, NFP-split partition correctness against pipeline.calendar_filters.is_nfp_day, DST-imbalance UNVERIFIED-not-DEAD, OOS-power floor respected, BH-FDR composition at K=27, single-use SHA gate raises on second invocation, no holiday contamination in canonical (real DB-touching test gated by env var). Tests use in-memory DuckDB; live gold.db is touched ONLY by an env-var-gated integration smoke test that defaults to skip.
- Reads: gold.db (read-only via integration test only, env-gated); hypothesis YAML (read). Writes: NONE to DB; NEW files on disk per scope_lock.
- No allocator/config/deployment change. No pipeline/ or trading_app/ touched. Strategy_discovery untouched.

## Acceptance (all required before deleting this stage file)
- git mv from drafts/ to docs/audit/hypotheses/ succeeds AND prereg SHA matches the pre-move SHA (proves no body change).
- Runner unit tests pass — show output.
- Runner --check-prereq mode prints PASS for: prereg loadable, K-budget PASS, is_nyse_holiday wired, orb_outcomes coverage non-empty for MNQ NYSE_PREOPEN E2 CB1. (Requires gold.db read; if locked by live bot, skip with a clear "gold.db locked by PID X" message — do not kill.)
- python pipeline/check_drift.py PASSES — show output.
- dead-code sweep: grep confirms the runner module is wired to its test, no orphan helpers.
- self-review against institutional-rigor § 1 + § 2: line-level audit of the runner before claiming done.

## NOT done by this stage (deferred to Stage 4b)
- Running the runner against canonical layers.
- Emitting docs/audit/results/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.md.
- Emitting the row-level CSV.
- Any write to experimental_strategies (deferred per user decision: verdict-MD/CSV only first, separate post-verdict authorization required for any DB write).
- BH-FDR composition over real cells (the math is implemented and unit-tested with synthetic data; production composition happens in 4b).
- Pinecone upsert / MEMORY.md verdict entry.

## NOT done by this stage (out of scope entirely)
- Any change to pipeline/, trading_app/, or strategy_discovery.
- Any change to validated_setups, allocator state, live config, lane_allocation.json.
- Promotion to live trading.
- Theory-grant upgrade. Prereg locks theory_grant: false.
- v2 prereg with full US_DATA_830 economic-calendar split.
