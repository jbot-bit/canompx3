# Rebuild outcomes + full chain — MNQ only

task: Run `bash scripts/tools/run_rebuild_with_sync.sh MNQ` end-to-end. Refresh orb_outcomes through 2026-05-20 (catching the 1-day lag vs daily_features), re-discover, re-validate, rebuild edge_families, run health_check, surface promotion candidates, sync to Pinecone. Capture pre/post diff of validated_setups (active count, row hash, promoted/demoted strategy_ids) and report allocator-impact risk. Do not run MES or MGC.

mode: TRIVIAL

agent: claude (opus 4.7)

## Scope Lock
- scripts/tools/run_rebuild_with_sync.sh — execute only, no edits
- gold.db — read+write via the wrapper's canonical pipeline scripts
- rebuild_baseline_mnq.json — local diff snapshot file (already written)

## Blast Radius
- WRITES: orb_outcomes (MNQ rows, 3 apertures), validated_setups (MNQ rows), edge_families (MNQ), REPO_MAP.md, Pinecone vectors. ~2.1M orb_outcomes rows in scope; ~789 validated_setups rows in scope.
- READS: bars_1m (MNQ), daily_features (MNQ, all apertures).
- LIVE-IMPACT: validated_setups changes can shift allocator inputs at next rebalance. Currently deployed MNQ lanes (MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100, MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08, MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15, MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12) could be promoted/demoted if walk-forward stats shift. No automatic rebalance — allocator runs separately.
- Brisbane Sunday 2026-05-24, no US sessions today; Monday Brisbane evening = US Monday cash open. ~24h buffer to inspect output before next live signal.
- Idempotency: outcome_builder uses --force; strategy_validator uses INSERT OR REPLACE per project convention.
- Rollback: orb_outcomes / validated_setups can be re-derived by re-running same chain. Pinecone sync is the least-reversible step (vector deletes require explicit removal). Stops short before live config / lane_allocation.json.

## Acceptance
- All 10 wrapper steps complete (Steps 1-10).
- health_check.py passes.
- check_drift.py passes.
- post-run diff shows: validated_active count, hash change, list of promoted strategy_ids (in post but not pre as active), demoted strategy_ids (in pre as active, not in post as active), orb_outcomes max_trading_day advanced to 2026-05-20.
- Stage file deleted after acceptance report delivered to user.
