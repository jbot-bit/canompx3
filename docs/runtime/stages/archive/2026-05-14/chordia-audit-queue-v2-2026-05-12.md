# Stage: chordia-audit-queue-v2-2026-05-12

task: Generate ranked Chordia-audit candidate queue (Mode-A canonical recompute + lit-grounded gates) producing CSV + 5 audit MDs. Read-only research; no allocator/DB/live mutation; no pre-reg writes. Top-3 recommendation MD with verbatim literature excerpts.

mode: IMPLEMENTATION

scope_lock:
  - research/chordia_queue_recompute.py
  - docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md
  - docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv
  - docs/audit/results/2026-05-12-chordia-audit-queue-candidates.md
  - docs/audit/results/2026-05-12-chordia-audit-queue-blocked-reasons.md
  - docs/audit/results/2026-05-12-chordia-audit-queue-top3-prereg-recommendation.md
  - docs/plans/2026-05-12-chordia-audit-queue-v2-plan.md
  - HANDOFF.md

## Blast Radius

- research/chordia_queue_recompute.py — NEW read-only script. Zero callers. Delegates to: research.oos_power, research.filter_utils, trading_app.chordia, trading_app.holdout_policy, trading_app.config.ALL_FILTERS, pipeline.paths.GOLD_DB_PATH. Imports only canonical helpers; no re-encoding.
- 5 audit MDs + 1 CSV under docs/audit/results/ — NEW artifacts. Zero existing readers.
- docs/plans/...v2-plan.md — already committed in stage 0 (51b1cb03).
- HANDOFF.md — append-only entry.
- Reads: gold.db (read-only via duckdb read_only=True). docs/runtime/lane_allocation.json (read-only). docs/runtime/chordia_audit_log.yaml (read-only).
- Writes: ONLY the listed scope_lock files. Zero writes to pipeline/, trading_app/, scripts/, schema, validated_setups, allocator state.
- No pre-reg yaml authored. No bootstrap-runtime-control flags. No live state mutation.
