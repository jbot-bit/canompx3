---
mode: IMPLEMENTATION
slug: ml-v3-stage-3-execution
task: Execute the pre-registered ML V3 pooled meta-label research sprint per binding 22-step contract in the hypothesis YAML
agent: Claude Code (institutional ML V3 sprint)
created: 2026-04-11
parent_plan: docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence.yaml
---

# Stage: ML V3 — Stage 3: Research Sprint Execution

## Scope Lock

- scripts/research/ml_v3_pooled_discovery.py
- docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md
- docs/audit/ml_v3/

## Blast Radius

Research script that reads from gold.db read-only, imports `trading_app.ml.features.transform_to_features`, `trading_app.config.ALL_FILTERS`, `pipeline.cost_model.COST_SPECS`, `pipeline.paths.GOLD_DB_PATH`. No writes to production tables. No changes to production code paths. Outputs go to `docs/audit/ml_v3/` and `docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md`, both in SAFE_DIRS. Tests: runs a self-contained smoke test of `orb_volume_norm` rolling normalisation inside gate G6. Drift: no changes to `pipeline/check_drift.py` or its rule set. Downstream: the postmortem feeds into the Stage 4 decision (deploy or delete). If the verdict is DEAD (high likelihood given debug-run CPCV AUC 0.5002 on trial 1), Stage 4 will delete `trading_app/ml/` and update the Blueprint NO-GO registry. If VERDICT is POSITIVE, Stage 4 deploys to shadow mode.

## Purpose

Execute the pre-registered 22-step Stage 3 contract from the hypothesis YAML (commit a165ff80 v2) binding to the RR-stratified pooled meta-label experiment. Produce a postmortem documenting trial results, kill-criteria triggers, and the verdict.

## Acceptance Criteria

1. Script runs to completion with exit code 0 (survivor found) or 1 (DEAD).
2. Postmortem file written atomically to `docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md`.
3. Postmortem contains all required sections: metadata, G2 drops, per-trial tables (training stats, holdout stats, per-strategy table, kill triggers), narrative, caveats/deviations.
4. Any deviation from the hypothesis file is explicitly documented in the postmortem (not hidden).
5. No query against the sacred 2026 holdout before step 17 of the execution contract.

## Rollback Plan

If the script fails mid-execution, the tempfile-based postmortem write ensures no partial postmortem is left on disk. Re-run from the top. If the hypothesis file itself needs revision, a new amendment (v3) must be written, committed, and the run restarted with the new SHA stamped.
