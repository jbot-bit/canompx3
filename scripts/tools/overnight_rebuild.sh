#!/usr/bin/env bash
# Overnight rebuild: MNQ/MES/M2K outcomes + ML retrain + sync
# MGC already rebuilt on scratch DB. MGC/MNQ ML already retrained.
set -e
cd /c/Users/joshd/canompx3

LOG="/c/Users/joshd/canompx3/research/output/overnight_rebuild_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== OVERNIGHT REBUILD ==="
echo "Started: $(date)"
echo "Scratch DB: C:/db/gold.db"
echo ""

# ─── Step 1: Rebuild MNQ, MES, M2K on scratch DB ───
for INST in MNQ MES M2K; do
    echo ""
    echo "========================================"
    echo "Rebuilding $INST..."
    echo "========================================"

    # Walk-forward enabled for all instruments (Mar 2026). All have 5+ years of data.
    python trading_app/outcome_builder.py --instrument "$INST" --force
    python trading_app/strategy_discovery.py --instrument "$INST"
    python trading_app/strategy_validator.py \
        --instrument "$INST" --min-sample 50 \
        --no-regime-waivers --min-years-positive-pct 0.75
    python scripts/migrations/retire_e3_strategies.py
    python scripts/tools/build_edge_families.py --instrument "$INST"

    echo "$INST rebuild complete at $(date)"
done

# ─── Step 2: Post-rebuild checks on scratch DB ───
echo ""
echo "=== POST-REBUILD CHECKS ==="
python scripts/tools/gen_repo_map.py
python pipeline/health_check.py
echo "Health check passed at $(date)"

# ─── Step 3: Copy scratch DB to project root ───
echo ""
echo "=== COPYING SCRATCH DB TO PROJECT ROOT ==="
cp "C:/db/gold.db" "C:/Users/joshd/canompx3/gold.db"
echo "DB copied at $(date)"

# ─── Step 4: Retrain ML for MES + M2K (MGC/MNQ already done) ───
echo ""
echo "=== ML RETRAIN (MES + M2K) ==="
echo "ML retrain: MES"
python -m trading_app.ml.meta_label --instrument MES --single-config \
    --rr-target 2.5 --config-selection max_samples --skip-filter
echo "MES ML complete at $(date)"

echo ""
echo "ML retrain: M2K"
python -m trading_app.ml.meta_label --instrument M2K --single-config \
    --rr-target 1.0 --config-selection max_samples --skip-filter
echo "M2K ML complete at $(date)"

# ─── Step 5: Pinecone sync ───
echo ""
echo "=== PINECONE SYNC ==="
python scripts/tools/sync_pinecone.py
echo "Sync complete at $(date)"

# ─── Step 6: Final health check on project root DB ───
echo ""
echo "=== FINAL HEALTH CHECK (project root DB) ==="
DUCKDB_PATH=gold.db python pipeline/health_check.py
echo "Final health check passed at $(date)"

echo ""
echo "=== ALL DONE ==="
echo "Log: $LOG"
echo "Finished: $(date)"
