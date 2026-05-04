#!/usr/bin/env bash
# Wrapper: run full rebuild chain + pinecone sync for a single instrument
# Usage: bash scripts/tools/run_rebuild_with_sync.sh [INSTRUMENT]
# Default: MGC
#
# This extends the standard rebuild chain (outcome_builder -> discovery
# -> validator -> edge_families -> family_rr_locks) with E3 retirement
# and Pinecone knowledge sync.
#
# For a full all-instrument rebuild WITHOUT sync, see full_rebuild.sh.

set -e

cd "$(dirname "$0")/../.."

INSTRUMENT="${1:-MGC}"

# Write FAILED manifest on error (non-fatal — must not abort rebuild)
trap 'python scripts/tools/pipeline_status.py --write-manifest --instrument "$INSTRUMENT" --status-value FAILED --trigger SHELL 2>/dev/null || true' ERR

# Walk-forward enabled for all instruments (Mar 2026).
# All 4 active instruments have 5+ years of data — sufficient for WF.
# MGC uses WF_START_OVERRIDE=2022-01-01 in config.py (regime shift).
WF_FLAG=""

echo "=========================================="
echo "Rebuild + Pinecone Sync for $INSTRUMENT"
echo "Started: $(date)"
echo "=========================================="

# Step 1: Rebuild outcomes for ALL apertures (5m, 15m, 30m)
# NOTE: daily_features must already be built for each aperture.
# If stale, run first: python pipeline/build_daily_features.py --instrument X --orb-minutes Y
echo ""
echo "Step 1/10: Rebuilding outcomes (O5 + O15 + O30)..."
for OM in 5 15 30; do
    echo "  -- outcome_builder --orb-minutes $OM --"
    python trading_app/outcome_builder.py --instrument "$INSTRUMENT" --force --orb-minutes "$OM"
done

# Step 2: Discover strategies for ALL apertures
echo ""
echo "Step 2/10: Discovering strategies (O5 + O15 + O30)..."
for OM in 5 15 30; do
    echo "  -- strategy_discovery --orb-minutes $OM --"
    python trading_app/strategy_discovery.py --instrument "$INSTRUMENT" --orb-minutes "$OM"
done

# Step 3: Validate strategies
echo ""
echo "Step 3/10: Validating strategies..."
python trading_app/strategy_validator.py \
    --instrument "$INSTRUMENT" --min-sample 30 \
    --no-regime-waivers --min-years-positive-pct 0.75 \
    $WF_FLAG

# Step 4: Retire E3 strategies (validator promotes E3 to active; this fixes it)
echo ""
echo "Step 4/10: Retiring E3 strategies..."
python scripts/migrations/retire_e3_strategies.py

# Step 5: Build edge families
echo ""
echo "Step 5/10: Building edge families..."
python scripts/tools/build_edge_families.py --instrument "$INSTRUMENT"

# Step 6: Recompute family RR locks (SharpeDD criterion)
echo ""
echo "Step 6/10: Recomputing family RR locks..."
python scripts/tools/select_family_rr.py

# Step 7: Regenerate REPO_MAP (tracks file inventory drift)
echo ""
echo "Step 7/10: Regenerating REPO_MAP.md..."
python scripts/tools/gen_repo_map.py

# Step 8: Post-rebuild health check (drift + integrity + tests)
echo ""
echo "Step 8/10: Running post-rebuild health check..."
python pipeline/health_check.py

# Step 9: Surface promotion candidates (PM review queue)
echo ""
echo "Step 9/10: Surfacing promotion candidates..."
python scripts/tools/generate_promotion_candidates.py --format terminal --no-open

# Step 10: Sync to Pinecone
echo ""
echo "Step 10/10: Syncing knowledge to Pinecone..."
python scripts/tools/sync_pinecone.py

# Write rebuild manifest (success)
echo ""
echo "--- Writing rebuild manifest ---"
python scripts/tools/pipeline_status.py --write-manifest --instrument "$INSTRUMENT" --status-value COMPLETED --trigger SHELL 2>/dev/null || true

echo ""
echo "=========================================="
echo "Rebuild + sync complete for $INSTRUMENT"
echo "Finished: $(date)"
echo "=========================================="
