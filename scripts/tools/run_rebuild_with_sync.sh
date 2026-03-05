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

# Per-instrument walkforward flags (MNQ has 5yr data, uses WF)
case "$INSTRUMENT" in
    MNQ) WF_FLAG="" ;;
    *)   WF_FLAG="--no-walkforward" ;;
esac

echo "=========================================="
echo "Rebuild + Pinecone Sync for $INSTRUMENT"
echo "Started: $(date)"
echo "=========================================="

# Step 1: Rebuild outcomes for ALL apertures (5m, 15m, 30m)
# NOTE: daily_features must already be built for each aperture.
# If stale, run first: python pipeline/build_daily_features.py --instrument X --orb-minutes Y
echo ""
echo "Step 1/9: Rebuilding outcomes (O5 + O15 + O30)..."
for OM in 5 15 30; do
    echo "  -- outcome_builder --orb-minutes $OM --"
    python trading_app/outcome_builder.py --instrument "$INSTRUMENT" --force --orb-minutes "$OM"
done

# Step 2: Discover strategies for ALL apertures
echo ""
echo "Step 2/9: Discovering strategies (O5 + O15 + O30)..."
for OM in 5 15 30; do
    echo "  -- strategy_discovery --orb-minutes $OM --"
    python trading_app/strategy_discovery.py --instrument "$INSTRUMENT" --orb-minutes "$OM"
done

# Step 3: Validate strategies
echo ""
echo "Step 3/9: Validating strategies..."
python trading_app/strategy_validator.py \
    --instrument "$INSTRUMENT" --min-sample 50 \
    --no-regime-waivers --min-years-positive-pct 0.75 \
    $WF_FLAG

# Step 4: Retire E3 strategies (validator promotes E3 to active; this fixes it)
echo ""
echo "Step 4/9: Retiring E3 strategies..."
python scripts/migrations/retire_e3_strategies.py

# Step 5: Build edge families
echo ""
echo "Step 5/9: Building edge families..."
python scripts/tools/build_edge_families.py --instrument "$INSTRUMENT"

# Step 6: Recompute family RR locks (SharpeDD criterion)
echo ""
echo "Step 6/9: Recomputing family RR locks..."
python scripts/tools/select_family_rr.py

# Step 7: Regenerate REPO_MAP (tracks file inventory drift)
echo ""
echo "Step 7/9: Regenerating REPO_MAP.md..."
python scripts/tools/gen_repo_map.py

# Step 8: Post-rebuild health check (drift + integrity + tests)
echo ""
echo "Step 8/9: Running post-rebuild health check..."
python pipeline/health_check.py

# Step 9: Sync to Pinecone
echo ""
echo "Step 9/9: Syncing knowledge to Pinecone..."
python scripts/tools/sync_pinecone.py

echo ""
echo "=========================================="
echo "Rebuild + sync complete for $INSTRUMENT"
echo "Finished: $(date)"
echo "=========================================="
