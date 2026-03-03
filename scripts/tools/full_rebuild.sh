#!/usr/bin/env bash
# Full rebuild chain for all 4 instruments
# Run from project root: bash scripts/tools/full_rebuild.sh
# NOTE: Pinecone sync is NOT included here. Use run_rebuild_with_sync.sh
# for single-instrument rebuilds with Pinecone knowledge sync.
set -e

cd "$(dirname "$0")/../.."

echo "============================================"
echo "FULL REBUILD — ALL 4 INSTRUMENTS"
echo "Started: $(date)"
echo "============================================"

# Instrument configs: name, start, end, extra validator flags
declare -A STARTS ENDS EXTRA_FLAGS
STARTS[MGC]="2016-02-01"; ENDS[MGC]="2026-02-04"; EXTRA_FLAGS[MGC]="--no-walkforward"
STARTS[MNQ]="2021-02-04"; ENDS[MNQ]="2026-02-03"; EXTRA_FLAGS[MNQ]=""
STARTS[MES]="2019-02-12"; ENDS[MES]="2026-02-11"; EXTRA_FLAGS[MES]="--no-walkforward"
STARTS[M2K]="2021-02-22"; ENDS[M2K]="2026-02-20"; EXTRA_FLAGS[M2K]="--no-walkforward"

for INST in MGC MNQ MES M2K; do
    echo ""
    echo "============================================"
    echo "[$INST] Starting full chain (${STARTS[$INST]} to ${ENDS[$INST]})"
    echo "============================================"

    echo ""
    echo "--- [$INST] Step 1/5: Outcome Builder (--force) ---"
    python trading_app/outcome_builder.py \
        --instrument "$INST" --force \
        --start "${STARTS[$INST]}" --end "${ENDS[$INST]}" 2>&1 | tail -5
    echo "[$INST] Outcomes done: $(date)"

    echo ""
    echo "--- [$INST] Step 2/5: Strategy Discovery ---"
    python trading_app/strategy_discovery.py --instrument "$INST" 2>&1 | tail -5
    echo "[$INST] Discovery done: $(date)"

    echo ""
    echo "--- [$INST] Step 3/5: Strategy Validator ---"
    python trading_app/strategy_validator.py \
        --instrument "$INST" --min-sample 50 \
        --no-regime-waivers --min-years-positive-pct 0.75 \
        ${EXTRA_FLAGS[$INST]} 2>&1 | tail -10
    echo "[$INST] Validation done: $(date)"

    echo ""
    echo "--- [$INST] Step 4/5: Retire E3 strategies ---"
    python scripts/migrations/retire_e3_strategies.py 2>&1 | tail -3
    echo "[$INST] E3 retirement done: $(date)"

    echo ""
    echo "--- [$INST] Step 5/5: Edge Families ---"
    python scripts/tools/build_edge_families.py --instrument "$INST" 2>&1 | tail -5
    echo "[$INST] Edge families done: $(date)"

    echo ""
    echo "============================================"
    echo "[$INST] COMPLETE"
    echo "============================================"
done

echo ""
echo "============================================"
echo "ALL INSTRUMENTS COMPLETE"
echo "Finished: $(date)"
echo "============================================"

# Post-rebuild summary
echo ""
echo "--- Post-rebuild: validated_setups count ---"
python -c "
import duckdb
con = duckdb.connect('gold.db', read_only=True)
total = con.execute(\"SELECT COUNT(*) FROM validated_setups WHERE status = 'active'\").fetchone()[0]
print(f'Total active validated: {total}')
for inst in ['MGC', 'MNQ', 'MES', 'M2K']:
    n = con.execute(f\"SELECT COUNT(*) FROM validated_setups WHERE status = 'active' AND instrument = '{inst}'\").fetchone()[0]
    fdr = con.execute(f\"SELECT COUNT(*) FROM validated_setups WHERE status = 'active' AND instrument = '{inst}' AND fdr_significant = true\").fetchone()[0]
    print(f'  {inst}: {n} total, {fdr} FDR-pass')
con.close()
"

# Post-rebuild: Regenerate REPO_MAP
echo ""
echo "--- Post-rebuild: Regenerating REPO_MAP.md ---"
python scripts/tools/gen_repo_map.py

# Post-rebuild: Full health check
echo ""
echo "--- Post-rebuild: Health check ---"
python pipeline/health_check.py
