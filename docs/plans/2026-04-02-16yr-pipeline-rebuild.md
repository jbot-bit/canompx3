# 16-Year Pipeline Rebuild Plan

**Created:** 2026-04-02
**Status:** READY TO EXECUTE
**Prerequisite:** gold.db exclusive access (no other writers)
**Estimated time:** 3-4 hours
**Trigger:** 1m bars extended to 2010-06-06 for MGC/MES/MNQ — downstream layers missing

## Context

bars_1m now covers 2010→2026 for all 3 active instruments.
Downstream layers (5m, daily_features, outcomes) only cover 2016→2026 (MGC/MNQ) or 2019→2026 (MES).
Per Carver (Systematic Trading Ch.3 Table 5): SR=0.5 needs 20yr to confirm. 16yr is a major improvement over 7-10yr.

## Gaps

| Layer | MGC gap | MES gap | MNQ gap |
|-------|---------|---------|---------|
| bars_1m | DONE | DONE | DONE |
| bars_5m | 2010-2016 MISSING | 2010-2019 MISSING | 2010-2016 MISSING |
| daily_features | 2010-2016 MISSING | 2010-2019 MISSING | 2010-2016 MISSING |
| orb_outcomes | 2010-2016 MISSING | 2010-2019 MISSING | 2010-2016 MISSING |

## Verified Clean

- Data quality: 0 NULLs, 0 duplicates, 0 high<low in 2010-2016 bars
- Continuity: smooth transition at boundary dates (no gaps)
- Contracts: 5-6/year (correct quarterly rollover)
- DST: resolvers produce correct Brisbane times for 2010-2016
- E2 honest entry: committed (orb_end_utc scan + filter exclusions)
- Cost model: fixed per-tick, correct for all eras
- No hardcoded dates in pipeline code

## Execution

```bash
export PYTHONPATH=$(pwd)

# ================================================================
# PHASE 1: Data Layers — Gap Fill (~45 min)
# ================================================================

# 5m bars
python pipeline/build_bars_5m.py --instrument MGC --start 2010-06-06 --end 2016-02-01
python pipeline/build_bars_5m.py --instrument MES --start 2010-06-06 --end 2019-02-12
python pipeline/build_bars_5m.py --instrument MNQ --start 2010-06-06 --end 2016-02-01

# Daily features — ALL 3 apertures for row integrity
for OM in 5 15 30; do
  python pipeline/build_daily_features.py --instrument MGC --start 2010-06-06 --end 2016-02-01 --orb-minutes $OM
  python pipeline/build_daily_features.py --instrument MES --start 2010-06-06 --end 2019-02-12 --orb-minutes $OM
  python pipeline/build_daily_features.py --instrument MNQ --start 2010-06-06 --end 2016-02-01 --orb-minutes $OM
done

# ================================================================
# PHASE 2: ATR Percentile Patch (~5 min)
# ================================================================

python -c "
from pipeline.daily_backfill import _patch_atr_percentiles
from pipeline.paths import GOLD_DB_PATH
for inst in ['MGC', 'MES', 'MNQ']:
    _patch_atr_percentiles(str(GOLD_DB_PATH), inst)
"

# ================================================================
# PHASE 3: Outcome Rebuild — O5 ONLY (~1-2 hours)
# ================================================================
# Force rebuild ALL dates — ATR context shifted for 2016-2018
# O5 ONLY — all 210 validated strategies are O5

python -m trading_app.outcome_builder --instrument MGC --force --orb-minutes 5
python -m trading_app.outcome_builder --instrument MES --force --orb-minutes 5
python -m trading_app.outcome_builder --instrument MNQ --force --orb-minutes 5

# ================================================================
# PHASE 4: Discovery — O5 ONLY + HOLDOUT ENFORCED (~30 min)
# ================================================================
# CRITICAL: --holdout-date 2026-01-01 protects sacred holdout

python -m trading_app.strategy_discovery --instrument MGC --orb-minutes 5 --holdout-date 2026-01-01
python -m trading_app.strategy_discovery --instrument MES --orb-minutes 5 --holdout-date 2026-01-01
python -m trading_app.strategy_discovery --instrument MNQ --orb-minutes 5 --holdout-date 2026-01-01

# ================================================================
# PHASE 5: Validation — WFE Gate Active (~30 min)
# ================================================================

python -m trading_app.strategy_validator --instrument MGC --min-sample 30 --no-regime-waivers --min-years-positive-pct 0.75
python -m trading_app.strategy_validator --instrument MES --min-sample 30 --no-regime-waivers --min-years-positive-pct 0.75
python -m trading_app.strategy_validator --instrument MNQ --min-sample 30 --no-regime-waivers --min-years-positive-pct 0.75

# ================================================================
# PHASE 6: Post-Rebuild (~15 min)
# ================================================================

python scripts/migrations/retire_e3_strategies.py
python scripts/tools/build_edge_families.py --instrument MGC
python scripts/tools/build_edge_families.py --instrument MES
python scripts/tools/build_edge_families.py --instrument MNQ
python scripts/tools/select_family_rr.py
python pipeline/check_drift.py
python pipeline/health_check.py
```

## What NOT to Do

- Do NOT rebuild O15/O30 outcomes or discovery — 0/210 validated strategies use them
- Do NOT recompute ATR percentiles for existing dates — impact is 23/210 strategies, minimal for recent dates
- Do NOT run discovery without --holdout-date — 2026 is sacred
- Do NOT use full_rebuild.sh without updating its stale dates (still says 2016/2021)

## Post-Rebuild Checklist

- [ ] Verify strategy count (expect DROP from current 210)
- [ ] Check per-instrument breakdown (MGC likely drops further)
- [ ] Verify WFE gate killed overfit strategies (WFE < 0.50)
- [ ] Check era-dependency flags (>50% R from one year)
- [ ] Run /trade-book to see new deployed lane candidates
- [ ] Update prop_profiles if lanes change
- [ ] Update full_rebuild.sh start dates to 2010-06-06

## Fix After Rebuild

- [ ] Update full_rebuild.sh: MGC/MES/MNQ start dates → 2010-06-06
- [ ] Update run_rebuild_with_sync.sh: add --holdout-date 2026-01-01 default

## Expected Outcomes (per Carver/Pardo/Aronson)

- Strategy count DROPS (16yr is harder than 7-10yr)
- MGC drops further (gold bear 2010-2019 adds more negative years)
- MNQ regime-dependent strategies tested against 2010-2019 (pre-COVID)
- WFE >= 0.50 gate kills 8+ overfit strategies
- Surviving strategies have institutional-grade confidence
