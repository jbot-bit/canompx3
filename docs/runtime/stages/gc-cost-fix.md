mode: IMPLEMENTATION
task: "Fix GC cost spec (spread/slippage 10x too low) and re-run COST_LT10 discovery"
updated: 2026-04-10T19:00:00+10:00
scope_lock:
  - pipeline/cost_model.py
blast_radius: "GC cost spec only. No impact on MGC/MNQ/MES. Existing GC orb_outcomes pnl_r unaffected (price-based). GC pnl_dollars will change. COST_LT filter pass rates on GC will change."
acceptance:
  - "GC spread_doubled=20.00, slippage=20.00, commission_rt=17.40"
  - "COST_LT10 threshold same in points for GC and MGC (both ~5.74 pts)"
  - "Re-run discovery + validation for COST_LT10 hypothesis"
