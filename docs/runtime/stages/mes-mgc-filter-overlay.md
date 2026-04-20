---
stage: mes-mgc-filter-overlay-family
mode: IMPLEMENTATION
task: "Filter-overlay family on MES + MGC (cross-instrument completion of PR #53 null). Tests whether 5 canonical filters (COST_LT12, ORB_G5, OVNRNG_50, ATR_P50, VWAP_MID_ALIGNED) rescue any MES/MGC cell that failed unfiltered-baseline at K=176."
updated: "2026-04-21"
scope_lock:
  - "docs/audit/hypotheses/2026-04-21-mes-mgc-filter-overlay-family-v1.yaml"
  - "research/mes_mgc_filter_overlay_family_v1.py"
  - "docs/audit/results/2026-04-21-mes-mgc-filter-overlay-family-v1.md"
  - "HANDOFF.md"
acceptance:
  - "Pre-reg LOCKED with Phase 0 grounding (Bailey MinBTL, Chordia t>=3.0, Harvey-Liu BH-FDR, Fitschen/Chan momentum)."
  - "Canonical filter delegation: uses research.filter_utils.filter_signal — no re-encoding of filter logic."
  - "K_family bounded <=210 (14 instrument-sessions × 3 RRs × 5 filters) — pre-committed, within Bailey 300 budget."
  - "Overnight-look-ahead gate respected: OVNRNG_50 only routed to sessions >=17:00 Brisbane (LONDON_METALS onwards). Asian-window sessions excluded for this filter per overnight_range look-ahead rule in StrategyFilter docstring."
  - "daily_features triple-join on (trading_day, symbol, orb_minutes) per RULE 9."
  - "Result MD reports filter-on ExpR, t, q_BH, WFE, N_OOS, ratio_OOS_IS, era_min, fire_rate per cell."
  - "Drift check passes."
---
