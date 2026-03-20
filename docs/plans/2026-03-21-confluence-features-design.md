# Multi-ORB Confluence Features — Draft Design

**Status:** DRAFT — research still in progress. Do not implement.
**Date:** 2026-03-21
**Context:** User observations from live MNQ trading about multi-ORB interactions, VWAP confluences, and pre-session behavior.

---

## Core Insight

Most of the user's observations are ALREADY captured by existing ML features (level proximity, cross-session counts, VWAP). The ML at RR2.0 O30 passed bootstrap (p=0.005) WITH these features. The question is whether we can squeeze more from 2-3 genuinely new features.

## User Observations → Existing Features

| Observation | Already Built? | Feature |
|---|---|---|
| Prior session ORB levels as S/R | YES | `nearest_level_to_high_R`, `nearest_level_to_low_R`, `levels_within_1R/2R` |
| Multi-ORB directional interactions | YES | `prior_sessions_broken`, `prior_sessions_long/short` |
| Current ORB nested in prior ORB | YES | `orb_nested_in_prior` |
| VWAP position relative to ORB | PARTIAL | `orb_vwap` exists, but not crossed with prior ORB levels |
| Prior day high/low proximity | PARTIAL | `prev_day_range` global; PDH/PDL blacklisted for Asian sessions |
| Pre-session "baiting" behavior | PARTIAL | `orb_pre_velocity` captures direction, not sweep pattern |
| Prior ORB ranges from night before | YES | Level proximity includes all prior sessions |
| Round numbers | NO | High data-mining risk |
| Trend lines | NO | Too subjective to systematize |

## Genuinely New Feature Candidates

1. **`vwap_in_prior_orb`** — Binary: is pre-break VWAP between nearest prior ORB high/low? Mechanism: VWAP = fair value inside old range = double resistance.
2. **`pdh_pdl_proximity_R`** — Distance to PDH/PDL in R (US sessions only, session-aware blacklist). Mechanism: institutional overnight positioning.
3. **`pre_session_sweep`** — Binary: did price touch both ORB edges in last 5min pre-session? Mechanism: algorithmic liquidity hunting. WARNING: break quality adjacent (NO-GO territory).

## Proposed Path (When Ready)

- **Phase 0:** Audit existing model feature importance — are level proximity features contributing or being E6-dropped?
- **Phase 1:** Build 2-3 new features (if Phase 0 says proceed)
- **Phase 2:** Univariate quartile signal check (no optimization)
- **Phase 3:** ML integration + bootstrap verification
- **No changes to grid search or raw baseline strategies.**

## Key Risks

- Pre-session sweep may be break quality 2.0 (same NO-GO) — test separately
- PDH/PDL may confirm prior-day context = NOISE finding
- New features may add noise → AUC drops (E6 mitigates)
- Round numbers = data mining trap

## Decision: ML Path, Not Grid

Features belong in the ML layer. Adding as grid filters would multiply hypothesis space → FDR kills everything. ML handles interactions naturally + bootstrap validates.
