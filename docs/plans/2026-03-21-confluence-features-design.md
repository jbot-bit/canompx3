# Multi-ORB Confluence Features — Draft Design

**Status:** DRAFT — research still in progress. Do not implement.
**Date:** 2026-03-21
**Context:** User observations from live MNQ trading about multi-ORB interactions, VWAP confluences, and pre-session behavior.

---

## ⚠ ML STATUS: UNPROVEN (4 FAILs — see ml_methodology_audit.md)

The ML meta-labeling system has **4 unresolved methodology failures** identified on Mar 21 2026:

1. **Negative baselines violate de Prado (AIFML Ch 3.6):** Meta-labeling assumes positive-edge primary model. 5/7 "survivors" trained on negative-baseline sessions.
2. **Bootstrap resolution floor:** p=0.005 is the minimum reportable value at 200 permutations (Phipson & Smyth 2010). Need 5000+ for reliable p-values.
3. **Underpowered samples:** EPV (events per variable) = 2.4 for NYSE_OPEN O30 (55 positives / 23 features). Literature requires EPV >= 10 (Peduzzi et al. 1996).
4. **Selection bias:** 7 sessions selected from 12, then tested. White 2000: selection-then-test = data snooping.

**Implication for this design:** Any claim that "ML confirms these patterns" is WRONG. The existing ML features (level proximity, cross-session counts) exist in the code but are NOT validated. Adding more features to the current ML system would make the sample size problem WORSE (more features, same 55 positives).

---

## Core Insight

Most of the user's observations map to features that EXIST in the ML codebase (level proximity, cross-session counts, VWAP). However, those features are **unproven** — the ML system using them has 4 methodology failures. The features should be tested as **standalone univariate signals on positive baselines FIRST**, independent of ML.

## User Observations → Existing Features

| Observation | Feature Exists? | Feature Name | Validated? |
|---|---|---|---|
| Prior session ORB levels as S/R | YES | `nearest_level_to_high_R`, `nearest_level_to_low_R`, `levels_within_1R/2R` | **NO** — in unproven ML |
| Multi-ORB directional interactions | YES | `prior_sessions_broken`, `prior_sessions_long/short` | **NO** — in unproven ML |
| Current ORB nested in prior ORB | YES | `orb_nested_in_prior` | **NO** — in unproven ML |
| VWAP position relative to ORB | PARTIAL | `orb_vwap` exists, not crossed with prior ORB levels | **NO** |
| Prior day high/low proximity | PARTIAL | `prev_day_range` global; PDH/PDL blacklisted for Asian sessions | Prior-day context = NOISE (tested) |
| Pre-session "baiting" behavior | PARTIAL | `orb_pre_velocity` captures direction, not sweep pattern | **NO** |
| Prior ORB ranges from night before | YES | Level proximity includes all prior sessions | **NO** — in unproven ML |
| Round numbers | NO | Not built | High data-mining risk |
| Trend lines | NO | Too subjective to systematize | — |

## Genuinely New Feature Candidates

1. **`vwap_in_prior_orb`** — Binary: is pre-break VWAP between nearest prior ORB high/low? Mechanism: VWAP = fair value inside old range = double resistance.
2. **`pdh_pdl_proximity_R`** — Distance to PDH/PDL in R (US sessions only, session-aware blacklist). Mechanism: institutional overnight positioning.
3. **`pre_session_sweep`** — Binary: did price touch both ORB edges in last 5min pre-session? Mechanism: algorithmic liquidity hunting. WARNING: break quality adjacent (NO-GO territory).

## Proposed Path (When Ready)

**Critical ordering: univariate FIRST, ML LAST (and only after 4 FAILs are fixed).**

- **Phase 0:** Audit existing features as standalone univariate signals on POSITIVE baselines. Quartile split per feature × session. Does the top quartile have meaningfully different ExpR? No ML, no optimization — just describe.
- **Phase 1:** For features showing univariate signal, test as simple binary filters on positive-baseline sessions. FDR correction at honest test count.
- **Phase 2:** Build 2-3 new features (if Phase 0/1 show existing features have signal worth extending).
- **Phase 3:** ML integration ONLY IF the 4 methodology FAILs are resolved:
  - Retrain on positive-baseline sessions only (de Prado compliant)
  - 5000+ bootstrap permutations
  - EPV >= 10 (reduce features or pool data)
  - Pre-register test sessions (no selection bias)
- **No changes to grid search or raw baseline strategies.**

## Key Risks

- Pre-session sweep may be break quality 2.0 (same NO-GO) — test separately
- PDH/PDL may confirm prior-day context = NOISE finding
- Adding features to current ML makes EPV problem WORSE (more features, same positives)
- Round numbers = data mining trap
- Existing features may show zero univariate signal (in which case ML wasn't learning from them either)

## Decision: Univariate First, ML Path Only After Fixes

Features should be tested as standalone signals on positive baselines BEFORE touching ML. The ML system has 4 unresolved methodology failures. Adding features to a broken system doesn't fix the system.

Grid search is still the WRONG path (combinatorial explosion). The correct sequence is:
1. Univariate signal scan (standalone)
2. Simple binary filter test (if signal found)
3. ML integration (only after 4 FAILs resolved)
