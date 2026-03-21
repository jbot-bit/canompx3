# ML Methodology Fix — Design Doc (V2: Post-Audit)

**Status:** APPROVED for phased implementation
**Date:** 2026-03-21 (V2 after zero-context audit + truth-check)
**Audit:** `docs/plans/2026-03-21-ml-zero-context-audit.md` (28 questions, 5 kill shots)

**Key constraint:** Layer 1 (O5 RR1.0 raw baseline) is CLEAN (p<1e-9). Do NOT touch it.

---

## Problem Statement

The current ML system produces numbers that cannot be trusted. 6 proven bugs:

| # | Bug | Evidence | Severity |
|---|-----|----------|----------|
| A | Non-deterministic config selection | `ORDER BY cnt DESC` with no tiebreaker (`features.py:979`) | Medium |
| B | EUROPE_FLOW cross-session lookahead in winter | Static `SESSION_CHRONOLOGICAL_ORDER` — EF index 5 includes LM index 4, but in winter EF (17:00) precedes LM (18:00). ~42% of EF rows contaminated. | High |
| C | Constant-column drop on full data (train+test) | `session_data = X_session.iloc[session_indices]` not `train_idx` (`meta_label.py:360`) | Low |
| D | No universe-wide BH/FDR | Bootstrap reports per-config p<0.05. No FDR code exists. BH at 7 tests → 1 survivor. BH at 68 (full sweep) → 0 survivors. | Critical |
| E | No positive-baseline gate | Training proceeds on sessions with Sharpe=nan (10/12 at O30 RR2.0). De Prado (AIFML Ch 3.6) violation. | Critical |
| F | EPV = 2.2 | 25 E6 features, ~55 positives per session at O30 RR2.0. (Peduzzi 1996: need ≥10) | Critical |

**All 5K bootstrap p-values are from the BROKEN system. "3/7 passed" does not survive BH FDR even at family=7.**

### Additional Context from Baseline Queries

| Config | Positive Sessions | Best ML Territory? |
|--------|------------------|--------------------|
| O5 RR1.0 | **12/12** | YES — all positive, high N (~1300/session), EPV=86 with 5 features |
| O5 RR2.0 | 11/12 | Good |
| O15 RR1.0 | 12/12 | Good |
| O15 RR2.0 | 9/12 | Moderate |
| O30 RR1.0 | 12/12 | Good |
| O30 RR2.0 | **6/12** | NO — old ML territory. NYSE_OPEN (-0.136R) and US_DATA_1000 (-0.125R) are negative. |

---

## Implementation: 6 Fixes, Phased

Each fix is implemented, verified, and committed independently. No batching without verification.

### Fix A: Deterministic Config Selection
- **File:** `features.py` ~line 979
- **Change:** Add tiebreaker to `ORDER BY cnt DESC` → `ORDER BY cnt DESC, rr_target ASC, entry_model ASC, orb_minutes ASC`
- **Blast radius:** Query change only. All callers of `load_single_config_feature_matrix` get deterministic results.
- **Verify:** Run query twice, confirm same config selected both times.

### Fix B: EUROPE_FLOW / LONDON_METALS DST Cross-Session Lookahead
- **File:** `meta_label.py` `_get_session_features()` ~line 196
- **Change:** Drop cross-session features for EUROPE_FLOW and LONDON_METALS (they swap order by season — neither can safely include the other).
- **Alternative considered:** Dynamic per-date ordering. Rejected — too complex for the benefit.
- **Blast radius:** Only affects cross-session features for 2 sessions. No schema change.
- **Verify:** After fix, EF and LM models train without `prior_sessions_broken`, `prior_sessions_long`, `prior_sessions_short`, `nearest_level_to_high_R`, etc.

### Fix C: Train-Only Constant-Column Drop
- **File:** `meta_label.py` ~line 360
- **Change:** `session_data = X_session.iloc[session_indices]` → `session_data = X_session.iloc[train_idx]`
- **Blast radius:** Negligible — same columns drop in practice (entry_model one-hots are constant in train and test).
- **Verify:** Log which columns drop, confirm same set as before.

### Fix E: Positive Baseline Gate
- **File:** `meta_label.py` — inside per-session loop, before training
- **Change:** Compute raw ExpR for this (session, aperture, RR) from pnl_r in the train split. If ExpR ≤ 0, skip with reason `"negative_baseline_expr={value}"`.
- **Gate uses TRAIN split only** — no test-set info.
- **Blast radius:** Blocks negative-baseline sessions from training. At O30 RR2.0, kills 6/12 sessions (including NYSE_OPEN and US_DATA_1000). At O5 RR1.0, kills 0/12.
- **Verify:** Run training, confirm skipped sessions match the baseline query results above.

### Fix F: Feature Reduction (EPV Fix)
- **File:** `config.py` (new `ML_CORE_FEATURES` list), `meta_label.py` (apply selection)
- **Change:** Define 5 features with structural mechanisms:
  - `orb_size_norm` — ORB size IS the edge (Blueprint §2)
  - `atr_20_pct` — vol regime rank (confirmed)
  - `gap_open_points_norm` — overnight institutional repositioning
  - `orb_pre_velocity_norm` — pre-session momentum
  - `prior_sessions_broken` — cross-session flow (confirmed #1 importance)
- **Expert prior, not data-driven selection** — avoids scan-on-train bias (Hastie/Tibshirani ESL §7.10). If univariate scan later contradicts, revisit.
- **Blast radius:** Changes `compute_config_hash()` → all existing models invalidated. Drift check `check_ml_config_hash_match` fires. Expected.
- **Verify:** After fix, `X_e6.shape[1]` should be ~5 (plus any surviving categoricals). EPV at O5 RR1.0 ≈ 430/5 = 86.

### Fix D: Universe-Wide BH/FDR (Post-Bootstrap)
- **File:** `ml_bootstrap_test.py` — add BH correction in summary section
- **Change:** After all configs are bootstrapped, collect all p-values, apply BH FDR at q=0.05 across the full tested family. Report both raw p-values and FDR-adjusted q-values.
- **Blast radius:** None — post-processing only.
- **Verify:** Run BH on results, confirm which configs survive.

---

## Pre-Registration (commit before any retrain)

```
Universe: MNQ E2, all 12 sessions, O5/O15/O30, RR1.0/RR1.5/RR2.0
Total configs: 108 (12 × 3 × 3)
Entry model: E2 only
Data source: bypass_validated=True (full universe)
Gates: baseline ExpR > 0 (train split), EPV ≥ 10
Bootstrap: 5000 perms, Phipson & Smyth correction
Multiple testing: BH FDR q=0.05 across ALL bootstrapped configs
Acceptance: whatever survives FDR is real. 0 survivors = ML dead.
```

---

## Version Gate (Already Deployed)

`ML_METHODOLOGY_VERSION = 2` in config.py. Existing V1 model rejected at inference. Commit `9853817`.

---

## Risk: Zero Survivors

If no config survives after honest methodology + BH FDR:
- ML for ORB is DEAD — add to NO-GO registry
- Layer 1 raw baseline remains the tradeable portfolio
- Confluence univariate features become the research path (no ML)
- This is an honest answer worth having
