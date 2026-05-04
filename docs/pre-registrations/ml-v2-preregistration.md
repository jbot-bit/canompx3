# ML V2 Pre-Registration

**Committed BEFORE any retrain. Results cannot retroactively change this spec.**

## Universe

- Instrument: MNQ
- Entry model: E2
- Sessions: all 12 (from `pipeline.dst.SESSION_CATALOG`)
- Apertures: O5, O15, O30
- RR targets: 1.0, 1.5, 2.0
- Total configs: 108 (12 sessions x 3 apertures x 3 RR)

## Methodology (V2 fixes applied)

- Features: `ML_CORE_FEATURES` (5 expert-prior features)
  - `orb_size_norm`, `atr_20_pct`, `gap_open_points_norm`,
    `orb_pre_velocity_norm`, `prior_sessions_broken`
- Data source: `bypass_validated=True` (full orb_outcomes universe)
- Positive baseline gate: train split ExpR > 0 (Fix E)
- EPV gate: >= 10 (guaranteed with 5 features)
- EF/LM cross-session features dropped (Fix B)
- Deterministic config selection (Fix A)
- Train-only constant-column drop (Fix C)

## Config Selection Protocol

For each of 12 sessions:
1. Train RF on all 9 (aperture x RR) combos that pass the baseline gate
2. Rank by CPCV AUC computed on the 60% train split
3. Select the top-ranked config for bootstrap testing
4. If CPCV AUC < 0.50 for ALL combos, skip the session

Selection is committed before any bootstrap runs.

## Statistical Testing

- Bootstrap: 5000 permutations, Phipson & Smyth (2010) p-value correction
- Family unit: session (K=12)
- BH FDR: q=0.05 applied at K=12 (promotion decision)
- K=108 reported as conservative footnote (per RESEARCH_RULES.md)
- Cross-session consistency: >= 2 BH survivors required for ML ALIVE

## Decision Rules (defined before results)

| Outcome | Action |
|---------|--------|
| 0 sessions survive BH at K=12 | ML DEAD. Add to Blueprint NO-GO. |
| 1 session survives | ML CONDITIONAL. Investigate only. No Phase 2. |
| >= 2 sessions survive | ML ALIVE. Proceed to Phase 2. |

## What this spec locks

- The 5 features cannot be changed after seeing bootstrap results
- The K=12 family cannot be changed to K=108 to kill results
- The K=108 cannot be used instead of K=12 to rescue results
- The >= 2 survivor gate cannot be lowered to >= 1
- The q=0.05 threshold cannot be changed
