# Correction cycle multi-angle synthesis — 2026-04-19

**Generated:** 2026-04-19
**Purpose:** Open-minded re-read of the 9-correction cycle's data. The headline verdicts are "FAMILY KILL × 2, REGIME × 4, 0/13 drift" — but the data contains several angles not fully actioned in the KILL-framed commits. This doc surfaces them for committee consideration.

## Meta-check: am I tunnel-visioning on KILL verdicts?

Cycle commits emphasised Pathway A Chordia gate passes/fails. That is the primary promotion criterion per `pre_registered_criteria.md`. But Pathway A is ONE framing. The data under KILL verdicts still contains:

1. Near-miss signals (under-powered, not null)
2. Inverse-filter anti-signal patterns (feature predicts NON-edge)
3. Cross-direction NEGATIVE-edge cells (fade candidates)
4. Regime-drift retirement candidates that WERE identified but not voted

This doc opens those four angles.

---

## Angle 1 — MES K=40 cross-direction ATR_P70 inverse signal (NEW HYPOTHESIS — pre-reg required before any "finding" claim)

Looking at the 5 × ATR_P70 × 2-direction cells in the MES K=40 scan (commit `76475a34`):

| Session | ATR_L t | ATR_L ExpR | ATR_S t | ATR_S ExpR | Both-dirs negative? |
|---|---:|---:|---:|---:|---|
| TOKYO_OPEN (H07/H08) | -0.40 | -0.025 | -0.99 | -0.061 | BOTH NEGATIVE |
| SINGAPORE_OPEN (H15/H16) | -0.86 | -0.051 | -2.12 | **-0.132** | BOTH NEGATIVE |
| EUROPE_FLOW (H23/H24) | -1.72 | -0.107 | **-3.07** | **-0.190** | BOTH NEGATIVE (deep) |
| LONDON_METALS (H31/H32) | -1.27 | -0.081 | -1.06 | -0.069 | BOTH NEGATIVE |
| COMEX_SETTLE (H39/H40) | -0.54 | -0.035 | -1.58 | -0.110 | BOTH NEGATIVE |

**10/10 ATR_P70 cells show NEGATIVE ExpR in MES.** This is not random — top-30% ATR days on MES produce losing breakouts on BOTH long and short directions across all 5 sessions tested.

**Interpretation:** ATR_P70 (HIGH-ATR days) is a MES-specific SKIP signal. Trading E2 breakouts on top-ATR MES days loses money both directions. The inverse — `NOT(ATR_P70)` = bottom-70% ATR — is the non-losing regime.

**Pattern is sibling to the deployed MNQ filters** like COST_LT12, ATR_P50, OVNRNG_100 — filters that positively select low-friction / normal-regime days. The MES equivalent may be `ATR_LT70` (inverse of `ATR_P70`).

**Not auto-deployable.** This is a post-hoc pattern from the K=40 data; requires a NEW pre-reg testing `ATR_LT70` (or equivalent) as a directional-signal FILTER on MES. Possibly combined with ORB_G5/G8 size screens.

**Action:** Pre-register a follow-up MES Pathway A family testing `ATR_LT70` (bottom-70% ATR) as an overlay filter. Target 5 MES sessions × 2 directions × RR 1.0/1.5 = K=20. File as `2026-04-20-mes-atr-lt70-inverse-filter-v1.yaml` if approved.

---

## Angle 2 — MNQ genuine retirement candidates (regime-drift excess decay)

Correction 3 (`9937ebf6`) surfaced 13 MNQ lanes with Sharpe drops > 0.50 beyond the portfolio's −0.41 decay. The committee pack reframe rejected the 4 original CRITICAL lanes as false-positives, but DID NOT formally queue the honest retirement candidates.

**4 lanes with drops > 1.00 (excess > 0.60 beyond portfolio):**

| Lane | Early Sh | Late Sh | Drop | Excess vs portfolio |
|---|---:|---:|---:|---:|
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM | 1.59 | 0.18 | −1.41 | −1.00 |
| MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM | 1.48 | 0.21 | −1.28 | −0.87 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12 | 1.42 | 0.16 | −1.26 | −0.85 |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 | 0.74 | −0.43 | −1.17 | −0.76 |

**US_DATA_1000 ORB_G5 O15** has NEGATIVE late Sharpe — it's losing money in 2024-25 IS. That's not regime stress; that's a dead edge.

**Action:** These 4 should queue for committee retirement vote ahead of the previously-flagged (now-reframed) 4 CRITICAL lanes. Recommended classification: DECAY_CANDIDATE, vote this week.

Additional 9 lanes with drops 0.50-1.00 form a second-tier review queue (excess −0.10 to −0.60). Lower urgency but non-trivial.

---

## Angle 3 — MES H35 CMX_G8_L positive near-miss (potential shadow candidate)

From the K=40 scan, highest positive t-stat:

- **H35 MES COMEX_SETTLE ORB_G8 long RR1.5:** N=106, ExpR=+0.196, t=+1.71, boot_p not reported (probably ~0.08), 4+ positive years.

Not above Chordia t>=3.00, but:
- Same session where MES has its ONE validated lane (`MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` — Phase 7 confirmed t=3.66).
- Same filter (ORB_G8) as that validated lane.
- Same direction (long).
- Different RR (1.5 vs 1.0) and different session (COMEX_SETTLE vs CME_PRECLOSE).

**Interpretation:** This may be within-family of the validated MES lane, not a new independent signal. But it's a candidate for shadow-park (same treatment as C4 MGC-long) to accumulate N and re-evaluate when the IS grows.

**Caveat:** The 2026-04-19 MES broader scan already tested H2 MES CME_PRECLOSE COST_LT12 long RR1.0 (t=2.67, KILL), H3 MES CME_PRECLOSE ATR_P70 long RR1.0 (t=2.59, KILL). H35 CMX_G8_L RR1.5 is in a related family that's been chipped at multiple angles.

**Action:** Optional shadow pre-reg. Low priority compared to Angles 1 and 2. A conservative read says "just track in the regular regime-check" — not every positive-near-miss needs a formal pre-reg.

---

## Angle 4 — Cross-correction connections

### Regime-drift + comprehensive-scan cross-check
The 13 rel_vol_HIGH_Q3 BH-global cells (Correction 6) all hold under IS-only quantile. But 4 of the 5 (instrument, session) lanes in that finding are NOT the same as the MNQ excess-decay retirement candidates (Correction 3). Exceptions:
- MNQ COMEX_SETTLE short (rel_vol survivor) — Correction 3 shows `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` drop +1.28 (SHARPE UP, not down) → rel_vol's overlap signal is in a regime-HEALTHY portion of the portfolio. Edge likely real.
- MNQ SINGAPORE_OPEN short (rel_vol survivor) — `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` drop +0.96 (SHARPE UP). Same — rel_vol in a healthy subset.

**Implication:** the rel_vol_HIGH_Q3 finding survives both the quantile sensitivity AND the regime-drift test. If anything its supporting lanes are OUTPERFORMING in the stressed period.

### MES coverage: what's left after K=40?
Not tested in K=40: NYSE_OPEN, US_DATA_830 (both N_IS > 1600). Reserved as next K=20-40 pre-reg pair per pre-reg scope statement. Recommended before Committee decides MES is structurally flat.

---

## Summary of under-actioned opportunities

| # | Opportunity | Priority | Action |
|---|---|---|---|
| 1 | MES `ATR_LT70` inverse-filter pre-reg | HIGH | New Pathway A pre-reg (K=20) |
| 2 | MNQ 4-lane retirement queue (excess decay) | HIGH | Committee vote (formal) |
| 3 | MES H35 CMX_G8_L shadow | LOW | Track via regime-check; optional pre-reg |
| 4 | MES NYSE_OPEN + US_DATA_830 coverage | MED | Second K=20-40 MES pre-reg |
| 5 | 9 second-tier decay lanes (drop 0.5-1.0) | MED | Committee review queue |

**Framing check:** the KILL verdicts in the correction cycle ARE correct at the Pathway A Chordia gate. But "family KILL at t>=3.00" does not mean "the data has nothing to say." The above 5 items are the data's other angles, visible once the Pathway-A-only lens is set down.

## Reproduction

- Correction 3: `research/regime_drift_control_critical_lanes.py`
- Correction 6: `research/rel_vol_is_only_quantile_sensitivity.py`
- Correction 7: `research/mgc_mode_a_rediscovery_orbg5_short_v1_scan.py`
- Correction 8: `research/mes_comprehensive_mode_a_feature_v1_scan.py`

Read-only. No writes.
