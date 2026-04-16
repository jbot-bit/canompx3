# Volume Confluence Scan — 2-factor AND Combinations

**Date:** 2026-04-15
**Pre-reg:** `docs/audit/hypotheses/2026-04-15-volume-exploitation-research-plan.md`
**Classification:** EXPLORATORY (no validated_setups writes).
**Total cells:** 432
**Trustworthy:** 380 (not extreme-fire, not arithmetic-only)
**Strict survivors:** 1 (|t|>=3 + dir_match + N>=50)

## BH-FDR at each K framing
- K_global (432): 4 pass
- K_family (avg K~52): 5 pass
- K_lane (avg K~74): 12 pass
- Promising (|t|>=2.5 + dir_match): 4

## Strict Survivors

| Scope | Instr | Session | Apt | RR | Dir | Pass | Composite | N_on | Fire% | ExpR_on | WR_Δ | Δ_IS | Δ_OOS | t | p | Anchor_corr | BH_g | BH_f | BH_l |
|-------|-------|---------|-----|----|----|------|-----------|------|-------|---------|------|------|-------|---|---|-------------|------|------|------|
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | unfiltered | rel_vol_HIGH_Q3_AND_F6_INSIDE_PDR | 148 | 19.2% | +0.351 | +0.154 | +0.366 | +0.276 | +3.51 | 0.0005 | 0.61 | Y | Y | Y |

## BH-FDR per-family Survivors (top 30 by |t|)

| Scope | Instr | Session | Dir | Pass | Composite | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p |
|-------|-------|---------|----|------|-----------|------|-------|---------|------|-------|---|---|
| deployed | MNQ | COMEX_SETTLE | short | unfiltered | rel_vol_HIGH_Q3_AND_garch_vol_pct_LT30 | 99 | 12.9% | +0.511 | +0.522 | +nan | +4.39 | 0.00002 |
| deployed | MNQ | TOKYO_OPEN | long | unfiltered | rel_vol_HIGH_Q3_AND_bb_volume_ratio_HIGH | 156 | 17.8% | +0.395 | +0.394 | +nan | +4.15 | 0.00005 |
| deployed | MNQ | TOKYO_OPEN | long | unfiltered | rel_vol_LOW_Q1_AND_bb_volume_ratio_LOW | 186 | 21.3% | -0.188 | -0.333 | +nan | -3.78 | 0.00019 |
| deployed | MNQ | COMEX_SETTLE | short | unfiltered | rel_vol_HIGH_Q3_AND_F6_INSIDE_PDR | 148 | 19.2% | +0.351 | +0.366 | +0.276 | +3.51 | 0.00054 |
| deployed | MNQ | COMEX_SETTLE | short | unfiltered | rel_vol_HIGH_Q3_AND_bb_volume_ratio_HIGH | 170 | 22.1% | +0.324 | +0.345 | -0.476 | +3.51 | 0.00053 |

## Honest Kill-Criteria Check (from pre-reg §6.4)

- **0 family BH:** Not triggered
- **≥20 global BH:** Not triggered
- **Concentration in one lane:** CHECK MANUAL — review per-lane counts