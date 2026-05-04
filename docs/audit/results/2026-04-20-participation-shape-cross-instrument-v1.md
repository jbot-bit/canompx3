# Participation-shape cross-instrument — v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-20-participation-shape-cross-instrument-v1.yaml`

**Script:** `research/participation_shape_cross_instrument_v1.py`

**Scope:** MNQ + MES + MGC at 5m E2 CB1 RR1.5 unfiltered (primary); 3x3x3 shape-map secondary.


## Primary verdicts (per-instrument Pathway-B K=1)

| Instrument | N | β₁ (rank) | t | one-tailed p | per-cell agreement | Verdict |
|---|---:|---:|---:|---:|---:|---|
| MNQ | 17828 | +0.27775 | +9.589 | 0.0000 | 91.7% (22/24) | **MNQ_MONOTONIC_CONFIRMED** |
| MES | 16014 | +0.33025 | +11.802 | 0.0000 | 100.0% (22/22) | **MES_MONOTONIC_CONFIRMED** |
| MGC | 7444 | +0.29975 | +7.541 | 0.0000 | 100.0% (18/18) | **MGC_MONOTONIC_CONFIRMED** |

- **MNQ:** Pooled β1=+0.27775 t=+9.589 significant; per-lane agreement 91.7% >= 50%. Monotonic-up replicates on MNQ.
- **MES:** Pooled β1=+0.33025 t=+11.802 significant; per-lane agreement 100.0% >= 50%. Monotonic-up replicates on MES.
- **MGC:** Pooled β1=+0.29975 t=+7.541 significant; per-lane agreement 100.0% >= 50%. Monotonic-up replicates on MGC.

## Combined cross-instrument interpretation

3/3 CONFIRMED — monotonic-up is the universal ORB 5m E2 RR1.5 spec. PR #41 MNQ inverted-U likely a shape-artefact atop a monotonic-up base; PR #43 Q4-band contract may be under-scoped or mis-shaped.

## Secondary shape-map (descriptive — not decision-bearing)

Per-cell descriptive classification: `MONOTONIC_UP` if rank β₁ > 0 and |t| ≥ 2.0; `INVERTED_U` if quad β₂ < 0 and |t| ≥ 2.0; `BOTH` if both; `NULL` otherwise. `INSUFFICIENT_N` if n < 100.

| Instrument | Aperture | RR | N | β₁ (rank) | t(rank) | β₂ (quad) | t(quad) | Shape |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| MES | 5 | 1.0 | 16592 | +0.31567 | +14.294 | -0.00059 | -2.484 | BOTH |
| MES | 5 | 1.5 | 16014 | +0.33025 | +11.802 | -0.00061 | -1.834 | MONOTONIC_UP |
| MES | 5 | 2.0 | 15638 | +0.32218 | +9.812 | -0.00054 | -1.564 | MONOTONIC_UP |
| MES | 15 | 1.0 | 14387 | +0.21551 | +8.590 | -0.00036 | -3.759 | BOTH |
| MES | 15 | 1.5 | 13896 | +0.21337 | +6.738 | -0.00032 | -2.960 | BOTH |
| MES | 15 | 2.0 | 13485 | +0.22053 | +5.965 | -0.00028 | -2.424 | BOTH |
| MES | 30 | 1.0 | 13301 | +0.19197 | +7.158 | -0.00023 | -3.787 | BOTH |
| MES | 30 | 1.5 | 12602 | +0.15916 | +4.688 | -0.00015 | -2.171 | BOTH |
| MES | 30 | 2.0 | 12043 | +0.13928 | +3.512 | -0.00017 | -1.887 | MONOTONIC_UP |
| MGC | 5 | 1.0 | 7617 | +0.29605 | +9.454 | -0.00024 | -0.317 | MONOTONIC_UP |
| MGC | 5 | 1.5 | 7444 | +0.29975 | +7.541 | -0.00021 | -0.273 | MONOTONIC_UP |
| MGC | 5 | 2.0 | 7313 | +0.31241 | +6.708 | -0.00020 | -0.216 | MONOTONIC_UP |
| MGC | 15 | 1.0 | 7078 | +0.18659 | +5.365 | -0.00015 | -0.939 | MONOTONIC_UP |
| MGC | 15 | 1.5 | 6711 | +0.15295 | +3.426 | -0.00010 | -0.326 | MONOTONIC_UP |
| MGC | 15 | 2.0 | 6455 | +0.16552 | +3.145 | -0.00009 | -0.203 | MONOTONIC_UP |
| MGC | 30 | 1.0 | 6332 | +0.22298 | +5.892 | -0.00012 | -2.166 | BOTH |
| MGC | 30 | 1.5 | 5866 | +0.20099 | +4.101 | -0.00008 | -1.880 | MONOTONIC_UP |
| MGC | 30 | 2.0 | 5568 | +0.20152 | +3.460 | -0.00007 | -0.379 | MONOTONIC_UP |
| MNQ | 5 | 1.0 | 18410 | +0.24117 | +10.706 | -0.00154 | -6.533 | BOTH |
| MNQ | 5 | 1.5 | 17828 | +0.27775 | +9.589 | -0.00156 | -5.189 | BOTH |
| MNQ | 5 | 2.0 | 17429 | +0.25468 | +7.425 | -0.00153 | -4.192 | BOTH |
| MNQ | 15 | 1.0 | 16105 | +0.17408 | +6.936 | -0.00038 | -2.698 | BOTH |
| MNQ | 15 | 1.5 | 15501 | +0.16331 | +5.091 | -0.00021 | -1.506 | MONOTONIC_UP |
| MNQ | 15 | 2.0 | 15031 | +0.17745 | +4.700 | -0.00018 | -1.049 | MONOTONIC_UP |
| MNQ | 30 | 1.0 | 14785 | +0.13820 | +5.165 | -0.00024 | -1.460 | MONOTONIC_UP |
| MNQ | 30 | 1.5 | 14008 | +0.12756 | +3.744 | +0.00003 | +0.141 | MONOTONIC_UP |
| MNQ | 30 | 2.0 | 13467 | +0.09713 | +2.433 | +0.00019 | +0.721 | MONOTONIC_UP |

## Not done by this result

- No capital action.
- Does NOT modify Q4-band MNQ contract (PR #43).
- Does NOT revise PRs #41/#42/#45 verdicts.
- Does NOT test filtered universes / size filters / multi-variate specs.
- Does NOT run OOS validation.
