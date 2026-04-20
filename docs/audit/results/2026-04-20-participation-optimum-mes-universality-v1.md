# Participation-optimum universality — MES v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-20-participation-optimum-mes-universality-v1.yaml` (LOCKED, commit_sha=`77e33f06`)
**Script:** `research/participation_optimum_mes_universality_v1.py`
**Parent MNQ merged:** PR #41, commit `126ed6b8`
**Scope:** MES × 11 active sessions × both directions × 5m E2 CB1 RR1.5 unfiltered, IS only

## Verdict: **MES_NO_REPLICATION**

> Pooled beta2=-0.00061 t=-1.834 p=0.0334 does NOT clear t<=-3.0. MNQ finding does NOT replicate on MES — mechanism is MNQ-specific.

## Cross-instrument comparison

| instrument | N (IS pooled) | β₂ | t | one-tailed p | per-lane agreement |
|---|---:|---:|---:|---:|---:|
| MNQ (parent) | 17,828 | -0.00156 | -5.189 | 0.0000 | 87.5% |
| MES (this)   | 16,014 | -0.00061 | -1.834 | 0.0334 | 95.5% |

## Integrity

- rel_vol non-null on IS 5m: 99.7% (threshold ≥ 90%)
- IS 5m N: 16058
- Sessions tested: 11
- Cells loaded: 22 (max)

## Pooled regression (5m, IS, lane_FE)

| param | value |
|---|---:|
| N | 16014 |
| β₀ | -0.06819 |
| β₁ | +0.04359 |
| **β₂ (rel_vol²)** | **-0.00061** |
| SE(β₂) | 0.00033 |
| t(β₂) | -1.834 |
| one-tailed p | 0.0334 |

## Per-cell regression

| session | direction | N | β₂ | t(β₂) | one-tailed p | sign |
|---|---|---:|---:|---:|---:|:---:|
| CME_PRECLOSE | long | 614 | -0.0008 | -1.131 | 0.1293 | neg |
| CME_PRECLOSE | short | 607 | -0.0010 | -0.385 | 0.3501 | neg |
| CME_REOPEN | long | 363 | -0.0020 | -0.813 | 0.2085 | neg |
| CME_REOPEN | short | 389 | -0.0030 | -2.052 | 0.0204 | neg |
| COMEX_SETTLE | long | 884 | -0.0098 | -2.717 | 0.0034 | neg |
| COMEX_SETTLE | short | 753 | -0.0015 | -1.233 | 0.1090 | neg |
| EUROPE_FLOW | long | 849 | -0.0012 | -1.254 | 0.1051 | neg |
| EUROPE_FLOW | short | 864 | -0.0010 | -2.791 | 0.0027 | neg |
| LONDON_METALS | long | 872 | -0.0032 | -1.059 | 0.1448 | neg |
| LONDON_METALS | short | 840 | -0.0039 | -0.488 | 0.3129 | neg |
| NYSE_CLOSE | long | 290 | -0.0013 | -0.956 | 0.1700 | neg |
| NYSE_CLOSE | short | 268 | -0.0012 | -0.196 | 0.4223 | neg |
| NYSE_OPEN | long | 842 | -0.0072 | -0.641 | 0.2608 | neg |
| NYSE_OPEN | short | 835 | +0.0007 | +0.102 | 0.5406 | pos |
| SINGAPORE_OPEN | long | 890 | -0.0008 | -0.212 | 0.4160 | neg |
| SINGAPORE_OPEN | short | 826 | -0.0004 | -0.179 | 0.4289 | neg |
| TOKYO_OPEN | long | 841 | -0.0005 | -0.155 | 0.4383 | neg |
| TOKYO_OPEN | short | 875 | -0.0018 | -3.851 | 0.0001 | neg |
| US_DATA_1000 | long | 859 | -0.0032 | -1.056 | 0.1457 | neg |
| US_DATA_1000 | short | 810 | -0.0038 | -0.237 | 0.4063 | neg |
| US_DATA_830 | long | 826 | -0.0046 | -2.280 | 0.0114 | neg |
| US_DATA_830 | short | 817 | -0.0009 | -0.264 | 0.3959 | neg |

- Valid cells (N≥50): 22/22
- Cells with β₂ < 0: 21/22 = **95.5%**

## Not done by this result

- No capital action.
- Does NOT test MGC, 15m/30m, E3/E4, other RRs.
- If CONFIRMED_*, unblocks a multi-instrument deployment-shape pre-reg; does not deploy on its own.
