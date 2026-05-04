# Participation-optimum universality test v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-20-participation-optimum-universality-v1.yaml` (LOCKED, commit_sha=`677bf381`)
**Script:** `research/participation_optimum_universality_v1.py`
**Scope:** MNQ × 12 sessions × both directions × 5m E2 CB1 RR1.5 unfiltered, IS only
**IS window:** trading_day < 2026-01-01

## Verdict: **CONFIRMED_UNIVERSAL**

> Pooled β₂=-0.00156 t=-5.189 p=0.0000 significantly negative. Per-lane sign-agreement = 87.5% (>= 50%). Participation-optimum is a general MNQ mechanism.

## Integrity

- rel_vol non-null fraction on IS 5m: 99.7% (threshold ≥ 90%)
- IS 5m pooled N: 17884
- Cells loaded: 12 sessions × 2 directions = 24 (expected)
- Parity (COMEX_SETTLE short β₂ < 0): YES
  - COMEX_SETTLE short: β₂=-0.00324, t=-1.679, N=764

## Primary pooled regression (5m, IS, lane_FE)

| param | value |
|---|---:|
| N | 17828 |
| β₀ (intercept) | -0.16825 |
| β₁ (rel_vol) | +0.06230 |
| **β₂ (rel_vol²)** | **-0.00156** |
| SE(β₂) | 0.00030 |
| t(β₂) | -5.189 |
| one-tailed p (β₂ < 0) | 0.0000 |
| Chordia threshold | t ≤ -3.0 (with prior theory) |

## Per-cell regression (K=24, 5m, IS)

| session | direction | N | β₂ | t(β₂) | one-tailed p | sign |
|---|---|---:|---:|---:|---:|:---:|
| BRISBANE_1025 | long | 904 | -0.0007 | -0.382 | 0.3514 | neg |
| BRISBANE_1025 | short | 811 | -0.0033 | -0.393 | 0.3471 | neg |
| CME_PRECLOSE | long | 655 | -0.0009 | -0.838 | 0.2011 | neg |
| CME_PRECLOSE | short | 638 | -0.0030 | -0.866 | 0.1934 | neg |
| CME_REOPEN | long | 382 | -0.0010 | -0.410 | 0.3408 | neg |
| CME_REOPEN | short | 372 | -0.0046 | -0.727 | 0.2337 | neg |
| COMEX_SETTLE | long | 867 | +0.0123 | +1.138 | 0.8723 | pos |
| COMEX_SETTLE | short | 764 | -0.0032 | -1.679 | 0.0468 | neg |
| EUROPE_FLOW | long | 847 | -0.0019 | -0.527 | 0.2992 | neg |
| EUROPE_FLOW | short | 865 | -0.0024 | -0.497 | 0.3096 | neg |
| LONDON_METALS | long | 859 | -0.0008 | -0.929 | 0.1767 | neg |
| LONDON_METALS | short | 853 | -0.0073 | -3.288 | 0.0005 | neg |
| NYSE_CLOSE | long | 290 | -0.0025 | -0.645 | 0.2597 | neg |
| NYSE_CLOSE | short | 319 | -0.0013 | -0.184 | 0.4270 | neg |
| NYSE_OPEN | long | 828 | -0.0006 | -0.030 | 0.4879 | neg |
| NYSE_OPEN | short | 817 | +0.0112 | +1.419 | 0.9218 | pos |
| SINGAPORE_OPEN | long | 881 | -0.0007 | -0.412 | 0.3401 | neg |
| SINGAPORE_OPEN | short | 835 | -0.0011 | -0.583 | 0.2799 | neg |
| TOKYO_OPEN | long | 855 | -0.0019 | -1.596 | 0.0554 | neg |
| TOKYO_OPEN | short | 861 | -0.0020 | -1.431 | 0.0763 | neg |
| US_DATA_1000 | long | 866 | -0.0066 | -0.724 | 0.2347 | neg |
| US_DATA_1000 | short | 803 | -0.0025 | -0.154 | 0.4388 | neg |
| US_DATA_830 | long | 812 | -0.0029 | -0.911 | 0.1813 | neg |
| US_DATA_830 | short | 844 | +0.0002 | +0.039 | 0.5156 | pos |

- Valid cells (N≥50): 24/24
- Cells with β₂ < 0: 21/24 = **87.5%**
- RULE 14 threshold: ≥ 50% (CONFIRMED_UNIVERSAL), 25-50% (CONFIRMED_HETEROGENEOUS), < 25% (KILL_SIMPSON)

## Robustness — 15m aperture pooled

- N: 15501
- β₂: -0.00021, t: -1.506, one-tailed p: 0.0660

## Robustness — 30m aperture pooled

- N: 14008
- β₂: +0.00003, t: +0.141, one-tailed p: 0.5562

## Not done by this result

- No capital, allocator, sizing, or filter change.
- Does NOT test MES/MGC, E3/E4 entries, RR 1.0/2.0, or overlay-filter-conditional peaks.
- A CONFIRMED_UNIVERSAL or CONFIRMED_HETEROGENEOUS verdict only unblocks writing a deployment-shape follow-on pre-reg; it does not deploy anything on its own.
