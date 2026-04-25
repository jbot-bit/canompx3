# Participation-optimum universality — MGC v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-20-participation-optimum-mgc-universality-v1.yaml`
**Script:** `research/participation_optimum_mgc_universality_v1.py`
**Parent MNQ merged:** PR #41, commit `126ed6b8`
**Parent MES merged:** PR #42, commit `75b0d0bc`
**Scope:** MGC x 9 canonical sessions x both directions x 5m E2 CB1 RR1.5 unfiltered, IS only

## Verdict: **MGC_NO_REPLICATION**

> Pooled beta2=-0.00021 t=-0.273 p=0.3923 does NOT clear t<=-3.0 or p<0.05. MNQ finding does NOT replicate on MGC — mechanism is index-specific or attenuated on gold.

## Cross-instrument comparison

| instrument | N (IS pooled) | beta2 | t | one-tailed p | per-lane agreement |
|---|---:|---:|---:|---:|---:|
| MNQ (parent) | 17,828 | -0.00156 | -5.189 | 0.0000 | 87.5% |
| MES (parent) | n/a | -0.00061 | -1.834 | 0.0333 | 95.5% |
| MGC (this)   | 7,444 | -0.00021 | -0.273 | 0.3923 | 66.7% |

## Integrity

- rel_vol non-null on IS 5m: 99.5% (threshold >= 90%)
- IS 5m N (raw): 7485
- IS 5m N (rel_vol available): 7444
- Sessions tested: 9
- Cells loaded: 18 (max)

## Pooled regression (5m, IS, lane_FE, HC3)

| param | value |
|---|---:|
| N | 7444 |
| beta0 | -0.24712 |
| beta1 | +0.03406 |
| **beta2 (rel_vol^2)** | **-0.00021** |
| SE(beta2) | 0.00077 |
| t(beta2) | -0.273 |
| one-tailed p | 0.3923 |

## Per-cell regression

| session | direction | N | beta2 | t(beta2) | one-tailed p | sign |
|---|---|---:|---:|---:|---:|:---:|
| CME_REOPEN | long | 216 | -0.0002 | -0.344 | 0.3655 | neg |
| CME_REOPEN | short | 192 | -0.0021 | -0.818 | 0.2073 | neg |
| COMEX_SETTLE | long | 430 | -0.0010 | -0.220 | 0.4132 | neg |
| COMEX_SETTLE | short | 412 | +0.0015 | +0.889 | 0.8127 | pos |
| EUROPE_FLOW | long | 490 | +0.0045 | +0.758 | 0.7756 | pos |
| EUROPE_FLOW | short | 422 | +0.0030 | +0.825 | 0.7951 | pos |
| LONDON_METALS | long | 467 | -0.0018 | -1.091 | 0.1379 | neg |
| LONDON_METALS | short | 445 | -0.0025 | -0.478 | 0.3163 | neg |
| NYSE_OPEN | long | 439 | +0.0038 | +0.621 | 0.7326 | pos |
| NYSE_OPEN | short | 458 | -0.0078 | -0.974 | 0.1652 | neg |
| SINGAPORE_OPEN | long | 464 | -0.0012 | -1.297 | 0.0977 | neg |
| SINGAPORE_OPEN | short | 449 | -0.0068 | -1.091 | 0.1380 | neg |
| TOKYO_OPEN | long | 453 | -0.0000 | -0.018 | 0.4929 | neg |
| TOKYO_OPEN | short | 460 | +0.0005 | +0.580 | 0.7189 | pos |
| US_DATA_1000 | long | 439 | -0.0020 | -0.113 | 0.4549 | neg |
| US_DATA_1000 | short | 379 | +0.0023 | +0.355 | 0.6385 | pos |
| US_DATA_830 | long | 408 | -0.0026 | -0.491 | 0.3119 | neg |
| US_DATA_830 | short | 421 | -0.0009 | -0.412 | 0.3403 | neg |

- Valid cells (N>=50): 18/18
- Cells with beta2 < 0: 12/18 = **66.7%**

## Not done by this result

- No capital action.
- Does NOT modify the Q4-band MNQ deployment-shape contract.
- Does NOT test 15m/30m, E3/E4, other RRs, or any MGC filter.
- If MGC_CONFIRMED_*, unblocks a multi-instrument deployment-shape pre-reg; does not deploy on its own.
