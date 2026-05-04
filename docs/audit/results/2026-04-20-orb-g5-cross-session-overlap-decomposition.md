# ORB_G5 cross-session overlap decomposition

**Generated:** 2026-04-20T07:30:39+00:00
**Script:** `research/orb_g5_cross_session_overlap_decomposition.py`
**IS window:** `trading_day < 2026-01-01` (Mode A, imported from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`).
**Filter source:** canonical `trading_app.config.ALL_FILTERS['ORB_G5']` via `research.filter_utils.filter_signal` (no re-encoded logic).

## Audited claim

Three of six deployed MNQ lanes in the 2026-04-18 allocator DEPLOY set use the `ORB_G5` size filter: MNQ EUROPE_FLOW, MNQ COMEX_SETTLE, and MNQ US_DATA_1000 (O15 variant). Portfolio sizing treats them as three independent signals. **Adversarial question:** do they fire on the same trading days (one driver counted three times), or on disjoint day sets (three genuinely diversifying signals)?

## Pre-committed decision rule

Locked before this run in `docs/runtime/stages/orb_g5_cross_session_overlap.md`:

**Driver statistic:** max pairwise Pearson correlation on 0/1 fire masks (= phi coefficient on binary). Chosen over Jaccard because Jaccard is mechanically inflated at high marginal fire rates (two 80%-fire lanes under pure independence give Jaccard ~0.67).

- **max pairwise ρ > 0.5** (IS) -> DROP_CANDIDATE: recommend dropping one of the overlapping lanes.
- **0.25 < max ρ <= 0.5** (IS) -> PARTIAL_OVERLAP: flag; no capital action.
- **max ρ <= 0.25** (IS) -> CLEAN_DIVERSIFICATION.

Thresholds mapped to Bailey-Lopez de Prado 2014 Appendix A.3 Eq. 9 `N̂ = ρ̂ + (1−ρ̂)·M` (M=3): ρ=0.5 → N̂≈2.0; ρ=0.25 → N̂≈2.5; ρ=0.0 → N̂=3.0. Per-pair max preferred over ρ̂ (average) per MEMORY.md 2026-04-20 rule (pooled averages hide heterogeneous cells).

## Lanes under test

| ID | strategy_id | Session | ORB min | RR | Rows loaded | Rows IS | Fire-days IS | Fire-days full |
|---|---|---|---:|---:|---:|---:|---:|---:|
| L1 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | EUROPE_FLOW | 5 | 1.5 | 1786 | 1717 | 1582 | 1651 |
| L2 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | COMEX_SETTLE | 5 | 1.5 | 1698 | 1635 | 1555 | 1618 |
| L3 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | US_DATA_1000 | 15 | 1.5 | 1549 | 1494 | 1490 | 1545 |

`Rows loaded` = E2 trade rows from `orb_outcomes` JOIN `daily_features` (pnl_r non-null, break_dir in {long,short}). `Fire-days` = subset where ORB_G5 canonical filter also fires (orb_size >= 5.0 points).

## Pairwise overlap matrix — Mode A IS

| Pair | A | B | \|A\| | \|B\| | \|A ∩ B\| | \|A ∪ B\| | Jaccard | % of min |
|---|---|---|---:|---:|---:|---:|---:|---:|
| L1 x L2 | EUROPE_FLOW | COMEX_SETTLE | 1582 | 1555 | 1476 | 1661 | 0.889 | 94.9 |
| L1 x L3 | EUROPE_FLOW | US_DATA_1000 | 1582 | 1490 | 1376 | 1696 | 0.811 | 92.3 |
| L2 x L3 | COMEX_SETTLE | US_DATA_1000 | 1555 | 1490 | 1365 | 1680 | 0.812 | 91.6 |

## Simultaneous-fire distribution — Mode A IS

| # lanes firing | Trading-days with this count |
|---:|---:|
| 1 | 81 |
| 2 | 329 |
| 3 | 1296 |
| **total fire-days** | **1706** |

Sum per-lane fire-days: 4627. Union: 1706. Redundancy = 1 − union/sum = 0.631.

## Pairwise Pearson correlation — Mode A IS

| | L1 | L2 | L3 |
|---|---:|---:|---:|
| L1 | 1.000 | 0.333 | 0.040 |
| L2 | 0.333 | 1.000 | -0.013 |
| L3 | 0.040 | -0.013 | 1.000 |

## Nyholt 2004 M-effective — Mode A IS

**Meff = 2.950** (out of m=3 lanes).

Eigenvalues of the correlation matrix: [0.663 1.003 1.334].

## Bailey-Lopez de Prado 2014 Eq. 9 implied independent trials — Mode A IS

Canonical formula from `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` Appendix A.3 Eq. 9: `N̂ = ρ̂ + (1 − ρ̂)·M`, where `ρ̂` is the average off-diagonal correlation and `M = 3` lanes.

**Observed:** ρ̂ = 0.120 → **N̂ = 2.760** independent trials (out of M = 3).

Interpretation: ρ̂ near 0 → N̂ ≈ M (full independence); ρ̂ near 1 → N̂ ≈ 1 (one effective signal). BL 2014 Appendix A.3 clips ρ̂ to the valid range; this implementation clips negative ρ̂ to 0 for conservative reporting.

## Pairwise overlap matrix — Full sample (IS + 2026 OOS)

_Reference only; verdict is driven by the Mode A IS table above._

| Pair | A | B | \|A\| | \|B\| | \|A ∩ B\| | \|A ∪ B\| | Jaccard | % of min |
|---|---|---|---:|---:|---:|---:|---:|---:|
| L1 x L2 | EUROPE_FLOW | COMEX_SETTLE | 1651 | 1618 | 1539 | 1730 | 0.890 | 95.1 |
| L1 x L3 | EUROPE_FLOW | US_DATA_1000 | 1651 | 1545 | 1431 | 1765 | 0.811 | 92.6 |
| L2 x L3 | COMEX_SETTLE | US_DATA_1000 | 1618 | 1545 | 1416 | 1747 | 0.811 | 91.7 |

Full-sample summary: union fire-days 1775 (1=83, 2=345, 3=1347); Meff = 2.952; ρ̂ = 0.115; N̂ = 2.770 (BL 2014 Eq.9); max pair ρ = 0.326; max Jaccard = 0.890.

## Verdict

**Mode A IS max pairwise ρ: 0.333** (pair: L1 × L2)

Supporting: avg ρ̂ = 0.120 → N̂ = 2.760 (BL14); Nyholt Meff = 2.950; max pair Jaccard = 0.889.

**Label:** `PARTIAL_OVERLAP`

max pairwise ρ 0.333 in (0.25, 0.5] — partial concentration flagged; no capital action yet.

## Caveats

- **L3 (US_DATA_1000 O15) ORB_G5 fire rate = 99.7%.** On this lane the filter is effectively a pass-through gate; the fire mask is a near-constant vector. Pair-ρ involving this lane has limited variance-based statistical power. Interpret its ~0 correlations as 'cannot detect coupling from a constant vector' rather than strong evidence of independence.

- This audit measures **decision-level** correlation (fire/no-fire), not **return-level** correlation. Two lanes with ρ=0 on fire masks could still produce correlated P&L if they fire on different days but share the same regime driver. A return-correlation follow-up (P&L per trade_day per lane) is the natural next audit if return coupling becomes the question.

Most-overlapping pair: **L1 x L2** (EUROPE_FLOW vs COMEX_SETTLE, Jaccard=0.889). Monitor; no capital action at this Jaccard.

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/orb_g5_cross_session_overlap_decomposition.py
```

No randomness. Fire masks computed via canonical `trading_app.config.ALL_FILTERS['ORB_G5']` through `research.filter_utils.filter_signal`. No writes to `validated_setups` / `experimental_strategies` / `live_config`.

