# Phase 3a — DSR ranking empirical verification

- rebalance_date: `2026-04-18`
- profile: `topstep_50k_mnq_auto`
- lanes audited: `30`
- canonical deps: `trading_app.dsr.compute_sr0/compute_dsr` + `trading_app.lane_allocator.*`
- validator pattern source: `trading_app/strategy_validator.py:2180-2229`
- OOS consumption: zero (uses validator-stored Sharpe + Mode-A-aware var_sr from canonical_experimental_strategies)
- one-shot lock enforced

## DSR inputs (canonical)

- N_eff (edge_families distinct count): `21`
- var_sr by entry_model: `{'E1': 0.047, 'E2': 0.006712097170233147}`
- N_eff sensitivity bands also reported: `[5, 12, 36, 72, 253]`

Per the 2026-04-15 rel_vol v2 stress-test lesson (`.claude/rules/backtesting-methodology.md` historical failure log), DSR is reported at multiple N_eff because single-N_eff DSR can mislead. Per `trading_app/dsr.py` docstring line 35 DSR is informational, not a hard gate.

## Selection comparison (top max_slots = 7)

- `sel_raw`   (6): ['MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5', 'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5', 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5', 'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15', 'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12', 'MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15']
- `sel_dsr`   (7): ['MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100', 'MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5', 'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100', 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5', 'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30', 'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12', 'MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15']
- `sel_combo` (6): ['MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100', 'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100', 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5', 'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30', 'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12', 'MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15']

- |raw ∩ dsr|   = 2, dsr adds ['MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100', 'MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5', 'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100', 'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30', 'MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15'], dsr removes ['MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5', 'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5', 'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15', 'MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15']
- |raw ∩ combo| = 2, combo adds ['MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100', 'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100', 'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30', 'MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15'], combo removes ['MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5', 'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5', 'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15', 'MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15']

- non-tied selection delta vs raw: dsr=7, combo=6

## Materiality verdict

**RANKING_MATERIAL** — DSR flips 7 slot(s), combo flips 6 slot(s). A2b-2 patch is BUG_MATERIAL on the current rebalance.

## Per-lane ranking + DSR sensitivity

| rank_raw | strategy_id | inst | session | RR | filter | annual_r | eff_r | SR0 | DSR_can | DSR_n5 | DSR_n12 | DSR_n36 | DSR_n72 | DSR_n253 | rank_dsr | rank_combo | sel_raw | sel_dsr | sel_combo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  1 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1` | MNQ | EUROPE_FLOW | 1.5 | `ORB_G5` | `+46.1` | `+46.1` | `0.158` | `0.0002` | `0.1072` | `0.0031` | `0.0000` | `0.0000` | `0.0000` | 29 | 28 | Y | . | . |
|  2 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_` | MNQ | SINGAPORE_OPEN | 1.5 | `ATR_P50` | `+44.0` | `+44.0` | `0.158` | `0.0239` | `0.4551` | `0.0932` | `0.0053` | `0.0006` | `0.0000` | 14 | 12 | Y | . | . |
|  3 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_` | MNQ | SINGAPORE_OPEN | 1.5 | `ATR_P50` | `+44.0` | `+44.0` | `0.158` | `0.0551` | `0.6042` | `0.1733` | `0.0149` | `0.0022` | `0.0000` | 11 |  8 | . | Y | Y |
|  4 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB` | MNQ | COMEX_SETTLE | 1.5 | `ORB_G5` | `+41.0` | `+41.0` | `0.158` | `0.0105` | `0.4937` | `0.0669` | `0.0013` | `0.0001` | `0.0000` | 18 | 17 | Y | . | . |
|  5 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1` | MNQ | EUROPE_FLOW | 1.5 | `COST_LT12` | `+40.6` | `+40.6` | `0.158` | `0.0180` | `0.4297` | `0.0778` | `0.0036` | `0.0004` | `0.0000` | 15 | 13 | . | . | . |
|  6 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB` | MNQ | COMEX_SETTLE | 1.5 | `OVNRNG_100` | `+40.1` | `+40.1` | `0.158` | `0.7183` | `0.9727` | `0.8536` | `0.5645` | `0.3721` | `0.1322` |  2 |  1 | . | . | Y |
|  7 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1` | MNQ | EUROPE_FLOW | 1.5 | `CROSS_SGP_MOMENTUM` | `+39.2` | `+39.2` | `0.158` | `0.0087` | `0.3211` | `0.0442` | `0.0015` | `0.0001` | `0.0000` | 19 | 19 | . | . | . |
|  8 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1` | MNQ | EUROPE_FLOW | 2.0 | `ORB_G5` | `+38.1` | `+38.1` | `0.158` | `0.0004` | `0.1594` | `0.0060` | `0.0000` | `0.0000` | `0.0000` | 27 | 27 | . | . | . |
|  9 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1` | MNQ | EUROPE_FLOW | 1.5 | `OVNRNG_100` | `+37.4` | `+37.4` | `0.158` | `0.4098` | `0.8746` | `0.6015` | `0.2568` | `0.1244` | `0.0249` |  5 |  3 | . | Y | Y |
| 10 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1` | MNQ | EUROPE_FLOW | 1.0 | `ORB_G5` | `+34.5` | `+34.5` | `0.158` | `0.0007` | `0.1816` | `0.0086` | `0.0001` | `0.0000` | `0.0000` | 26 | 26 | . | . | . |
| 11 | `MNQ_COMEX_SETTLE_E2_RR2.0_CB` | MNQ | COMEX_SETTLE | 2.0 | `ORB_G5` | `+34.2` | `+34.2` | `0.158` | `0.0002` | `0.1101` | `0.0033` | `0.0000` | `0.0000` | `0.0000` | 28 | 29 | . | Y | . |
| 12 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1` | MNQ | EUROPE_FLOW | 1.0 | `COST_LT12` | `+32.6` | `+32.6` | `0.158` | `0.0385` | `0.5436` | `0.1343` | `0.0094` | `0.0012` | `0.0000` | 12 | 11 | . | . | . |
| 13 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB` | MNQ | COMEX_SETTLE | 1.0 | `OVNRNG_100` | `+30.8` | `+30.8` | `0.158` | `0.7560` | `0.9769` | `0.8753` | `0.6148` | `0.4283` | `0.1726` |  1 |  2 | . | Y | . |
| 14 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_O` | MNQ | NYSE_OPEN | 1.0 | `ORB_G5` | `+29.0` | `+29.0` | `0.158` | `0.0061` | `0.4204` | `0.0452` | `0.0006` | `0.0000` | `0.0000` | 20 | 20 | Y | Y | Y |
| 15 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_C` | MNQ | NYSE_OPEN | 1.0 | `COST_LT12` | `+29.0` | `+29.0` | `0.158` | `0.0050` | `0.3894` | `0.0386` | `0.0005` | `0.0000` | `0.0000` | 21 | 21 | . | . | . |
| 16 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB` | MNQ | COMEX_SETTLE | 1.0 | `ORB_G5` | `+27.4` | `+27.4` | `0.158` | `0.0127` | `0.5073` | `0.0748` | `0.0017` | `0.0001` | `0.0000` | 17 | 18 | . | . | . |
| 17 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1` | MNQ | EUROPE_FLOW | 1.0 | `OVNRNG_100` | `+26.8` | `+26.8` | `0.158` | `0.2670` | `0.7643` | `0.4409` | `0.1500` | `0.0637` | `0.0104` |  6 |  5 | . | . | . |
| 18 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1` | MNQ | EUROPE_FLOW | 2.0 | `CROSS_SGP_MOMENTUM` | `+26.6` | `+26.6` | `0.158` | `0.0172` | `0.4303` | `0.0761` | `0.0033` | `0.0003` | `0.0000` | 16 | 15 | . | . | . |
| 19 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB` | MNQ | COMEX_SETTLE | 1.0 | `COST_LT12` | `+24.9` | `+24.9` | `0.158` | `0.1015` | `0.7851` | `0.2926` | `0.0280` | `0.0039` | `0.0001` |  8 |  7 | . | . | . |
| 20 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1` | MNQ | EUROPE_FLOW | 1.0 | `CROSS_SGP_MOMENTUM` | `+23.5` | `+23.5` | `0.158` | `0.0283` | `0.4861` | `0.1063` | `0.0065` | `0.0008` | `0.0000` | 13 | 14 | . | . | . |
| 21 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_` | MNQ | TOKYO_OPEN | 1.5 | `COST_LT12` | `+20.3` | `+20.3` | `0.158` | `0.0845` | `0.6699` | `0.2311` | `0.0264` | `0.0047` | `0.0001` |  9 | 10 | Y | Y | Y |
| 22 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_` | MNQ | TOKYO_OPEN | 1.5 | `ORB_G5` | `+19.1` | `+19.1` | `0.158` | `0.0024` | `0.3062` | `0.0227` | `0.0002` | `0.0000` | `0.0000` | 24 | 22 | . | . | . |
| 23 | `MNQ_US_DATA_1000_E2_RR1.5_CB` | MNQ | US_DATA_1000 | 1.5 | `ORB_G5` | `+18.8` | `+18.8` | `0.158` | `0.0015` | `0.2229` | `0.0141` | `0.0001` | `0.0000` | `0.0000` | 25 | 25 | Y | . | . |
| 24 | `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_` | MNQ | TOKYO_OPEN | 2.0 | `ORB_G5` | `+18.6` | `+18.6` | `0.158` | `0.0002` | `0.1070` | `0.0029` | `0.0000` | `0.0000` | `0.0000` | 30 | 30 | . | . | . |
| 25 | `MNQ_US_DATA_1000_E2_RR1.5_CB` | MNQ | US_DATA_1000 | 1.5 | `VWAP_MID_ALIGNED` | `+16.0` | `+16.0` | `0.158` | `0.6601` | `0.9770` | `0.8342` | `0.4696` | `0.2574` | `0.0569` |  3 |  4 | . | Y | Y |
| 26 | `MNQ_US_DATA_1000_E2_RR1.0_CB` | MNQ | US_DATA_1000 | 1.0 | `VWAP_MID_ALIGNED` | `+13.8` | `+13.8` | `0.158` | `0.4883` | `0.9400` | `0.7017` | `0.3020` | `0.1370` | `0.0215` |  4 |  6 | . | . | . |
| 27 | `MNQ_US_DATA_1000_E2_RR2.0_CB` | MNQ | US_DATA_1000 | 2.0 | `VWAP_MID_ALIGNED` | `+12.1` | `+12.1` | `0.158` | `0.1874` | `0.7437` | `0.3655` | `0.0863` | `0.0271` | `0.0023` |  7 |  9 | . | . | . |
| 28 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_C` | MNQ | NYSE_OPEN | 1.5 | `COST_LT12` | `+12.0` | `+12.0` | `0.158` | `0.0033` | `0.3408` | `0.0283` | `0.0003` | `0.0000` | `0.0000` | 23 | 24 | . | . | . |
| 29 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_O` | MNQ | NYSE_OPEN | 1.5 | `ORB_G5` | `+12.0` | `+12.0` | `0.158` | `0.0037` | `0.3618` | `0.0317` | `0.0003` | `0.0000` | `0.0000` | 22 | 23 | . | . | . |
| 30 | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_` | MNQ | TOKYO_OPEN | 1.0 | `COST_LT12` | `+7.1` | `+7.1` | `0.158` | `0.0628` | `0.5960` | `0.1825` | `0.0188` | `0.0032` | `0.0001` | 10 | 16 | . | . | . |

## Self-consistency

`build_allocation` under raw `_effective_annual_r` reproduces `docs/runtime/lane_allocation.json` exactly (HALT otherwise). Monkey-patch path under `selection_under_objective(score_fn=score_raw)` reproduces the same set (additional sanity check; HALT otherwise).

## Known limitation — Mode-B grandfathered DSR inputs

This audit consumes `validated_setups.{sharpe_ratio, sample_size, skewness, kurtosis_excess}` as the validator does (`strategy_validator.py:2206-2215`). Per the 2026-04-19 Mode-A revalidation finding (`docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`), all 38 active rows drift materially from strict Mode A. A separate audit (Phase 3b candidate) is needed to recompute lane DSR against Mode-A-fresh Sharpe/skew/kurt before any Stage-2 patch ships. This Phase 3a result is apples-to-apples with current allocator behavior; it is NOT the Mode-A-true ranking.

## Next phase

Result feeds:

- A2b-1 PAUSED note (BUG_COSMETIC verdict + this Phase 3a evidence reorders priority) — `docs/audit/hypotheses/2026-04-18-a2b-1-regime-gate-filtered-patch-preregistered.md`
- Multi-phase roadmap update — `docs/plans/2026-04-18-multi-phase-audit-roadmap.md` Phase 3 promoted
- Phase 3 Stage-1 scope doc — `docs/audit/hypotheses/2026-04-18-a2b-2-dsr-ranking-preregistered.md` (TO WRITE, informed by this MD)

