# Phase 3b — Mode-A-true DSR ranking comparison

- rebalance_date: `2026-04-18`
- profile: `topstep_50k_mnq_auto`
- Mode-A IS boundary: `trading_day < 2026-01-01`
- lanes audited: `30`
- canonical SQL pattern: mirrors `research/mode_a_revalidation_active_setups.py::compute_mode_a` lines 146-209
- DSR inputs delta vs Phase 3a: per-lane Sharpe/skew/kurt RECOMPUTED under Mode-A; var_sr_by_em + n_eff UNCHANGED (cross-strategy values; out of scope per A2b-2 §5)
- prerequisite for: A2b-2 Stage-2 implementation per K1 of `docs/audit/hypotheses/2026-04-18-a2b-2-dsr-ranking-preregistered.md`
- one-shot lock + zero new OOS consumption

## DSR cross-strategy inputs (unchanged from Phase 3a)

- N_eff (edge_families distinct count): `21`
- var_sr by entry_model: `{'E1': 0.047, 'E2': 0.006712097170233147}`

## Selection comparison (top max_slots = 7) — Mode-A inputs

- `sel_raw`        (6): ['MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5', 'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5', 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5', 'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15', 'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12', 'MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15']
- `sel_dsr_mode_a` (7): ['MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100', 'MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5', 'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100', 'MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12', 'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30', 'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12', 'MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15']
- `sel_combo_mode_a` (7): ['MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100', 'MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5', 'MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100', 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12', 'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30', 'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12', 'MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15']

Cross-reference with Phase 3a (Mode-B inputs):

- Phase 3a sel_dsr   ∩ Phase 3b sel_dsr:   `6` of `7`
- Phase 3a sel_combo ∩ Phase 3b sel_combo: `4` of `6`

## Verdict

Non-tied selection delta vs raw under Mode-A: `dsr=11, combo=11`

**RANKING_MATERIAL_PRESERVED-WITH-PARTIAL-AGREEMENT** — Mode-A DSR rank still flips selection vs raw (`dsr=11` slots, `combo=11` slots), but the Mode-A selection set is NOT identical to the Mode-B (Phase 3a) selection set:

- `sel_dsr` overlap with Phase 3a: `6 of 7` = 86%
- `sel_combo` overlap with Phase 3a: `4 of 6` = 67%

**The broad shape holds across both runs** — OVNRNG_100, VWAP_MID_ALIGNED, and ATR_P50_O30 are systematically promoted in both Mode-B and Mode-A; ORB_G5 lanes are systematically demoted in both. **Specific lane variants shift** between the two regimes (e.g., NYSE_OPEN under Mode-A `sel_dsr` picks RR1.5/COST_LT12 instead of Phase 3a's RR1.0/ORB_G5; same session, different filter+RR — within the family of high-DSR alternatives).

The 67% overlap on `sel_combo` is meaningful disagreement: the multiplicative score `annual_r × DSR` is more sensitive to the underlying Sharpe values, so Mode-A correction shifts more lanes there than under DSR-alone ranking.

Direction overlap on FLIPPED lanes (lanes either added or removed vs raw): of the lanes Phase 3a flipped under DSR, `9` of `9` are ALSO flipped under Mode-A. The remaining `0` were Mode-B artifacts; the patch's flip-direction is robust.

**A2b-2 K1 prerequisite SATISFIED** — direction confirmed; Stage-2 may proceed pending user approval of patch shape (A/B/C/D in scope §11). Note the ~30% selection-instability across the two runs argues for additional caution at first-rebalance gate K6.

## Per-lane Mode-A DSR table

| rank_raw | strategy_id | inst | session | RR | filter | Mode-A N | Mode-A Sharpe | Mode-A skew | Mode-A kurt | SR0 | DSR_mode_a | DSR_phase3a | drift | rank_dsr_mA | rank_combo_mA | sel_raw | sel_dsr_mA | sel_combo_mA |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
|  1 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_O` | MNQ | EUROPE_FLOW | 1.5 | `ORB_G5` | `773` | `+0.0684` | `+0.095` | `-1.975` | `0.158` | `0.0065` | `0.0002` | `+0.0063` | 25 | 22 | Y | . | . |
|  2 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB` | MNQ | SINGAPORE_OPEN | 1.5 | `ATR_P50` | `496` | `+0.1777` | `-0.084` | `-1.981` | `0.158` | `0.6724` | `0.0239` | `+0.6485` |  3 |  2 | Y | . | . |
|  3 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB` | MNQ | SINGAPORE_OPEN | 1.5 | `ATR_P50` | `485` | `+0.1878` | `-0.071` | `-1.990` | `0.158` | `0.7462` | `0.0551` | `+0.6911` |  2 |  1 | . | Y | Y |
|  4 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_` | MNQ | COMEX_SETTLE | 1.5 | `ORB_G5` | `829` | `+0.0805` | `+0.101` | `-1.978` | `0.158` | `0.0131` | `0.0105` | `+0.0026` | 19 | 17 | Y | . | . |
|  5 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_C` | MNQ | EUROPE_FLOW | 1.5 | `COST_LT12` | `521` | `+0.0907` | `+0.092` | `-1.991` | `0.158` | `0.0631` | `0.0180` | `+0.0451` | 12 |  9 | . | . | . |
|  6 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_` | MNQ | COMEX_SETTLE | 1.5 | `OVNRNG_100` | `278` | `+0.1592` | `-0.020` | `-1.996` | `0.158` | `0.5110` | `0.7183` | `-0.2073` |  4 |  4 | . | . | . |
|  7 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_C` | MNQ | EUROPE_FLOW | 1.5 | `CROSS_SGP_MOMENTUM` | `535` | `+0.0735` | `+0.065` | `-1.958` | `0.158` | `0.0259` | `0.0087` | `+0.0172` | 17 | 14 | . | . | . |
|  8 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_O` | MNQ | EUROPE_FLOW | 2.0 | `ORB_G5` | `773` | `+0.0556` | `+0.427` | `-1.803` | `0.158` | `0.0021` | `0.0004` | `+0.0017` | 28 | 27 | . | . | . |
|  9 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_O` | MNQ | EUROPE_FLOW | 1.5 | `OVNRNG_100` | `263` | `+0.1017` | `+0.076` | `-1.992` | `0.158` | `0.1821` | `0.4098` | `-0.2277` |  7 |  6 | . | Y | Y |
| 10 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_O` | MNQ | EUROPE_FLOW | 1.0 | `ORB_G5` | `773` | `+0.0397` | `-0.293` | `-1.893` | `0.158` | `0.0006` | `0.0007` | `-0.0001` | 30 | 29 | . | . | . |
| 11 | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_` | MNQ | COMEX_SETTLE | 2.0 | `ORB_G5` | `814` | `+0.0785` | `+0.405` | `-1.825` | `0.158` | `0.0110` | `0.0002` | `+0.0108` | 22 | 21 | . | Y | Y |
| 12 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_C` | MNQ | EUROPE_FLOW | 1.0 | `COST_LT12` | `521` | `+0.0589` | `-0.278` | `-1.920` | `0.158` | `0.0129` | `0.0385` | `-0.0256` | 20 | 20 | . | . | . |
| 13 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_` | MNQ | COMEX_SETTLE | 1.0 | `OVNRNG_100` | `283` | `+0.2035` | `-0.536` | `-1.705` | `0.158` | `0.7683` | `0.7560` | `+0.0123` |  1 |  3 | . | Y | Y |
| 14 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB` | MNQ | NYSE_OPEN | 1.0 | `ORB_G5` | `856` | `+0.0693` | `-0.212` | `-1.953` | `0.158` | `0.0052` | `0.0061` | `-0.0009` | 27 | 25 | Y | . | . |
| 15 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COS` | MNQ | NYSE_OPEN | 1.0 | `COST_LT12` | `844` | `+0.0724` | `-0.216` | `-1.952` | `0.158` | `0.0071` | `0.0050` | `+0.0021` | 24 | 24 | . | . | Y |
| 16 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_` | MNQ | COMEX_SETTLE | 1.0 | `ORB_G5` | `836` | `+0.0865` | `-0.348` | `-1.861` | `0.158` | `0.0216` | `0.0127` | `+0.0089` | 18 | 16 | . | . | . |
| 17 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_O` | MNQ | EUROPE_FLOW | 1.0 | `OVNRNG_100` | `263` | `+0.0609` | `-0.274` | `-1.921` | `0.158` | `0.0604` | `0.2670` | `-0.2066` | 13 | 11 | . | . | . |
| 18 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_C` | MNQ | EUROPE_FLOW | 2.0 | `CROSS_SGP_MOMENTUM` | `535` | `+0.0856` | `+0.348` | `-1.845` | `0.158` | `0.0459` | `0.0172` | `+0.0287` | 14 | 13 | . | . | . |
| 19 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_` | MNQ | COMEX_SETTLE | 1.0 | `COST_LT12` | `669` | `+0.1137` | `-0.373` | `-1.855` | `0.158` | `0.1338` | `0.1015` | `+0.0323` |  9 |  8 | . | . | . |
| 20 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_C` | MNQ | EUROPE_FLOW | 1.0 | `CROSS_SGP_MOMENTUM` | `535` | `+0.0577` | `-0.353` | `-1.829` | `0.158` | `0.0112` | `0.0283` | `-0.0171` | 21 | 23 | . | . | . |
| 21 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_CO` | MNQ | TOKYO_OPEN | 1.5 | `COST_LT12` | `469` | `+0.0900` | `+0.085` | `-1.991` | `0.158` | `0.0712` | `0.0845` | `-0.0133` | 10 | 12 | Y | Y | Y |
| 22 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_OR` | MNQ | TOKYO_OPEN | 1.5 | `ORB_G5` | `786` | `+0.0893` | `+0.032` | `-1.980` | `0.158` | `0.0278` | `0.0024` | `+0.0254` | 16 | 18 | . | . | . |
| 23 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_` | MNQ | US_DATA_1000 | 1.5 | `ORB_G5` | `800` | `+0.0472` | `+0.250` | `-1.935` | `0.158` | `0.0009` | `0.0015` | `-0.0006` | 29 | 30 | Y | . | . |
| 24 | `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_OR` | MNQ | TOKYO_OPEN | 2.0 | `ORB_G5` | `785` | `+0.0936` | `+0.326` | `-1.877` | `0.158` | `0.0347` | `0.0002` | `+0.0345` | 15 | 15 | . | . | . |
| 25 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_` | MNQ | US_DATA_1000 | 1.5 | `VWAP_MID_ALIGNED` | `436` | `+0.1532` | `+0.033` | `-1.997` | `0.158` | `0.4643` | `0.6601` | `-0.1958` |  5 |  5 | . | Y | Y |
| 26 | `MNQ_US_DATA_1000_E2_RR1.0_CB1_` | MNQ | US_DATA_1000 | 1.0 | `VWAP_MID_ALIGNED` | `460` | `+0.1396` | `-0.354` | `-1.870` | `0.158` | `0.3543` | `0.4883` | `-0.1340` |  6 |  7 | . | . | . |
| 27 | `MNQ_US_DATA_1000_E2_RR2.0_CB1_` | MNQ | US_DATA_1000 | 2.0 | `VWAP_MID_ALIGNED` | `398` | `+0.1050` | `+0.414` | `-1.826` | `0.158` | `0.1423` | `0.1874` | `-0.0451` |  8 | 10 | . | . | . |
| 28 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COS` | MNQ | NYSE_OPEN | 1.5 | `COST_LT12` | `817` | `+0.0739` | `+0.196` | `-1.962` | `0.158` | `0.0081` | `0.0033` | `+0.0048` | 23 | 26 | . | Y | . |
| 29 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB` | MNQ | NYSE_OPEN | 1.5 | `ORB_G5` | `829` | `+0.0715` | `+0.199` | `-1.959` | `0.158` | `0.0064` | `0.0037` | `+0.0027` | 26 | 28 | . | . | . |
| 30 | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_CO` | MNQ | TOKYO_OPEN | 1.0 | `COST_LT12` | `469` | `+0.0866` | `-0.345` | `-1.879` | `0.158` | `0.0652` | `0.0628` | `+0.0024` | 11 | 19 | . | . | . |

## Provenance

- Phase 3a Mode-B baseline: `docs/audit/results/2026-04-18-dsr-ranking-empirical-verification.md`
- A2b-2 Stage-1 scope: `docs/audit/hypotheses/2026-04-18-a2b-2-dsr-ranking-preregistered.md` (K1 binding)
- Mode-A revalidation context: `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`
- Canonical SQL pattern: `research/mode_a_revalidation_active_setups.py::compute_mode_a` lines 146-209
- DSR canonical: `trading_app/dsr.py`
- Holdout boundary: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`

