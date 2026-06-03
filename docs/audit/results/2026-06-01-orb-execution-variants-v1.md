# ORB Execution Variants v1

**Pre-reg:** `docs/audit/hypotheses/2026-06-01-orb-execution-variants-v1.yaml`
**Canonical inputs:** `bars_1m`, `daily_features`, `orb_outcomes` only.
**Selectable K:** `114`
**Holdout policy:** pre-2026 selects; `2026-01-01` onward is descriptive only.
**Full cell CSV:** `docs/audit/results/2026-06-01-orb-execution-variants-v1-cells.csv`

## Executive Verdict

No candidate clears the powered policy gates. Priority additions: **none**.
Primary same-direction answer: **KILL**. Best same-direction cell is `same_dir_reentry__CME_PRECLOSE__rr2__wait5` with delta `0.0334`, BH-family `0.0525`, DSR `0.0000`, era-stable `False`, and 2026 descriptive delta `-0.0278`.
Closest non-primary signal: `fakeout_reversal__US_DATA_1000__rr2__wait5` is an opposite-direction reversal, not the user's same-direction re-entry failure mode; verdict `NARROW`.

This report does not infer stop-hunt intent. The measurable object is: parent E2 entry stops, then a bounded execution modification is tested per original ORB opportunity.
All EV numbers include skipped trades as zero and include the first stopped trade before any re-entry recovery.

## Priority Table

| Rank | Verdict | Candidate | Role | Parent EV | Modified EV | Delta EV | DD Delta | p | BH family | DSR | WFE | 2026 delta |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | NARROW | `fakeout_reversal__US_DATA_1000__rr2__wait5` | standalone | 0.1110 | 0.2288 | 0.1178 | -9.4282 | 0.0000 | 0.0000 | 0.0000 | 0.6466 | 0.1057 |
| 2 | NARROW | `fakeout_reversal__NYSE_OPEN__rr2__wait5` | standalone | 0.1080 | 0.1975 | 0.0895 | 3.1587 | 0.0002 | 0.0004 | 0.0000 | 1.1408 | 0.0102 |
| 3 | NARROW | `fakeout_reversal__US_DATA_1000__rr1.5__wait5` | standalone | 0.0985 | 0.1845 | 0.0859 | -8.9636 | 0.0000 | 0.0001 | 0.0000 | 0.7452 | 0.0867 |
| 4 | NARROW | `fakeout_reversal__US_DATA_1000__rr1__wait5` | standalone | 0.0864 | 0.1534 | 0.0670 | -6.7946 | 0.0000 | 0.0000 | 0.0000 | 0.8479 | 0.0207 |
| 5 | NARROW | `fakeout_reversal__CME_PRECLOSE__rr2__wait5` | standalone | 0.1194 | 0.1781 | 0.0587 | 7.3453 | 0.0000 | 0.0001 | 0.0000 | 1.4349 | 0.1115 |
| 6 | NARROW | `fakeout_reversal__NYSE_OPEN__rr1.5__wait5` | standalone | 0.0988 | 0.1553 | 0.0566 | 11.0337 | 0.0038 | 0.0069 | 0.0000 | 2.1077 | 0.0640 |
| 7 | NARROW | `fakeout_reversal__NYSE_OPEN__rr1__wait5` | standalone | 0.0777 | 0.1260 | 0.0483 | -1.6311 | 0.0007 | 0.0017 | 0.0000 | 1.2225 | 0.1074 |
| 8 | NARROW | `fakeout_reversal__CME_PRECLOSE__rr1.5__wait5` | standalone | 0.0982 | 0.1461 | 0.0478 | 4.6455 | 0.0001 | 0.0003 | 0.0000 | 1.8185 | 0.0680 |
| 9 | NARROW | `fakeout_reversal__CME_PRECLOSE__rr1__wait5` | standalone | 0.0760 | 0.1023 | 0.0263 | 8.1799 | 0.0045 | 0.0074 | 0.0000 | 2.2511 | 0.0460 |
| 10 | KILL | `same_dir_reentry__CME_PRECLOSE__rr2__wait5` | execution | 0.1194 | 0.1528 | 0.0334 | 10.4968 | 0.0117 | 0.0525 | 0.0000 | 2.2857 | -0.0278 |
| 11 | KILL | `fakeout_reversal__US_DATA_1000__rr2__wait0` | standalone | 0.1110 | 0.1409 | 0.0299 | 6.6135 | 0.2150 | 0.2977 | 0.0000 | 0.1206 | 0.0272 |
| 12 | KILL | `fakeout_reversal__NYSE_OPEN__rr2__wait0` | standalone | 0.1080 | 0.1343 | 0.0262 | 14.6796 | 0.2672 | 0.3436 | 0.0000 | 0.9415 | -0.0013 |
| 13 | KILL | `same_dir_reentry__CME_PRECLOSE__rr1.5__wait5` | execution | 0.0982 | 0.1195 | 0.0212 | 8.5161 | 0.0580 | 0.1322 | 0.0000 | 2.5169 | -0.0349 |
| 14 | KILL | `same_dir_reentry__US_DATA_1000__rr1__wait5` | execution | 0.0864 | 0.1032 | 0.0169 | -2.5890 | 0.1726 | 0.2913 | 0.0000 | 0.4152 | 0.0142 |
| 15 | KILL | `same_dir_reentry__US_DATA_1000__rr1.5__wait5` | execution | 0.0985 | 0.1137 | 0.0152 | 11.3648 | 0.3634 | 0.5772 | 0.0000 | 0.2240 | 0.0343 |
| 16 | KILL | `fakeout_reversal__US_DATA_1000__rr1.5__wait0` | standalone | 0.0985 | 0.1123 | 0.0138 | -0.9186 | 0.4868 | 0.5477 | 0.0000 | -0.0672 | 0.0340 |
| 17 | KILL | `same_dir_reentry__CME_PRECLOSE__rr1__wait5` | execution | 0.0760 | 0.0888 | 0.0128 | 9.6270 | 0.1408 | 0.2535 | 0.0000 | 10.7377 | -0.0286 |
| 18 | KILL | `fakeout_reversal__US_DATA_1000__rr1__wait0` | standalone | 0.0864 | 0.0978 | 0.0114 | -3.8853 | 0.4314 | 0.5177 | 0.0000 | -0.0600 | -0.0356 |
| 19 | KILL | `same_dir_reentry__US_DATA_1000__rr1.5__wait2` | execution | 0.0985 | 0.1093 | 0.0107 | 12.3648 | 0.5213 | 0.7054 | 0.0000 | 0.1078 | 0.0227 |
| 20 | KILL | `same_dir_reentry__US_DATA_1000__rr1__wait2` | execution | 0.0864 | 0.0953 | 0.0089 | -1.5344 | 0.4725 | 0.7054 | 0.0000 | 0.3865 | 0.0026 |
| 21 | KILL | `same_dir_reentry__US_DATA_1000__rr1.5__wait0` | execution | 0.0985 | 0.1060 | 0.0074 | 17.0571 | 0.6574 | 0.7718 | 0.0000 | -0.0521 | -0.0052 |
| 22 | KILL | `same_dir_reentry__US_DATA_1000__rr1__wait0` | execution | 0.0864 | 0.0936 | 0.0073 | -1.5344 | 0.5582 | 0.7054 | 0.0000 | 0.2284 | -0.0197 |
| 23 | KILL | `same_dir_reentry__US_DATA_1000__rr2__wait5` | execution | 0.1110 | 0.1153 | 0.0043 | 12.3513 | 0.8296 | 0.8615 | 0.0000 | -0.4454 | 0.0128 |
| 24 | KILL | `same_dir_reentry__CME_PRECLOSE__rr2__wait2` | execution | 0.1194 | 0.1235 | 0.0041 | 13.0285 | 0.7647 | 0.8603 | 0.0000 | -6.9126 | -0.0581 |
| 25 | KILL | `fakeout_reversal__NYSE_OPEN__rr1.5__wait0` | standalone | 0.0988 | 0.1001 | 0.0013 | 20.2797 | 0.9474 | 0.9474 | 0.0000 | -2.8505 | 0.0806 |

## Family Summary

| Family | Best candidate | Verdicts | Best delta EV | Best p | Best BH family | Best DSR |
|---|---|---|---:|---:|---:|---:|
| fakeout_reversal_after_stop | `fakeout_reversal__US_DATA_1000__rr2__wait5` | KILL=9, NARROW=9 | 0.1178 | 0.0000 | 0.0000 | 0.0000 |
| initial_entry_confirmation_delay | `initial_confirmation_delay__NYSE_OPEN__rr2__cb5` | KILL=27 | -0.0729 | 0.0048 | 0.0048 | 0.0000 |
| one_loss_session_throttle | `one_loss_throttle__rr1__half_after_first_loss` | KILL=6 | -0.0156 | 0.0000 | 0.0000 | 0.0000 |
| pre_entry_size_atr_filter | `pre_entry_filter__CME_PRECLOSE__rr1__orb_size_q67` | KILL=18 | -0.0146 | 0.4012 | 0.4012 | 0.0000 |
| random_non_open_range_control | `control_non_open_range__CME_PRECLOSE__rr2__offset73_wait5` | PARK=9 | 0.0000 | NA | NA | NA |
| retest_hold_before_stop | `retest_hold__CME_PRECLOSE__rr1__wait0` | KILL=18 | -0.0612 | 0.0000 | 0.0000 | 0.0000 |
| same_direction_reentry_after_stop | `same_dir_reentry__CME_PRECLOSE__rr2__wait5` | KILL=27 | 0.0334 | 0.0117 | 0.0525 | 0.0000 |
| shuffled_reentry_date_control | `control_shuffled_date_same_dir__US_DATA_1000__rr2__wait5` | PARK=9 | 0.3884 | 0.0000 | NA | NA |

## Controls

- Known-bad E2 lookahead injection: `orb_NYSE_OPEN_break_ts`, `rel_vol_NYSE_OPEN`, and `pnl_r` are rejected before the run.
- Inverted-direction trigger: tested as `fakeout_reversal_after_stop`; it is scored separately from same-direction re-entry and is not allowed to rescue that family.
- Shuffled re-entry trigger dates: tested as `shuffled_reentry_date_control` by shifting the next trading day's path onto the parent day; not eligible for ranking.
- Random non-open range window: tested as `random_non_open_range_control` using a deterministic 5m range 73 minutes after session OR start; not eligible for ranking.
- Order-flow, footprint, delta, and absorption variants: `PARK_NEW_DATA`; current 1m OHLCV cannot measure them honestly.
- Control read: shuffled-date controls print large positive deltas, which is a construction-sensitivity warning, not evidence. It reinforces the decision not to promote a path narrative from these 1m OHLCV controls.

## Candidate Detail

### `fakeout_reversal__US_DATA_1000__rr2__wait5`

- role: `standalone`
- measurable trigger rule: After parent E2 full stop, wait 5 bars, then enter the opposite ORB side; max one reversal.
- parent population: MNQ US_DATA_1000 O5 E2 CB1 RR2
- original parent EV: `0.1110`
- modified policy EV, including first loss/skips: `0.2288`
- delta drawdown / tail loss: `-9.4282` / `-1.0000`
- sample and span: `N=1718`, `2019-05-06..2025-12-31`
- p-value / BH K / DSR / WFE / era stability: `0.0000` / `114` / `0.0000` / `0.6466` / `True`
- 2026 descriptive result, non-selection: `N=86`, delta `0.1057`
- verdict: `NARROW`

### `fakeout_reversal__NYSE_OPEN__rr2__wait5`

- role: `standalone`
- measurable trigger rule: After parent E2 full stop, wait 5 bars, then enter the opposite ORB side; max one reversal.
- parent population: MNQ NYSE_OPEN O5 E2 CB1 RR2
- original parent EV: `0.1080`
- modified policy EV, including first loss/skips: `0.1975`
- delta drawdown / tail loss: `3.1587` / `-1.0000`
- sample and span: `N=1719`, `2019-05-06..2025-12-31`
- p-value / BH K / DSR / WFE / era stability: `0.0002` / `114` / `0.0000` / `1.1408` / `True`
- 2026 descriptive result, non-selection: `N=87`, delta `0.0102`
- verdict: `NARROW`

### `fakeout_reversal__US_DATA_1000__rr1.5__wait5`

- role: `standalone`
- measurable trigger rule: After parent E2 full stop, wait 5 bars, then enter the opposite ORB side; max one reversal.
- parent population: MNQ US_DATA_1000 O5 E2 CB1 RR1.5
- original parent EV: `0.0985`
- modified policy EV, including first loss/skips: `0.1845`
- delta drawdown / tail loss: `-8.9636` / `-1.0000`
- sample and span: `N=1718`, `2019-05-06..2025-12-31`
- p-value / BH K / DSR / WFE / era stability: `0.0000` / `114` / `0.0000` / `0.7452` / `True`
- 2026 descriptive result, non-selection: `N=86`, delta `0.0867`
- verdict: `NARROW`

### `fakeout_reversal__US_DATA_1000__rr1__wait5`

- role: `standalone`
- measurable trigger rule: After parent E2 full stop, wait 5 bars, then enter the opposite ORB side; max one reversal.
- parent population: MNQ US_DATA_1000 O5 E2 CB1 RR1
- original parent EV: `0.0864`
- modified policy EV, including first loss/skips: `0.1534`
- delta drawdown / tail loss: `-6.7946` / `-1.0000`
- sample and span: `N=1718`, `2019-05-06..2025-12-31`
- p-value / BH K / DSR / WFE / era stability: `0.0000` / `114` / `0.0000` / `0.8479` / `True`
- 2026 descriptive result, non-selection: `N=86`, delta `0.0207`
- verdict: `NARROW`

### `fakeout_reversal__CME_PRECLOSE__rr2__wait5`

- role: `standalone`
- measurable trigger rule: After parent E2 full stop, wait 5 bars, then enter the opposite ORB side; max one reversal.
- parent population: MNQ CME_PRECLOSE O5 E2 CB1 RR2
- original parent EV: `0.1194`
- modified policy EV, including first loss/skips: `0.1781`
- delta drawdown / tail loss: `7.3453` / `-1.0000`
- sample and span: `N=1643`, `2019-05-06..2025-12-31`
- p-value / BH K / DSR / WFE / era stability: `0.0000` / `114` / `0.0000` / `1.4349` / `True`
- 2026 descriptive result, non-selection: `N=83`, delta `0.1115`
- verdict: `NARROW`

### `fakeout_reversal__NYSE_OPEN__rr1.5__wait5`

- role: `standalone`
- measurable trigger rule: After parent E2 full stop, wait 5 bars, then enter the opposite ORB side; max one reversal.
- parent population: MNQ NYSE_OPEN O5 E2 CB1 RR1.5
- original parent EV: `0.0988`
- modified policy EV, including first loss/skips: `0.1553`
- delta drawdown / tail loss: `11.0337` / `-1.0000`
- sample and span: `N=1719`, `2019-05-06..2025-12-31`
- p-value / BH K / DSR / WFE / era stability: `0.0038` / `114` / `0.0000` / `2.1077` / `False`
- 2026 descriptive result, non-selection: `N=87`, delta `0.0640`
- verdict: `NARROW`

### `fakeout_reversal__NYSE_OPEN__rr1__wait5`

- role: `standalone`
- measurable trigger rule: After parent E2 full stop, wait 5 bars, then enter the opposite ORB side; max one reversal.
- parent population: MNQ NYSE_OPEN O5 E2 CB1 RR1
- original parent EV: `0.0777`
- modified policy EV, including first loss/skips: `0.1260`
- delta drawdown / tail loss: `-1.6311` / `-1.0000`
- sample and span: `N=1719`, `2019-05-06..2025-12-31`
- p-value / BH K / DSR / WFE / era stability: `0.0007` / `114` / `0.0000` / `1.2225` / `True`
- 2026 descriptive result, non-selection: `N=87`, delta `0.1074`
- verdict: `NARROW`

### `fakeout_reversal__CME_PRECLOSE__rr1.5__wait5`

- role: `standalone`
- measurable trigger rule: After parent E2 full stop, wait 5 bars, then enter the opposite ORB side; max one reversal.
- parent population: MNQ CME_PRECLOSE O5 E2 CB1 RR1.5
- original parent EV: `0.0982`
- modified policy EV, including first loss/skips: `0.1461`
- delta drawdown / tail loss: `4.6455` / `-1.0000`
- sample and span: `N=1643`, `2019-05-06..2025-12-31`
- p-value / BH K / DSR / WFE / era stability: `0.0001` / `114` / `0.0000` / `1.8185` / `True`
- 2026 descriptive result, non-selection: `N=83`, delta `0.0680`
- verdict: `NARROW`

### `fakeout_reversal__CME_PRECLOSE__rr1__wait5`

- role: `standalone`
- measurable trigger rule: After parent E2 full stop, wait 5 bars, then enter the opposite ORB side; max one reversal.
- parent population: MNQ CME_PRECLOSE O5 E2 CB1 RR1
- original parent EV: `0.0760`
- modified policy EV, including first loss/skips: `0.1023`
- delta drawdown / tail loss: `8.1799` / `-1.0000`
- sample and span: `N=1643`, `2019-05-06..2025-12-31`
- p-value / BH K / DSR / WFE / era stability: `0.0045` / `114` / `0.0000` / `2.2511` / `True`
- 2026 descriptive result, non-selection: `N=83`, delta `0.0460`
- verdict: `NARROW`

## SURVIVED SCRUTINY

- None.

## DID NOT SURVIVE

- `NARROW`: 9 cells.
- `PARK`: 18 cells.
- `KILL`: 105 cells.

## CAVEATS

- DSR uses the fixed universe K declared here and the cross-sectional variance of candidate delta Sharpes. It is a research-validation cross-check, not deployment permission.
- Re-entry path simulation uses 1m OHLC bars and conservative same-bar stop/target ordering. It cannot observe queue position or footprint absorption.
- The shuffled-date control is intentionally non-selectable and came back positive; that makes narrative explanations weaker, not stronger.
- 2026 rows are descriptive monitoring only; no threshold, session, or candidate ranking is selected from them.

## NEXT STEPS

- Only `CONTINUE` rows, if any, should move to a separate deployment-readiness route.
- `NARROW` rows need a fresh prereg or an implementation simplification before retest.
- `PARK_NEW_DATA` order-flow claims need actual order-flow data before they can be ranked.
