# ORB Retest Entry Pilot V1

Research-only pilot locked by `docs/audit/hypotheses/2026-04-20-orb-retest-entry-pilot-v1.yaml`.

## Scope

- Instruments: MES, MGC, MNQ
- Sessions: TOKYO_OPEN, EUROPE_FLOW, NYSE_OPEN, US_DATA_1000, COMEX_SETTLE, CME_PRECLOSE
- Aperture: O5
- RR targets: 1.0 / 1.5 / 2.0
- Baseline: canonical `E2` on the same retest-eligible days
- Holdout: 2026-01-01 onwards is diagnostic OOS only

## Event definition

- Require canonical ORB break in `daily_features`.
- After the break, require one later 1-minute close one tick beyond the ORB boundary in the breakout direction.
- Then enter at the first later touch of the ORB boundary.
- Stop = opposite ORB boundary; target = fixed RR multiple.
- Ambiguous stop+target bars resolve as loss.

## Family verdict

- Locked family K: 54
- Retest trades captured: IS 69744, OOS 3522
- Primary survivors (BH + paired-delta + N>=50 + OOS direction check): 1
- Trading-relevant survivors (same as above AND retest avg IS > 0): 0

No trading-relevant survivors.

The bounded first-touch-to-ORB-boundary continuation shape did not justify production-code integration on the locked scope.

Important audit note: one cell improved materially versus E2 after BH correction, but the retest path still had negative IS expectancy. That is a risk-reduction observation, not an edge.

## Warm cells (informational only)

- MES CME_PRECLOSE RR2.0: IS retest avg=-0.2426R, matched E2=-0.3651R, delta=+0.0718R, n_pairs=652, p=0.0179
- MES CME_PRECLOSE RR1.5: IS retest avg=-0.1565R, matched E2=-0.2669R, delta=+0.0640R, n_pairs=727, p=0.0203
- MNQ CME_PRECLOSE RR2.0: IS retest avg=-0.2492R, matched E2=-0.3244R, delta=+0.0722R, n_pairs=704, p=0.0238
- MGC US_DATA_1000 RR2.0: IS retest avg=-0.1708R, matched E2=-0.2242R, delta=+0.0454R, n_pairs=661, p=0.0846
- MNQ CME_PRECLOSE RR1.5: IS retest avg=-0.1262R, matched E2=-0.1777R, delta=+0.0398R, n_pairs=784, p=0.1594
- MES CME_PRECLOSE RR1.0: IS retest avg=-0.0645R, matched E2=-0.1025R, delta=+0.0235R, n_pairs=865, p=0.3099
- MGC COMEX_SETTLE RR2.0: IS retest avg=-0.2951R, matched E2=-0.3252R, delta=+0.0167R, n_pairs=690, p=0.4053
- MES EUROPE_FLOW RR2.0: IS retest avg=-0.1915R, matched E2=-0.2042R, delta=+0.0127R, n_pairs=1652, p=0.5390

## Strong negative cells

- MNQ TOKYO_OPEN RR1.5: IS retest avg=-0.0914R vs matched E2=+0.0646R, delta=-0.1560R, n_pairs=1691, p=0.0000
- MNQ TOKYO_OPEN RR1.0: IS retest avg=-0.0894R vs matched E2=+0.0447R, delta=-0.1340R, n_pairs=1692, p=0.0000
- MNQ TOKYO_OPEN RR2.0: IS retest avg=-0.1067R vs matched E2=+0.0538R, delta=-0.1605R, n_pairs=1689, p=0.0000
- MNQ EUROPE_FLOW RR1.0: IS retest avg=-0.0712R vs matched E2=+0.0332R, delta=-0.1044R, n_pairs=1664, p=0.0000
- MNQ NYSE_OPEN RR1.0: IS retest avg=-0.0356R vs matched E2=+0.0555R, delta=-0.0917R, n_pairs=1495, p=0.0000
- MNQ EUROPE_FLOW RR1.5: IS retest avg=-0.0830R vs matched E2=+0.0281R, delta=-0.1112R, n_pairs=1664, p=0.0000
- MNQ EUROPE_FLOW RR2.0: IS retest avg=-0.0863R vs matched E2=+0.0374R, delta=-0.1236R, n_pairs=1664, p=0.0000
- MNQ COMEX_SETTLE RR1.0: IS retest avg=-0.0519R vs matched E2=+0.0423R, delta=-0.0942R, n_pairs=1499, p=0.0000

## Caveats

- This is a research-only pilot, not a validator result and not a deployment recommendation.
- The retest rule is deliberately narrow; broader non-ORB pullback families remain untested here.
- No queue-position model is applied; the fill rule is resting-limit-on-touch with conservative ambiguity handling.
- If this pilot is dead, that kills the bounded ORB-integration route only. It does not kill standalone SC2 event families.

## Artefacts

- Cell CSV: `research/output/orb_retest_entry_pilot_v1_cells.csv`
- Script: `research/orb_retest_entry_pilot_v1.py`
