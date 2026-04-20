# Prior-Day Direction-Split ORB Overlays V1

Locked by `docs/audit/hypotheses/2026-04-20-prior-day-direction-split-orb-overlays-v1.yaml`.

## Scope

- Instrument: MNQ
- Sessions: NYSE_OPEN, COMEX_SETTLE, US_DATA_1000
- Aperture: O5
- Entry model: E2
- Confirm bars: CB1
- RR targets: 1.0, 1.5
- Directions: long, short
- Features: F1_NEAR_PDH_15, F5_BELOW_PDL, F6_INSIDE_PDR
- Mode A: trading_day < 2026-01-01 is IS; 2026+ is descriptive OOS only

## Resource grounding

- `resources/Algorithmic_Trading_Chan.pdf`: intraday/systematic strategy framing justifies testing bounded, executable pattern families instead of ad hoc chart lore.
- `resources/Robert Carver - Systematic Trading.pdf`: supports treating signals as conditioners/sizers rather than assuming every useful feature must be a standalone strategy.
- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`: theory-first discipline; use a small pre-registered family instead of backtest fishing.
- `resources/Two_Million_Trading_Strategies_FDR.pdf`: honest family-level multiple-testing control is mandatory.

## Family verdict

- Locked family K: 36
- Primary survivors: 7
- Conditional cells: 1

### Primary survivors

- NYSE_OPEN RR1.0 short F5_BELOW_PDL: TAKE | IS on/off ExpR +0.3531/+0.0531 | delta +0.3000 | t=3.39 | q=0.0062 | N_on/N_off=120/715
- NYSE_OPEN RR1.5 short F5_BELOW_PDL: TAKE | IS on/off ExpR +0.4688/+0.0447 | delta +0.4241 | t=3.59 | q=0.0041 | N_on/N_off=117/702
- COMEX_SETTLE RR1.0 long F6_INSIDE_PDR: AVOID | IS on/off ExpR -0.0296/+0.1651 | delta -0.1947 | t=-3.24 | q=0.0075 | N_on/N_off=433/443
- US_DATA_1000 RR1.0 long F5_BELOW_PDL: TAKE | IS on/off ExpR +0.3258/-0.0112 | delta +0.3370 | t=4.02 | q=0.0030 | N_on/N_off=136/745
- US_DATA_1000 RR1.0 long F6_INSIDE_PDR: AVOID | IS on/off ExpR -0.0434/+0.1583 | delta -0.2017 | t=-3.15 | q=0.0086 | N_on/N_off=513/368
- US_DATA_1000 RR1.5 long F5_BELOW_PDL: TAKE | IS on/off ExpR +0.4022/-0.0024 | delta +0.4046 | t=3.65 | q=0.0041 | N_on/N_off=135/731
- US_DATA_1000 RR1.5 long F6_INSIDE_PDR: AVOID | IS on/off ExpR -0.0588/+0.2261 | delta -0.2849 | t=-3.52 | q=0.0041 | N_on/N_off=503/363

### Conditional cells

- NYSE_OPEN RR1.0 long F6_INSIDE_PDR: AVOID | delta -0.1760 | t=-2.67 | q=0.0353 | N_on/N_off=505/352 | OOS dir match=None

## Strongest positive deltas

- NYSE_OPEN RR1.5 short F5_BELOW_PDL: role=TAKE delta=+0.4241 q=0.0041 N_on/N_off=117/702
- US_DATA_1000 RR1.5 long F5_BELOW_PDL: role=TAKE delta=+0.4046 q=0.0041 N_on/N_off=135/731
- US_DATA_1000 RR1.0 long F5_BELOW_PDL: role=TAKE delta=+0.3370 q=0.0030 N_on/N_off=136/745
- NYSE_OPEN RR1.0 short F5_BELOW_PDL: role=TAKE delta=+0.3000 q=0.0062 N_on/N_off=120/715
- COMEX_SETTLE RR1.5 short F5_BELOW_PDL: role=TAKE delta=+0.1535 q=0.3247 N_on/N_off=165/601
- US_DATA_1000 RR1.5 long F1_NEAR_PDH_15: role=TAKE delta=+0.1325 q=0.3585 N_on/N_off=171/695
- COMEX_SETTLE RR1.5 long F5_BELOW_PDL: role=TAKE delta=+0.1316 q=0.3585 N_on/N_off=173/696
- US_DATA_1000 RR1.0 short F1_NEAR_PDH_15: role=TAKE delta=+0.1114 q=0.3585 N_on/N_off=155/664

## Strongest negative deltas

- US_DATA_1000 RR1.5 long F6_INSIDE_PDR: role=AVOID delta=-0.2849 q=0.0041 N_on/N_off=503/363
- NYSE_OPEN RR1.5 long F6_INSIDE_PDR: role=AVOID delta=-0.2066 q=0.0592 N_on/N_off=492/338
- NYSE_OPEN RR1.5 short F6_INSIDE_PDR: role=AVOID delta=-0.2057 q=0.0676 N_on/N_off=523/296
- US_DATA_1000 RR1.0 long F6_INSIDE_PDR: role=AVOID delta=-0.2017 q=0.0086 N_on/N_off=513/368
- COMEX_SETTLE RR1.0 long F6_INSIDE_PDR: role=AVOID delta=-0.1947 q=0.0075 N_on/N_off=433/443
- NYSE_OPEN RR1.0 long F6_INSIDE_PDR: role=AVOID delta=-0.1760 q=0.0353 N_on/N_off=505/352
- COMEX_SETTLE RR1.0 long F1_NEAR_PDH_15: role=AVOID delta=-0.1723 q=0.0761 N_on/N_off=176/700
- COMEX_SETTLE RR1.5 long F6_INSIDE_PDR: role=AVOID delta=-0.1714 q=0.0786 N_on/N_off=429/440

## Caveats

- This is a context-overlay family only. It does not validate a standalone trade class.
- 2026 OOS remains descriptive unless both on/off groups reach the pre-registered 30-trade floor.
- TAKE vs AVOID is derived from one signed delta per cell, not from a doubled post-hoc hypothesis set.

## Artefacts

- CSV: `research/output/prior_day_direction_split_orb_overlays_v1.csv`
- Script: `research/prior_day_direction_split_orb_overlays_v1.py`
