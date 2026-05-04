# Dalton 80% Rule Test Notes

- Session anchor: 0900
- Symbols: MGC, MES, MNQ
- Profile bin size: 0.5

## Variants
- touch_A_B: both first two 30m brackets overlap prior VA
- close_A_B: both first two 30m bracket closes are inside prior VA
- first_touch_1h: first touch/re-entry within first hour

## Rule outcome
Success = opposite VA boundary hit before same-side boundary reclaim.
Ambiguous same-bar hit = failure (conservative).

## Summary
- MES close_A_B above: setups=29, resolved=29, wins=9, hit_rate=31.0%
- MES close_A_B below: setups=21, resolved=21, wins=7, hit_rate=33.3%
- MES first_touch_1h above: setups=139, resolved=139, wins=7, hit_rate=5.0%
- MES first_touch_1h below: setups=93, resolved=93, wins=6, hit_rate=6.5%
- MES touch_A_B above: setups=81, resolved=81, wins=19, hit_rate=23.5%
- MES touch_A_B below: setups=57, resolved=57, wins=15, hit_rate=26.3%
- MGC close_A_B above: setups=40, resolved=40, wins=11, hit_rate=27.5%
- MGC close_A_B below: setups=27, resolved=27, wins=8, hit_rate=29.6%
- MGC first_touch_1h above: setups=144, resolved=144, wins=5, hit_rate=3.5%
- MGC first_touch_1h below: setups=116, resolved=116, wins=4, hit_rate=3.4%
- MGC touch_A_B above: setups=88, resolved=88, wins=18, hit_rate=20.5%
- MGC touch_A_B below: setups=73, resolved=73, wins=17, hit_rate=23.3%
- MNQ close_A_B above: setups=6, resolved=6, wins=3, hit_rate=50.0%
- MNQ close_A_B below: setups=4, resolved=4, wins=2, hit_rate=50.0%
- MNQ first_touch_1h above: setups=46, resolved=46, wins=6, hit_rate=13.0%
- MNQ first_touch_1h below: setups=43, resolved=43, wins=8, hit_rate=18.6%
- MNQ touch_A_B above: setups=27, resolved=27, wins=11, hit_rate=40.7%
- MNQ touch_A_B below: setups=26, resolved=26, wins=13, hit_rate=50.0%