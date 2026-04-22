# 2026-04-22 MNQ Layered Candidate Board v1

Canonical read-only discovery board on `orb_outcomes x daily_features`.

- Parent lanes: 6
- Signal specs per lane: 40
- Actual tested rows: 104
- Holdout split: IS < 2026-01-01, OOS >= 2026-01-01

## Top Board

| Lane | Signal | Role | N_on_IS | ExpR_on_IS | ExpR_off_IS | Delta_IS | t | p_bh | N_on_OOS | Delta_OOS | OOS sign |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NYSE_OPEN RR1.5 short | F2_NEAR_PDL_15__AND__F5_BELOW_PDL | TAKE | 40 | +0.5000 | +0.0850 | +0.4150 | +2.1705 | 0.3077 | 2 | +0.2605 | True |
| US_DATA_1000 RR1.0 long | F5_BELOW_PDL | TAKE | 136 | +0.3258 | -0.0112 | +0.3370 | +4.0176 | 0.0087 | 8 | +0.0375 | True |
| NYSE_OPEN RR1.5 long | F3_NEAR_PIVOT_15 | AVOID | 216 | -0.1092 | +0.1516 | -0.2608 | -2.8062 | 0.0913 | 4 | -0.7158 | True |
| US_DATA_1000 RR1.0 long | F3_NEAR_PIVOT_50 | AVOID | 618 | -0.0344 | +0.2177 | -0.2521 | -3.6937 | 0.0116 | 19 | -0.2711 | True |
| US_DATA_1000 RR1.0 long | F2_NEAR_PDL_15__AND__F5_BELOW_PDL | TAKE | 41 | +0.3037 | +0.0280 | +0.2757 | +1.9069 | 0.4223 | 2 | +0.0299 | True |
| US_DATA_1000 RR1.0 long | F3_NEAR_PIVOT_50__AND__F6_INSIDE_PDR | AVOID | 482 | -0.0612 | +0.1642 | -0.2254 | -3.5564 | 0.0116 | 17 | -0.0528 | True |
| NYSE_OPEN RR1.5 short | F2_NEAR_PDL_15__AND__F6_INSIDE_PDR | AVOID | 77 | -0.1304 | +0.1297 | -0.2601 | -1.8674 | 0.4223 | 1 | -1.0074 | True |
| US_DATA_1000 RR1.5 short | F2_NEAR_PDL_15__AND__F5_BELOW_PDL | AVOID | 46 | -0.1279 | +0.1406 | -0.2686 | -1.5280 | 0.6001 | 3 | -1.0538 | True |
| US_DATA_1000 RR1.5 short | F3_NEAR_PIVOT_50__AND__F5_BELOW_PDL | AVOID | 51 | -0.1234 | +0.1421 | -0.2655 | -1.5888 | 0.5824 | 1 | -0.9880 | True |
| NYSE_OPEN RR1.5 long | F3_NEAR_PIVOT_15__AND__F6_INSIDE_PDR | AVOID | 200 | -0.0855 | +0.1375 | -0.2231 | -2.3314 | 0.2141 | 4 | -0.7158 | True |
| NYSE_OPEN RR1.5 long | F4_ABOVE_PDH | TAKE | 216 | +0.2463 | +0.0266 | +0.2198 | +2.3253 | 0.2141 | 5 | +0.8972 | True |
| US_DATA_1000 RR1.0 long | F6_INSIDE_PDR | AVOID | 513 | -0.0434 | +0.1583 | -0.2017 | -3.1530 | 0.0349 | 20 | -0.1573 | True |
| NYSE_OPEN RR1.5 long | F6_INSIDE_PDR | AVOID | 492 | -0.0004 | +0.2063 | -0.2066 | -2.4433 | 0.2141 | 21 | -0.7023 | True |
| CME_PRECLOSE RR1.0 long | F2_NEAR_PDL_15__AND__F5_BELOW_PDL | AVOID | 42 | -0.0870 | +0.1481 | -0.2351 | -1.6000 | 0.5824 | 2 | -0.0767 | True |
| NYSE_OPEN RR1.5 short | F3_NEAR_PIVOT_50 | AVOID | 622 | +0.0589 | +0.2516 | -0.1927 | -1.9562 | 0.3810 | 23 | -0.0570 | True |

## CME_PRECLOSE RR1.0 long

### Best Take States

- `F1_NEAR_PDH_15__AND__F6_INSIDE_PDR`: `N_on_IS=74`, `ExpR_on_IS=+0.1458`, `Delta_IS=+0.0121`, `t=+0.1125`, `BH=0.9764`, `N_on_OOS=5`, `Delta_OOS=+0.0985`
- `F3_NEAR_PIVOT_15`: `N_on_IS=122`, `ExpR_on_IS=+0.1351`, `Delta_IS=+0.0002`, `t=+0.0027`, `BH=0.9978`, `N_on_OOS=7`, `Delta_OOS=+0.4029`
- `F1_NEAR_PDH_15__AND__F4_ABOVE_PDH`: `N_on_IS=66`, `ExpR_on_IS=+0.2852`, `Delta_IS=+0.1650`, `t=+1.5356`, `BH=0.6001`, `N_on_OOS=3`, `Delta_OOS=-1.1448`
- `F3_NEAR_PIVOT_50__AND__F4_ABOVE_PDH`: `N_on_IS=63`, `ExpR_on_IS=+0.2267`, `Delta_IS=+0.1003`, `t=+0.8944`, `BH=0.7338`, `N_on_OOS=3`, `Delta_OOS=-1.1448`
- `F1_NEAR_PDH_15`: `N_on_IS=140`, `ExpR_on_IS=+0.2115`, `Delta_IS=+0.0945`, `t=+1.1667`, `BH=0.7066`, `N_on_OOS=8`, `Delta_OOS=-0.4530`
- `F4_ABOVE_PDH`: `N_on_IS=252`, `ExpR_on_IS=+0.1913`, `Delta_IS=+0.0855`, `t=+1.2560`, `BH=0.6883`, `N_on_OOS=10`, `Delta_OOS=-0.9712`
- `F1_NEAR_PDH_15__AND__F3_NEAR_PIVOT_50`: `N_on_IS=106`, `ExpR_on_IS=+0.2067`, `Delta_IS=+0.0838`, `t=+0.9282`, `BH=0.7338`, `N_on_OOS=6`, `Delta_OOS=-0.9029`

### Worst On-States (Avoid Candidates)

- `F2_NEAR_PDL_15__AND__F5_BELOW_PDL`: `N_on_IS=42`, `ExpR_on_IS=-0.0870`, `Delta_IS=-0.2351`, `t=-1.6000`, `BH=0.5824`, `N_on_OOS=2`, `Delta_OOS=-0.0767`
- `F2_NEAR_PDL_15__AND__F3_NEAR_PIVOT_50`: `N_on_IS=70`, `ExpR_on_IS=-0.0174`, `Delta_IS=-0.1681`, `t=-1.4718`, `BH=0.6045`, `N_on_OOS=3`, `Delta_OOS=-0.4346`
- `F2_NEAR_PDL_15`: `N_on_IS=99`, `ExpR_on_IS=-0.0020`, `Delta_IS=-0.1579`, `t=-1.5996`, `BH=0.5824`, `N_on_OOS=3`, `Delta_OOS=-0.4346`
- `F3_NEAR_PIVOT_50__AND__F5_BELOW_PDL`: `N_on_IS=31`, `ExpR_on_IS=-0.0074`, `Delta_IS=-0.1485`, `t=-0.8812`, `BH=0.7409`, `N_on_OOS=2`, `Delta_OOS=-0.0767`
- `F2_NEAR_PDL_15__AND__F6_INSIDE_PDR`: `N_on_IS=57`, `ExpR_on_IS=+0.0607`, `Delta_IS=-0.0804`, `t=-0.6384`, `BH=0.8407`, `N_on_OOS=1`, `Delta_OOS=-1.0685`
- `F5_BELOW_PDL`: `N_on_IS=164`, `ExpR_on_IS=+0.0785`, `Delta_IS=-0.0723`, `t=-0.8936`, `BH=0.7338`, `N_on_OOS=5`, `Delta_OOS=+0.5709`
- `F3_NEAR_PIVOT_50__AND__F6_INSIDE_PDR`: `N_on_IS=279`, `ExpR_on_IS=+0.1110`, `Delta_IS=-0.0383`, `t=-0.5656`, `BH=0.8801`, `N_on_OOS=13`, `Delta_OOS=+0.2163`
- `F3_NEAR_PIVOT_50`: `N_on_IS=373`, `ExpR_on_IS=+0.1207`, `Delta_IS=-0.0285`, `t=-0.4353`, `BH=0.8903`, `N_on_OOS=18`, `Delta_OOS=-0.2137`

## CME_PRECLOSE RR1.0 short

### Best Take States

- `F5_BELOW_PDL`: `N_on_IS=140`, `ExpR_on_IS=+0.1477`, `Delta_IS=+0.1077`, `t=+1.2489`, `BH=0.6883`, `N_on_OOS=11`, `Delta_OOS=+0.0181`
- `F1_NEAR_PDH_15__AND__F4_ABOVE_PDH`: `N_on_IS=44`, `ExpR_on_IS=+0.1570`, `Delta_IS=+0.1021`, `t=+0.7377`, `BH=0.7788`, `N_on_OOS=3`, `Delta_OOS=+0.3945`
- `F3_NEAR_PIVOT_50__AND__F5_BELOW_PDL`: `N_on_IS=30`, `ExpR_on_IS=+0.2770`, `Delta_IS=+0.2254`, `t=+1.4087`, `BH=0.6045`, `N_on_OOS=0`, `Delta_OOS=nan`
- `F3_NEAR_PIVOT_15`: `N_on_IS=130`, `ExpR_on_IS=+0.1450`, `Delta_IS=+0.1026`, `t=+1.1734`, `BH=0.7066`, `N_on_OOS=3`, `Delta_OOS=-0.2461`
- `F3_NEAR_PIVOT_15__AND__F6_INSIDE_PDR`: `N_on_IS=124`, `ExpR_on_IS=+0.1408`, `Delta_IS=+0.0964`, `t=+1.0820`, `BH=0.7289`, `N_on_OOS=3`, `Delta_OOS=-0.2461`
- `F3_NEAR_PIVOT_50`: `N_on_IS=359`, `ExpR_on_IS=+0.0992`, `Delta_IS=+0.0769`, `t=+1.1218`, `BH=0.7236`, `N_on_OOS=13`, `Delta_OOS=-0.4432`
- `F3_NEAR_PIVOT_50__AND__F4_ABOVE_PDH`: `N_on_IS=44`, `ExpR_on_IS=+0.1232`, `Delta_IS=+0.0661`, `t=+0.4689`, `BH=0.8903`, `N_on_OOS=1`, `Delta_OOS=-0.8781`
- `F6_INSIDE_PDR`: `N_on_IS=325`, `ExpR_on_IS=+0.0885`, `Delta_IS=+0.0503`, `t=+0.7322`, `BH=0.7788`, `N_on_OOS=13`, `Delta_OOS=-0.4432`

### Worst On-States (Avoid Candidates)

- `F4_ABOVE_PDH`: `N_on_IS=244`, `ExpR_on_IS=-0.0247`, `Delta_IS=-0.1310`, `t=-1.8208`, `BH=0.4236`, `N_on_OOS=11`, `Delta_OOS=+0.4621`
- `F1_NEAR_PDH_15__AND__F6_INSIDE_PDR`: `N_on_IS=57`, `ExpR_on_IS=+0.0009`, `Delta_IS=-0.0655`, `t=-0.5124`, `BH=0.8903`, `N_on_OOS=4`, `Delta_OOS=-0.4439`
- `F2_NEAR_PDL_15__AND__F3_NEAR_PIVOT_50`: `N_on_IS=56`, `ExpR_on_IS=+0.0551`, `Delta_IS=-0.0066`, `t=-0.0515`, `BH=0.9974`, `N_on_OOS=2`, `Delta_OOS=-0.9048`

## NYSE_OPEN RR1.5 long

### Best Take States

- `F4_ABOVE_PDH`: `N_on_IS=216`, `ExpR_on_IS=+0.2463`, `Delta_IS=+0.2198`, `t=+2.3253`, `BH=0.2141`, `N_on_OOS=5`, `Delta_OOS=+0.8972`
- `F5_BELOW_PDL`: `N_on_IS=121`, `ExpR_on_IS=+0.1246`, `Delta_IS=+0.0478`, `t=+0.4002`, `BH=0.8903`, `N_on_OOS=2`, `Delta_OOS=+0.0012`
- `F3_NEAR_PIVOT_50__AND__F4_ABOVE_PDH`: `N_on_IS=95`, `ExpR_on_IS=+0.2477`, `Delta_IS=+0.1852`, `t=+1.4229`, `BH=0.6045`, `N_on_OOS=0`, `Delta_OOS=nan`
- `F3_NEAR_PIVOT_50__AND__F5_BELOW_PDL`: `N_on_IS=45`, `ExpR_on_IS=+0.1807`, `Delta_IS=+0.1025`, `t=+0.5482`, `BH=0.8801`, `N_on_OOS=0`, `Delta_OOS=nan`
- `F1_NEAR_PDH_15__AND__F4_ABOVE_PDH`: `N_on_IS=84`, `ExpR_on_IS=+0.1349`, `Delta_IS=+0.0569`, `t=+0.4125`, `BH=0.8903`, `N_on_OOS=0`, `Delta_OOS=nan`

### Worst On-States (Avoid Candidates)

- `F1_NEAR_PDH_15__AND__F3_NEAR_PIVOT_15`: `N_on_IS=30`, `ExpR_on_IS=-0.2066`, `Delta_IS=-0.3012`, `t=-1.4157`, `BH=0.6045`, `N_on_OOS=0`, `Delta_OOS=nan`
- `F3_NEAR_PIVOT_15`: `N_on_IS=216`, `ExpR_on_IS=-0.1092`, `Delta_IS=-0.2608`, `t=-2.8062`, `BH=0.0913`, `N_on_OOS=4`, `Delta_OOS=-0.7158`
- `F3_NEAR_PIVOT_15__AND__F6_INSIDE_PDR`: `N_on_IS=200`, `ExpR_on_IS=-0.0855`, `Delta_IS=-0.2231`, `t=-2.3314`, `BH=0.2141`, `N_on_OOS=4`, `Delta_OOS=-0.7158`
- `F6_INSIDE_PDR`: `N_on_IS=492`, `ExpR_on_IS=-0.0004`, `Delta_IS=-0.2066`, `t=-2.4433`, `BH=0.2141`, `N_on_OOS=21`, `Delta_OOS=-0.7023`
- `F3_NEAR_PIVOT_50__AND__F6_INSIDE_PDR`: `N_on_IS=468`, `ExpR_on_IS=+0.0044`, `Delta_IS=-0.1820`, `t=-2.1720`, `BH=0.2851`, `N_on_OOS=20`, `Delta_OOS=-0.4299`
- `F2_NEAR_PDL_15`: `N_on_IS=113`, `ExpR_on_IS=-0.0332`, `Delta_IS=-0.1354`, `t=-1.1203`, `BH=0.7236`, `N_on_OOS=4`, `Delta_OOS=+0.0068`
- `F2_NEAR_PDL_15__AND__F3_NEAR_PIVOT_50`: `N_on_IS=90`, `ExpR_on_IS=-0.0295`, `Delta_IS=-0.1270`, `t=-0.9515`, `BH=0.7338`, `N_on_OOS=3`, `Delta_OOS=+0.4682`
- `F2_NEAR_PDL_15__AND__F6_INSIDE_PDR`: `N_on_IS=60`, `ExpR_on_IS=-0.0293`, `Delta_IS=-0.1219`, `t=-0.7584`, `BH=0.7788`, `N_on_OOS=3`, `Delta_OOS=+0.4682`

## NYSE_OPEN RR1.5 short

### Best Take States

- `F2_NEAR_PDL_15__AND__F5_BELOW_PDL`: `N_on_IS=40`, `ExpR_on_IS=+0.5000`, `Delta_IS=+0.4150`, `t=+2.1705`, `BH=0.3077`, `N_on_OOS=2`, `Delta_OOS=+0.2605`
- `F5_BELOW_PDL`: `N_on_IS=117`, `ExpR_on_IS=+0.4688`, `Delta_IS=+0.4241`, `t=+3.5876`, `BH=0.0116`, `N_on_OOS=8`, `Delta_OOS=-0.0722`
- `F3_NEAR_PIVOT_50__AND__F5_BELOW_PDL`: `N_on_IS=45`, `ExpR_on_IS=+0.2256`, `Delta_IS=+0.1273`, `t=+0.6847`, `BH=0.8127`, `N_on_OOS=3`, `Delta_OOS=-0.1756`
- `F1_NEAR_PDH_15__AND__F4_ABOVE_PDH`: `N_on_IS=83`, `ExpR_on_IS=+0.2111`, `Delta_IS=+0.1178`, `t=+0.8449`, `BH=0.7480`, `N_on_OOS=3`, `Delta_OOS=-0.1799`
- `F1_NEAR_PDH_15`: `N_on_IS=189`, `ExpR_on_IS=+0.1129`, `Delta_IS=+0.0099`, `t=+0.0999`, `BH=0.9769`, `N_on_OOS=10`, `Delta_OOS=-0.3257`
- `F3_NEAR_PIVOT_50__AND__F4_ABOVE_PDH`: `N_on_IS=80`, `ExpR_on_IS=+0.1066`, `Delta_IS=+0.0015`, `t=+0.0107`, `BH=0.9978`, `N_on_OOS=1`, `Delta_OOS=-1.0074`

### Worst On-States (Avoid Candidates)

- `F2_NEAR_PDL_15__AND__F3_NEAR_PIVOT_15`: `N_on_IS=35`, `ExpR_on_IS=-0.2047`, `Delta_IS=-0.3238`, `t=-1.6679`, `BH=0.5824`, `N_on_OOS=0`, `Delta_OOS=nan`
- `F2_NEAR_PDL_15__AND__F6_INSIDE_PDR`: `N_on_IS=77`, `ExpR_on_IS=-0.1304`, `Delta_IS=-0.2601`, `t=-1.8674`, `BH=0.4223`, `N_on_OOS=1`, `Delta_OOS=-1.0074`
- `F6_INSIDE_PDR`: `N_on_IS=523`, `ExpR_on_IS=+0.0309`, `Delta_IS=-0.2057`, `t=-2.3562`, `BH=0.2141`, `N_on_OOS=24`, `Delta_OOS=+0.0986`
- `F3_NEAR_PIVOT_50`: `N_on_IS=622`, `ExpR_on_IS=+0.0589`, `Delta_IS=-0.1927`, `t=-1.9562`, `BH=0.3810`, `N_on_OOS=23`, `Delta_OOS=-0.0570`
- `F3_NEAR_PIVOT_50__AND__F6_INSIDE_PDR`: `N_on_IS=495`, `ExpR_on_IS=+0.0353`, `Delta_IS=-0.1768`, `t=-2.0607`, `BH=0.3177`, `N_on_OOS=19`, `Delta_OOS=+0.0915`
- `F1_NEAR_PDH_15__AND__F6_INSIDE_PDR`: `N_on_IS=103`, `ExpR_on_IS=+0.0196`, `Delta_IS=-0.0980`, `t=-0.7811`, `BH=0.7687`, `N_on_OOS=7`, `Delta_OOS=-0.3366`
- `F3_NEAR_PIVOT_15`: `N_on_IS=223`, `ExpR_on_IS=+0.0408`, `Delta_IS=-0.0885`, `t=-0.9445`, `BH=0.7338`, `N_on_OOS=9`, `Delta_OOS=-0.2206`
- `F3_NEAR_PIVOT_15__AND__F6_INSIDE_PDR`: `N_on_IS=212`, `ExpR_on_IS=+0.0416`, `Delta_IS=-0.0859`, `t=-0.9006`, `BH=0.7338`, `N_on_OOS=9`, `Delta_OOS=-0.2206`

## US_DATA_1000 RR1.0 long

### Best Take States

- `F5_BELOW_PDL`: `N_on_IS=136`, `ExpR_on_IS=+0.3258`, `Delta_IS=+0.3370`, `t=+4.0176`, `BH=0.0087`, `N_on_OOS=8`, `Delta_OOS=+0.0375`
- `F2_NEAR_PDL_15__AND__F5_BELOW_PDL`: `N_on_IS=41`, `ExpR_on_IS=+0.3037`, `Delta_IS=+0.2757`, `t=+1.9069`, `BH=0.4223`, `N_on_OOS=2`, `Delta_OOS=+0.0299`
- `F2_NEAR_PDL_15`: `N_on_IS=110`, `ExpR_on_IS=+0.1426`, `Delta_IS=+0.1163`, `t=+1.2142`, `BH=0.6934`, `N_on_OOS=4`, `Delta_OOS=+0.5934`
- `F1_NEAR_PDH_15__AND__F6_INSIDE_PDR`: `N_on_IS=98`, `ExpR_on_IS=+0.1335`, `Delta_IS=+0.1043`, `t=+1.0462`, `BH=0.7289`, `N_on_OOS=2`, `Delta_OOS=+1.0428`
- `F1_NEAR_PDH_15`: `N_on_IS=175`, `ExpR_on_IS=+0.0930`, `Delta_IS=+0.0650`, `t=+0.8252`, `BH=0.7480`, `N_on_OOS=3`, `Delta_OOS=+0.3717`
- `F1_NEAR_PDH_15__AND__F3_NEAR_PIVOT_50`: `N_on_IS=143`, `ExpR_on_IS=+0.0841`, `Delta_IS=+0.0517`, `t=+0.6074`, `BH=0.8577`, `N_on_OOS=3`, `Delta_OOS=+0.3717`
- `F2_NEAR_PDL_15__AND__F3_NEAR_PIVOT_50`: `N_on_IS=95`, `ExpR_on_IS=+0.0795`, `Delta_IS=+0.0433`, `t=+0.4213`, `BH=0.8903`, `N_on_OOS=2`, `Delta_OOS=+1.0849`
- `F4_ABOVE_PDH`: `N_on_IS=231`, `ExpR_on_IS=+0.0647`, `Delta_IS=+0.0323`, `t=+0.4490`, `BH=0.8903`, `N_on_OOS=7`, `Delta_OOS=+0.1994`

### Worst On-States (Avoid Candidates)

- `F3_NEAR_PIVOT_50`: `N_on_IS=618`, `ExpR_on_IS=-0.0344`, `Delta_IS=-0.2521`, `t=-3.6937`, `BH=0.0116`, `N_on_OOS=19`, `Delta_OOS=-0.2711`
- `F3_NEAR_PIVOT_50__AND__F6_INSIDE_PDR`: `N_on_IS=482`, `ExpR_on_IS=-0.0612`, `Delta_IS=-0.2254`, `t=-3.5564`, `BH=0.0116`, `N_on_OOS=17`, `Delta_OOS=-0.0528`
- `F6_INSIDE_PDR`: `N_on_IS=513`, `ExpR_on_IS=-0.0434`, `Delta_IS=-0.2017`, `t=-3.1530`, `BH=0.0349`, `N_on_OOS=20`, `Delta_OOS=-0.1573`
- `F3_NEAR_PIVOT_15`: `N_on_IS=222`, `ExpR_on_IS=-0.0377`, `Delta_IS=-0.1051`, `t=-1.4298`, `BH=0.6045`, `N_on_OOS=9`, `Delta_OOS=-0.3943`
- `F3_NEAR_PIVOT_15__AND__F6_INSIDE_PDR`: `N_on_IS=212`, `ExpR_on_IS=-0.0376`, `Delta_IS=-0.1033`, `t=-1.3845`, `BH=0.6045`, `N_on_OOS=9`, `Delta_OOS=-0.3943`
- `F3_NEAR_PIVOT_50__AND__F4_ABOVE_PDH`: `N_on_IS=86`, `ExpR_on_IS=-0.0105`, `Delta_IS=-0.0569`, `t=-0.5370`, `BH=0.8801`, `N_on_OOS=2`, `Delta_OOS=-1.0042`

## US_DATA_1000 RR1.5 short

### Best Take States

- `F1_NEAR_PDH_15__AND__F4_ABOVE_PDH`: `N_on_IS=59`, `ExpR_on_IS=+0.3082`, `Delta_IS=+0.1973`, `t=+1.2423`, `BH=0.6883`, `N_on_OOS=4`, `Delta_OOS=+0.2887`
- `F3_NEAR_PIVOT_15__AND__F6_INSIDE_PDR`: `N_on_IS=202`, `ExpR_on_IS=+0.1898`, `Delta_IS=+0.0860`, `t=+0.8933`, `BH=0.7338`, `N_on_OOS=6`, `Delta_OOS=+0.8287`
- `F3_NEAR_PIVOT_15`: `N_on_IS=213`, `ExpR_on_IS=+0.1836`, `Delta_IS=+0.0791`, `t=+0.8366`, `BH=0.7480`, `N_on_OOS=6`, `Delta_OOS=+0.8287`
- `F4_ABOVE_PDH`: `N_on_IS=200`, `ExpR_on_IS=+0.1596`, `Delta_IS=+0.0455`, `t=+0.4735`, `BH=0.8903`, `N_on_OOS=9`, `Delta_OOS=+0.5155`
- `F2_NEAR_PDL_15__AND__F6_INSIDE_PDR`: `N_on_IS=54`, `ExpR_on_IS=+0.1540`, `Delta_IS=+0.0307`, `t=+0.1807`, `BH=0.9586`, `N_on_OOS=1`, `Delta_OOS=-0.9880`
- `F6_INSIDE_PDR`: `N_on_IS=453`, `ExpR_on_IS=+0.1336`, `Delta_IS=+0.0188`, `t=+0.2236`, `BH=0.9586`, `N_on_OOS=19`, `Delta_OOS=-0.1306`
- `F1_NEAR_PDH_15`: `N_on_IS=151`, `ExpR_on_IS=+0.1262`, `Delta_IS=+0.0010`, `t=+0.0098`, `BH=0.9978`, `N_on_OOS=8`, `Delta_OOS=-0.0657`

### Worst On-States (Avoid Candidates)

- `F2_NEAR_PDL_15__AND__F5_BELOW_PDL`: `N_on_IS=46`, `ExpR_on_IS=-0.1279`, `Delta_IS=-0.2686`, `t=-1.5280`, `BH=0.6001`, `N_on_OOS=3`, `Delta_OOS=-1.0538`
- `F3_NEAR_PIVOT_50__AND__F5_BELOW_PDL`: `N_on_IS=51`, `ExpR_on_IS=-0.1234`, `Delta_IS=-0.2655`, `t=-1.5888`, `BH=0.5824`, `N_on_OOS=1`, `Delta_OOS=-0.9880`
- `F2_NEAR_PDL_15__AND__F3_NEAR_PIVOT_50`: `N_on_IS=81`, `ExpR_on_IS=-0.0050`, `Delta_IS=-0.1448`, `t=-1.0475`, `BH=0.7289`, `N_on_OOS=0`, `Delta_OOS=nan`
- `F1_NEAR_PDH_15__AND__F6_INSIDE_PDR`: `N_on_IS=92`, `ExpR_on_IS=+0.0094`, `Delta_IS=-0.1308`, `t=-1.0139`, `BH=0.7295`, `N_on_OOS=4`, `Delta_OOS=-0.4020`
- `F2_NEAR_PDL_15`: `N_on_IS=100`, `ExpR_on_IS=+0.0243`, `Delta_IS=-0.1153`, `t=-0.9091`, `BH=0.7338`, `N_on_OOS=4`, `Delta_OOS=-1.0902`
- `F1_NEAR_PDH_15__AND__F3_NEAR_PIVOT_50`: `N_on_IS=126`, `ExpR_on_IS=+0.0295`, `Delta_IS=-0.1136`, `t=-1.0063`, `BH=0.7295`, `N_on_OOS=5`, `Delta_OOS=+0.0113`
- `F3_NEAR_PIVOT_50`: `N_on_IS=543`, `ExpR_on_IS=+0.0944`, `Delta_IS=-0.0944`, `t=-1.0585`, `BH=0.7289`, `N_on_OOS=19`, `Delta_OOS=+0.1696`
- `F5_BELOW_PDL`: `N_on_IS=154`, `ExpR_on_IS=+0.0566`, `Delta_IS=-0.0849`, `t=-0.7954`, `BH=0.7660`, `N_on_OOS=5`, `Delta_OOS=-0.5472`

