# Fast Lead-Lag Scan

- Slice: E1/CB2/RR2.5
- min_cond: 80
- Condition: leader break direction == follower break direction

## Top summary
- MES_0900 -> MNQ_1000: N_on=252/512, avgR on/off +0.1623/-0.1562, Δ(on-off)=+0.3185, WR on/off 35.7%/26.2%
- MNQ_2300 -> MES_0030: N_on=211/1585, avgR on/off +0.1022/-0.1741, Δ(on-off)=+0.2764, WR on/off 34.6%/26.2%
- MNQ_0900 -> MES_1000: N_on=232/1796, avgR on/off +0.0521/-0.2234, Δ(on-off)=+0.2754, WR on/off 35.3%/27.2%
- MNQ_0900 -> MNQ_1000: N_on=251/512, avgR on/off +0.1284/-0.1223, Δ(on-off)=+0.2508, WR on/off 34.7%/27.2%
- MNQ_1000 -> MES_0030: N_on=231/1585, avgR on/off +0.0431/-0.1681, Δ(on-off)=+0.2113, WR on/off 32.5%/26.4%
- MNQ_1800 -> MES_0030: N_on=208/1585, avgR on/off +0.0347/-0.1633, Δ(on-off)=+0.1981, WR on/off 32.2%/26.6%
- MGC_0900 -> MNQ_1000: N_on=217/512, avgR on/off +0.1099/-0.0798, Δ(on-off)=+0.1897, WR on/off 34.1%/28.5%
- MGC_1000 -> MNQ_1100: N_on=250/512, avgR on/off +0.0761/-0.1026, Δ(on-off)=+0.1787, WR on/off 33.6%/28.2%
- MNQ_0900 -> MGC_1000: N_on=247/2547, avgR on/off -0.2107/-0.3841, Δ(on-off)=+0.1734, WR on/off 30.4%/28.7%
- MNQ_1000 -> MGC_1100: N_on=260/2525, avgR on/off -0.1522/-0.3043, Δ(on-off)=+0.1521, WR on/off 29.2%/28.8%
- MNQ_1100 -> MGC_1800: N_on=247/2516, avgR on/off -0.1954/-0.3446, Δ(on-off)=+0.1492, WR on/off 28.7%/27.9%
- MES_0900 -> MES_1100: N_on=889/1787, avgR on/off -0.2038/-0.3461, Δ(on-off)=+0.1422, WR on/off 29.0%/23.7%

## Quick OOS
- MNQ_0900 -> MES_1000: train Δ=+0.0023, test Δ=+0.7372, n_test_on=117
- MES_0900 -> MES_1000: train Δ=+0.0172, test Δ=+0.5781, n_test_on=118
- MES_0900 -> MNQ_1000: train Δ=+0.2148, test Δ=+0.4733, n_test_on=129
- MES_0900 -> MES_1100: train Δ=+0.0923, test Δ=+0.4672, n_test_on=124
- MNQ_0900 -> MNQ_1000: train Δ=+0.1020, test Δ=+0.4626, n_test_on=130
- MNQ_0900 -> MES_1100: train Δ=-0.2625, test Δ=+0.4074, n_test_on=123
- MGC_1000 -> MNQ_1100: train Δ=-0.0984, test Δ=+0.3374, n_test_on=135
- MGC_0900 -> MNQ_1000: train Δ=+0.0243, test Δ=+0.3196, n_test_on=115
- MES_0900 -> MNQ_1100: train Δ=-0.1844, test Δ=+0.3025, n_test_on=119
- MNQ_0900 -> MNQ_1100: train Δ=-0.0196, test Δ=+0.2145, n_test_on=128
- MES_2300 -> MES_0030: train Δ=+0.1102, test Δ=+0.2138, n_test_on=111
- MES_1000 -> MES_0030: train Δ=+0.1084, test Δ=+0.2017, n_test_on=119