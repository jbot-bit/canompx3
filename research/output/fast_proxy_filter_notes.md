# Fast Proxy Filter Scan

- Slice: E1 / CB2 / RR2.5
- min_on: 80

## Top summary rows
- one_way_proxy | MGC 0900: N=378/2438, avgR -0.4481->-0.0640 (Δ=+0.3841), WR 25.8%->33.1%
- one_way_proxy | MNQ 1100: N=104/512, avgR -0.0154->+0.2299 (Δ=+0.2453), WR 30.9%->36.5%
- one_way_proxy | MGC 1000: N=481/2547, avgR -0.3673->-0.1712 (Δ=+0.1961), WR 28.9%->29.3%
- one_way_proxy | MES 1000: N=353/1796, avgR -0.1878->-0.0091 (Δ=+0.1787), WR 28.2%->31.2%
- one_way_proxy | MGC 1100: N=451/2525, avgR -0.2887->-0.1113 (Δ=+0.1773), WR 28.8%->29.0%
- one_way_proxy | MES 0900: N=350/1701, avgR -0.2741->-0.1211 (Δ=+0.1530), WR 26.6%->27.7%
- one_way_proxy | MES 1100: N=363/1787, avgR -0.2753->-0.1346 (Δ=+0.1407), WR 26.4%->27.5%
- one_way_proxy | MNQ 1000: N=107/512, avgR +0.0006->+0.1365 (Δ=+0.1359), WR 30.9%->33.6%
- continue_only | MGC 0900: N=1375/2438, avgR -0.4481->-0.3434 (Δ=+0.1047), WR 25.8%->28.5%
- fast_break_only | MGC 0900: N=1199/2438, avgR -0.4481->-0.3776 (Δ=+0.0705), WR 25.8%->27.5%
- one_way_proxy | MNQ 0900: N=101/479, avgR -0.0638->+0.0048 (Δ=+0.0686), WR 29.2%->29.7%
- fast_break_only | MGC 1000: N=1347/2547, avgR -0.3673->-0.3056 (Δ=+0.0617), WR 28.9%->29.6%

## Quick OOS (last-year holdout)
- continue_only | MGC 0900: train Δ=+0.1559, test Δ=+0.9533, n_test_on=243
- continue_only | MNQ 0900: train Δ=+0.2114, test Δ=+0.9029, n_test_on=241
- continue_only | MES 1000: train Δ=+0.0567, test Δ=+0.8783, n_test_on=255
- continue_only | MES 1100: train Δ=-0.0063, test Δ=+0.8699, n_test_on=253
- continue_only | MGC 1100: train Δ=+0.0644, test Δ=+0.8530, n_test_on=254
- continue_only | MES 0900: train Δ=+0.1108, test Δ=+0.7556, n_test_on=239
- one_way_proxy | MES 1000: train Δ=+0.1408, test Δ=+0.4825, n_test_on=86
- fast_break_only | MES 1000: train Δ=+0.0205, test Δ=+0.1895, n_test_on=213
- one_way_proxy | MGC 0900: train Δ=+0.3804, test Δ=+0.1645, n_test_on=153
- continue_only | MGC 1000: train Δ=+0.0960, test Δ=+0.1550, n_test_on=249