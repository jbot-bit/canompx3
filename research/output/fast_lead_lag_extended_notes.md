# Fast Lead-Lag Extended (MES/MNQ/M2K/M6E)

- Slice: E1/CB2/RR2.5
- Min label N: 800
- Min ON/OFF: 80
- No-lookahead: leader_break_ts <= follower entry_ts

## Top rows
- MES_0900 -> MES_CME_OPEN: N_on=613/697, avgR on/off -0.2156/-0.6555, Δ=+0.4399, WR on/off 26.3%/11.9%
- MES_US_DATA_OPEN -> M2K_US_DATA_OPEN: N_on=794/1154, avgR on/off -0.1397/-0.4008, Δ=+0.2611, WR on/off 29.5%/20.8%
- M2K_1000 -> M2K_US_POST_EQUITY: N_on=498/990, avgR on/off -0.0392/-0.2757, Δ=+0.2365, WR on/off 30.3%/23.0%
- MES_CME_OPEN -> MES_0900: N_on=541/1701, avgR on/off -0.1314/-0.3406, Δ=+0.2092, WR on/off 29.0%/25.5%
- M6E_1800 -> M2K_US_EQUITY_OPEN: N_on=408/840, avgR on/off -0.1741/-0.3808, Δ=+0.2067, WR on/off 25.2%/19.0%
- MES_1000 -> M2K_US_POST_EQUITY: N_on=493/985, avgR on/off -0.0588/-0.2588, Δ=+0.1999, WR on/off 29.8%/23.4%
- M2K_1100 -> MES_1100: N_on=1006/1278, avgR on/off -0.2750/-0.4600, Δ=+0.1850, WR on/off 26.2%/20.2%
- MES_1000 -> M2K_US_EQUITY_OPEN: N_on=413/836, avgR on/off -0.1848/-0.3670, Δ=+0.1822, WR on/off 24.9%/19.4%
- MES_1000 -> M2K_US_DATA_OPEN: N_on=571/1154, avgR on/off -0.1316/-0.3088, Δ=+0.1772, WR on/off 29.8%/23.8%
- M6E_US_EQUITY_OPEN -> M2K_US_POST_EQUITY: N_on=505/990, avgR on/off -0.0797/-0.2369, Δ=+0.1573, WR on/off 29.1%/24.1%
- M2K_CME_OPEN -> M2K_LONDON_OPEN: N_on=183/1277, avgR on/off -0.0679/-0.2212, Δ=+0.1533, WR on/off 32.2%/27.4%
- M2K_1000 -> MES_1000: N_on=1007/1282, avgR on/off -0.1717/-0.3243, Δ=+0.1526, WR on/off 28.4%/24.4%
- M6E_US_POST_EQUITY -> M2K_0030: N_on=359/1031, avgR on/off -0.1519/-0.3017, Δ=+0.1497, WR on/off 27.3%/22.2%
- M2K_CME_OPEN -> M2K_0900: N_on=384/1223, avgR on/off -0.2758/-0.4252, Δ=+0.1493, WR on/off 25.0%/24.7%
- MES_LONDON_OPEN -> M6E_US_DATA_OPEN: N_on=495/1019, avgR on/off -0.3158/-0.4643, Δ=+0.1485, WR on/off 27.3%/21.2%
- MES_0900 -> MES_1100: N_on=889/1787, avgR on/off -0.2038/-0.3461, Δ=+0.1422, WR on/off 29.0%/23.7%
- MES_LONDON_OPEN -> M2K_LONDON_OPEN: N_on=906/1271, avgR on/off -0.1577/-0.2977, Δ=+0.1400, WR on/off 29.4%/25.2%
- M6E_LONDON_OPEN -> MES_LONDON_OPEN: N_on=567/1267, avgR on/off -0.0814/-0.2206, Δ=+0.1393, WR on/off 30.3%/26.0%
- MES_US_DATA_OPEN -> M6E_US_EQUITY_OPEN: N_on=506/1121, avgR on/off -0.2547/-0.3896, Δ=+0.1349, WR on/off 29.4%/24.1%
- MES_US_POST_EQUITY -> M2K_CME_CLOSE: N_on=207/432, avgR on/off -0.6921/-0.8265, Δ=+0.1344, WR on/off 11.1%/6.2%