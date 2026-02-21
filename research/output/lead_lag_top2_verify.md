# Lead-Lag Top2 Verification (No-Lookahead)

Pairs:
- MES_0900 -> MNQ_1000
- MNQ_0900 -> MES_1000

Guard:
- include row only if leader_break_ts <= follower entry_ts

## Aggregate
- MES_0900_to_MNQ_1000: N=482, ON=252, OFF=230, avgR on/off +0.1623/-0.1836, Δ=+0.3460, WR on/off 35.7%/25.2%
- MNQ_0900_to_MES_1000: N=483, ON=232, OFF=251, avgR on/off +0.0521/-0.3542, Δ=+0.4063, WR on/off 35.3%/22.3%

## Yearly uplift
- MES_0900_to_MNQ_1000: years uplift>0 = 3/3
- MNQ_0900_to_MES_1000: years uplift>0 = 3/3

## Decision rule
- KEEP if aggregate Δ>0, yearly positive majority, and OOS mostly positive.
- WATCH if mixed but still positive in latest OOS.
- KILL if OOS negative or unstable.