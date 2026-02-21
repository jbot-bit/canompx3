# Dalton filter ON/OFF uplift (anchor-level)

No-lookahead applied (entry_ts >= A/B gate_ts).

- MNQ 0900: N=1589, ON=90 (5.7%), avgR all +0.1488, ON +0.8429, OFF +0.1071, Δ(on-off)=+0.7357, WR on/off 62.2%/32.3%, DD all/on 131.67/15.46
- MES 0900: N=6274, ON=128 (2.0%), avgR all -0.1199, ON +0.1739, OFF -0.1260, Δ(on-off)=+0.2999, WR on/off 43.0%/28.7%, DD all/on 1026.50/36.00
- MES 1100: N=6427, ON=168 (2.6%), avgR all -0.2962, ON -0.1427, OFF -0.3003, Δ(on-off)=+0.1576, WR on/off 37.5%/29.5%, DD all/on 1902.73/63.52
- MGC 0900: N=5184, ON=126 (2.4%), avgR all -0.2846, ON -0.2707, OFF -0.2850, Δ(on-off)=+0.0143, WR on/off 23.0%/27.0%, DD all/on 1536.82/55.08
- MGC 1000: N=5220, ON=162 (3.1%), avgR all -0.1632, ON -0.2601, OFF -0.1601, Δ(on-off)=-0.1001, WR on/off 24.7%/32.7%, DD all/on 980.29/48.02
- MES 1000: N=7300, ON=246 (3.4%), avgR all -0.1810, ON -0.4184, OFF -0.1727, Δ(on-off)=-0.2457, WR on/off 17.9%/26.7%, DD all/on 1726.35/102.32
- MNQ 1100: N=1874, ON=36 (1.9%), avgR all +0.0527, ON -0.3610, OFF +0.0609, Δ(on-off)=-0.4218, WR on/off 27.8%/38.7%, DD all/on 203.94/20.00
- MGC 1100: N=5748, ON=126 (2.2%), avgR all -0.1925, ON -0.9866, OFF -0.1747, Δ(on-off)=-0.8120, WR on/off 0.8%/34.7%, DD all/on 1305.54/123.31

Candidate keepers (uplift>0 and DD not worse): 4
- MNQ 0900: Δ=+0.7357, ΔDD=-116.21
- MES 0900: Δ=+0.2999, ΔDD=-990.50
- MES 1100: Δ=+0.1576, ΔDD=-1839.21
- MGC 0900: Δ=+0.0143, ΔDD=-1481.74