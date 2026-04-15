# Comprehensive Deployed-Lane Scan — Institutional Grade

**Date:** 2026-04-15
**Total cells scanned:** 1925
**Trustworthy cells** (not extreme-fire, not tautology, not arithmetic-only): 1855
**Strict survivors** (|t|>=3 + dir_match + N>=50 + trustworthy): 19
**BH-FDR global survivors** (q=0.05): 3
**BH-FDR per-family survivors** (q=0.05 within family): 35
**Promising** (|t|>=2.5 + dir_match + N>=50 + trustworthy): 40

## Scope definitions
- **deployed**: Alpha — 6 live MNQ lanes, test overlays within their filter subset
- **twin**: Beta — MES/MGC version of deployed (session,apt,RR) with same filter
- **non_deployed**: Delta — top non-deployed sessions × 3 instruments (no filter)

## Methodological controls
- TWO-PASS: every deployed lane tested both `unfiltered` (full lane universe) and `filtered` (within deployed-filter subset). Overlay valid only if signal holds in `filtered` pass.
- T0 tautology: |corr| > 0.7 with deployed filter → flagged tautology, excluded from survivors.
- Extreme fire rate: <5% or >95% → flagged untrustworthy.
- ARITHMETIC_ONLY: WR_spread < 3% AND |delta_is| > 0.10 → flagged as cost-screen not signal.
- Per-family BH-FDR alongside global BH-FDR.

## Strict Survivors (deploy candidates)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Family | Pass | N_on | Fire% | ExpR_on | WR_Δ | Δ_IS | Δ_OOS | t | p | BH_g | BH_f |
|-------|-------|---------|-----|----|----|---------|--------|------|------|-------|---------|------|------|-------|---|---|------|------|
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 276 | 32.5% | +0.092 | +0.140 | +0.328 | +0.116 | +4.46 | 0.0000 | Y | Y |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 284 | 37.7% | +0.057 | +0.138 | +0.325 | +0.451 | +4.15 | 0.0000 | Y | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 202 | 34.0% | +0.102 | +0.140 | +0.330 | +0.364 | +3.56 | 0.0004 | . | Y |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 232 | 30.3% | -0.156 | -0.121 | -0.305 | -0.366 | -3.50 | 0.0005 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_LOW_Q1 | volume | unfiltered | 289 | 33.1% | -0.105 | -0.103 | -0.267 | -0.697 | -3.45 | 0.0006 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 280 | 32.6% | +0.259 | +0.109 | +0.277 | +0.083 | +3.44 | 0.0006 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | atr_vel_LOW | volatility | unfiltered | 291 | 33.4% | -0.103 | -0.113 | -0.265 | -0.450 | -3.40 | 0.0007 | . | . |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 278 | 32.4% | +0.252 | +0.131 | +0.265 | +0.760 | +3.35 | 0.0009 | . | Y |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | short | ovn_range_pct_GT80 | overnight | unfiltered | 165 | 20.5% | +0.137 | +0.120 | +0.304 | +0.091 | +3.28 | 0.0012 | . | . |
| twin | MES | NYSE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 312 | 37.2% | +0.143 | +0.105 | +0.209 | +0.156 | +3.28 | 0.0011 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_LOW | volume | unfiltered | 314 | 36.1% | -0.086 | -0.121 | -0.250 | -0.243 | -3.23 | 0.0013 | . | Y |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | filtered | 74 | 37.4% | +0.203 | +0.221 | +0.493 | +0.831 | +3.19 | 0.0018 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | long | break_delay_LT2 | timing | unfiltered | 370 | 59.6% | +0.116 | +0.121 | +0.272 | +0.439 | +3.17 | 0.0016 | . | . |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 188 | 31.3% | +0.084 | +0.159 | +0.292 | +0.333 | +3.17 | 0.0017 | . | Y |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 208 | 33.0% | -0.169 | -0.131 | -0.294 | -0.645 | -3.10 | 0.0020 | . | Y |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | long | near_session_london_high | level | unfiltered | 391 | 47.2% | -0.262 | -0.072 | -0.221 | -0.260 | -3.10 | 0.0020 | . | . |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 127 | 34.7% | +0.194 | +0.161 | +0.359 | +1.003 | +3.08 | 0.0023 | . | Y |
| twin | MGC | NYSE_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 173 | 36.7% | +0.133 | +0.157 | +0.245 | +0.620 | +3.08 | 0.0022 | . | Y |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 241 | 29.8% | +0.133 | +0.132 | +0.258 | +0.157 | +3.03 | 0.0026 | . | Y |

## BH-FDR Survivors — Global (q=0.05)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Pass | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p | BH_crit |
|-------|-------|---------|-----|----|----|---------|------|------|-------|---------|------|-------|---|---|---------|
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 295 | 38.6% | +0.299 | +0.395 | -0.150 | +4.68 | 0.00000 | 0.00003 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 276 | 32.5% | +0.092 | +0.328 | +0.116 | +4.46 | 0.00001 | 0.00005 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 284 | 37.7% | +0.057 | +0.325 | +0.451 | +4.15 | 0.00004 | 0.00008 |

## BH-FDR Survivors — Per-Family (q=0.05 within family)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Family | Pass | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p |
|-------|-------|---------|-----|----|----|---------|--------|------|------|-------|---------|------|-------|---|---|
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 295 | 38.6% | +0.299 | +0.395 | -0.150 | +4.68 | 0.00000 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 276 | 32.5% | +0.092 | +0.328 | +0.116 | +4.46 | 0.00001 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 284 | 37.7% | +0.057 | +0.325 | +0.451 | +4.15 | 0.00004 |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 111 | 33.7% | -0.351 | -0.479 | +nan | -3.82 | 0.00017 |
| non_deployed | MES | NYSE_CLOSE | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 96 | 32.9% | +0.193 | +0.492 | +nan | +3.80 | 0.00020 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 202 | 34.0% | +0.102 | +0.330 | +0.364 | +3.56 | 0.00042 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 284 | 33.3% | +0.260 | +0.284 | -0.246 | +3.53 | 0.00044 |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 232 | 30.3% | -0.156 | -0.305 | -0.366 | -3.50 | 0.00051 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 210 | 28.2% | -0.345 | -0.276 | +0.072 | -3.48 | 0.00056 |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 91 | 31.6% | +0.353 | +0.492 | +nan | +3.47 | 0.00066 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_LOW_Q1 | volume | unfiltered | 289 | 33.1% | -0.105 | -0.267 | -0.697 | -3.45 | 0.00061 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 280 | 32.6% | +0.259 | +0.277 | +0.083 | +3.44 | 0.00062 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 278 | 32.4% | +0.252 | +0.265 | +0.760 | +3.35 | 0.00086 |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 126 | 31.3% | -0.294 | -0.368 | +nan | -3.30 | 0.00108 |
| twin | MES | NYSE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 312 | 37.2% | +0.143 | +0.209 | +0.156 | +3.28 | 0.00111 |
| twin | MES | EUROPE_FLOW | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 278 | 32.1% | -0.373 | -0.220 | +0.184 | -3.26 | 0.00119 |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.5 | long | bb_volume_ratio_LOW | volume | unfiltered | 94 | 32.2% | -0.282 | -0.439 | +nan | -3.23 | 0.00146 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_LOW | volume | unfiltered | 314 | 36.1% | -0.086 | -0.250 | -0.243 | -3.23 | 0.00131 |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | filtered | 74 | 37.4% | +0.203 | +0.493 | +0.831 | +3.19 | 0.00178 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 188 | 31.3% | +0.084 | +0.292 | +0.333 | +3.17 | 0.00167 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 208 | 33.0% | -0.169 | -0.294 | -0.645 | -3.10 | 0.00204 |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 127 | 34.7% | +0.194 | +0.359 | +1.003 | +3.08 | 0.00231 |
| twin | MGC | NYSE_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 173 | 36.7% | +0.133 | +0.245 | +0.620 | +3.08 | 0.00222 |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 241 | 29.8% | +0.133 | +0.258 | +0.157 | +3.03 | 0.00260 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 272 | 31.6% | +0.020 | +0.220 | -0.417 | +3.02 | 0.00266 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 293 | 33.6% | +0.225 | +0.236 | +0.027 | +2.99 | 0.00287 |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 239 | 29.2% | -0.268 | -0.228 | -0.289 | -2.95 | 0.00339 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 291 | 33.3% | +0.006 | +0.213 | +0.055 | +2.94 | 0.00343 |
| non_deployed | MES | US_DATA_830 | O30 | 1.0 | long | bb_volume_ratio_LOW | volume | unfiltered | 297 | 35.6% | -0.185 | -0.191 | -0.611 | -2.92 | 0.00365 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 221 | 35.0% | +0.212 | +0.279 | +0.525 | +2.91 | 0.00376 |
| deployed | MNQ | EUROPE_FLOW | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 292 | 33.2% | -0.126 | -0.226 | +1.041 | -2.90 | 0.00383 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_LOW | volume | unfiltered | 297 | 35.0% | -0.260 | -0.202 | +0.114 | -2.86 | 0.00437 |
| twin | MGC | COMEX_SETTLE | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 154 | 34.1% | -0.024 | +0.276 | +nan | +2.84 | 0.00477 |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.5 | long | bb_volume_ratio_LOW | volume | unfiltered | 297 | 36.2% | -0.193 | -0.227 | +0.586 | -2.81 | 0.00514 |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 253 | 30.0% | -0.125 | -0.233 | -0.262 | -2.77 | 0.00574 |

## Promising cells (candidates for next-round T0-T8)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Pass | N_on | ExpR_on | WR_Δ | Δ_IS | Δ_OOS | t | p |
|-------|-------|---------|-----|----|----|---------|------|------|---------|------|------|-------|---|---|
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 276 | +0.092 | +0.140 | +0.328 | +0.116 | +4.46 | 0.0000 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 284 | +0.057 | +0.138 | +0.325 | +0.451 | +4.15 | 0.0000 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 202 | +0.102 | +0.140 | +0.330 | +0.364 | +3.56 | 0.0004 |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_LOW_Q1 | unfiltered | 232 | -0.156 | -0.121 | -0.305 | -0.366 | -3.50 | 0.0005 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_LOW_Q1 | unfiltered | 289 | -0.105 | -0.103 | -0.267 | -0.697 | -3.45 | 0.0006 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 280 | +0.259 | +0.109 | +0.277 | +0.083 | +3.44 | 0.0006 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | atr_vel_LOW | unfiltered | 291 | -0.103 | -0.113 | -0.265 | -0.450 | -3.40 | 0.0007 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_HIGH | unfiltered | 278 | +0.252 | +0.131 | +0.265 | +0.760 | +3.35 | 0.0009 |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | short | ovn_range_pct_GT80 | unfiltered | 165 | +0.137 | +0.120 | +0.304 | +0.091 | +3.28 | 0.0012 |
| twin | MES | NYSE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 312 | +0.143 | +0.105 | +0.209 | +0.156 | +3.28 | 0.0011 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_LOW | unfiltered | 314 | -0.086 | -0.121 | -0.250 | -0.243 | -3.23 | 0.0013 |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | short | bb_volume_ratio_HIGH | filtered | 74 | +0.203 | +0.221 | +0.493 | +0.831 | +3.19 | 0.0018 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | long | break_delay_LT2 | unfiltered | 370 | +0.116 | +0.121 | +0.272 | +0.439 | +3.17 | 0.0016 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_HIGH | unfiltered | 188 | +0.084 | +0.159 | +0.292 | +0.333 | +3.17 | 0.0017 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_LOW | unfiltered | 208 | -0.169 | -0.131 | -0.294 | -0.645 | -3.10 | 0.0020 |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | long | near_session_london_high | unfiltered | 391 | -0.262 | -0.072 | -0.221 | -0.260 | -3.10 | 0.0020 |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | short | bb_volume_ratio_HIGH | unfiltered | 127 | +0.194 | +0.161 | +0.359 | +1.003 | +3.08 | 0.0023 |
| twin | MGC | NYSE_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | unfiltered | 173 | +0.133 | +0.157 | +0.245 | +0.620 | +3.08 | 0.0022 |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.5 | long | bb_volume_ratio_HIGH | unfiltered | 241 | +0.133 | +0.132 | +0.258 | +0.157 | +3.03 | 0.0026 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | short | bb_volume_ratio_HIGH | unfiltered | 293 | +0.225 | +0.112 | +0.236 | +0.027 | +2.99 | 0.0029 |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | short | bb_volume_ratio_LOW | unfiltered | 239 | -0.268 | -0.116 | -0.228 | -0.289 | -2.95 | 0.0034 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 291 | +0.006 | +0.090 | +0.213 | +0.055 | +2.94 | 0.0034 |
| non_deployed | MES | US_DATA_830 | O30 | 1.0 | long | bb_volume_ratio_LOW | unfiltered | 297 | -0.185 | -0.114 | -0.191 | -0.611 | -2.92 | 0.0037 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 221 | +0.212 | +0.109 | +0.279 | +0.525 | +2.91 | 0.0038 |
| twin | MES | NYSE_OPEN | O5 | 1.0 | long | atr_20_pct_GT80 | filtered | 65 | -0.320 | -0.225 | -0.425 | -0.265 | -2.83 | 0.0054 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | short | break_delay_LT2 | unfiltered | 427 | +0.117 | +0.114 | +0.265 | +0.671 | +2.80 | 0.0053 |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.5 | short | bb_volume_ratio_LOW | unfiltered | 253 | -0.125 | -0.111 | -0.233 | -0.262 | -2.77 | 0.0057 |
| twin | MES | NYSE_OPEN | O5 | 1.0 | long | is_friday_TRUE | unfiltered | 171 | +0.163 | +0.108 | +0.210 | +0.670 | +2.77 | 0.0060 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | long | bb_volume_ratio_HIGH | unfiltered | 211 | +0.168 | +0.129 | +0.245 | +0.564 | +2.76 | 0.0061 |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | long | near_session_london_low | unfiltered | 190 | -0.118 | -0.110 | -0.250 | -0.283 | -2.72 | 0.0068 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_LOW_Q1 | unfiltered | 287 | -0.253 | -0.062 | -0.188 | -0.166 | -2.70 | 0.0071 |
| non_deployed | MNQ | BRISBANE_1025 | O15 | 2.0 | long | break_delay_GT10 | unfiltered | 376 | -0.076 | -0.094 | -0.236 | -0.334 | -2.69 | 0.0072 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | atr_vel_HIGH | unfiltered | 279 | +0.003 | +0.084 | +0.197 | +0.445 | +2.68 | 0.0075 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_LOW | unfiltered | 213 | -0.272 | -0.121 | -0.238 | -0.454 | -2.67 | 0.0078 |
| deployed | MNQ | US_DATA_1000 | O5 | 1.5 | short | pre_velocity_LOW | unfiltered | 255 | +0.288 | +0.090 | +0.238 | +0.774 | +2.64 | 0.0086 |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | atr_20_pct_GT80 | unfiltered | 189 | +0.249 | +0.094 | +0.255 | +0.625 | +2.62 | 0.0091 |
| twin | MES | EUROPE_FLOW | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 259 | -0.020 | +0.070 | +0.196 | +0.436 | +2.58 | 0.0101 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 254 | +0.045 | +0.074 | +0.203 | +0.672 | +2.58 | 0.0103 |
| deployed | MNQ | US_DATA_1000 | O5 | 1.5 | long | bb_volume_ratio_HIGH | unfiltered | 257 | +0.219 | +0.104 | +0.224 | +1.046 | +2.57 | 0.0104 |
| non_deployed | MES | US_DATA_830 | O30 | 1.0 | short | near_session_asia_high | unfiltered | 324 | +0.001 | +0.101 | +0.162 | +0.258 | +2.52 | 0.0118 |

## Flagged cells (excluded despite |t|>=3) — for transparency

Excluded because: tautology |corr|>0.7 (0), extreme fire <5% or >95% (49), arithmetic-only WR flat (21)

| Scope | Instr | Session | Feature | Pass | t | Fire% | T0 corr | Reason |
|-------|-------|---------|---------|------|---|-------|---------|--------|
| deployed | MNQ | EUROPE_FLOW | is_nfp_TRUE | unfiltered | -3.49 | 4.2% | 0.00 | FIRE(4.2%) |

## Baseline Per-Lane (no feature overlay)

| Scope | Instr | Session | Apt | RR | Filter | N_is | N_oos | ExpR_is | ExpR_oos | Filter_fire% |
|-------|-------|---------|-----|----|--------|------|-------|---------|----------|--------------|
| deployed | MNQ | EUROPE_FLOW | O5 | 1.5 | ORB_G5 | 327 | 32 | +0.093 | +0.490 | 20.2% |
| deployed | MNQ | SINGAPORE_OPEN | O30 | 1.5 | ATR_P50 | 820 | 67 | +0.140 | +0.115 | 50.0% |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | OVNRNG_100 | 24 | 1 | -0.029 | -1.000 | 1.5% |
| deployed | MNQ | NYSE_OPEN | O5 | 1.0 | ORB_G5 | 310 | 44 | +0.053 | +0.300 | 20.2% |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | ORB_G5 | 310 | 48 | +0.179 | +0.248 | 20.1% |
| deployed | MNQ | US_DATA_1000 | O5 | 1.5 | VWAP_MID_ALIGNED | 846 | 28 | +0.113 | +0.220 | 50.3% |
| twin | MES | EUROPE_FLOW | O5 | 1.5 | ORB_G5 | 362 | 27 | -0.050 | +0.166 | 21.8% |
| twin | MGC | EUROPE_FLOW | O5 | 1.5 | ORB_G5 | 131 | 65 | +0.146 | -0.068 | 20.1% |
| twin | MES | SINGAPORE_OPEN | O30 | 1.5 | ATR_P50 | 820 | 66 | -0.034 | -0.060 | 50.1% |
| twin | MGC | SINGAPORE_OPEN | O30 | 1.5 | ATR_P50 | 417 | 60 | -0.010 | +0.177 | 50.0% |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | OVNRNG_100 | 35 | 1 | -0.196 | +0.854 | 2.1% |
| twin | MGC | COMEX_SETTLE | O5 | 1.5 | OVNRNG_100 | 29 | 12 | +0.100 | -0.800 | 4.5% |
| twin | MES | NYSE_OPEN | O5 | 1.0 | ORB_G5 | 331 | 42 | -0.012 | +0.049 | 21.1% |
| twin | MGC | NYSE_OPEN | O5 | 1.0 | ORB_G5 | 132 | 64 | -0.056 | +0.357 | 20.2% |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | ORB_G5 | 316 | 41 | +0.027 | +0.111 | 20.0% |
| twin | MGC | TOKYO_OPEN | O5 | 1.5 | ORB_G5 | 135 | 63 | +0.124 | +0.130 | 20.3% |
| twin | MES | US_DATA_1000 | O5 | 1.5 | VWAP_MID_ALIGNED | 877 | 27 | -0.030 | +0.122 | 52.1% |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | VWAP_MID_ALIGNED | 439 | 33 | -0.081 | +0.167 | 53.1% |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | NONE | 1296 | 57 | +0.089 | -0.184 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | NONE | 1223 | 54 | -0.055 | -0.247 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.5 | NONE | 611 | 26 | -0.012 | +0.183 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O5 | 1.5 | NONE | 558 | 26 | -0.213 | -0.062 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.5 | NONE | 1660 | 65 | -0.005 | -0.158 | 100.0% |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | NONE | 1647 | 64 | -0.126 | -0.192 | 100.0% |
| non_deployed | MES | US_DATA_830 | O30 | 1.0 | NONE | 1634 | 64 | -0.079 | -0.067 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O15 | 2.0 | NONE | 1720 | 67 | +0.015 | +0.187 | 100.0% |