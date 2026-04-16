# Comprehensive Scan — ALL Sessions × ALL Instruments × ALL Apertures × ALL RRs

**Date:** 2026-04-15
**Total cells scanned:** 14261
**Trustworthy cells** (not extreme-fire, not tautology, not arithmetic-only): 13635
**Strict survivors** (|t|>=3 + dir_match + N>=50 + trustworthy): 102

## BH-FDR pass counts at each K framing

- **K_global** (K=14261) strictest: 13 pass
- **K_family** (within feature-family, avg K~2550): 127 pass
- **K_lane** (within session+apt+rr+instr, avg K~56): 163 pass
- **K_session** (within session across instruments, avg K~1387): 57 pass
- **K_instrument** (within instrument, avg K~4815): 32 pass
- **K_feature** (within feature across lanes, avg K~491): 121 pass

**Promising** (|t|>=2.5 + dir_match + N>=50 + trustworthy): 265

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
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 288 | 37.9% | +0.077 | +0.155 | +0.303 | +0.366 | +4.89 | 0.0000 | Y | Y |
| non_deployed | MGC | LONDON_METALS | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 148 | 34.2% | +0.121 | +0.199 | +0.357 | +0.074 | +4.78 | 0.0000 | Y | Y |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 276 | 32.5% | +0.092 | +0.140 | +0.328 | +0.116 | +4.46 | 0.0000 | Y | Y |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 286 | 34.5% | +0.135 | +0.129 | +0.262 | +0.403 | +4.27 | 0.0000 | Y | Y |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 284 | 37.7% | +0.057 | +0.138 | +0.325 | +0.451 | +4.15 | 0.0000 | Y | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 284 | 34.5% | -0.051 | +0.113 | +0.229 | +0.272 | +4.04 | 0.0001 | . | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 285 | 31.5% | -0.058 | +0.111 | +0.223 | +0.591 | +4.01 | 0.0001 | . | Y |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | bb_volume_ratio_LOW | volume | unfiltered | 232 | 33.1% | -0.131 | -0.159 | -0.285 | -0.591 | -3.89 | 0.0001 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 2.0 | long | break_delay_LT2 | timing | unfiltered | 328 | 60.4% | +0.106 | +0.153 | +0.404 | +0.633 | +3.89 | 0.0001 | . | . |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 275 | 32.6% | -0.219 | -0.117 | -0.304 | -0.003 | -3.85 | 0.0001 | . | Y |
| non_deployed | MES | US_DATA_830 | O5 | 1.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 292 | 34.1% | -0.282 | -0.089 | -0.227 | -0.682 | -3.85 | 0.0001 | . | Y |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_LONG_INTERACT | overnight | unfiltered | 208 | 28.7% | -0.197 | -0.138 | -0.279 | -0.045 | -3.82 | 0.0002 | . | . |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_TRUE | overnight | unfiltered | 208 | 28.7% | -0.197 | -0.138 | -0.279 | -0.045 | -3.82 | 0.0002 | . | . |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 251 | 35.7% | +0.234 | +0.131 | +0.267 | +0.793 | +3.79 | 0.0002 | . | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 302 | 33.7% | -0.341 | -0.101 | -0.198 | -0.191 | -3.76 | 0.0002 | . | Y |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 302 | 35.8% | +0.105 | +0.141 | +0.224 | +0.446 | +3.74 | 0.0002 | . | Y |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 287 | 30.9% | +0.089 | +0.147 | +0.212 | +0.379 | +3.73 | 0.0002 | . | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 285 | 31.5% | -0.016 | +0.104 | +0.261 | +0.673 | +3.73 | 0.0002 | . | Y |
| non_deployed | MES | NYSE_CLOSE | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 126 | 32.0% | +0.216 | +0.181 | +0.314 | +0.065 | +3.68 | 0.0003 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 2.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 192 | 36.1% | +0.208 | +0.172 | +0.404 | +0.763 | +3.67 | 0.0003 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 234 | 34.0% | +0.131 | +0.123 | +0.245 | +0.488 | +3.61 | 0.0003 | . | Y |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 281 | 31.2% | +0.090 | +0.100 | +0.213 | +0.120 | +3.58 | 0.0004 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 202 | 34.0% | +0.102 | +0.140 | +0.330 | +0.364 | +3.56 | 0.0004 | . | Y |
| non_deployed | MES | LONDON_METALS | O30 | 1.5 | long | ovn_range_pct_GT80 | overnight | unfiltered | 179 | 20.9% | +0.231 | +0.132 | +0.341 | +0.690 | +3.54 | 0.0005 | . | . |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 232 | 30.3% | -0.156 | -0.121 | -0.305 | -0.366 | -3.50 | 0.0005 | . | Y |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 286 | 34.5% | +0.135 | +0.110 | +0.273 | +0.798 | +3.48 | 0.0006 | . | Y |
| non_deployed | MGC | NYSE_OPEN | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 171 | 36.6% | +0.152 | +0.175 | +0.355 | +0.705 | +3.47 | 0.0006 | . | Y |
| non_deployed | MES | EUROPE_FLOW | O15 | 1.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 186 | 21.2% | +0.143 | +0.101 | +0.246 | +0.334 | +3.47 | 0.0006 | . | . |
| non_deployed | MGC | LONDON_METALS | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 148 | 34.2% | +0.074 | +0.147 | +0.341 | +0.110 | +3.47 | 0.0006 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_LOW_Q1 | volume | unfiltered | 289 | 33.1% | -0.105 | -0.103 | -0.267 | -0.697 | -3.45 | 0.0006 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 280 | 32.6% | +0.259 | +0.109 | +0.277 | +0.083 | +3.44 | 0.0006 | . | Y |
| non_deployed | MES | TOKYO_OPEN | O5 | 1.0 | long | atr_vel_HIGH | volatility | unfiltered | 279 | 33.6% | +0.018 | +0.105 | +0.199 | +0.374 | +3.44 | 0.0006 | . | . |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 293 | 33.7% | +0.165 | +0.128 | +0.216 | +0.562 | +3.43 | 0.0006 | . | Y |
| non_deployed | MES | EUROPE_FLOW | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 260 | 31.0% | -0.022 | +0.099 | +0.207 | +0.024 | +3.42 | 0.0007 | . | Y |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 246 | 30.0% | +0.161 | +0.148 | +0.225 | +0.206 | +3.41 | 0.0007 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | atr_vel_LOW | volatility | unfiltered | 291 | 33.4% | -0.103 | -0.113 | -0.265 | -0.450 | -3.40 | 0.0007 | . | . |
| non_deployed | MES | US_DATA_1000 | O15 | 2.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 177 | 26.5% | -0.224 | -0.126 | -0.377 | -0.887 | -3.39 | 0.0008 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | long | break_delay_LT2 | timing | unfiltered | 426 | 58.7% | +0.115 | +0.117 | +0.214 | +0.450 | +3.38 | 0.0008 | . | . |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 267 | 32.1% | -0.327 | -0.089 | -0.185 | -0.517 | -3.36 | 0.0008 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 278 | 32.4% | +0.252 | +0.131 | +0.265 | +0.760 | +3.35 | 0.0009 | . | Y |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | short | bb_volume_ratio_LOW | volume | unfiltered | 256 | 29.9% | -0.090 | -0.127 | -0.218 | -0.364 | -3.34 | 0.0009 | . | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.5 | long | rel_vol_LOW_Q1 | volume | unfiltered | 302 | 33.7% | -0.336 | -0.086 | -0.215 | -0.138 | -3.34 | 0.0009 | . | Y |
| non_deployed | MES | US_DATA_1000 | O15 | 1.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 188 | 25.4% | -0.130 | -0.120 | -0.252 | -0.451 | -3.30 | 0.0011 | . | Y |
| non_deployed | MNQ | COMEX_SETTLE | O30 | 2.0 | long | ovn_took_pdh_TRUE | overnight | unfiltered | 152 | 28.7% | -0.404 | -0.135 | -0.385 | -0.488 | -3.29 | 0.0011 | . | . |
| non_deployed | MNQ | COMEX_SETTLE | O30 | 2.0 | long | ovn_took_pdh_LONG_INTERACT | overnight | unfiltered | 152 | 28.7% | -0.404 | -0.135 | -0.385 | -0.488 | -3.29 | 0.0011 | . | . |
| non_deployed | MNQ | LONDON_METALS | O15 | 2.0 | short | atr_vel_LOW | volatility | unfiltered | 265 | 31.8% | -0.219 | -0.111 | -0.318 | -0.492 | -3.28 | 0.0011 | . | . |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | short | ovn_range_pct_GT80 | overnight | unfiltered | 165 | 20.5% | +0.137 | +0.120 | +0.304 | +0.091 | +3.28 | 0.0012 | . | . |
| twin | MES | NYSE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 312 | 37.2% | +0.143 | +0.105 | +0.209 | +0.156 | +3.28 | 0.0011 | . | Y |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 1.0 | short | dow_thu | calendar | unfiltered | 158 | 20.8% | +0.248 | +0.136 | +0.267 | +0.552 | +3.27 | 0.0012 | . | . |
| non_deployed | MNQ | BRISBANE_1025 | O30 | 2.0 | long | is_monday_TRUE | calendar | unfiltered | 192 | 20.7% | +0.338 | +0.132 | +0.364 | +0.854 | +3.27 | 0.0012 | . | . |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | long | atr_vel_LOW | volatility | unfiltered | 291 | 33.4% | -0.104 | -0.105 | -0.203 | -0.432 | -3.25 | 0.0012 | . | . |
| non_deployed | MGC | COMEX_SETTLE | O30 | 1.0 | long | near_session_london_low | level | unfiltered | 79 | 26.8% | -0.318 | -0.181 | -0.354 | -0.146 | -3.25 | 0.0015 | . | . |
| non_deployed | MES | TOKYO_OPEN | O15 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 260 | 30.2% | -0.311 | -0.112 | -0.244 | -0.236 | -3.24 | 0.0013 | . | Y |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 270 | 33.0% | -0.174 | -0.092 | -0.243 | -0.983 | -3.24 | 0.0013 | . | Y |
| non_deployed | MES | LONDON_METALS | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 275 | 33.0% | +0.021 | +0.110 | +0.247 | +0.334 | +3.23 | 0.0013 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_LOW | volume | unfiltered | 314 | 36.1% | -0.086 | -0.121 | -0.250 | -0.243 | -3.23 | 0.0013 | . | Y |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 2.0 | long | near_session_london_low | level | unfiltered | 188 | 22.2% | -0.175 | -0.125 | -0.340 | -0.389 | -3.22 | 0.0014 | . | . |
| non_deployed | MGC | NYSE_OPEN | O5 | 2.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 171 | 37.0% | +0.173 | +0.159 | +0.388 | +1.398 | +3.21 | 0.0014 | . | Y |
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 254 | 28.7% | +0.060 | +0.093 | +0.199 | +0.505 | +3.21 | 0.0014 | . | Y |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 2.0 | long | bb_volume_ratio_LOW | volume | unfiltered | 314 | 36.2% | -0.090 | -0.116 | -0.291 | -0.176 | -3.21 | 0.0014 | . | Y |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 2.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 278 | 32.5% | +0.300 | +0.122 | +0.304 | +0.672 | +3.20 | 0.0015 | . | Y |
| non_deployed | MNQ | TOKYO_OPEN | O15 | 2.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 300 | 35.4% | +0.165 | +0.105 | +0.307 | +0.260 | +3.19 | 0.0015 | . | Y |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | filtered | 74 | 37.4% | +0.203 | +0.221 | +0.493 | +0.831 | +3.19 | 0.0018 | . | Y |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 1.0 | long | garch_vol_pct_GT70 | volatility | unfiltered | 198 | 23.4% | +0.247 | +0.091 | +0.230 | +0.236 | +3.18 | 0.0016 | . | . |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 2.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 289 | 33.1% | -0.097 | -0.093 | -0.289 | -0.730 | -3.18 | 0.0015 | . | Y |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 2.0 | short | dow_thu | calendar | unfiltered | 144 | 20.8% | +0.281 | +0.143 | +0.415 | +0.522 | +3.17 | 0.0017 | . | . |
| non_deployed | MGC | COMEX_SETTLE | O30 | 2.0 | short | near_session_london_high | level | unfiltered | 55 | 26.2% | -0.730 | -0.174 | -0.447 | -0.450 | -3.17 | 0.0018 | . | . |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | long | break_delay_LT2 | timing | unfiltered | 370 | 59.6% | +0.116 | +0.121 | +0.272 | +0.439 | +3.17 | 0.0016 | . | . |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 188 | 31.3% | +0.084 | +0.159 | +0.292 | +0.333 | +3.17 | 0.0017 | . | Y |
| non_deployed | MGC | EUROPE_FLOW | O30 | 2.0 | short | near_session_asia_high | level | unfiltered | 163 | 39.2% | -0.319 | -0.142 | -0.381 | -0.236 | -3.17 | 0.0017 | . | . |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 278 | 32.6% | +0.164 | +0.101 | +0.262 | +0.317 | +3.16 | 0.0017 | . | Y |
| non_deployed | MES | EUROPE_FLOW | O5 | 2.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 180 | 21.6% | +0.079 | +0.103 | +0.326 | +0.134 | +3.16 | 0.0018 | . | . |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 223 | 32.1% | +0.112 | +0.147 | +0.211 | +0.260 | +3.16 | 0.0017 | . | Y |
| non_deployed | MGC | US_DATA_1000 | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 137 | 35.2% | +0.129 | +0.155 | +0.280 | +0.627 | +3.15 | 0.0018 | . | Y |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 274 | 30.4% | +0.140 | +0.130 | +0.185 | +0.251 | +3.15 | 0.0017 | . | Y |
| non_deployed | MES | EUROPE_FLOW | O5 | 1.0 | short | garch_vol_pct_GT70 | volatility | unfiltered | 184 | 22.2% | -0.003 | +0.090 | +0.217 | +0.158 | +3.14 | 0.0019 | . | . |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 278 | 32.4% | +0.161 | +0.119 | +0.193 | +0.524 | +3.14 | 0.0018 | . | Y |
| non_deployed | MGC | US_DATA_830 | O5 | 2.0 | long | bb_volume_ratio_LOW | volume | unfiltered | 143 | 35.3% | -0.302 | -0.155 | -0.392 | -0.051 | -3.14 | 0.0018 | . | Y |
| non_deployed | MES | EUROPE_FLOW | O5 | 2.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 259 | 31.0% | +0.016 | +0.090 | +0.278 | +0.324 | +3.13 | 0.0019 | . | Y |
| non_deployed | MES | LONDON_METALS | O30 | 1.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 182 | 21.1% | +0.121 | +0.107 | +0.235 | +0.330 | +3.12 | 0.0020 | . | . |
| non_deployed | MES | TOKYO_OPEN | O15 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 298 | 34.9% | +0.014 | +0.099 | +0.237 | +0.261 | +3.12 | 0.0019 | . | Y |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 293 | 33.7% | +0.152 | +0.116 | +0.251 | +1.091 | +3.11 | 0.0020 | . | Y |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 2.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 270 | 33.0% | -0.180 | -0.087 | -0.272 | -1.354 | -3.11 | 0.0020 | . | Y |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 208 | 33.0% | -0.169 | -0.131 | -0.294 | -0.645 | -3.10 | 0.0020 | . | Y |
| non_deployed | MES | LONDON_METALS | O5 | 2.0 | short | atr_vel_HIGH | volatility | unfiltered | 277 | 32.6% | +0.028 | +0.100 | +0.276 | +0.641 | +3.10 | 0.0020 | . | . |
| non_deployed | MES | TOKYO_OPEN | O15 | 2.0 | long | garch_vol_pct_LT30 | volatility | unfiltered | 311 | 35.5% | -0.200 | -0.095 | -0.266 | -0.847 | -3.10 | 0.0020 | . | . |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | long | near_session_london_high | level | unfiltered | 391 | 47.2% | -0.262 | -0.072 | -0.221 | -0.260 | -3.10 | 0.0020 | . | . |
| non_deployed | MES | CME_PRECLOSE | O5 | 2.0 | short | bb_volume_ratio_LOW | volume | unfiltered | 196 | 36.8% | -0.401 | -0.134 | -0.320 | -0.706 | -3.10 | 0.0021 | . | Y |
| non_deployed | MGC | LONDON_METALS | O5 | 2.0 | short | ovn_range_pct_GT80 | overnight | unfiltered | 101 | 23.8% | +0.185 | +0.141 | +0.421 | +0.362 | +3.09 | 0.0024 | . | . |
| non_deployed | MNQ | NYSE_OPEN | O5 | 2.0 | long | atr_20_pct_GT80 | volatility | unfiltered | 192 | 25.4% | -0.178 | -0.124 | -0.343 | -0.274 | -3.09 | 0.0021 | . | . |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 127 | 34.7% | +0.194 | +0.161 | +0.359 | +1.003 | +3.08 | 0.0023 | . | Y |
| twin | MGC | NYSE_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 173 | 36.7% | +0.133 | +0.157 | +0.245 | +0.620 | +3.08 | 0.0022 | . | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 2.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 285 | 31.5% | -0.007 | +0.079 | +0.253 | +0.884 | +3.07 | 0.0023 | . | Y |
| non_deployed | MES | COMEX_SETTLE | O5 | 2.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 284 | 37.7% | +0.054 | +0.097 | +0.282 | +0.541 | +3.07 | 0.0023 | . | Y |
| non_deployed | MES | CME_REOPEN | O15 | 1.5 | long | garch_vol_pct_GT70 | volatility | unfiltered | 66 | 26.6% | -0.522 | -0.216 | -0.421 | -0.519 | -3.06 | 0.0027 | . | . |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | atr_vel_HIGH | volatility | unfiltered | 287 | 32.7% | -0.095 | +0.090 | +0.169 | +0.580 | +3.05 | 0.0024 | . | . |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 2.0 | long | atr_vel_LOW | volatility | unfiltered | 290 | 33.4% | -0.090 | -0.100 | -0.279 | -0.429 | -3.05 | 0.0024 | . | . |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 241 | 29.8% | +0.133 | +0.132 | +0.258 | +0.157 | +3.03 | 0.0026 | . | Y |
| non_deployed | MGC | SINGAPORE_OPEN | O5 | 1.0 | long | bb_volume_ratio_LOW | volume | unfiltered | 157 | 34.9% | -0.256 | -0.140 | -0.232 | -0.244 | -3.02 | 0.0027 | . | Y |
| non_deployed | MGC | US_DATA_1000 | O15 | 2.0 | short | pre_velocity_HIGH | timing | unfiltered | 95 | 34.2% | -0.352 | -0.175 | -0.459 | -0.292 | -3.02 | 0.0028 | . | . |
| non_deployed | MGC | EUROPE_FLOW | O30 | 1.5 | short | near_session_asia_high | level | unfiltered | 164 | 38.6% | -0.276 | -0.139 | -0.315 | -0.320 | -3.01 | 0.0028 | . | . |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 2.0 | long | pre_velocity_HIGH | timing | unfiltered | 277 | 30.4% | +0.170 | +0.089 | +0.272 | +0.304 | +3.01 | 0.0028 | . | . |

## BH-FDR Survivors — Global (q=0.05)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Pass | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p | BH_crit |
|-------|-------|---------|-----|----|----|---------|------|------|-------|---------|------|-------|---|---|---------|
| non_deployed | MES | TOKYO_OPEN | O5 | 1.0 | long | rel_vol_HIGH_Q3 | unfiltered | 276 | 32.5% | +0.094 | +0.311 | -0.046 | +5.43 | 0.00000 | 0.00000 |
| non_deployed | MNQ | CME_REOPEN | O5 | 1.5 | long | bb_volume_ratio_LOW | unfiltered | 132 | 34.3% | -0.455 | -0.568 | +0.877 | -5.10 | 0.00000 | 0.00001 |
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 288 | 37.9% | +0.077 | +0.303 | +0.366 | +4.89 | 0.00000 | 0.00001 |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 298 | 38.6% | +0.254 | +0.318 | -0.522 | +4.88 | 0.00000 | 0.00001 |
| non_deployed | MGC | LONDON_METALS | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 148 | 34.2% | +0.121 | +0.357 | +0.074 | +4.78 | 0.00000 | 0.00002 |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 295 | 38.6% | +0.299 | +0.395 | -0.150 | +4.68 | 0.00000 | 0.00002 |
| non_deployed | MES | NYSE_CLOSE | O5 | 2.0 | long | bb_volume_ratio_HIGH | unfiltered | 77 | 32.3% | +0.218 | +0.759 | +nan | +4.61 | 0.00001 | 0.00003 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 276 | 32.5% | +0.092 | +0.328 | +0.116 | +4.46 | 0.00001 | 0.00002 |
| non_deployed | MES | CME_REOPEN | O5 | 2.0 | short | rel_vol_LOW_Q1 | unfiltered | 110 | 29.4% | -0.567 | -0.510 | +0.118 | -4.29 | 0.00002 | 0.00004 |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 286 | 34.5% | +0.135 | +0.262 | +0.403 | +4.27 | 0.00002 | 0.00004 |
| non_deployed | MES | US_DATA_1000 | O5 | 2.0 | long | break_delay_GT10 | unfiltered | 151 | 17.7% | -0.428 | -0.431 | +nan | -4.23 | 0.00003 | 0.00004 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | short | break_delay_LT2 | unfiltered | 438 | 62.9% | +0.072 | +0.276 | -0.376 | +4.16 | 0.00004 | 0.00005 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 284 | 37.7% | +0.057 | +0.325 | +0.451 | +4.15 | 0.00004 | 0.00005 |

## BH-FDR Survivors — Per-Family (q=0.05 within family)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Family | Pass | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p |
|-------|-------|---------|-----|----|----|---------|--------|------|------|-------|---------|------|-------|---|---|
| non_deployed | MES | TOKYO_OPEN | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 276 | 32.5% | +0.094 | +0.311 | -0.046 | +5.43 | 0.00000 |
| non_deployed | MNQ | CME_REOPEN | O5 | 1.5 | long | bb_volume_ratio_LOW | volume | unfiltered | 132 | 34.3% | -0.455 | -0.568 | +0.877 | -5.10 | 0.00000 |
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 288 | 37.9% | +0.077 | +0.303 | +0.366 | +4.89 | 0.00000 |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 298 | 38.6% | +0.254 | +0.318 | -0.522 | +4.88 | 0.00000 |
| non_deployed | MGC | LONDON_METALS | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 148 | 34.2% | +0.121 | +0.357 | +0.074 | +4.78 | 0.00000 |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 295 | 38.6% | +0.299 | +0.395 | -0.150 | +4.68 | 0.00000 |
| non_deployed | MES | NYSE_CLOSE | O5 | 2.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 77 | 32.3% | +0.218 | +0.759 | +nan | +4.61 | 0.00001 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 276 | 32.5% | +0.092 | +0.328 | +0.116 | +4.46 | 0.00001 |
| non_deployed | MES | CME_REOPEN | O5 | 2.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 110 | 29.4% | -0.567 | -0.510 | +0.118 | -4.29 | 0.00002 |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 286 | 34.5% | +0.135 | +0.262 | +0.403 | +4.27 | 0.00002 |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 2.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 76 | 32.3% | +0.379 | +0.765 | +nan | +4.18 | 0.00005 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | short | break_delay_LT2 | timing | unfiltered | 438 | 62.9% | +0.072 | +0.276 | -0.376 | +4.16 | 0.00004 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 284 | 37.7% | +0.057 | +0.325 | +0.451 | +4.15 | 0.00004 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 284 | 34.5% | -0.051 | +0.229 | +0.272 | +4.04 | 0.00006 |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 284 | 33.3% | +0.229 | +0.248 | -0.158 | +4.03 | 0.00006 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 285 | 31.5% | -0.058 | +0.223 | +0.591 | +4.01 | 0.00007 |
| non_deployed | MES | TOKYO_OPEN | O5 | 2.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 276 | 32.6% | +0.154 | +0.351 | -0.113 | +4.00 | 0.00007 |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 293 | 33.6% | +0.221 | +0.239 | -0.129 | +3.98 | 0.00008 |
| non_deployed | MES | TOKYO_OPEN | O15 | 1.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 281 | 32.7% | -0.262 | -0.239 | +0.139 | -3.96 | 0.00009 |
| non_deployed | MES | NYSE_CLOSE | O15 | 2.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 41 | 39.4% | +0.309 | +0.956 | +nan | +3.94 | 0.00021 |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 2.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 294 | 38.8% | +0.234 | +0.391 | -0.039 | +3.94 | 0.00009 |
| non_deployed | MNQ | CME_REOPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 151 | 35.9% | +0.241 | +0.352 | -0.630 | +3.90 | 0.00012 |
| non_deployed | MES | NYSE_CLOSE | O15 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 46 | 38.3% | +0.264 | +0.779 | +nan | +3.90 | 0.00020 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | bb_volume_ratio_LOW | volume | unfiltered | 232 | 33.1% | -0.131 | -0.285 | -0.591 | -3.89 | 0.00012 |
| non_deployed | MES | CME_REOPEN | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 113 | 28.8% | -0.485 | -0.412 | +0.137 | -3.87 | 0.00014 |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 275 | 32.6% | -0.219 | -0.304 | -0.003 | -3.85 | 0.00013 |
| non_deployed | MES | US_DATA_830 | O5 | 1.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 292 | 34.1% | -0.282 | -0.227 | -0.682 | -3.85 | 0.00013 |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 111 | 33.7% | -0.351 | -0.479 | +nan | -3.82 | 0.00017 |
| non_deployed | MES | NYSE_CLOSE | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 96 | 32.9% | +0.193 | +0.492 | +nan | +3.80 | 0.00020 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 251 | 35.7% | +0.234 | +0.267 | +0.793 | +3.79 | 0.00017 |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | long | bb_volume_ratio_LOW | volume | unfiltered | 127 | 35.4% | -0.428 | -0.502 | +1.053 | -3.79 | 0.00018 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 302 | 33.7% | -0.341 | -0.198 | -0.191 | -3.76 | 0.00018 |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 302 | 35.8% | +0.105 | +0.224 | +0.446 | +3.74 | 0.00020 |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 287 | 30.9% | +0.089 | +0.212 | +0.379 | +3.73 | 0.00021 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 285 | 31.5% | -0.016 | +0.261 | +0.673 | +3.73 | 0.00021 |
| non_deployed | MES | NYSE_CLOSE | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 126 | 32.0% | +0.216 | +0.314 | +0.065 | +3.68 | 0.00028 |
| non_deployed | MES | NYSE_CLOSE | O15 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 33 | 31.7% | -0.677 | -0.677 | +nan | -3.68 | 0.00041 |
| non_deployed | MES | CME_PRECLOSE | O5 | 2.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 192 | 36.1% | +0.208 | +0.404 | +0.763 | +3.67 | 0.00028 |
| non_deployed | MNQ | CME_REOPEN | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 112 | 29.1% | +0.243 | +0.461 | -0.395 | +3.64 | 0.00034 |
| non_deployed | MES | NYSE_CLOSE | O15 | 2.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 35 | 33.7% | +0.353 | +0.936 | +nan | +3.64 | 0.00062 |

## Promising cells (candidates for next-round T0-T8)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Pass | N_on | ExpR_on | WR_Δ | Δ_IS | Δ_OOS | t | p |
|-------|-------|---------|-----|----|----|---------|------|------|---------|------|------|-------|---|---|
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 288 | +0.077 | +0.155 | +0.303 | +0.366 | +4.89 | 0.0000 |
| non_deployed | MGC | LONDON_METALS | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 148 | +0.121 | +0.199 | +0.357 | +0.074 | +4.78 | 0.0000 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 276 | +0.092 | +0.140 | +0.328 | +0.116 | +4.46 | 0.0000 |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 286 | +0.135 | +0.129 | +0.262 | +0.403 | +4.27 | 0.0000 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 284 | +0.057 | +0.138 | +0.325 | +0.451 | +4.15 | 0.0000 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 284 | -0.051 | +0.113 | +0.229 | +0.272 | +4.04 | 0.0001 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_HIGH_Q3 | unfiltered | 285 | -0.058 | +0.111 | +0.223 | +0.591 | +4.01 | 0.0001 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | bb_volume_ratio_LOW | unfiltered | 232 | -0.131 | -0.159 | -0.285 | -0.591 | -3.89 | 0.0001 |
| non_deployed | MES | CME_PRECLOSE | O5 | 2.0 | long | break_delay_LT2 | unfiltered | 328 | +0.106 | +0.153 | +0.404 | +0.633 | +3.89 | 0.0001 |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | short | rel_vol_LOW_Q1 | unfiltered | 275 | -0.219 | -0.117 | -0.304 | -0.003 | -3.85 | 0.0001 |
| non_deployed | MES | US_DATA_830 | O5 | 1.0 | long | rel_vol_LOW_Q1 | unfiltered | 292 | -0.282 | -0.089 | -0.227 | -0.682 | -3.85 | 0.0001 |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_TRUE | unfiltered | 208 | -0.197 | -0.138 | -0.279 | -0.045 | -3.82 | 0.0002 |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_LONG_INTERACT | unfiltered | 208 | -0.197 | -0.138 | -0.279 | -0.045 | -3.82 | 0.0002 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 251 | +0.234 | +0.131 | +0.267 | +0.793 | +3.79 | 0.0002 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_LOW_Q1 | unfiltered | 302 | -0.341 | -0.101 | -0.198 | -0.191 | -3.76 | 0.0002 |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | unfiltered | 302 | +0.105 | +0.141 | +0.224 | +0.446 | +3.74 | 0.0002 |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | long | bb_volume_ratio_HIGH | unfiltered | 287 | +0.089 | +0.147 | +0.212 | +0.379 | +3.73 | 0.0002 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 285 | -0.016 | +0.104 | +0.261 | +0.673 | +3.73 | 0.0002 |
| non_deployed | MES | NYSE_CLOSE | O5 | 1.0 | long | bb_volume_ratio_HIGH | unfiltered | 126 | +0.216 | +0.181 | +0.314 | +0.065 | +3.68 | 0.0003 |
| non_deployed | MES | CME_PRECLOSE | O5 | 2.0 | long | bb_volume_ratio_HIGH | unfiltered | 192 | +0.208 | +0.172 | +0.404 | +0.763 | +3.67 | 0.0003 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 234 | +0.131 | +0.123 | +0.245 | +0.488 | +3.61 | 0.0003 |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | long | rel_vol_HIGH_Q3 | unfiltered | 281 | +0.090 | +0.100 | +0.213 | +0.120 | +3.58 | 0.0004 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 202 | +0.102 | +0.140 | +0.330 | +0.364 | +3.56 | 0.0004 |
| non_deployed | MES | LONDON_METALS | O30 | 1.5 | long | ovn_range_pct_GT80 | unfiltered | 179 | +0.231 | +0.132 | +0.341 | +0.690 | +3.54 | 0.0005 |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_LOW_Q1 | unfiltered | 232 | -0.156 | -0.121 | -0.305 | -0.366 | -3.50 | 0.0005 |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 286 | +0.135 | +0.110 | +0.273 | +0.798 | +3.48 | 0.0006 |
| non_deployed | MGC | NYSE_OPEN | O5 | 1.5 | short | bb_volume_ratio_HIGH | unfiltered | 171 | +0.152 | +0.175 | +0.355 | +0.705 | +3.47 | 0.0006 |
| non_deployed | MES | EUROPE_FLOW | O15 | 1.0 | long | ovn_range_pct_GT80 | unfiltered | 186 | +0.143 | +0.101 | +0.246 | +0.334 | +3.47 | 0.0006 |
| non_deployed | MGC | LONDON_METALS | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 148 | +0.074 | +0.147 | +0.341 | +0.110 | +3.47 | 0.0006 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_LOW_Q1 | unfiltered | 289 | -0.105 | -0.103 | -0.267 | -0.697 | -3.45 | 0.0006 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 280 | +0.259 | +0.109 | +0.277 | +0.083 | +3.44 | 0.0006 |
| non_deployed | MES | TOKYO_OPEN | O5 | 1.0 | long | atr_vel_HIGH | unfiltered | 279 | +0.018 | +0.105 | +0.199 | +0.374 | +3.44 | 0.0006 |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.0 | short | bb_volume_ratio_HIGH | unfiltered | 293 | +0.165 | +0.128 | +0.216 | +0.562 | +3.43 | 0.0006 |
| non_deployed | MES | EUROPE_FLOW | O5 | 1.0 | long | rel_vol_HIGH_Q3 | unfiltered | 260 | -0.022 | +0.099 | +0.207 | +0.024 | +3.42 | 0.0007 |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.0 | long | bb_volume_ratio_HIGH | unfiltered | 246 | +0.161 | +0.148 | +0.225 | +0.206 | +3.41 | 0.0007 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | atr_vel_LOW | unfiltered | 291 | -0.103 | -0.113 | -0.265 | -0.450 | -3.40 | 0.0007 |
| non_deployed | MES | US_DATA_1000 | O15 | 2.0 | short | rel_vol_LOW_Q1 | unfiltered | 177 | -0.224 | -0.126 | -0.377 | -0.887 | -3.39 | 0.0008 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | long | break_delay_LT2 | unfiltered | 426 | +0.115 | +0.117 | +0.214 | +0.450 | +3.38 | 0.0008 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_LOW_Q1 | unfiltered | 267 | -0.327 | -0.089 | -0.185 | -0.517 | -3.36 | 0.0008 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_HIGH | unfiltered | 278 | +0.252 | +0.131 | +0.265 | +0.760 | +3.35 | 0.0009 |

## Flagged cells (excluded despite |t|>=3) — for transparency

Excluded because: tautology |corr|>0.7 (0), extreme fire <5% or >95% (455), arithmetic-only WR flat (176)

| Scope | Instr | Session | Feature | Pass | t | Fire% | T0 corr | Reason |
|-------|-------|---------|---------|------|---|-------|---------|--------|
| non_deployed | MNQ | EUROPE_FLOW | is_nfp_TRUE | unfiltered | -4.93 | 4.2% | 0.00 | FIRE(4.2%) |
| deployed | MNQ | EUROPE_FLOW | is_nfp_TRUE | unfiltered | -3.49 | 4.2% | 0.00 | FIRE(4.2%) |
| non_deployed | MES | SINGAPORE_OPEN | break_bar_continues_TRUE | unfiltered | +3.03 | 96.7% | 0.00 | FIRE(96.7%) |

## Baseline Per-Lane (no feature overlay)

| Scope | Instr | Session | Apt | RR | Filter | N_is | N_oos | ExpR_is | ExpR_oos | Filter_fire% |
|-------|-------|---------|-----|----|--------|------|-------|---------|----------|--------------|
| non_deployed | MNQ | CME_REOPEN | O5 | 1.0 | NONE | 882 | 49 | +0.017 | +0.091 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O5 | 1.5 | NONE | 754 | 46 | -0.068 | +0.136 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | NONE | 711 | 46 | -0.068 | +0.241 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O15 | 1.0 | NONE | 566 | 39 | +0.036 | +0.381 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O15 | 1.5 | NONE | 506 | 38 | -0.013 | +0.323 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O15 | 2.0 | NONE | 486 | 38 | -0.121 | +0.132 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O30 | 1.0 | NONE | 350 | 29 | +0.018 | +0.333 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O30 | 1.5 | NONE | 333 | 29 | -0.092 | +0.164 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O30 | 2.0 | NONE | 327 | 29 | -0.122 | -0.103 | 100.0% |
| non_deployed | MES | CME_REOPEN | O5 | 1.0 | NONE | 861 | 51 | -0.100 | -0.089 | 100.0% |
| non_deployed | MES | CME_REOPEN | O5 | 1.5 | NONE | 752 | 51 | -0.193 | -0.085 | 100.0% |
| non_deployed | MES | CME_REOPEN | O5 | 2.0 | NONE | 709 | 50 | -0.237 | -0.044 | 100.0% |
| non_deployed | MES | CME_REOPEN | O15 | 1.0 | NONE | 564 | 40 | -0.054 | +0.089 | 100.0% |
| non_deployed | MES | CME_REOPEN | O15 | 1.5 | NONE | 510 | 38 | -0.116 | +0.134 | 100.0% |
| non_deployed | MES | CME_REOPEN | O15 | 2.0 | NONE | 496 | 38 | -0.193 | -0.006 | 100.0% |
| non_deployed | MES | CME_REOPEN | O30 | 1.0 | NONE | 347 | 30 | -0.047 | +0.044 | 100.0% |
| non_deployed | MES | CME_REOPEN | O30 | 1.5 | NONE | 331 | 30 | -0.148 | -0.007 | 100.0% |
| non_deployed | MES | CME_REOPEN | O30 | 2.0 | NONE | 328 | 30 | -0.177 | -0.184 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | NONE | 464 | 47 | -0.140 | +0.112 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O5 | 1.5 | NONE | 409 | 44 | -0.252 | +0.262 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O5 | 2.0 | NONE | 384 | 44 | -0.295 | +0.450 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O15 | 1.0 | NONE | 304 | 34 | -0.103 | +0.311 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O15 | 1.5 | NONE | 284 | 31 | -0.227 | +0.561 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O15 | 2.0 | NONE | 273 | 30 | -0.250 | +0.742 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O30 | 1.0 | NONE | 195 | 25 | -0.155 | +0.633 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O30 | 1.5 | NONE | 188 | 23 | -0.190 | +0.897 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O30 | 2.0 | NONE | 186 | 20 | -0.198 | +0.884 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | NONE | 1721 | 67 | +0.047 | +0.044 | 100.0% |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | ORB_G5 | 310 | 48 | +0.179 | +0.248 | 20.1% |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 2.0 | NONE | 1720 | 67 | +0.066 | +0.229 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O15 | 1.0 | NONE | 1720 | 67 | +0.058 | -0.109 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O15 | 1.5 | NONE | 1720 | 67 | +0.056 | +0.041 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O15 | 2.0 | NONE | 1719 | 67 | +0.042 | +0.164 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O30 | 1.0 | NONE | 1709 | 67 | +0.038 | +0.187 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O30 | 1.5 | NONE | 1704 | 67 | +0.036 | +0.265 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O30 | 2.0 | NONE | 1699 | 66 | +0.008 | +0.145 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O5 | 1.0 | NONE | 1720 | 67 | -0.126 | -0.039 | 100.0% |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | ORB_G5 | 316 | 41 | +0.027 | +0.111 | 20.0% |
| non_deployed | MES | TOKYO_OPEN | O5 | 2.0 | NONE | 1719 | 67 | -0.125 | +0.138 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O15 | 1.0 | NONE | 1718 | 67 | -0.094 | -0.156 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O15 | 1.5 | NONE | 1718 | 67 | -0.098 | +0.022 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O15 | 2.0 | NONE | 1716 | 67 | -0.102 | +0.185 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O30 | 1.0 | NONE | 1704 | 67 | -0.069 | +0.135 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O30 | 1.5 | NONE | 1699 | 66 | -0.111 | +0.299 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O30 | 2.0 | NONE | 1693 | 66 | -0.111 | +0.351 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O5 | 1.0 | NONE | 917 | 66 | -0.214 | +0.032 | 100.0% |
| twin | MGC | TOKYO_OPEN | O5 | 1.5 | ORB_G5 | 135 | 63 | +0.124 | +0.130 | 20.3% |
| non_deployed | MGC | TOKYO_OPEN | O5 | 2.0 | NONE | 917 | 66 | -0.239 | +0.072 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O15 | 1.0 | NONE | 916 | 66 | -0.113 | +0.074 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O15 | 1.5 | NONE | 914 | 66 | -0.105 | +0.160 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O15 | 2.0 | NONE | 913 | 66 | -0.120 | +0.218 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O30 | 1.0 | NONE | 914 | 66 | -0.065 | +0.114 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O30 | 1.5 | NONE | 909 | 66 | -0.066 | +0.212 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O30 | 2.0 | NONE | 904 | 66 | -0.062 | +0.280 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | NONE | 1721 | 67 | -0.012 | +0.281 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.5 | NONE | 1721 | 67 | -0.010 | +0.215 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 2.0 | NONE | 1720 | 67 | +0.007 | +0.332 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O15 | 1.0 | NONE | 1718 | 67 | +0.028 | +0.079 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O15 | 1.5 | NONE | 1718 | 67 | +0.050 | +0.099 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O15 | 2.0 | NONE | 1718 | 67 | +0.064 | +0.148 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O30 | 1.0 | NONE | 1707 | 67 | +0.035 | +0.152 | 100.0% |
| deployed | MNQ | SINGAPORE_OPEN | O30 | 1.5 | ATR_P50 | 820 | 67 | +0.140 | +0.115 | 50.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O30 | 2.0 | NONE | 1703 | 67 | +0.100 | +0.037 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | NONE | 1720 | 67 | -0.206 | -0.014 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.5 | NONE | 1720 | 67 | -0.212 | +0.038 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 2.0 | NONE | 1720 | 67 | -0.219 | -0.022 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O15 | 1.0 | NONE | 1718 | 67 | -0.138 | -0.043 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O15 | 1.5 | NONE | 1718 | 67 | -0.129 | -0.036 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O15 | 2.0 | NONE | 1717 | 67 | -0.112 | +0.077 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O30 | 1.0 | NONE | 1707 | 67 | -0.084 | -0.016 | 100.0% |
| twin | MES | SINGAPORE_OPEN | O30 | 1.5 | ATR_P50 | 820 | 66 | -0.034 | -0.060 | 50.1% |
| non_deployed | MES | SINGAPORE_OPEN | O30 | 2.0 | NONE | 1699 | 67 | -0.069 | -0.095 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O5 | 1.0 | NONE | 917 | 65 | -0.121 | +0.338 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O5 | 1.5 | NONE | 917 | 65 | -0.132 | +0.414 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O5 | 2.0 | NONE | 917 | 65 | -0.125 | +0.385 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O15 | 1.0 | NONE | 912 | 65 | -0.120 | +0.137 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O15 | 1.5 | NONE | 912 | 63 | -0.126 | +0.272 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O15 | 2.0 | NONE | 908 | 62 | -0.120 | +0.409 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O30 | 1.0 | NONE | 897 | 62 | -0.061 | +0.102 | 100.0% |
| twin | MGC | SINGAPORE_OPEN | O30 | 1.5 | ATR_P50 | 417 | 60 | -0.010 | +0.177 | 50.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O30 | 2.0 | NONE | 879 | 57 | -0.036 | +0.178 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.0 | NONE | 1716 | 67 | +0.028 | -0.043 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | NONE | 1716 | 67 | +0.037 | +0.057 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O5 | 2.0 | NONE | 1716 | 67 | +0.030 | +0.015 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O15 | 1.0 | NONE | 1714 | 67 | +0.025 | -0.074 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O15 | 1.5 | NONE | 1708 | 67 | +0.041 | +0.015 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O15 | 2.0 | NONE | 1704 | 67 | +0.010 | +0.044 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O30 | 1.0 | NONE | 1692 | 67 | +0.002 | +0.078 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O30 | 1.5 | NONE | 1676 | 67 | +0.041 | +0.167 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O30 | 2.0 | NONE | 1668 | 66 | +0.048 | +0.287 | 100.0% |
| non_deployed | MES | LONDON_METALS | O5 | 1.0 | NONE | 1717 | 67 | -0.106 | -0.064 | 100.0% |
| non_deployed | MES | LONDON_METALS | O5 | 1.5 | NONE | 1716 | 67 | -0.114 | -0.053 | 100.0% |
| non_deployed | MES | LONDON_METALS | O5 | 2.0 | NONE | 1716 | 67 | -0.128 | -0.100 | 100.0% |
| non_deployed | MES | LONDON_METALS | O15 | 1.0 | NONE | 1710 | 67 | -0.085 | -0.065 | 100.0% |
| non_deployed | MES | LONDON_METALS | O15 | 1.5 | NONE | 1700 | 67 | -0.087 | -0.035 | 100.0% |
| non_deployed | MES | LONDON_METALS | O15 | 2.0 | NONE | 1693 | 67 | -0.099 | +0.034 | 100.0% |
| non_deployed | MES | LONDON_METALS | O30 | 1.0 | NONE | 1683 | 67 | -0.070 | +0.013 | 100.0% |
| non_deployed | MES | LONDON_METALS | O30 | 1.5 | NONE | 1666 | 67 | -0.047 | -0.048 | 100.0% |
| non_deployed | MES | LONDON_METALS | O30 | 2.0 | NONE | 1642 | 67 | -0.052 | +0.099 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O5 | 1.0 | NONE | 916 | 66 | -0.134 | +0.014 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O5 | 1.5 | NONE | 916 | 66 | -0.153 | +0.055 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O5 | 2.0 | NONE | 915 | 66 | -0.128 | -0.070 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O15 | 1.0 | NONE | 916 | 66 | -0.064 | -0.017 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O15 | 1.5 | NONE | 911 | 66 | -0.074 | -0.168 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O15 | 2.0 | NONE | 903 | 66 | -0.082 | -0.129 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O30 | 1.0 | NONE | 908 | 65 | +0.008 | -0.165 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O30 | 1.5 | NONE | 896 | 65 | -0.027 | -0.141 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O30 | 2.0 | NONE | 880 | 64 | -0.029 | -0.272 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O5 | 1.0 | NONE | 1717 | 67 | +0.038 | +0.141 | 100.0% |
| deployed | MNQ | EUROPE_FLOW | O5 | 1.5 | ORB_G5 | 327 | 32 | +0.093 | +0.490 | 20.2% |
| non_deployed | MNQ | EUROPE_FLOW | O5 | 2.0 | NONE | 1717 | 67 | +0.062 | +0.210 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O15 | 1.0 | NONE | 1718 | 67 | +0.017 | +0.082 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O15 | 1.5 | NONE | 1717 | 67 | +0.033 | +0.213 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O15 | 2.0 | NONE | 1712 | 67 | +0.039 | +0.067 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O30 | 1.0 | NONE | 1706 | 67 | +0.029 | -0.018 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O30 | 1.5 | NONE | 1698 | 67 | +0.017 | +0.048 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O30 | 2.0 | NONE | 1692 | 67 | +0.026 | +0.083 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O5 | 1.0 | NONE | 1718 | 67 | -0.169 | -0.087 | 100.0% |
| twin | MES | EUROPE_FLOW | O5 | 1.5 | ORB_G5 | 362 | 27 | -0.050 | +0.166 | 21.8% |
| non_deployed | MES | EUROPE_FLOW | O5 | 2.0 | NONE | 1716 | 67 | -0.191 | -0.148 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O15 | 1.0 | NONE | 1718 | 67 | -0.096 | -0.145 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O15 | 1.5 | NONE | 1716 | 67 | -0.125 | -0.163 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O15 | 2.0 | NONE | 1713 | 67 | -0.121 | -0.116 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O30 | 1.0 | NONE | 1707 | 66 | -0.071 | +0.033 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O30 | 1.5 | NONE | 1699 | 66 | -0.086 | +0.118 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O30 | 2.0 | NONE | 1687 | 66 | -0.085 | +0.134 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O5 | 1.0 | NONE | 916 | 66 | -0.129 | -0.044 | 100.0% |
| twin | MGC | EUROPE_FLOW | O5 | 1.5 | ORB_G5 | 131 | 65 | +0.146 | -0.068 | 20.1% |
| non_deployed | MGC | EUROPE_FLOW | O5 | 2.0 | NONE | 916 | 66 | -0.126 | -0.028 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O15 | 1.0 | NONE | 915 | 66 | -0.077 | +0.035 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O15 | 1.5 | NONE | 912 | 66 | -0.095 | +0.077 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O15 | 2.0 | NONE | 908 | 66 | -0.064 | +0.121 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O30 | 1.0 | NONE | 908 | 66 | -0.048 | +0.047 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O30 | 1.5 | NONE | 904 | 66 | -0.039 | +0.056 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O30 | 2.0 | NONE | 890 | 66 | -0.012 | +0.005 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.0 | NONE | 1683 | 65 | +0.016 | -0.208 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.5 | NONE | 1660 | 65 | -0.005 | -0.158 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O5 | 2.0 | NONE | 1645 | 65 | -0.010 | -0.206 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O15 | 1.0 | NONE | 1664 | 65 | -0.003 | -0.139 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O15 | 1.5 | NONE | 1638 | 65 | -0.005 | -0.150 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O15 | 2.0 | NONE | 1618 | 65 | -0.034 | -0.200 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O30 | 1.0 | NONE | 1643 | 65 | +0.019 | -0.073 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O30 | 1.5 | NONE | 1610 | 65 | -0.023 | -0.028 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O30 | 2.0 | NONE | 1585 | 64 | -0.018 | -0.043 | 100.0% |
| non_deployed | MES | US_DATA_830 | O5 | 1.0 | NONE | 1673 | 65 | -0.101 | -0.084 | 100.0% |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | NONE | 1647 | 64 | -0.126 | -0.192 | 100.0% |
| non_deployed | MES | US_DATA_830 | O5 | 2.0 | NONE | 1628 | 64 | -0.127 | -0.330 | 100.0% |
| non_deployed | MES | US_DATA_830 | O15 | 1.0 | NONE | 1659 | 65 | -0.076 | +0.020 | 100.0% |
| non_deployed | MES | US_DATA_830 | O15 | 1.5 | NONE | 1625 | 64 | -0.093 | -0.145 | 100.0% |
| non_deployed | MES | US_DATA_830 | O15 | 2.0 | NONE | 1601 | 64 | -0.114 | -0.103 | 100.0% |
| non_deployed | MES | US_DATA_830 | O30 | 1.0 | NONE | 1634 | 64 | -0.079 | -0.067 | 100.0% |
| non_deployed | MES | US_DATA_830 | O30 | 1.5 | NONE | 1589 | 64 | -0.118 | -0.053 | 100.0% |
| non_deployed | MES | US_DATA_830 | O30 | 2.0 | NONE | 1550 | 63 | -0.121 | -0.069 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O5 | 1.0 | NONE | 859 | 64 | -0.087 | -0.088 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O5 | 1.5 | NONE | 833 | 64 | -0.083 | +0.028 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O5 | 2.0 | NONE | 816 | 64 | -0.064 | -0.033 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O15 | 1.0 | NONE | 808 | 64 | -0.018 | -0.156 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O15 | 1.5 | NONE | 765 | 63 | -0.020 | -0.120 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O15 | 2.0 | NONE | 716 | 62 | -0.063 | -0.115 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O30 | 1.0 | NONE | 748 | 60 | +0.025 | +0.035 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O30 | 1.5 | NONE | 668 | 57 | +0.011 | +0.062 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O30 | 2.0 | NONE | 609 | 56 | -0.026 | +0.142 | 100.0% |
| deployed | MNQ | NYSE_OPEN | O5 | 1.0 | ORB_G5 | 310 | 44 | +0.053 | +0.300 | 20.2% |
| non_deployed | MNQ | NYSE_OPEN | O5 | 1.5 | NONE | 1649 | 63 | +0.094 | +0.093 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O5 | 2.0 | NONE | 1601 | 60 | +0.084 | +0.083 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O15 | 1.0 | NONE | 1544 | 55 | +0.097 | +0.257 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O15 | 1.5 | NONE | 1393 | 44 | +0.068 | +0.290 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O15 | 2.0 | NONE | 1259 | 38 | +0.011 | +0.246 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O30 | 1.0 | NONE | 1270 | 42 | +0.107 | +0.368 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O30 | 1.5 | NONE | 1037 | 26 | +0.021 | +0.143 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O30 | 2.0 | NONE | 870 | 21 | -0.166 | -0.010 | 100.0% |
| twin | MES | NYSE_OPEN | O5 | 1.0 | ORB_G5 | 331 | 42 | -0.012 | +0.049 | 21.1% |
| non_deployed | MES | NYSE_OPEN | O5 | 1.5 | NONE | 1681 | 65 | +0.018 | -0.016 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O5 | 2.0 | NONE | 1651 | 63 | +0.044 | -0.140 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O15 | 1.0 | NONE | 1624 | 60 | +0.031 | +0.060 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O15 | 1.5 | NONE | 1524 | 49 | +0.050 | -0.020 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O15 | 2.0 | NONE | 1416 | 48 | +0.016 | +0.079 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O30 | 1.0 | NONE | 1414 | 45 | +0.032 | +0.079 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O30 | 1.5 | NONE | 1220 | 34 | -0.010 | +0.001 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O30 | 2.0 | NONE | 1062 | 28 | -0.125 | -0.169 | 100.0% |
| twin | MGC | NYSE_OPEN | O5 | 1.0 | ORB_G5 | 132 | 64 | -0.056 | +0.357 | 20.2% |
| non_deployed | MGC | NYSE_OPEN | O5 | 1.5 | NONE | 901 | 64 | -0.061 | +0.241 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O5 | 2.0 | NONE | 882 | 63 | -0.070 | +0.235 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O15 | 1.0 | NONE | 826 | 59 | +0.023 | +0.156 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O15 | 1.5 | NONE | 737 | 55 | -0.034 | +0.109 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O15 | 2.0 | NONE | 675 | 49 | -0.097 | +0.013 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O30 | 1.0 | NONE | 639 | 49 | +0.062 | -0.041 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O30 | 1.5 | NONE | 519 | 43 | -0.010 | +0.027 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O30 | 2.0 | NONE | 443 | 39 | -0.118 | -0.019 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O5 | 1.0 | NONE | 1700 | 64 | +0.086 | +0.006 | 100.0% |
| deployed | MNQ | US_DATA_1000 | O5 | 1.5 | VWAP_MID_ALIGNED | 846 | 28 | +0.113 | +0.220 | 50.3% |
| non_deployed | MNQ | US_DATA_1000 | O5 | 2.0 | NONE | 1638 | 61 | +0.090 | -0.089 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O15 | 1.0 | NONE | 1593 | 60 | +0.096 | +0.180 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O15 | 1.5 | NONE | 1494 | 55 | +0.105 | +0.161 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O15 | 2.0 | NONE | 1376 | 46 | +0.051 | +0.087 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O30 | 1.0 | NONE | 1387 | 44 | +0.103 | +0.165 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O30 | 1.5 | NONE | 1181 | 33 | +0.047 | -0.179 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O30 | 2.0 | NONE | 1019 | 28 | -0.093 | -0.473 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O5 | 1.0 | NONE | 1700 | 65 | -0.029 | +0.037 | 100.0% |
| twin | MES | US_DATA_1000 | O5 | 1.5 | VWAP_MID_ALIGNED | 877 | 27 | -0.030 | +0.122 | 52.1% |
| non_deployed | MES | US_DATA_1000 | O5 | 2.0 | NONE | 1637 | 59 | -0.014 | +0.049 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O15 | 1.0 | NONE | 1627 | 60 | +0.024 | +0.182 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O15 | 1.5 | NONE | 1552 | 50 | +0.042 | +0.199 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O15 | 2.0 | NONE | 1434 | 44 | -0.007 | -0.085 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O30 | 1.0 | NONE | 1449 | 45 | +0.023 | +0.068 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O30 | 1.5 | NONE | 1270 | 39 | +0.001 | -0.012 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O30 | 2.0 | NONE | 1108 | 35 | -0.123 | -0.173 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O5 | 1.0 | NONE | 868 | 65 | -0.029 | +0.035 | 100.0% |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | VWAP_MID_ALIGNED | 439 | 33 | -0.081 | +0.167 | 53.1% |
| non_deployed | MGC | US_DATA_1000 | O5 | 2.0 | NONE | 781 | 64 | -0.081 | -0.007 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O15 | 1.0 | NONE | 742 | 62 | +0.004 | +0.159 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O15 | 1.5 | NONE | 651 | 58 | -0.007 | +0.045 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O15 | 2.0 | NONE | 583 | 50 | -0.084 | -0.185 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O30 | 1.0 | NONE | 569 | 52 | +0.061 | -0.022 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O30 | 1.5 | NONE | 439 | 44 | -0.076 | -0.059 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O30 | 2.0 | NONE | 382 | 36 | -0.180 | -0.268 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 1.0 | NONE | 1649 | 63 | +0.064 | +0.114 | 100.0% |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | OVNRNG_100 | 24 | 1 | -0.029 | -1.000 | 1.5% |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 2.0 | NONE | 1614 | 60 | +0.045 | +0.043 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 1.0 | NONE | 1573 | 58 | +0.041 | +0.003 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 1.5 | NONE | 1497 | 55 | +0.012 | -0.161 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 2.0 | NONE | 1410 | 51 | -0.049 | -0.144 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O30 | 1.0 | NONE | 1383 | 50 | +0.049 | +0.094 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O30 | 1.5 | NONE | 1221 | 42 | -0.040 | +0.161 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O30 | 2.0 | NONE | 1094 | 35 | -0.114 | +0.001 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | NONE | 1653 | 63 | -0.096 | +0.018 | 100.0% |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | OVNRNG_100 | 35 | 1 | -0.196 | +0.854 | 2.1% |
| non_deployed | MES | COMEX_SETTLE | O5 | 2.0 | NONE | 1630 | 61 | -0.098 | -0.176 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O15 | 1.0 | NONE | 1586 | 61 | -0.068 | +0.036 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O15 | 1.5 | NONE | 1517 | 58 | -0.103 | -0.038 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O15 | 2.0 | NONE | 1440 | 51 | -0.151 | -0.078 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | NONE | 1425 | 54 | -0.042 | -0.058 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.5 | NONE | 1289 | 50 | -0.077 | -0.056 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O30 | 2.0 | NONE | 1165 | 44 | -0.156 | -0.172 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O5 | 1.0 | NONE | 882 | 64 | -0.160 | -0.131 | 100.0% |
| twin | MGC | COMEX_SETTLE | O5 | 1.5 | OVNRNG_100 | 29 | 12 | +0.100 | -0.800 | 4.5% |
| non_deployed | MGC | COMEX_SETTLE | O5 | 2.0 | NONE | 817 | 61 | -0.228 | -0.317 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O15 | 1.0 | NONE | 771 | 60 | -0.148 | -0.110 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O15 | 1.5 | NONE | 656 | 57 | -0.227 | -0.124 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O15 | 2.0 | NONE | 606 | 52 | -0.283 | -0.228 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O30 | 1.0 | NONE | 586 | 54 | -0.091 | -0.037 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O30 | 1.5 | NONE | 479 | 49 | -0.189 | +0.028 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O30 | 2.0 | NONE | 423 | 40 | -0.274 | -0.132 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | NONE | 1452 | 61 | +0.099 | -0.016 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | NONE | 1296 | 57 | +0.089 | -0.184 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 2.0 | NONE | 1156 | 56 | +0.040 | -0.153 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.0 | NONE | 288 | 15 | +0.217 | +0.164 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.5 | NONE | 200 | 12 | +0.072 | +0.214 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 2.0 | NONE | 159 | 10 | -0.171 | +0.161 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O30 | 1.0 | NONE | 66 | 4 | +0.232 | +0.935 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O30 | 1.5 | NONE | 39 | 3 | -0.131 | +1.429 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O30 | 2.0 | NONE | 34 | 3 | -0.233 | +1.915 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | NONE | 1424 | 60 | -0.002 | -0.159 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | NONE | 1223 | 54 | -0.055 | -0.247 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O5 | 2.0 | NONE | 1070 | 52 | -0.125 | -0.265 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O15 | 1.0 | NONE | 179 | 17 | +0.194 | -0.008 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O15 | 1.5 | NONE | 122 | 14 | +0.069 | -0.173 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O15 | 2.0 | NONE | 96 | 12 | -0.118 | -0.300 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O30 | 1.0 | NONE | 63 | 5 | +0.307 | +0.846 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O30 | 1.5 | NONE | 33 | 4 | -0.073 | +1.311 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O30 | 2.0 | NONE | 26 | 3 | -0.357 | +1.799 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.0 | NONE | 804 | 39 | +0.083 | +0.503 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.5 | NONE | 611 | 26 | -0.012 | +0.183 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 2.0 | NONE | 514 | 23 | -0.171 | +0.112 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O15 | 1.0 | NONE | 326 | 16 | +0.115 | +0.315 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O15 | 1.5 | NONE | 230 | 11 | -0.029 | +0.300 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O15 | 2.0 | NONE | 190 | 9 | -0.176 | +0.275 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O30 | 1.0 | NONE | 196 | 6 | +0.135 | +0.924 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O30 | 1.5 | NONE | 130 | 3 | -0.077 | +1.395 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O30 | 2.0 | NONE | 104 | 1 | -0.305 | +1.905 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O5 | 1.0 | NONE | 744 | 31 | -0.056 | -0.021 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O5 | 1.5 | NONE | 558 | 26 | -0.213 | -0.062 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O5 | 2.0 | NONE | 476 | 24 | -0.351 | -0.010 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O15 | 1.0 | NONE | 318 | 16 | -0.037 | +0.017 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O15 | 1.5 | NONE | 226 | 14 | -0.218 | +0.122 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O15 | 2.0 | NONE | 195 | 11 | -0.323 | -0.030 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O30 | 1.0 | NONE | 198 | 8 | -0.033 | +0.586 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O30 | 1.5 | NONE | 133 | 4 | -0.301 | +0.694 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O30 | 2.0 | NONE | 112 | 3 | -0.486 | +0.807 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | NONE | 1721 | 67 | -0.044 | +0.178 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.5 | NONE | 1720 | 67 | -0.032 | +0.094 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 2.0 | NONE | 1720 | 67 | -0.009 | +0.187 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O15 | 1.0 | NONE | 1721 | 67 | +0.007 | +0.163 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O15 | 1.5 | NONE | 1720 | 67 | -0.019 | +0.241 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O15 | 2.0 | NONE | 1720 | 67 | +0.015 | +0.187 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O30 | 1.0 | NONE | 1717 | 67 | -0.011 | +0.291 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O30 | 1.5 | NONE | 1714 | 67 | -0.009 | +0.254 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O30 | 2.0 | NONE | 1712 | 66 | +0.004 | +0.089 | 100.0% |