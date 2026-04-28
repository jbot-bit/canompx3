# Comprehensive Scan — ALL Sessions × ALL Instruments × ALL Apertures × ALL RRs

**Date:** 2026-04-15
**Total cells scanned:** 17075
**Trustworthy cells** (not extreme-fire, not tautology, not arithmetic-only): 16206
**Strict survivors** (|t|>=3 + dir_match + N>=50 + trustworthy): 139

## BH-FDR pass counts at each K framing

- **K_global** (K=17075) strictest: 28 pass
- **K_family** (within feature-family, avg K~3220): 192 pass
- **K_lane** (within session+apt+rr+instr, avg K~62): 233 pass
- **K_session** (within session across instruments, avg K~1540): 70 pass
- **K_instrument** (within instrument, avg K~5818): 31 pass
- **K_feature** (within feature across lanes, avg K~540): 212 pass

**Promising** (|t|>=2.5 + dir_match + N>=50 + trustworthy): 341

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
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 289 | 37.9% | +0.078 | +0.154 | +0.306 | +0.366 | +4.95 | 0.0000 | Y | Y |
| non_deployed | MGC | LONDON_METALS | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 149 | 34.3% | +0.125 | +0.203 | +0.364 | +0.074 | +4.88 | 0.0000 | Y | Y |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 276 | 32.4% | +0.092 | +0.140 | +0.328 | +0.116 | +4.46 | 0.0000 | Y | Y |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 289 | 37.9% | +0.069 | +0.143 | +0.339 | +0.451 | +4.39 | 0.0000 | Y | Y |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 287 | 34.7% | +0.131 | +0.125 | +0.256 | +0.483 | +4.18 | 0.0000 | Y | Y |
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | short | pit_range_atr_HIGH | volatility | unfiltered | 121 | 32.3% | -0.009 | +0.126 | +0.271 | +0.165 | +4.09 | 0.0001 | Y | Y |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_LONG_INTERACT | overnight | unfiltered | 248 | 28.8% | -0.188 | -0.131 | -0.254 | -0.070 | -4.08 | 0.0001 | Y | . |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_TRUE | overnight | unfiltered | 248 | 28.8% | -0.188 | -0.131 | -0.254 | -0.070 | -4.08 | 0.0001 | Y | . |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | short | pit_range_atr_LOW | volatility | unfiltered | 240 | 32.4% | -0.221 | -0.127 | -0.303 | -0.088 | -4.04 | 0.0001 | Y | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 286 | 34.6% | -0.052 | +0.112 | +0.228 | +0.272 | +4.03 | 0.0001 | Y | Y |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | bb_volume_ratio_LOW | volume | unfiltered | 265 | 32.8% | -0.135 | -0.155 | -0.254 | -0.591 | -3.98 | 0.0001 | Y | Y |
| non_deployed | MES | US_DATA_830 | O5 | 1.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 298 | 34.0% | -0.274 | -0.091 | -0.231 | -0.682 | -3.98 | 0.0001 | Y | Y |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 2.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 175 | 28.5% | +0.279 | +0.133 | +0.255 | +0.458 | +3.94 | 0.0001 | . | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 286 | 31.4% | -0.061 | +0.108 | +0.219 | +0.591 | +3.93 | 0.0001 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | long | break_delay_LT2 | timing | unfiltered | 463 | 55.7% | +0.106 | +0.138 | +0.214 | +0.402 | +3.87 | 0.0001 | . | . |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 175 | 28.5% | +0.221 | +0.139 | +0.197 | +0.460 | +3.85 | 0.0001 | . | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 301 | 33.6% | -0.344 | -0.104 | -0.202 | -0.191 | -3.85 | 0.0001 | . | Y |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 275 | 32.7% | -0.219 | -0.117 | -0.304 | -0.003 | -3.85 | 0.0001 | . | Y |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 238 | 32.2% | +0.197 | +0.140 | +0.202 | +0.014 | +3.78 | 0.0002 | . | Y |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 302 | 35.8% | +0.105 | +0.141 | +0.224 | +0.446 | +3.74 | 0.0002 | . | Y |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 287 | 30.9% | +0.089 | +0.147 | +0.212 | +0.379 | +3.73 | 0.0002 | . | Y |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 235 | 30.1% | -0.161 | -0.134 | -0.316 | -0.366 | -3.68 | 0.0003 | . | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 286 | 31.4% | -0.020 | +0.101 | +0.256 | +0.673 | +3.67 | 0.0003 | . | Y |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 175 | 28.5% | +0.246 | +0.129 | +0.217 | +0.428 | +3.63 | 0.0003 | . | Y |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 214 | 34.6% | -0.025 | -0.150 | -0.160 | -0.502 | -3.63 | 0.0003 | . | Y |
| non_deployed | MGC | LONDON_METALS | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 149 | 34.3% | +0.080 | +0.152 | +0.353 | +0.110 | +3.59 | 0.0004 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 2.0 | long | break_delay_LT2 | timing | unfiltered | 463 | 55.7% | +0.182 | +0.087 | +0.259 | +0.503 | +3.59 | 0.0004 | . | . |
| non_deployed | MES | LONDON_METALS | O30 | 1.5 | long | ovn_range_pct_GT80 | overnight | unfiltered | 183 | 20.9% | +0.231 | +0.132 | +0.338 | +0.690 | +3.58 | 0.0004 | . | . |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 281 | 31.3% | +0.090 | +0.100 | +0.213 | +0.120 | +3.58 | 0.0004 | . | Y |
| non_deployed | MES | US_DATA_830 | O5 | 2.0 | long | near_session_london_high | level | unfiltered | 399 | 46.2% | -0.260 | -0.103 | -0.279 | -0.031 | -3.54 | 0.0004 | . | . |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 281 | 32.7% | +0.263 | +0.112 | +0.283 | +0.083 | +3.52 | 0.0005 | . | Y |
| non_deployed | MGC | NYSE_OPEN | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 174 | 36.8% | +0.152 | +0.179 | +0.354 | +0.816 | +3.52 | 0.0005 | . | Y |
| non_deployed | MNQ | CME_REOPEN | O15 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 185 | 28.4% | +0.141 | +0.146 | +0.236 | +0.690 | +3.50 | 0.0005 | . | Y |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 289 | 36.0% | +0.176 | +0.113 | +0.217 | +0.776 | +3.48 | 0.0005 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | short | bb_volume_ratio_LOW | volume | unfiltered | 281 | 34.6% | -0.184 | -0.153 | -0.208 | -0.359 | -3.47 | 0.0006 | . | Y |
| non_deployed | MES | EUROPE_FLOW | O15 | 1.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 186 | 21.1% | +0.143 | +0.101 | +0.246 | +0.334 | +3.47 | 0.0006 | . | . |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | long | near_session_london_high | level | unfiltered | 399 | 46.2% | -0.249 | -0.089 | -0.239 | -0.260 | -3.47 | 0.0006 | . | . |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_LOW_Q1 | volume | unfiltered | 289 | 33.1% | -0.105 | -0.103 | -0.267 | -0.697 | -3.45 | 0.0006 | . | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.5 | long | rel_vol_LOW_Q1 | volume | unfiltered | 301 | 33.6% | -0.340 | -0.089 | -0.221 | -0.138 | -3.45 | 0.0006 | . | Y |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 293 | 33.7% | +0.165 | +0.128 | +0.216 | +0.562 | +3.43 | 0.0006 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 256 | 32.4% | +0.087 | +0.163 | +0.250 | +0.296 | +3.42 | 0.0007 | . | Y |
| non_deployed | MES | EUROPE_FLOW | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 262 | 31.2% | -0.022 | +0.098 | +0.206 | +0.024 | +3.42 | 0.0007 | . | Y |
| non_deployed | MES | TOKYO_OPEN | O5 | 1.0 | long | atr_vel_HIGH | volatility | unfiltered | 281 | 33.7% | +0.016 | +0.104 | +0.197 | +0.374 | +3.41 | 0.0007 | . | . |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 287 | 34.7% | +0.131 | +0.108 | +0.267 | +0.601 | +3.41 | 0.0007 | . | Y |
| non_deployed | MES | NYSE_CLOSE | O15 | 2.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 123 | 30.4% | -0.253 | -0.158 | -0.230 | -0.269 | -3.41 | 0.0008 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | atr_vel_LOW | volatility | unfiltered | 291 | 33.4% | -0.103 | -0.113 | -0.265 | -0.450 | -3.40 | 0.0007 | . | . |
| non_deployed | MES | US_DATA_1000 | O15 | 2.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 198 | 25.5% | -0.180 | -0.119 | -0.341 | -0.955 | -3.40 | 0.0008 | . | Y |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 250 | 29.9% | +0.166 | +0.141 | +0.220 | +0.098 | +3.39 | 0.0008 | . | Y |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 265 | 32.8% | -0.136 | -0.122 | -0.254 | -0.487 | -3.38 | 0.0008 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | long | break_delay_LT2 | timing | unfiltered | 463 | 55.7% | +0.142 | +0.104 | +0.221 | +0.336 | +3.38 | 0.0008 | . | . |
| non_deployed | MGC | COMEX_SETTLE | O30 | 1.0 | long | near_session_london_low | level | unfiltered | 119 | 26.9% | -0.278 | -0.163 | -0.256 | -0.140 | -3.37 | 0.0009 | . | . |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 267 | 32.3% | -0.327 | -0.089 | -0.185 | -0.517 | -3.36 | 0.0008 | . | Y |
| non_deployed | MGC | TOKYO_OPEN | O5 | 1.0 | short | pit_range_atr_HIGH | volatility | unfiltered | 148 | 33.6% | -0.081 | +0.109 | +0.260 | +0.769 | +3.35 | 0.0009 | . | . |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 278 | 32.4% | +0.252 | +0.131 | +0.265 | +0.760 | +3.35 | 0.0009 | . | Y |
| non_deployed | MNQ | LONDON_METALS | O15 | 2.0 | short | atr_vel_LOW | volatility | unfiltered | 264 | 31.6% | -0.223 | -0.111 | -0.323 | -0.350 | -3.34 | 0.0009 | . | . |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | short | bb_volume_ratio_LOW | volume | unfiltered | 256 | 29.9% | -0.090 | -0.127 | -0.218 | -0.364 | -3.34 | 0.0009 | . | Y |
| non_deployed | MGC | US_DATA_830 | O5 | 2.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 144 | 31.7% | -0.275 | -0.129 | -0.380 | -0.516 | -3.34 | 0.0009 | . | Y |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 1.0 | short | dow_thu | calendar | unfiltered | 162 | 20.8% | +0.246 | +0.132 | +0.265 | +0.552 | +3.33 | 0.0010 | . | . |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | short | ovn_range_pct_GT80 | overnight | unfiltered | 170 | 20.3% | +0.146 | +0.125 | +0.300 | +0.037 | +3.33 | 0.0010 | . | . |
| non_deployed | MES | COMEX_SETTLE | O5 | 2.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 289 | 37.9% | +0.070 | +0.104 | +0.302 | +0.541 | +3.32 | 0.0009 | . | Y |
| non_deployed | MES | NYSE_CLOSE | O15 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 159 | 36.3% | +0.043 | +0.162 | +0.225 | +0.032 | +3.32 | 0.0010 | . | Y |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 2.0 | short | dow_thu | calendar | unfiltered | 162 | 20.8% | +0.307 | +0.129 | +0.389 | +0.360 | +3.30 | 0.0011 | . | . |
| twin | MES | NYSE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 313 | 37.0% | +0.141 | +0.105 | +0.210 | +0.156 | +3.30 | 0.0010 | . | Y |
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 254 | 28.8% | +0.063 | +0.095 | +0.204 | +0.495 | +3.30 | 0.0011 | . | Y |
| non_deployed | MNQ | NYSE_OPEN | O5 | 2.0 | long | atr_20_pct_GT80 | volatility | unfiltered | 204 | 24.8% | -0.146 | -0.133 | -0.345 | -0.240 | -3.29 | 0.0011 | . | . |
| non_deployed | MES | NYSE_CLOSE | O15 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 159 | 36.3% | +0.023 | +0.171 | +0.200 | +0.132 | +3.29 | 0.0011 | . | Y |
| non_deployed | MGC | COMEX_SETTLE | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 167 | 34.6% | +0.007 | +0.119 | +0.237 | +0.130 | +3.28 | 0.0011 | . | Y |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 2.0 | long | bb_volume_ratio_LOW | volume | unfiltered | 314 | 36.1% | -0.093 | -0.117 | -0.297 | -0.176 | -3.28 | 0.0011 | . | Y |
| non_deployed | MGC | EUROPE_FLOW | O30 | 2.0 | short | near_session_asia_high | level | unfiltered | 167 | 38.3% | -0.303 | -0.144 | -0.384 | -0.236 | -3.28 | 0.0011 | . | . |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | filtered | 85 | 38.4% | +0.181 | +0.206 | +0.457 | +0.831 | +3.28 | 0.0013 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 265 | 33.7% | +0.084 | +0.104 | +0.198 | +0.452 | +3.27 | 0.0011 | . | Y |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 146 | 35.2% | +0.192 | +0.153 | +0.341 | +0.848 | +3.26 | 0.0012 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 256 | 32.4% | +0.086 | +0.167 | +0.197 | +0.235 | +3.26 | 0.0012 | . | Y |
| non_deployed | MGC | US_DATA_1000 | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 146 | 35.2% | +0.116 | +0.153 | +0.274 | +0.627 | +3.25 | 0.0013 | . | Y |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | long | atr_vel_LOW | volatility | unfiltered | 291 | 33.4% | -0.104 | -0.105 | -0.203 | -0.432 | -3.25 | 0.0012 | . | . |
| non_deployed | MES | TOKYO_OPEN | O15 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 260 | 30.2% | -0.311 | -0.112 | -0.244 | -0.236 | -3.24 | 0.0013 | . | Y |
| non_deployed | MES | NYSE_CLOSE | O15 | 2.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 159 | 36.3% | +0.062 | +0.153 | +0.239 | +0.072 | +3.24 | 0.0013 | . | Y |
| non_deployed | MGC | US_DATA_830 | O5 | 1.5 | long | rel_vol_LOW_Q1 | volume | unfiltered | 144 | 31.7% | -0.268 | -0.119 | -0.325 | -0.615 | -3.24 | 0.0013 | . | Y |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 270 | 32.9% | -0.174 | -0.092 | -0.243 | -0.983 | -3.24 | 0.0013 | . | Y |
| non_deployed | MNQ | BRISBANE_1025 | O30 | 2.0 | long | is_monday_TRUE | calendar | unfiltered | 192 | 20.6% | +0.338 | +0.129 | +0.361 | +0.854 | +3.23 | 0.0014 | . | . |
| non_deployed | MES | NYSE_CLOSE | O15 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 123 | 30.4% | -0.249 | -0.156 | -0.210 | -0.426 | -3.23 | 0.0014 | . | Y |
| non_deployed | MGC | NYSE_OPEN | O5 | 2.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 174 | 36.8% | +0.174 | +0.156 | +0.383 | +1.559 | +3.23 | 0.0013 | . | Y |
| non_deployed | MES | LONDON_METALS | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 275 | 32.9% | +0.021 | +0.110 | +0.247 | +0.334 | +3.23 | 0.0013 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_LOW | volume | unfiltered | 314 | 36.1% | -0.086 | -0.121 | -0.250 | -0.243 | -3.23 | 0.0013 | . | Y |
| non_deployed | MES | COMEX_SETTLE | O30 | 2.0 | long | ovn_took_pdh_TRUE | overnight | unfiltered | 248 | 28.8% | -0.194 | -0.107 | -0.256 | -0.184 | -3.22 | 0.0014 | . | . |
| non_deployed | MES | COMEX_SETTLE | O30 | 2.0 | long | ovn_took_pdh_LONG_INTERACT | overnight | unfiltered | 248 | 28.8% | -0.194 | -0.107 | -0.256 | -0.184 | -3.22 | 0.0014 | . | . |
| non_deployed | MES | US_DATA_1000 | O15 | 1.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 198 | 25.5% | -0.128 | -0.108 | -0.237 | -0.394 | -3.22 | 0.0014 | . | Y |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 2.0 | long | near_session_london_low | level | unfiltered | 192 | 21.9% | -0.158 | -0.120 | -0.333 | -0.460 | -3.21 | 0.0015 | . | . |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 275 | 30.4% | +0.142 | +0.132 | +0.188 | +0.251 | +3.20 | 0.0015 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | filtered | 268 | 34.0% | +0.277 | +0.107 | +0.268 | +0.083 | +3.19 | 0.0015 | . | Y |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 2.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 289 | 33.1% | -0.097 | -0.094 | -0.290 | -0.730 | -3.19 | 0.0015 | . | Y |
| twin | MGC | NYSE_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 174 | 36.8% | +0.136 | +0.162 | +0.253 | +0.648 | +3.19 | 0.0015 | . | Y |
| non_deployed | MES | COMEX_SETTLE | O15 | 2.0 | long | ovn_took_pdh_LONG_INTERACT | overnight | unfiltered | 254 | 29.7% | -0.272 | -0.107 | -0.265 | -0.084 | -3.19 | 0.0015 | . | . |
| non_deployed | MES | COMEX_SETTLE | O15 | 2.0 | long | ovn_took_pdh_TRUE | overnight | unfiltered | 254 | 29.7% | -0.272 | -0.107 | -0.265 | -0.084 | -3.19 | 0.0015 | . | . |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 160 | 22.1% | +0.219 | +0.152 | +0.203 | +0.434 | +3.19 | 0.0016 | . | . |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 2.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 278 | 32.4% | +0.300 | +0.121 | +0.302 | +0.672 | +3.19 | 0.0015 | . | Y |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 1.0 | long | garch_vol_pct_GT70 | volatility | unfiltered | 199 | 23.6% | +0.245 | +0.088 | +0.229 | +0.227 | +3.18 | 0.0016 | . | . |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 265 | 33.7% | +0.070 | +0.111 | +0.229 | +0.294 | +3.18 | 0.0016 | . | Y |
| non_deployed | MGC | US_DATA_1000 | O15 | 2.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 156 | 36.6% | +0.249 | +0.155 | +0.347 | +1.083 | +3.18 | 0.0017 | . | Y |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 278 | 32.5% | +0.164 | +0.101 | +0.262 | +0.317 | +3.16 | 0.0017 | . | Y |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 2.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 235 | 30.1% | -0.215 | -0.113 | -0.308 | -0.707 | -3.15 | 0.0018 | . | Y |
| non_deployed | MES | NYSE_CLOSE | O15 | 1.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 123 | 30.4% | -0.240 | -0.158 | -0.192 | -0.397 | -3.14 | 0.0019 | . | Y |
| non_deployed | MES | EUROPE_FLOW | O5 | 1.0 | short | garch_vol_pct_GT70 | volatility | unfiltered | 184 | 22.8% | -0.003 | +0.090 | +0.217 | +0.158 | +3.14 | 0.0019 | . | . |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.5 | long | rel_vol_LOW_Q1 | volume | unfiltered | 214 | 34.6% | -0.009 | -0.135 | -0.152 | -0.545 | -3.14 | 0.0018 | . | Y |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 278 | 32.4% | +0.161 | +0.119 | +0.193 | +0.524 | +3.14 | 0.0018 | . | Y |
| non_deployed | MNQ | TOKYO_OPEN | O15 | 2.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 301 | 35.6% | +0.162 | +0.103 | +0.301 | +0.260 | +3.14 | 0.0018 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | filtered | 276 | 33.5% | +0.256 | +0.121 | +0.257 | +0.027 | +3.14 | 0.0018 | . | Y |
| twin | MES | NYSE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | filtered | 286 | 39.0% | +0.153 | +0.112 | +0.214 | +0.156 | +3.13 | 0.0018 | . | Y |
| non_deployed | MES | LONDON_METALS | O30 | 1.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 183 | 20.9% | +0.121 | +0.109 | +0.234 | +0.330 | +3.12 | 0.0020 | . | . |
| non_deployed | MES | EUROPE_FLOW | O5 | 2.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 180 | 21.5% | +0.079 | +0.101 | +0.322 | +0.134 | +3.12 | 0.0020 | . | . |
| non_deployed | MES | TOKYO_OPEN | O15 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 298 | 34.9% | +0.014 | +0.099 | +0.237 | +0.261 | +3.12 | 0.0019 | . | Y |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 2.0 | short | break_delay_LT2 | timing | unfiltered | 204 | 38.7% | +0.154 | +0.160 | +0.192 | +0.025 | +3.12 | 0.0020 | . | . |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 293 | 33.7% | +0.152 | +0.116 | +0.251 | +1.091 | +3.11 | 0.0020 | . | Y |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 2.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 270 | 32.9% | -0.180 | -0.087 | -0.272 | -1.354 | -3.11 | 0.0020 | . | Y |
| non_deployed | MNQ | CME_REOPEN | O15 | 2.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 185 | 28.4% | +0.116 | +0.137 | +0.235 | +0.181 | +3.11 | 0.0021 | . | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | atr_vel_HIGH | volatility | unfiltered | 288 | 32.6% | -0.094 | +0.092 | +0.172 | +0.580 | +3.11 | 0.0020 | . | . |
| non_deployed | MES | LONDON_METALS | O30 | 2.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 183 | 20.9% | +0.242 | +0.117 | +0.344 | +0.828 | +3.10 | 0.0021 | . | . |
| non_deployed | MGC | LONDON_METALS | O30 | 2.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 106 | 23.5% | +0.365 | +0.165 | +0.447 | +0.092 | +3.09 | 0.0023 | . | . |
| non_deployed | MES | EUROPE_FLOW | O5 | 2.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 262 | 31.2% | +0.014 | +0.087 | +0.273 | +0.324 | +3.09 | 0.0021 | . | Y |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_LOW | volume | filtered | 291 | 36.5% | -0.058 | -0.118 | -0.252 | -0.243 | -3.08 | 0.0022 | . | Y |
| non_deployed | MGC | LONDON_METALS | O5 | 2.0 | short | ovn_range_pct_GT80 | overnight | unfiltered | 101 | 23.7% | +0.185 | +0.139 | +0.418 | +0.362 | +3.08 | 0.0025 | . | . |
| non_deployed | MGC | US_DATA_1000 | O15 | 2.0 | short | pre_velocity_HIGH | timing | unfiltered | 129 | 30.3% | -0.215 | -0.157 | -0.339 | -0.134 | -3.07 | 0.0024 | . | . |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | short | break_delay_LT2 | timing | unfiltered | 508 | 62.9% | +0.119 | +0.103 | +0.225 | +0.633 | +3.06 | 0.0023 | . | . |
| twin | MES | EUROPE_FLOW | O5 | 1.5 | short | pit_range_atr_HIGH | volatility | unfiltered | 280 | 32.8% | -0.073 | +0.091 | +0.222 | +0.176 | +3.06 | 0.0023 | . | . |
| non_deployed | MGC | US_DATA_1000 | O15 | 1.5 | short | bb_volume_ratio_HIGH | volume | unfiltered | 156 | 36.6% | +0.221 | +0.145 | +0.299 | +0.747 | +3.06 | 0.0024 | . | Y |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 2.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 214 | 34.6% | -0.005 | -0.137 | -0.156 | -0.609 | -3.05 | 0.0024 | . | Y |
| non_deployed | MES | NYSE_OPEN | O30 | 1.0 | long | is_friday_TRUE | calendar | unfiltered | 171 | 18.7% | +0.199 | +0.131 | +0.218 | +0.257 | +3.05 | 0.0025 | . | . |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | filtered | 250 | 31.9% | +0.275 | +0.122 | +0.256 | +0.760 | +3.04 | 0.0025 | . | Y |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 2.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 118 | 19.3% | +0.260 | +0.133 | +0.201 | +0.202 | +3.04 | 0.0027 | . | . |
| non_deployed | MNQ | CME_REOPEN | O30 | 2.0 | long | atr_20_pct_GT80 | volatility | unfiltered | 121 | 27.2% | -0.243 | -0.133 | -0.254 | -0.641 | -3.04 | 0.0026 | . | . |
| non_deployed | MES | TOKYO_OPEN | O15 | 2.0 | long | garch_vol_pct_LT30 | volatility | unfiltered | 312 | 35.3% | -0.194 | -0.094 | -0.261 | -0.847 | -3.04 | 0.0025 | . | . |
| non_deployed | MGC | SINGAPORE_OPEN | O5 | 1.0 | long | bb_volume_ratio_LOW | volume | unfiltered | 157 | 34.8% | -0.256 | -0.140 | -0.232 | -0.244 | -3.02 | 0.0027 | . | Y |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | short | bb_volume_ratio_LOW | volume | unfiltered | 281 | 34.6% | -0.219 | -0.126 | -0.207 | -0.597 | -3.02 | 0.0026 | . | Y |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 2.0 | long | atr_vel_LOW | volatility | unfiltered | 291 | 33.4% | -0.087 | -0.098 | -0.276 | -0.429 | -3.02 | 0.0026 | . | . |
| non_deployed | MES | CME_PRECLOSE | O5 | 2.0 | short | bb_volume_ratio_LOW | volume | unfiltered | 281 | 34.6% | -0.215 | -0.112 | -0.222 | -0.572 | -3.02 | 0.0027 | . | Y |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 2.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 286 | 31.4% | -0.010 | +0.077 | +0.248 | +0.884 | +3.02 | 0.0027 | . | Y |
| non_deployed | MES | NYSE_CLOSE | O30 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 109 | 35.0% | +0.031 | +0.124 | +0.195 | +0.079 | +3.01 | 0.0030 | . | Y |
| non_deployed | MES | LONDON_METALS | O30 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 296 | 36.9% | +0.051 | +0.096 | +0.199 | +0.446 | +3.01 | 0.0027 | . | Y |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 2.0 | long | pre_velocity_HIGH | timing | unfiltered | 277 | 30.3% | +0.170 | +0.088 | +0.271 | +0.304 | +3.00 | 0.0028 | . | . |

## BH-FDR Survivors — Global (q=0.05)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Pass | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p | BH_crit |
|-------|-------|---------|-----|----|----|---------|------|------|-------|---------|------|-------|---|---|---------|
| non_deployed | MGC | CME_REOPEN | O5 | 2.0 | short | pit_range_atr_LOW | unfiltered | 137 | 32.8% | -0.592 | -0.452 | +nan | -6.32 | 0.00000 | 0.00000 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.5 | short | pit_range_atr_LOW | unfiltered | 137 | 32.8% | -0.525 | -0.391 | +nan | -5.64 | 0.00000 | 0.00001 |
| non_deployed | MES | TOKYO_OPEN | O5 | 1.0 | long | rel_vol_HIGH_Q3 | unfiltered | 276 | 32.4% | +0.094 | +0.311 | -0.046 | +5.43 | 0.00000 | 0.00001 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | short | pit_range_atr_LOW | unfiltered | 137 | 32.8% | -0.421 | -0.345 | +nan | -5.35 | 0.00000 | 0.00001 |
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 289 | 37.9% | +0.078 | +0.306 | +0.366 | +4.95 | 0.00000 | 0.00001 |
| non_deployed | MGC | CME_REOPEN | O15 | 1.0 | long | atr_20_pct_LT20 | unfiltered | 43 | 12.0% | -0.402 | -0.298 | +nan | -4.92 | 0.00000 | 0.00002 |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 301 | 38.9% | +0.254 | +0.317 | -0.522 | +4.89 | 0.00000 | 0.00002 |
| non_deployed | MGC | LONDON_METALS | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 149 | 34.3% | +0.125 | +0.364 | +0.074 | +4.88 | 0.00000 | 0.00002 |
| non_deployed | MGC | CME_REOPEN | O30 | 1.0 | long | atr_20_pct_LT20 | unfiltered | 31 | 13.1% | -0.382 | -0.326 | +nan | -4.73 | 0.00001 | 0.00003 |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 301 | 38.9% | +0.293 | +0.382 | -0.150 | +4.61 | 0.00000 | 0.00003 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 276 | 32.4% | +0.092 | +0.328 | +0.116 | +4.46 | 0.00001 | 0.00003 |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | long | bb_volume_ratio_HIGH | unfiltered | 230 | 29.8% | +0.249 | +0.342 | -0.014 | +4.44 | 0.00001 | 0.00004 |
| non_deployed | MNQ | CME_REOPEN | O5 | 1.5 | long | bb_volume_ratio_LOW | unfiltered | 275 | 35.2% | -0.181 | -0.264 | +0.476 | -4.42 | 0.00001 | 0.00004 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 289 | 37.9% | +0.069 | +0.339 | +0.451 | +4.39 | 0.00001 | 0.00004 |
| non_deployed | MNQ | CME_REOPEN | O5 | 1.5 | long | bb_volume_ratio_HIGH | unfiltered | 230 | 29.8% | +0.190 | +0.287 | -0.070 | +4.21 | 0.00003 | 0.00005 |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 287 | 34.7% | +0.131 | +0.256 | +0.483 | +4.18 | 0.00003 | 0.00005 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | short | break_delay_LT2 | unfiltered | 479 | 59.9% | +0.047 | +0.235 | -0.299 | +4.10 | 0.00005 | 0.00005 |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 285 | 33.2% | +0.232 | +0.252 | -0.158 | +4.10 | 0.00005 | 0.00006 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | short | pit_range_atr_HIGH | unfiltered | 121 | 32.3% | -0.009 | +0.271 | +0.165 | +4.09 | 0.00006 | 0.00006 |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_LONG_INTERACT | unfiltered | 248 | 28.8% | -0.188 | -0.254 | -0.070 | -4.08 | 0.00005 | 0.00006 |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_TRUE | unfiltered | 248 | 28.8% | -0.188 | -0.254 | -0.070 | -4.08 | 0.00005 | 0.00006 |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | short | pit_range_atr_LOW | unfiltered | 240 | 32.4% | -0.221 | -0.303 | -0.088 | -4.04 | 0.00006 | 0.00007 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 286 | 34.6% | -0.052 | +0.228 | +0.272 | +4.03 | 0.00006 | 0.00007 |
| non_deployed | MES | EUROPE_FLOW | O5 | 2.0 | long | pit_range_atr_LOW | unfiltered | 283 | 32.8% | -0.384 | -0.314 | +0.253 | -4.02 | 0.00007 | 0.00007 |
| non_deployed | MES | TOKYO_OPEN | O5 | 2.0 | long | rel_vol_HIGH_Q3 | unfiltered | 276 | 32.4% | +0.154 | +0.350 | -0.113 | +3.98 | 0.00008 | 0.00008 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | bb_volume_ratio_LOW | unfiltered | 265 | 32.8% | -0.135 | -0.254 | -0.591 | -3.98 | 0.00008 | 0.00008 |
| non_deployed | MES | US_DATA_830 | O5 | 1.0 | long | rel_vol_LOW_Q1 | unfiltered | 298 | 34.0% | -0.274 | -0.231 | -0.682 | -3.98 | 0.00008 | 0.00008 |
| non_deployed | MES | TOKYO_OPEN | O15 | 1.0 | short | rel_vol_LOW_Q1 | unfiltered | 281 | 32.7% | -0.262 | -0.239 | +0.139 | -3.96 | 0.00009 | 0.00009 |

## BH-FDR Survivors — Per-Family (q=0.05 within family)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Family | Pass | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p |
|-------|-------|---------|-----|----|----|---------|--------|------|------|-------|---------|------|-------|---|---|
| non_deployed | MGC | CME_REOPEN | O5 | 2.0 | short | pit_range_atr_LOW | volatility | unfiltered | 137 | 32.8% | -0.592 | -0.452 | +nan | -6.32 | 0.00000 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.5 | short | pit_range_atr_LOW | volatility | unfiltered | 137 | 32.8% | -0.525 | -0.391 | +nan | -5.64 | 0.00000 |
| non_deployed | MES | TOKYO_OPEN | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 276 | 32.4% | +0.094 | +0.311 | -0.046 | +5.43 | 0.00000 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | short | pit_range_atr_LOW | volatility | unfiltered | 137 | 32.8% | -0.421 | -0.345 | +nan | -5.35 | 0.00000 |
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 289 | 37.9% | +0.078 | +0.306 | +0.366 | +4.95 | 0.00000 |
| non_deployed | MGC | CME_REOPEN | O15 | 1.0 | long | atr_20_pct_LT20 | volatility | unfiltered | 43 | 12.0% | -0.402 | -0.298 | +nan | -4.92 | 0.00000 |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 301 | 38.9% | +0.254 | +0.317 | -0.522 | +4.89 | 0.00000 |
| non_deployed | MGC | LONDON_METALS | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 149 | 34.3% | +0.125 | +0.364 | +0.074 | +4.88 | 0.00000 |
| non_deployed | MGC | CME_REOPEN | O30 | 1.0 | long | atr_20_pct_LT20 | volatility | unfiltered | 31 | 13.1% | -0.382 | -0.326 | +nan | -4.73 | 0.00001 |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 301 | 38.9% | +0.293 | +0.382 | -0.150 | +4.61 | 0.00000 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 276 | 32.4% | +0.092 | +0.328 | +0.116 | +4.46 | 0.00001 |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 230 | 29.8% | +0.249 | +0.342 | -0.014 | +4.44 | 0.00001 |
| non_deployed | MNQ | CME_REOPEN | O5 | 1.5 | long | bb_volume_ratio_LOW | volume | unfiltered | 275 | 35.2% | -0.181 | -0.264 | +0.476 | -4.42 | 0.00001 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 289 | 37.9% | +0.069 | +0.339 | +0.451 | +4.39 | 0.00001 |
| non_deployed | MNQ | CME_REOPEN | O5 | 1.5 | long | bb_volume_ratio_HIGH | volume | unfiltered | 230 | 29.8% | +0.190 | +0.287 | -0.070 | +4.21 | 0.00003 |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 287 | 34.7% | +0.131 | +0.256 | +0.483 | +4.18 | 0.00003 |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 285 | 33.2% | +0.232 | +0.252 | -0.158 | +4.10 | 0.00005 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | short | pit_range_atr_HIGH | volatility | unfiltered | 121 | 32.3% | -0.009 | +0.271 | +0.165 | +4.09 | 0.00006 |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | short | pit_range_atr_LOW | volatility | unfiltered | 240 | 32.4% | -0.221 | -0.303 | -0.088 | -4.04 | 0.00006 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 286 | 34.6% | -0.052 | +0.228 | +0.272 | +4.03 | 0.00006 |
| non_deployed | MES | EUROPE_FLOW | O5 | 2.0 | long | pit_range_atr_LOW | volatility | unfiltered | 283 | 32.8% | -0.384 | -0.314 | +0.253 | -4.02 | 0.00007 |
| non_deployed | MES | TOKYO_OPEN | O5 | 2.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 276 | 32.4% | +0.154 | +0.350 | -0.113 | +3.98 | 0.00008 |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 293 | 33.6% | +0.221 | +0.239 | -0.129 | +3.98 | 0.00008 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | bb_volume_ratio_LOW | volume | unfiltered | 265 | 32.8% | -0.135 | -0.254 | -0.591 | -3.98 | 0.00008 |
| non_deployed | MES | US_DATA_830 | O5 | 1.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 298 | 34.0% | -0.274 | -0.231 | -0.682 | -3.98 | 0.00008 |
| non_deployed | MES | TOKYO_OPEN | O15 | 1.0 | short | rel_vol_LOW_Q1 | volume | unfiltered | 281 | 32.7% | -0.262 | -0.239 | +0.139 | -3.96 | 0.00009 |
| non_deployed | MES | CME_PRECLOSE | O30 | 1.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 113 | 35.6% | -0.122 | -0.157 | +nan | -3.94 | 0.00010 |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 2.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 175 | 28.5% | +0.279 | +0.255 | +0.458 | +3.94 | 0.00011 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 286 | 31.4% | -0.061 | +0.219 | +0.591 | +3.93 | 0.00010 |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | long | bb_volume_ratio_LOW | volume | unfiltered | 275 | 35.2% | -0.157 | -0.259 | +0.583 | -3.92 | 0.00010 |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 2.0 | short | rel_vol_HIGH_Q3 | volume | unfiltered | 301 | 38.9% | +0.231 | +0.378 | -0.039 | +3.90 | 0.00011 |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 175 | 28.5% | +0.221 | +0.197 | +0.460 | +3.85 | 0.00014 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_LOW_Q1 | volume | unfiltered | 301 | 33.6% | -0.344 | -0.202 | -0.191 | -3.85 | 0.00013 |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 275 | 32.7% | -0.219 | -0.304 | -0.003 | -3.85 | 0.00013 |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.5 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 238 | 32.2% | +0.236 | +0.235 | -0.251 | +3.83 | 0.00014 |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 2.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 238 | 32.2% | +0.252 | +0.253 | -0.333 | +3.79 | 0.00017 |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.0 | long | rel_vol_HIGH_Q3 | volume | unfiltered | 238 | 32.2% | +0.197 | +0.202 | +0.014 | +3.78 | 0.00018 |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | volume | unfiltered | 302 | 35.8% | +0.105 | +0.224 | +0.446 | +3.74 | 0.00020 |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | long | bb_volume_ratio_HIGH | volume | unfiltered | 287 | 30.9% | +0.089 | +0.212 | +0.379 | +3.73 | 0.00021 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_LOW_Q1 | volume | unfiltered | 212 | 28.3% | -0.351 | -0.290 | +0.072 | -3.68 | 0.00026 |

## Promising cells (candidates for next-round T0-T8)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Pass | N_on | ExpR_on | WR_Δ | Δ_IS | Δ_OOS | t | p |
|-------|-------|---------|-----|----|----|---------|------|------|---------|------|------|-------|---|---|
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 289 | +0.078 | +0.154 | +0.306 | +0.366 | +4.95 | 0.0000 |
| non_deployed | MGC | LONDON_METALS | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 149 | +0.125 | +0.203 | +0.364 | +0.074 | +4.88 | 0.0000 |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 276 | +0.092 | +0.140 | +0.328 | +0.116 | +4.46 | 0.0000 |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 289 | +0.069 | +0.143 | +0.339 | +0.451 | +4.39 | 0.0000 |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 287 | +0.131 | +0.125 | +0.256 | +0.483 | +4.18 | 0.0000 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | short | pit_range_atr_HIGH | unfiltered | 121 | -0.009 | +0.126 | +0.271 | +0.165 | +4.09 | 0.0001 |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_TRUE | unfiltered | 248 | -0.188 | -0.131 | -0.254 | -0.070 | -4.08 | 0.0001 |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_LONG_INTERACT | unfiltered | 248 | -0.188 | -0.131 | -0.254 | -0.070 | -4.08 | 0.0001 |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | short | pit_range_atr_LOW | unfiltered | 240 | -0.221 | -0.127 | -0.303 | -0.088 | -4.04 | 0.0001 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 286 | -0.052 | +0.112 | +0.228 | +0.272 | +4.03 | 0.0001 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | bb_volume_ratio_LOW | unfiltered | 265 | -0.135 | -0.155 | -0.254 | -0.591 | -3.98 | 0.0001 |
| non_deployed | MES | US_DATA_830 | O5 | 1.0 | long | rel_vol_LOW_Q1 | unfiltered | 298 | -0.274 | -0.091 | -0.231 | -0.682 | -3.98 | 0.0001 |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 2.0 | long | rel_vol_HIGH_Q3 | unfiltered | 175 | +0.279 | +0.133 | +0.255 | +0.458 | +3.94 | 0.0001 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_HIGH_Q3 | unfiltered | 286 | -0.061 | +0.108 | +0.219 | +0.591 | +3.93 | 0.0001 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | long | break_delay_LT2 | unfiltered | 463 | +0.106 | +0.138 | +0.214 | +0.402 | +3.87 | 0.0001 |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.0 | long | rel_vol_HIGH_Q3 | unfiltered | 175 | +0.221 | +0.139 | +0.197 | +0.460 | +3.85 | 0.0001 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | rel_vol_LOW_Q1 | unfiltered | 301 | -0.344 | -0.104 | -0.202 | -0.191 | -3.85 | 0.0001 |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | short | rel_vol_LOW_Q1 | unfiltered | 275 | -0.219 | -0.117 | -0.304 | -0.003 | -3.85 | 0.0001 |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.0 | long | rel_vol_HIGH_Q3 | unfiltered | 238 | +0.197 | +0.140 | +0.202 | +0.014 | +3.78 | 0.0002 |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | short | bb_volume_ratio_HIGH | unfiltered | 302 | +0.105 | +0.141 | +0.224 | +0.446 | +3.74 | 0.0002 |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | long | bb_volume_ratio_HIGH | unfiltered | 287 | +0.089 | +0.147 | +0.212 | +0.379 | +3.73 | 0.0002 |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | short | rel_vol_LOW_Q1 | unfiltered | 235 | -0.161 | -0.134 | -0.316 | -0.366 | -3.68 | 0.0003 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 286 | -0.020 | +0.101 | +0.256 | +0.673 | +3.67 | 0.0003 |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 175 | +0.246 | +0.129 | +0.217 | +0.428 | +3.63 | 0.0003 |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.0 | long | rel_vol_LOW_Q1 | unfiltered | 214 | -0.025 | -0.150 | -0.160 | -0.502 | -3.63 | 0.0003 |
| non_deployed | MGC | LONDON_METALS | O5 | 1.5 | short | rel_vol_HIGH_Q3 | unfiltered | 149 | +0.080 | +0.152 | +0.353 | +0.110 | +3.59 | 0.0004 |
| non_deployed | MES | CME_PRECLOSE | O5 | 2.0 | long | break_delay_LT2 | unfiltered | 463 | +0.182 | +0.087 | +0.259 | +0.503 | +3.59 | 0.0004 |
| non_deployed | MES | LONDON_METALS | O30 | 1.5 | long | ovn_range_pct_GT80 | unfiltered | 183 | +0.231 | +0.132 | +0.338 | +0.690 | +3.58 | 0.0004 |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | long | rel_vol_HIGH_Q3 | unfiltered | 281 | +0.090 | +0.100 | +0.213 | +0.120 | +3.58 | 0.0004 |
| non_deployed | MES | US_DATA_830 | O5 | 2.0 | long | near_session_london_high | unfiltered | 399 | -0.260 | -0.103 | -0.279 | -0.031 | -3.54 | 0.0004 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_HIGH_Q3 | unfiltered | 281 | +0.263 | +0.112 | +0.283 | +0.083 | +3.52 | 0.0005 |
| non_deployed | MGC | NYSE_OPEN | O5 | 1.5 | short | bb_volume_ratio_HIGH | unfiltered | 174 | +0.152 | +0.179 | +0.354 | +0.816 | +3.52 | 0.0005 |
| non_deployed | MNQ | CME_REOPEN | O15 | 1.5 | long | bb_volume_ratio_HIGH | unfiltered | 185 | +0.141 | +0.146 | +0.236 | +0.690 | +3.50 | 0.0005 |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | short | rel_vol_HIGH_Q3 | unfiltered | 289 | +0.176 | +0.113 | +0.217 | +0.776 | +3.48 | 0.0005 |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | short | bb_volume_ratio_LOW | unfiltered | 281 | -0.184 | -0.153 | -0.208 | -0.359 | -3.47 | 0.0006 |
| non_deployed | MES | EUROPE_FLOW | O15 | 1.0 | long | ovn_range_pct_GT80 | unfiltered | 186 | +0.143 | +0.101 | +0.246 | +0.334 | +3.47 | 0.0006 |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | long | near_session_london_high | unfiltered | 399 | -0.249 | -0.089 | -0.239 | -0.260 | -3.47 | 0.0006 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | rel_vol_LOW_Q1 | unfiltered | 289 | -0.105 | -0.103 | -0.267 | -0.697 | -3.45 | 0.0006 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.5 | long | rel_vol_LOW_Q1 | unfiltered | 301 | -0.340 | -0.089 | -0.221 | -0.138 | -3.45 | 0.0006 |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.0 | short | bb_volume_ratio_HIGH | unfiltered | 293 | +0.165 | +0.128 | +0.216 | +0.562 | +3.43 | 0.0006 |

## Flagged cells (excluded despite |t|>=3) — for transparency

Excluded because: tautology |corr|>0.7 (0), extreme fire <5% or >95% (575), arithmetic-only WR flat (309)

| Scope | Instr | Session | Feature | Pass | t | Fire% | T0 corr | Reason |
|-------|-------|---------|---------|------|---|-------|---------|--------|
| non_deployed | MNQ | EUROPE_FLOW | is_nfp_TRUE | unfiltered | -4.93 | 4.2% | 0.00 | FIRE(4.2%) |
| non_deployed | MES | CME_PRECLOSE | break_delay_GT10 | unfiltered | -3.68 | 3.6% | 0.00 | FIRE(3.6%) |
| deployed | MNQ | EUROPE_FLOW | is_nfp_TRUE | filtered | -3.55 | 4.0% | 0.00 | FIRE(4.0%) |
| non_deployed | MNQ | COMEX_SETTLE | is_nfp_TRUE | unfiltered | -3.51 | 4.9% | 0.00 | FIRE(4.9%) |
| deployed | MNQ | EUROPE_FLOW | is_nfp_TRUE | unfiltered | -3.49 | 4.2% | 0.04 | FIRE(4.2%) |
| non_deployed | MGC | CME_REOPEN | pit_range_atr_LOW | unfiltered | -3.47 | 33.9% | 0.00 | ARITH(WR_Δ=-0.007) |
| non_deployed | MNQ | COMEX_SETTLE | is_nfp_TRUE | unfiltered | -3.35 | 4.9% | 0.00 | FIRE(4.9%) |
| non_deployed | MES | CME_PRECLOSE | break_delay_GT10 | unfiltered | -3.26 | 3.6% | 0.00 | FIRE(3.6%) |
| non_deployed | MES | NYSE_CLOSE | near_session_london_high | unfiltered | -3.16 | 20.4% | 0.00 | ARITH(WR_Δ=-0.029) |
| non_deployed | MGC | CME_REOPEN | pit_range_atr_LOW | unfiltered | -3.14 | 32.6% | 0.00 | ARITH(WR_Δ=+0.001) |
| non_deployed | MNQ | COMEX_SETTLE | is_nfp_TRUE | unfiltered | -3.10 | 4.9% | 0.00 | FIRE(4.9%) |
| non_deployed | MES | CME_PRECLOSE | break_delay_GT10 | unfiltered | -3.06 | 3.6% | 0.00 | FIRE(3.6%) |
| non_deployed | MES | SINGAPORE_OPEN | break_bar_continues_TRUE | unfiltered | +3.03 | 96.8% | 0.00 | FIRE(96.8%) |

## Baseline Per-Lane (no feature overlay)

| Scope | Instr | Session | Apt | RR | Filter | N_is | N_oos | ExpR_is | ExpR_oos | Filter_fire% |
|-------|-------|---------|-----|----|--------|------|-------|---------|----------|--------------|
| non_deployed | MNQ | CME_REOPEN | O5 | 1.0 | NONE | 1497 | 63 | -0.025 | +0.074 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O5 | 1.5 | NONE | 1497 | 63 | -0.028 | +0.116 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | NONE | 1497 | 63 | -0.004 | +0.192 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O15 | 1.0 | NONE | 1242 | 56 | -0.023 | +0.297 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O15 | 1.5 | NONE | 1242 | 56 | -0.017 | +0.262 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O15 | 2.0 | NONE | 1242 | 56 | -0.043 | +0.132 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O30 | 1.0 | NONE | 881 | 41 | -0.021 | +0.240 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O30 | 1.5 | NONE | 881 | 41 | -0.054 | +0.120 | 100.0% |
| non_deployed | MNQ | CME_REOPEN | O30 | 2.0 | NONE | 881 | 41 | -0.061 | -0.068 | 100.0% |
| non_deployed | MES | CME_REOPEN | O5 | 1.0 | NONE | 1485 | 64 | -0.123 | -0.065 | 100.0% |
| non_deployed | MES | CME_REOPEN | O5 | 1.5 | NONE | 1485 | 64 | -0.136 | -0.061 | 100.0% |
| non_deployed | MES | CME_REOPEN | O5 | 2.0 | NONE | 1485 | 64 | -0.132 | -0.004 | 100.0% |
| non_deployed | MES | CME_REOPEN | O15 | 1.0 | NONE | 1228 | 52 | -0.099 | +0.066 | 100.0% |
| non_deployed | MES | CME_REOPEN | O15 | 1.5 | NONE | 1227 | 52 | -0.103 | +0.128 | 100.0% |
| non_deployed | MES | CME_REOPEN | O15 | 2.0 | NONE | 1227 | 52 | -0.124 | +0.025 | 100.0% |
| non_deployed | MES | CME_REOPEN | O30 | 1.0 | NONE | 842 | 40 | -0.087 | +0.038 | 100.0% |
| non_deployed | MES | CME_REOPEN | O30 | 1.5 | NONE | 842 | 40 | -0.116 | -0.001 | 100.0% |
| non_deployed | MES | CME_REOPEN | O30 | 2.0 | NONE | 842 | 40 | -0.124 | -0.133 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | NONE | 786 | 57 | -0.181 | +0.094 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O5 | 1.5 | NONE | 786 | 57 | -0.219 | +0.235 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O5 | 2.0 | NONE | 786 | 57 | -0.218 | +0.379 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O15 | 1.0 | NONE | 650 | 45 | -0.188 | +0.210 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O15 | 1.5 | NONE | 650 | 45 | -0.235 | +0.380 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O15 | 2.0 | NONE | 650 | 45 | -0.232 | +0.524 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O30 | 1.0 | NONE | 437 | 35 | -0.200 | +0.410 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O30 | 1.5 | NONE | 437 | 35 | -0.205 | +0.569 | 100.0% |
| non_deployed | MGC | CME_REOPEN | O30 | 2.0 | NONE | 437 | 35 | -0.205 | +0.570 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | NONE | 1721 | 67 | +0.047 | +0.044 | 100.0% |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | ORB_G5 | 1599 | 67 | +0.093 | +0.200 | 93.2% |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 2.0 | NONE | 1721 | 67 | +0.066 | +0.229 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O15 | 1.0 | NONE | 1720 | 67 | +0.058 | -0.109 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O15 | 1.5 | NONE | 1720 | 67 | +0.056 | +0.041 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O15 | 2.0 | NONE | 1720 | 67 | +0.043 | +0.164 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O30 | 1.0 | NONE | 1712 | 67 | +0.038 | +0.187 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O30 | 1.5 | NONE | 1712 | 67 | +0.038 | +0.265 | 100.0% |
| non_deployed | MNQ | TOKYO_OPEN | O30 | 2.0 | NONE | 1712 | 67 | +0.011 | +0.140 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O5 | 1.0 | NONE | 1720 | 67 | -0.126 | -0.039 | 100.0% |
| twin | MES | TOKYO_OPEN | O5 | 1.5 | ORB_G5 | 316 | 41 | +0.027 | +0.111 | 20.1% |
| non_deployed | MES | TOKYO_OPEN | O5 | 2.0 | NONE | 1720 | 67 | -0.125 | +0.138 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O15 | 1.0 | NONE | 1718 | 67 | -0.094 | -0.156 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O15 | 1.5 | NONE | 1718 | 67 | -0.098 | +0.022 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O15 | 2.0 | NONE | 1718 | 67 | -0.100 | +0.185 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O30 | 1.0 | NONE | 1707 | 67 | -0.069 | +0.135 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O30 | 1.5 | NONE | 1707 | 67 | -0.108 | +0.295 | 100.0% |
| non_deployed | MES | TOKYO_OPEN | O30 | 2.0 | NONE | 1707 | 67 | -0.106 | +0.345 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O5 | 1.0 | NONE | 917 | 66 | -0.214 | +0.032 | 100.0% |
| twin | MGC | TOKYO_OPEN | O5 | 1.5 | ORB_G5 | 76 | 60 | +0.248 | +0.186 | 14.4% |
| non_deployed | MGC | TOKYO_OPEN | O5 | 2.0 | NONE | 917 | 66 | -0.239 | +0.072 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O15 | 1.0 | NONE | 917 | 66 | -0.113 | +0.074 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O15 | 1.5 | NONE | 917 | 66 | -0.102 | +0.160 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O15 | 2.0 | NONE | 917 | 66 | -0.118 | +0.218 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O30 | 1.0 | NONE | 916 | 66 | -0.064 | +0.114 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O30 | 1.5 | NONE | 916 | 66 | -0.063 | +0.212 | 100.0% |
| non_deployed | MGC | TOKYO_OPEN | O30 | 2.0 | NONE | 916 | 66 | -0.057 | +0.280 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.0 | NONE | 1721 | 67 | -0.012 | +0.281 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 1.5 | NONE | 1721 | 67 | -0.010 | +0.215 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O5 | 2.0 | NONE | 1721 | 67 | +0.007 | +0.332 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O15 | 1.0 | NONE | 1719 | 67 | +0.029 | +0.079 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O15 | 1.5 | NONE | 1719 | 67 | +0.050 | +0.099 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O15 | 2.0 | NONE | 1719 | 67 | +0.065 | +0.148 | 100.0% |
| non_deployed | MNQ | SINGAPORE_OPEN | O30 | 1.0 | NONE | 1709 | 67 | +0.035 | +0.152 | 100.0% |
| deployed | MNQ | SINGAPORE_OPEN | O30 | 1.5 | ATR_P50 | 907 | 54 | +0.125 | +0.074 | 54.3% |
| non_deployed | MNQ | SINGAPORE_OPEN | O30 | 2.0 | NONE | 1709 | 67 | +0.100 | +0.037 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | NONE | 1720 | 67 | -0.206 | -0.014 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.5 | NONE | 1720 | 67 | -0.212 | +0.038 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 2.0 | NONE | 1720 | 67 | -0.219 | -0.022 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O15 | 1.0 | NONE | 1719 | 67 | -0.138 | -0.043 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O15 | 1.5 | NONE | 1719 | 67 | -0.129 | -0.036 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O15 | 2.0 | NONE | 1719 | 67 | -0.111 | +0.077 | 100.0% |
| non_deployed | MES | SINGAPORE_OPEN | O30 | 1.0 | NONE | 1711 | 67 | -0.084 | -0.016 | 100.0% |
| twin | MES | SINGAPORE_OPEN | O30 | 1.5 | ATR_P50 | 883 | 48 | -0.039 | -0.031 | 52.6% |
| non_deployed | MES | SINGAPORE_OPEN | O30 | 2.0 | NONE | 1711 | 67 | -0.066 | -0.095 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O5 | 1.0 | NONE | 917 | 65 | -0.121 | +0.338 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O5 | 1.5 | NONE | 917 | 65 | -0.132 | +0.414 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O5 | 2.0 | NONE | 917 | 65 | -0.125 | +0.385 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O15 | 1.0 | NONE | 914 | 65 | -0.120 | +0.137 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O15 | 1.5 | NONE | 914 | 65 | -0.126 | +0.278 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O15 | 2.0 | NONE | 914 | 65 | -0.118 | +0.411 | 100.0% |
| non_deployed | MGC | SINGAPORE_OPEN | O30 | 1.0 | NONE | 904 | 64 | -0.060 | +0.105 | 100.0% |
| twin | MGC | SINGAPORE_OPEN | O30 | 1.5 | ATR_P50 | 616 | 64 | +0.005 | +0.193 | 70.4% |
| non_deployed | MGC | SINGAPORE_OPEN | O30 | 2.0 | NONE | 904 | 64 | -0.021 | +0.205 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.0 | NONE | 1717 | 67 | +0.028 | -0.043 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O5 | 1.5 | NONE | 1717 | 67 | +0.038 | +0.057 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O5 | 2.0 | NONE | 1717 | 67 | +0.030 | +0.015 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O15 | 1.0 | NONE | 1716 | 67 | +0.025 | -0.074 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O15 | 1.5 | NONE | 1716 | 67 | +0.042 | +0.015 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O15 | 2.0 | NONE | 1716 | 67 | +0.012 | +0.044 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O30 | 1.0 | NONE | 1701 | 67 | +0.003 | +0.078 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O30 | 1.5 | NONE | 1701 | 67 | +0.044 | +0.167 | 100.0% |
| non_deployed | MNQ | LONDON_METALS | O30 | 2.0 | NONE | 1701 | 67 | +0.056 | +0.286 | 100.0% |
| non_deployed | MES | LONDON_METALS | O5 | 1.0 | NONE | 1718 | 67 | -0.105 | -0.064 | 100.0% |
| non_deployed | MES | LONDON_METALS | O5 | 1.5 | NONE | 1718 | 67 | -0.114 | -0.053 | 100.0% |
| non_deployed | MES | LONDON_METALS | O5 | 2.0 | NONE | 1718 | 67 | -0.127 | -0.100 | 100.0% |
| non_deployed | MES | LONDON_METALS | O15 | 1.0 | NONE | 1713 | 67 | -0.085 | -0.065 | 100.0% |
| non_deployed | MES | LONDON_METALS | O15 | 1.5 | NONE | 1713 | 67 | -0.085 | -0.035 | 100.0% |
| non_deployed | MES | LONDON_METALS | O15 | 2.0 | NONE | 1713 | 67 | -0.095 | +0.034 | 100.0% |
| non_deployed | MES | LONDON_METALS | O30 | 1.0 | NONE | 1698 | 67 | -0.070 | +0.013 | 100.0% |
| non_deployed | MES | LONDON_METALS | O30 | 1.5 | NONE | 1698 | 67 | -0.045 | -0.048 | 100.0% |
| non_deployed | MES | LONDON_METALS | O30 | 2.0 | NONE | 1698 | 67 | -0.037 | +0.099 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O5 | 1.0 | NONE | 916 | 66 | -0.134 | +0.014 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O5 | 1.5 | NONE | 916 | 66 | -0.153 | +0.055 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O5 | 2.0 | NONE | 916 | 66 | -0.128 | -0.070 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O15 | 1.0 | NONE | 916 | 66 | -0.064 | -0.017 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O15 | 1.5 | NONE | 916 | 66 | -0.073 | -0.168 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O15 | 2.0 | NONE | 916 | 66 | -0.076 | -0.129 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O30 | 1.0 | NONE | 915 | 65 | +0.008 | -0.165 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O30 | 1.5 | NONE | 915 | 65 | -0.023 | -0.141 | 100.0% |
| non_deployed | MGC | LONDON_METALS | O30 | 2.0 | NONE | 915 | 65 | -0.014 | -0.266 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O5 | 1.0 | NONE | 1717 | 67 | +0.038 | +0.141 | 100.0% |
| deployed | MNQ | EUROPE_FLOW | O5 | 1.5 | ORB_G5 | 1582 | 67 | +0.064 | +0.286 | 92.5% |
| non_deployed | MNQ | EUROPE_FLOW | O5 | 2.0 | NONE | 1717 | 67 | +0.062 | +0.210 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O15 | 1.0 | NONE | 1718 | 67 | +0.017 | +0.082 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O15 | 1.5 | NONE | 1718 | 67 | +0.033 | +0.213 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O15 | 2.0 | NONE | 1718 | 67 | +0.040 | +0.067 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O30 | 1.0 | NONE | 1710 | 67 | +0.028 | -0.018 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O30 | 1.5 | NONE | 1710 | 67 | +0.019 | +0.048 | 100.0% |
| non_deployed | MNQ | EUROPE_FLOW | O30 | 2.0 | NONE | 1710 | 67 | +0.030 | +0.083 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O5 | 1.0 | NONE | 1718 | 67 | -0.169 | -0.087 | 100.0% |
| twin | MES | EUROPE_FLOW | O5 | 1.5 | ORB_G5 | 363 | 27 | -0.048 | +0.166 | 21.9% |
| non_deployed | MES | EUROPE_FLOW | O5 | 2.0 | NONE | 1718 | 67 | -0.189 | -0.148 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O15 | 1.0 | NONE | 1718 | 67 | -0.096 | -0.145 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O15 | 1.5 | NONE | 1718 | 67 | -0.124 | -0.163 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O15 | 2.0 | NONE | 1718 | 67 | -0.119 | -0.116 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O30 | 1.0 | NONE | 1712 | 66 | -0.071 | +0.033 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O30 | 1.5 | NONE | 1712 | 66 | -0.085 | +0.118 | 100.0% |
| non_deployed | MES | EUROPE_FLOW | O30 | 2.0 | NONE | 1712 | 66 | -0.076 | +0.134 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O5 | 1.0 | NONE | 916 | 66 | -0.129 | -0.044 | 100.0% |
| twin | MGC | EUROPE_FLOW | O5 | 1.5 | ORB_G5 | 55 | 50 | +0.220 | +0.035 | 11.1% |
| non_deployed | MGC | EUROPE_FLOW | O5 | 2.0 | NONE | 916 | 66 | -0.126 | -0.028 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O15 | 1.0 | NONE | 916 | 66 | -0.077 | +0.035 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O15 | 1.5 | NONE | 916 | 66 | -0.095 | +0.077 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O15 | 2.0 | NONE | 916 | 66 | -0.062 | +0.121 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O30 | 1.0 | NONE | 913 | 66 | -0.048 | +0.047 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O30 | 1.5 | NONE | 913 | 66 | -0.038 | +0.056 | 100.0% |
| non_deployed | MGC | EUROPE_FLOW | O30 | 2.0 | NONE | 913 | 66 | -0.000 | +0.005 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.0 | NONE | 1716 | 67 | +0.019 | -0.195 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O5 | 1.5 | NONE | 1716 | 67 | +0.006 | -0.147 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O5 | 2.0 | NONE | 1716 | 67 | +0.009 | -0.193 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O15 | 1.0 | NONE | 1715 | 67 | +0.000 | -0.135 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O15 | 1.5 | NONE | 1715 | 67 | +0.004 | -0.145 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O15 | 2.0 | NONE | 1715 | 67 | -0.014 | -0.194 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O30 | 1.0 | NONE | 1714 | 66 | +0.021 | -0.070 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O30 | 1.5 | NONE | 1714 | 66 | -0.010 | -0.026 | 100.0% |
| non_deployed | MNQ | US_DATA_830 | O30 | 2.0 | NONE | 1714 | 66 | +0.007 | -0.014 | 100.0% |
| non_deployed | MES | US_DATA_830 | O5 | 1.0 | NONE | 1715 | 67 | -0.095 | -0.079 | 100.0% |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | NONE | 1715 | 67 | -0.107 | -0.172 | 100.0% |
| non_deployed | MES | US_DATA_830 | O5 | 2.0 | NONE | 1715 | 67 | -0.096 | -0.304 | 100.0% |
| non_deployed | MES | US_DATA_830 | O15 | 1.0 | NONE | 1718 | 67 | -0.070 | +0.018 | 100.0% |
| non_deployed | MES | US_DATA_830 | O15 | 1.5 | NONE | 1718 | 67 | -0.071 | -0.131 | 100.0% |
| non_deployed | MES | US_DATA_830 | O15 | 2.0 | NONE | 1718 | 67 | -0.075 | -0.091 | 100.0% |
| non_deployed | MES | US_DATA_830 | O30 | 1.0 | NONE | 1715 | 66 | -0.069 | -0.065 | 100.0% |
| non_deployed | MES | US_DATA_830 | O30 | 1.5 | NONE | 1715 | 66 | -0.090 | -0.052 | 100.0% |
| non_deployed | MES | US_DATA_830 | O30 | 2.0 | NONE | 1715 | 66 | -0.073 | -0.041 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O5 | 1.0 | NONE | 916 | 66 | -0.084 | -0.076 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O5 | 1.5 | NONE | 916 | 66 | -0.063 | +0.036 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O5 | 2.0 | NONE | 916 | 66 | -0.029 | -0.022 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O15 | 1.0 | NONE | 911 | 66 | -0.019 | -0.144 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O15 | 1.5 | NONE | 911 | 66 | -0.005 | -0.088 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O15 | 2.0 | NONE | 911 | 66 | -0.001 | -0.063 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O30 | 1.0 | NONE | 905 | 66 | +0.013 | +0.049 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O30 | 1.5 | NONE | 905 | 66 | +0.034 | +0.116 | 100.0% |
| non_deployed | MGC | US_DATA_830 | O30 | 2.0 | NONE | 905 | 66 | +0.061 | +0.201 | 100.0% |
| deployed | MNQ | NYSE_OPEN | O5 | 1.0 | ORB_G5 | 1713 | 66 | +0.078 | +0.133 | 99.7% |
| non_deployed | MNQ | NYSE_OPEN | O5 | 1.5 | NONE | 1718 | 66 | +0.098 | +0.105 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O5 | 2.0 | NONE | 1718 | 66 | +0.108 | +0.117 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O15 | 1.0 | NONE | 1714 | 66 | +0.089 | +0.237 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O15 | 1.5 | NONE | 1714 | 66 | +0.103 | +0.301 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O15 | 2.0 | NONE | 1714 | 66 | +0.132 | +0.352 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O30 | 1.0 | NONE | 1701 | 66 | +0.086 | +0.269 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O30 | 1.5 | NONE | 1701 | 66 | +0.105 | +0.214 | 100.0% |
| non_deployed | MNQ | NYSE_OPEN | O30 | 2.0 | NONE | 1701 | 66 | +0.112 | +0.226 | 100.0% |
| twin | MES | NYSE_OPEN | O5 | 1.0 | ORB_G5 | 1462 | 66 | +0.018 | +0.002 | 85.7% |
| non_deployed | MES | NYSE_OPEN | O5 | 1.5 | NONE | 1718 | 66 | +0.019 | -0.019 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O5 | 2.0 | NONE | 1718 | 66 | +0.056 | -0.128 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O15 | 1.0 | NONE | 1717 | 66 | +0.028 | +0.075 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O15 | 1.5 | NONE | 1717 | 66 | +0.065 | +0.077 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O15 | 2.0 | NONE | 1717 | 66 | +0.085 | +0.167 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O30 | 1.0 | NONE | 1710 | 66 | +0.024 | +0.092 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O30 | 1.5 | NONE | 1710 | 66 | +0.043 | +0.098 | 100.0% |
| non_deployed | MES | NYSE_OPEN | O30 | 2.0 | NONE | 1710 | 66 | +0.056 | +0.098 | 100.0% |
| twin | MGC | NYSE_OPEN | O5 | 1.0 | ORB_G5 | 240 | 64 | +0.002 | +0.357 | 31.3% |
| non_deployed | MGC | NYSE_OPEN | O5 | 1.5 | NONE | 917 | 66 | -0.060 | +0.245 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O5 | 2.0 | NONE | 917 | 66 | -0.059 | +0.233 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O15 | 1.0 | NONE | 913 | 66 | +0.012 | +0.135 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O15 | 1.5 | NONE | 913 | 66 | -0.014 | +0.101 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O15 | 2.0 | NONE | 913 | 66 | -0.006 | +0.100 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O30 | 1.0 | NONE | 894 | 66 | +0.025 | -0.003 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O30 | 1.5 | NONE | 894 | 66 | +0.022 | +0.075 | 100.0% |
| non_deployed | MGC | NYSE_OPEN | O30 | 2.0 | NONE | 894 | 66 | +0.038 | +0.081 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O5 | 1.0 | NONE | 1717 | 66 | +0.086 | +0.003 | 100.0% |
| deployed | MNQ | US_DATA_1000 | O5 | 1.5 | VWAP_MID_ALIGNED | 871 | 29 | +0.119 | +0.236 | 50.5% |
| non_deployed | MNQ | US_DATA_1000 | O5 | 2.0 | NONE | 1717 | 66 | +0.110 | -0.049 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O15 | 1.0 | NONE | 1716 | 66 | +0.092 | +0.166 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O15 | 1.5 | NONE | 1716 | 66 | +0.120 | +0.152 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O15 | 2.0 | NONE | 1716 | 66 | +0.130 | +0.211 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O30 | 1.0 | NONE | 1707 | 66 | +0.093 | +0.069 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O30 | 1.5 | NONE | 1707 | 66 | +0.111 | -0.071 | 100.0% |
| non_deployed | MNQ | US_DATA_1000 | O30 | 2.0 | NONE | 1707 | 66 | +0.117 | -0.125 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O5 | 1.0 | NONE | 1718 | 66 | -0.029 | +0.029 | 100.0% |
| twin | MES | US_DATA_1000 | O5 | 1.5 | VWAP_MID_ALIGNED | 902 | 28 | -0.021 | +0.106 | 52.1% |
| non_deployed | MES | US_DATA_1000 | O5 | 2.0 | NONE | 1718 | 66 | +0.007 | +0.038 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O15 | 1.0 | NONE | 1715 | 66 | +0.022 | +0.156 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O15 | 1.5 | NONE | 1715 | 66 | +0.054 | +0.153 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O15 | 2.0 | NONE | 1715 | 66 | +0.054 | +0.005 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O30 | 1.0 | NONE | 1704 | 65 | +0.019 | -0.004 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O30 | 1.5 | NONE | 1704 | 65 | +0.047 | -0.043 | 100.0% |
| non_deployed | MES | US_DATA_1000 | O30 | 2.0 | NONE | 1704 | 65 | +0.047 | -0.072 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O5 | 1.0 | NONE | 917 | 66 | -0.037 | +0.039 | 100.0% |
| twin | MGC | US_DATA_1000 | O5 | 1.5 | VWAP_MID_ALIGNED | 489 | 33 | -0.068 | +0.167 | 53.0% |
| non_deployed | MGC | US_DATA_1000 | O5 | 2.0 | NONE | 917 | 66 | -0.036 | +0.007 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O15 | 1.0 | NONE | 911 | 66 | -0.014 | +0.146 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O15 | 1.5 | NONE | 911 | 66 | +0.022 | +0.072 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O15 | 2.0 | NONE | 911 | 66 | +0.025 | -0.033 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O30 | 1.0 | NONE | 890 | 65 | +0.009 | -0.008 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O30 | 1.5 | NONE | 890 | 65 | +0.005 | +0.004 | 100.0% |
| non_deployed | MGC | US_DATA_1000 | O30 | 2.0 | NONE | 890 | 65 | +0.020 | -0.021 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 1.0 | NONE | 1657 | 64 | +0.064 | +0.114 | 100.0% |
| deployed | MNQ | COMEX_SETTLE | O5 | 1.5 | OVNRNG_100 | 529 | 61 | +0.209 | +0.104 | 34.3% |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 2.0 | NONE | 1657 | 64 | +0.054 | +0.055 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 1.0 | NONE | 1656 | 63 | +0.039 | -0.011 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 1.5 | NONE | 1656 | 63 | +0.027 | -0.108 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 2.0 | NONE | 1656 | 63 | +0.011 | -0.006 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O30 | 1.0 | NONE | 1644 | 62 | +0.037 | +0.060 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O30 | 1.5 | NONE | 1644 | 62 | +0.008 | +0.169 | 100.0% |
| non_deployed | MNQ | COMEX_SETTLE | O30 | 2.0 | NONE | 1644 | 62 | +0.030 | +0.180 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O5 | 1.0 | NONE | 1657 | 64 | -0.097 | +0.020 | 100.0% |
| twin | MES | COMEX_SETTLE | O5 | 1.5 | OVNRNG_100 | 13 | 3 | +0.095 | +0.355 | 0.9% |
| non_deployed | MES | COMEX_SETTLE | O5 | 2.0 | NONE | 1657 | 64 | -0.092 | -0.148 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O15 | 1.0 | NONE | 1654 | 63 | -0.069 | +0.034 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O15 | 1.5 | NONE | 1654 | 63 | -0.084 | +0.002 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O15 | 2.0 | NONE | 1654 | 63 | -0.094 | +0.078 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | NONE | 1645 | 62 | -0.050 | -0.047 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.5 | NONE | 1645 | 62 | -0.041 | -0.001 | 100.0% |
| non_deployed | MES | COMEX_SETTLE | O30 | 2.0 | NONE | 1645 | 62 | -0.040 | +0.019 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O5 | 1.0 | NONE | 915 | 65 | -0.163 | -0.120 | 100.0% |
| twin | MGC | COMEX_SETTLE | O5 | 1.5 | OVNRNG_100 | 4 | 26 | +0.985 | -0.381 | 3.0% |
| non_deployed | MGC | COMEX_SETTLE | O5 | 2.0 | NONE | 915 | 65 | -0.196 | -0.268 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O15 | 1.0 | NONE | 905 | 65 | -0.153 | -0.092 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O15 | 1.5 | NONE | 905 | 65 | -0.176 | -0.060 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O15 | 2.0 | NONE | 905 | 65 | -0.167 | -0.062 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O30 | 1.0 | NONE | 885 | 65 | -0.123 | -0.031 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O30 | 1.5 | NONE | 885 | 65 | -0.135 | +0.055 | 100.0% |
| non_deployed | MGC | COMEX_SETTLE | O30 | 2.0 | NONE | 885 | 65 | -0.119 | +0.059 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.0 | NONE | 1642 | 63 | +0.077 | -0.022 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 1.5 | NONE | 1642 | 63 | +0.099 | -0.161 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O5 | 2.0 | NONE | 1642 | 63 | +0.120 | -0.141 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.0 | NONE | 1130 | 49 | +0.055 | +0.007 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 1.5 | NONE | 1130 | 49 | +0.068 | +0.053 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 2.0 | NONE | 1130 | 49 | +0.070 | +0.079 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O30 | 1.0 | NONE | 611 | 28 | +0.027 | +0.224 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O30 | 1.5 | NONE | 611 | 28 | +0.024 | +0.253 | 100.0% |
| non_deployed | MNQ | CME_PRECLOSE | O30 | 2.0 | NONE | 611 | 28 | +0.030 | +0.305 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.0 | NONE | 1632 | 64 | -0.018 | -0.175 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O5 | 1.5 | NONE | 1632 | 64 | -0.019 | -0.239 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O5 | 2.0 | NONE | 1632 | 64 | +0.000 | -0.227 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O15 | 1.0 | NONE | 1046 | 47 | -0.004 | -0.112 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O15 | 1.5 | NONE | 1046 | 47 | -0.000 | -0.109 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O15 | 2.0 | NONE | 1046 | 47 | +0.003 | -0.106 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O30 | 1.0 | NONE | 567 | 28 | -0.015 | +0.134 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O30 | 1.5 | NONE | 567 | 28 | -0.028 | +0.207 | 100.0% |
| non_deployed | MES | CME_PRECLOSE | O30 | 2.0 | NONE | 567 | 28 | -0.028 | +0.233 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.0 | NONE | 1436 | 62 | +0.030 | +0.380 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.5 | NONE | 1436 | 62 | +0.038 | +0.278 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 2.0 | NONE | 1436 | 62 | +0.032 | +0.297 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O15 | 1.0 | NONE | 967 | 48 | +0.009 | +0.197 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O15 | 1.5 | NONE | 967 | 48 | +0.011 | +0.244 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O15 | 2.0 | NONE | 967 | 48 | +0.018 | +0.239 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O30 | 1.0 | NONE | 753 | 33 | +0.007 | +0.199 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O30 | 1.5 | NONE | 753 | 33 | +0.002 | +0.218 | 100.0% |
| non_deployed | MNQ | NYSE_CLOSE | O30 | 2.0 | NONE | 753 | 33 | +0.004 | +0.184 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O5 | 1.0 | NONE | 1398 | 60 | -0.089 | -0.033 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O5 | 1.5 | NONE | 1398 | 60 | -0.098 | +0.002 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O5 | 2.0 | NONE | 1398 | 60 | -0.094 | +0.067 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O15 | 1.0 | NONE | 963 | 45 | -0.098 | +0.031 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O15 | 1.5 | NONE | 963 | 45 | -0.097 | +0.104 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O15 | 2.0 | NONE | 963 | 45 | -0.092 | +0.117 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O30 | 1.0 | NONE | 753 | 32 | -0.095 | +0.150 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O30 | 1.5 | NONE | 753 | 32 | -0.101 | +0.137 | 100.0% |
| non_deployed | MES | NYSE_CLOSE | O30 | 2.0 | NONE | 753 | 32 | -0.105 | +0.168 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.0 | NONE | 1721 | 67 | -0.044 | +0.178 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 1.5 | NONE | 1721 | 67 | -0.032 | +0.094 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 2.0 | NONE | 1721 | 67 | -0.009 | +0.187 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O15 | 1.0 | NONE | 1721 | 67 | +0.007 | +0.163 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O15 | 1.5 | NONE | 1721 | 67 | -0.019 | +0.241 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O15 | 2.0 | NONE | 1721 | 67 | +0.016 | +0.187 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O30 | 1.0 | NONE | 1717 | 67 | -0.011 | +0.291 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O30 | 1.5 | NONE | 1717 | 67 | -0.007 | +0.254 | 100.0% |
| non_deployed | MNQ | BRISBANE_1025 | O30 | 2.0 | NONE | 1717 | 67 | +0.005 | +0.083 | 100.0% |

## Scope

Pre-E2-look-ahead-fix preservation copy of `2026-04-15-comprehensive-deployed-lane-scan.md`.
Captures the dirty-scan result before the 2026-04-28 break-bar gating fix landed in
`research/comprehensive_deployed_lane_scan.py`. Retained per Backtesting Rule 11
(audit trail) — never delete prior results.

## Verdict

SUPERSEDED by `2026-04-15-comprehensive-deployed-lane-scan.md` (post-fix). Contains
break-bar-look-ahead-contaminated cells (verified 40-43% E2 entry_ts < break_ts on the
prior 3 actionable cells). Do NOT cite for any deployment or research-baseline decision.

## Reproduction

Not reproducible on current code — the underlying script
`research/comprehensive_deployed_lane_scan.py` was hardened on 2026-04-28 to gate
break-bar features on E2. Use git checkout `pre-2026-04-28` if exact replay is required.

## Caveats / limitations

- Every cell using rel_vol_*, bb_volume_ratio_*, break_delay_*, break_bar_continues_* is contaminated
- 11 of 28 BH_global survivors flipped on OOS — that flip rate WAS the look-ahead artifact's signature
- After fix: 0 OOS-flips on the rebuild
