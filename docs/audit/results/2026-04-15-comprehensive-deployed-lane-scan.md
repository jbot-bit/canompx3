# Comprehensive Scan — ALL Sessions × ALL Instruments × ALL Apertures × ALL RRs

**Date:** 2026-04-15
**Total cells scanned:** 13415
**Trustworthy cells** (not extreme-fire, not tautology, not arithmetic-only): 12633
**Strict survivors** (|t|>=3 + dir_match + N>=50 + trustworthy): 42

## BH-FDR pass counts at each K framing

- **K_global** (K=13415) strictest: 5 pass
- **K_family** (within feature-family, avg K~3328): 8 pass
- **K_lane** (within session+apt+rr+instr, avg K~50): 87 pass
- **K_session** (within session across instruments, avg K~1228): 22 pass
- **K_instrument** (within instrument, avg K~4582): 7 pass
- **K_feature** (within feature across lanes, avg K~524): 28 pass

**Promising** (|t|>=2.5 + dir_match + N>=50 + trustworthy): 153

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
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | short | pit_range_atr_HIGH | volatility | unfiltered | 121 | 32.3% | -0.009 | +0.126 | +0.271 | +0.165 | +4.09 | 0.0001 | . | Y |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_LONG_INTERACT | overnight | unfiltered | 248 | 28.8% | -0.188 | -0.131 | -0.254 | -0.070 | -4.08 | 0.0001 | . | . |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_TRUE | overnight | unfiltered | 248 | 28.8% | -0.188 | -0.131 | -0.254 | -0.070 | -4.08 | 0.0001 | . | . |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | short | pit_range_atr_LOW | volatility | unfiltered | 240 | 32.4% | -0.221 | -0.127 | -0.303 | -0.088 | -4.04 | 0.0001 | . | Y |
| non_deployed | MES | LONDON_METALS | O30 | 1.5 | long | ovn_range_pct_GT80 | overnight | unfiltered | 183 | 20.9% | +0.231 | +0.132 | +0.338 | +0.690 | +3.58 | 0.0004 | . | . |
| non_deployed | MES | US_DATA_830 | O5 | 2.0 | long | near_session_london_high | level | unfiltered | 399 | 46.2% | -0.260 | -0.103 | -0.279 | -0.031 | -3.54 | 0.0004 | . | . |
| non_deployed | MES | EUROPE_FLOW | O15 | 1.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 186 | 21.1% | +0.143 | +0.101 | +0.246 | +0.334 | +3.47 | 0.0006 | . | . |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | long | near_session_london_high | level | unfiltered | 399 | 46.2% | -0.249 | -0.089 | -0.239 | -0.260 | -3.47 | 0.0006 | . | . |
| non_deployed | MES | TOKYO_OPEN | O5 | 1.0 | long | atr_vel_HIGH | volatility | unfiltered | 281 | 33.7% | +0.016 | +0.104 | +0.197 | +0.374 | +3.41 | 0.0007 | . | . |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | atr_vel_LOW | volatility | unfiltered | 291 | 33.4% | -0.103 | -0.113 | -0.265 | -0.450 | -3.40 | 0.0007 | . | . |
| non_deployed | MGC | COMEX_SETTLE | O30 | 1.0 | long | near_session_london_low | level | unfiltered | 119 | 26.9% | -0.278 | -0.163 | -0.256 | -0.140 | -3.37 | 0.0009 | . | . |
| non_deployed | MGC | TOKYO_OPEN | O5 | 1.0 | short | pit_range_atr_HIGH | volatility | unfiltered | 148 | 33.6% | -0.081 | +0.109 | +0.260 | +0.769 | +3.35 | 0.0009 | . | . |
| non_deployed | MNQ | LONDON_METALS | O15 | 2.0 | short | atr_vel_LOW | volatility | unfiltered | 264 | 31.6% | -0.223 | -0.111 | -0.323 | -0.350 | -3.34 | 0.0009 | . | . |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 1.0 | short | dow_thu | calendar | unfiltered | 162 | 20.8% | +0.246 | +0.132 | +0.265 | +0.552 | +3.33 | 0.0010 | . | . |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | short | ovn_range_pct_GT80 | overnight | unfiltered | 170 | 20.3% | +0.146 | +0.125 | +0.300 | +0.037 | +3.33 | 0.0010 | . | . |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 2.0 | short | dow_thu | calendar | unfiltered | 162 | 20.8% | +0.307 | +0.129 | +0.389 | +0.360 | +3.30 | 0.0011 | . | . |
| non_deployed | MNQ | NYSE_OPEN | O5 | 2.0 | long | atr_20_pct_GT80 | volatility | unfiltered | 204 | 24.8% | -0.146 | -0.133 | -0.345 | -0.240 | -3.29 | 0.0011 | . | . |
| non_deployed | MGC | EUROPE_FLOW | O30 | 2.0 | short | near_session_asia_high | level | unfiltered | 167 | 38.3% | -0.303 | -0.144 | -0.384 | -0.236 | -3.28 | 0.0011 | . | . |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | long | atr_vel_LOW | volatility | unfiltered | 291 | 33.4% | -0.104 | -0.105 | -0.203 | -0.432 | -3.25 | 0.0012 | . | . |
| non_deployed | MNQ | BRISBANE_1025 | O30 | 2.0 | long | is_monday_TRUE | calendar | unfiltered | 192 | 20.6% | +0.338 | +0.129 | +0.361 | +0.854 | +3.23 | 0.0014 | . | . |
| non_deployed | MES | COMEX_SETTLE | O30 | 2.0 | long | ovn_took_pdh_LONG_INTERACT | overnight | unfiltered | 248 | 28.8% | -0.194 | -0.107 | -0.256 | -0.184 | -3.22 | 0.0014 | . | . |
| non_deployed | MES | COMEX_SETTLE | O30 | 2.0 | long | ovn_took_pdh_TRUE | overnight | unfiltered | 248 | 28.8% | -0.194 | -0.107 | -0.256 | -0.184 | -3.22 | 0.0014 | . | . |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 2.0 | long | near_session_london_low | level | unfiltered | 192 | 21.9% | -0.158 | -0.120 | -0.333 | -0.460 | -3.21 | 0.0015 | . | . |
| non_deployed | MES | COMEX_SETTLE | O15 | 2.0 | long | ovn_took_pdh_TRUE | overnight | unfiltered | 254 | 29.7% | -0.272 | -0.107 | -0.265 | -0.084 | -3.19 | 0.0015 | . | . |
| non_deployed | MES | COMEX_SETTLE | O15 | 2.0 | long | ovn_took_pdh_LONG_INTERACT | overnight | unfiltered | 254 | 29.7% | -0.272 | -0.107 | -0.265 | -0.084 | -3.19 | 0.0015 | . | . |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 160 | 22.1% | +0.219 | +0.152 | +0.203 | +0.434 | +3.19 | 0.0016 | . | . |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 1.0 | long | garch_vol_pct_GT70 | volatility | unfiltered | 199 | 23.6% | +0.245 | +0.088 | +0.229 | +0.227 | +3.18 | 0.0016 | . | . |
| non_deployed | MES | EUROPE_FLOW | O5 | 1.0 | short | garch_vol_pct_GT70 | volatility | unfiltered | 184 | 22.8% | -0.003 | +0.090 | +0.217 | +0.158 | +3.14 | 0.0019 | . | . |
| non_deployed | MES | LONDON_METALS | O30 | 1.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 183 | 20.9% | +0.121 | +0.109 | +0.234 | +0.330 | +3.12 | 0.0020 | . | . |
| non_deployed | MES | EUROPE_FLOW | O5 | 2.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 180 | 21.5% | +0.079 | +0.101 | +0.322 | +0.134 | +3.12 | 0.0020 | . | . |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | atr_vel_HIGH | volatility | unfiltered | 288 | 32.6% | -0.094 | +0.092 | +0.172 | +0.580 | +3.11 | 0.0020 | . | . |
| non_deployed | MES | LONDON_METALS | O30 | 2.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 183 | 20.9% | +0.242 | +0.117 | +0.344 | +0.828 | +3.10 | 0.0021 | . | . |
| non_deployed | MGC | LONDON_METALS | O30 | 2.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 106 | 23.5% | +0.365 | +0.165 | +0.447 | +0.092 | +3.09 | 0.0023 | . | . |
| non_deployed | MGC | LONDON_METALS | O5 | 2.0 | short | ovn_range_pct_GT80 | overnight | unfiltered | 101 | 23.7% | +0.185 | +0.139 | +0.418 | +0.362 | +3.08 | 0.0025 | . | . |
| non_deployed | MGC | US_DATA_1000 | O15 | 2.0 | short | pre_velocity_HIGH | timing | unfiltered | 129 | 30.3% | -0.215 | -0.157 | -0.339 | -0.134 | -3.07 | 0.0024 | . | . |
| twin | MES | EUROPE_FLOW | O5 | 1.5 | short | pit_range_atr_HIGH | volatility | unfiltered | 280 | 32.8% | -0.073 | +0.091 | +0.222 | +0.176 | +3.06 | 0.0023 | . | . |
| non_deployed | MES | NYSE_OPEN | O30 | 1.0 | long | is_friday_TRUE | calendar | unfiltered | 171 | 18.7% | +0.199 | +0.131 | +0.218 | +0.257 | +3.05 | 0.0025 | . | . |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 2.0 | long | ovn_range_pct_GT80 | overnight | unfiltered | 118 | 19.3% | +0.260 | +0.133 | +0.201 | +0.202 | +3.04 | 0.0027 | . | . |
| non_deployed | MNQ | CME_REOPEN | O30 | 2.0 | long | atr_20_pct_GT80 | volatility | unfiltered | 121 | 27.2% | -0.243 | -0.133 | -0.254 | -0.641 | -3.04 | 0.0026 | . | . |
| non_deployed | MES | TOKYO_OPEN | O15 | 2.0 | long | garch_vol_pct_LT30 | volatility | unfiltered | 312 | 35.3% | -0.194 | -0.094 | -0.261 | -0.847 | -3.04 | 0.0025 | . | . |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 2.0 | long | atr_vel_LOW | volatility | unfiltered | 291 | 33.4% | -0.087 | -0.098 | -0.276 | -0.429 | -3.02 | 0.0026 | . | . |
| non_deployed | MNQ | BRISBANE_1025 | O5 | 2.0 | long | pre_velocity_HIGH | timing | unfiltered | 277 | 30.3% | +0.170 | +0.088 | +0.271 | +0.304 | +3.00 | 0.0028 | . | . |

## BH-FDR Survivors — Global (q=0.05)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Pass | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p | BH_crit |
|-------|-------|---------|-----|----|----|---------|------|------|-------|---------|------|-------|---|---|---------|
| non_deployed | MGC | CME_REOPEN | O5 | 2.0 | short | pit_range_atr_LOW | unfiltered | 137 | 32.8% | -0.592 | -0.452 | +nan | -6.32 | 0.00000 | 0.00000 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.5 | short | pit_range_atr_LOW | unfiltered | 137 | 32.8% | -0.525 | -0.391 | +nan | -5.64 | 0.00000 | 0.00001 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | short | pit_range_atr_LOW | unfiltered | 137 | 32.8% | -0.421 | -0.345 | +nan | -5.35 | 0.00000 | 0.00001 |
| non_deployed | MGC | CME_REOPEN | O15 | 1.0 | long | atr_20_pct_LT20 | unfiltered | 43 | 12.0% | -0.402 | -0.298 | +nan | -4.92 | 0.00000 | 0.00001 |
| non_deployed | MGC | CME_REOPEN | O30 | 1.0 | long | atr_20_pct_LT20 | unfiltered | 31 | 13.1% | -0.382 | -0.326 | +nan | -4.73 | 0.00001 | 0.00002 |

## BH-FDR Survivors — Per-Family (q=0.05 within family)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Family | Pass | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p |
|-------|-------|---------|-----|----|----|---------|--------|------|------|-------|---------|------|-------|---|---|
| non_deployed | MGC | CME_REOPEN | O5 | 2.0 | short | pit_range_atr_LOW | volatility | unfiltered | 137 | 32.8% | -0.592 | -0.452 | +nan | -6.32 | 0.00000 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.5 | short | pit_range_atr_LOW | volatility | unfiltered | 137 | 32.8% | -0.525 | -0.391 | +nan | -5.64 | 0.00000 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | short | pit_range_atr_LOW | volatility | unfiltered | 137 | 32.8% | -0.421 | -0.345 | +nan | -5.35 | 0.00000 |
| non_deployed | MGC | CME_REOPEN | O15 | 1.0 | long | atr_20_pct_LT20 | volatility | unfiltered | 43 | 12.0% | -0.402 | -0.298 | +nan | -4.92 | 0.00000 |
| non_deployed | MGC | CME_REOPEN | O30 | 1.0 | long | atr_20_pct_LT20 | volatility | unfiltered | 31 | 13.1% | -0.382 | -0.326 | +nan | -4.73 | 0.00001 |
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | short | pit_range_atr_HIGH | volatility | unfiltered | 121 | 32.3% | -0.009 | +0.271 | +0.165 | +4.09 | 0.00006 |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | short | pit_range_atr_LOW | volatility | unfiltered | 240 | 32.4% | -0.221 | -0.303 | -0.088 | -4.04 | 0.00006 |
| non_deployed | MES | EUROPE_FLOW | O5 | 2.0 | long | pit_range_atr_LOW | volatility | unfiltered | 283 | 32.8% | -0.384 | -0.314 | +0.253 | -4.02 | 0.00007 |

## Promising cells (candidates for next-round T0-T8)

| Scope | Instr | Session | Apt | RR | Dir | Feature | Pass | N_on | ExpR_on | WR_Δ | Δ_IS | Δ_OOS | t | p |
|-------|-------|---------|-----|----|----|---------|------|------|---------|------|------|-------|---|---|
| non_deployed | MGC | CME_REOPEN | O5 | 1.0 | short | pit_range_atr_HIGH | unfiltered | 121 | -0.009 | +0.126 | +0.271 | +0.165 | +4.09 | 0.0001 |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_LONG_INTERACT | unfiltered | 248 | -0.188 | -0.131 | -0.254 | -0.070 | -4.08 | 0.0001 |
| non_deployed | MES | COMEX_SETTLE | O30 | 1.0 | long | ovn_took_pdh_TRUE | unfiltered | 248 | -0.188 | -0.131 | -0.254 | -0.070 | -4.08 | 0.0001 |
| non_deployed | MNQ | CME_REOPEN | O5 | 2.0 | short | pit_range_atr_LOW | unfiltered | 240 | -0.221 | -0.127 | -0.303 | -0.088 | -4.04 | 0.0001 |
| non_deployed | MES | LONDON_METALS | O30 | 1.5 | long | ovn_range_pct_GT80 | unfiltered | 183 | +0.231 | +0.132 | +0.338 | +0.690 | +3.58 | 0.0004 |
| non_deployed | MES | US_DATA_830 | O5 | 2.0 | long | near_session_london_high | unfiltered | 399 | -0.260 | -0.103 | -0.279 | -0.031 | -3.54 | 0.0004 |
| non_deployed | MES | EUROPE_FLOW | O15 | 1.0 | long | ovn_range_pct_GT80 | unfiltered | 186 | +0.143 | +0.101 | +0.246 | +0.334 | +3.47 | 0.0006 |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | long | near_session_london_high | unfiltered | 399 | -0.249 | -0.089 | -0.239 | -0.260 | -3.47 | 0.0006 |
| non_deployed | MES | TOKYO_OPEN | O5 | 1.0 | long | atr_vel_HIGH | unfiltered | 281 | +0.016 | +0.104 | +0.197 | +0.374 | +3.41 | 0.0007 |
| deployed | MNQ | TOKYO_OPEN | O5 | 1.5 | long | atr_vel_LOW | unfiltered | 291 | -0.103 | -0.113 | -0.265 | -0.450 | -3.40 | 0.0007 |
| non_deployed | MGC | COMEX_SETTLE | O30 | 1.0 | long | near_session_london_low | unfiltered | 119 | -0.278 | -0.163 | -0.256 | -0.140 | -3.37 | 0.0009 |
| non_deployed | MGC | TOKYO_OPEN | O5 | 1.0 | short | pit_range_atr_HIGH | unfiltered | 148 | -0.081 | +0.109 | +0.260 | +0.769 | +3.35 | 0.0009 |
| non_deployed | MNQ | LONDON_METALS | O15 | 2.0 | short | atr_vel_LOW | unfiltered | 264 | -0.223 | -0.111 | -0.323 | -0.350 | -3.34 | 0.0009 |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 1.0 | short | dow_thu | unfiltered | 162 | +0.246 | +0.132 | +0.265 | +0.552 | +3.33 | 0.0010 |
| non_deployed | MES | US_DATA_830 | O5 | 1.5 | short | ovn_range_pct_GT80 | unfiltered | 170 | +0.146 | +0.125 | +0.300 | +0.037 | +3.33 | 0.0010 |
| non_deployed | MNQ | COMEX_SETTLE | O15 | 2.0 | short | dow_thu | unfiltered | 162 | +0.307 | +0.129 | +0.389 | +0.360 | +3.30 | 0.0011 |
| non_deployed | MNQ | NYSE_OPEN | O5 | 2.0 | long | atr_20_pct_GT80 | unfiltered | 204 | -0.146 | -0.133 | -0.345 | -0.240 | -3.29 | 0.0011 |
| non_deployed | MGC | EUROPE_FLOW | O30 | 2.0 | short | near_session_asia_high | unfiltered | 167 | -0.303 | -0.144 | -0.384 | -0.236 | -3.28 | 0.0011 |
| non_deployed | MNQ | TOKYO_OPEN | O5 | 1.0 | long | atr_vel_LOW | unfiltered | 291 | -0.104 | -0.105 | -0.203 | -0.432 | -3.25 | 0.0012 |
| non_deployed | MNQ | BRISBANE_1025 | O30 | 2.0 | long | is_monday_TRUE | unfiltered | 192 | +0.338 | +0.129 | +0.361 | +0.854 | +3.23 | 0.0014 |
| non_deployed | MES | COMEX_SETTLE | O30 | 2.0 | long | ovn_took_pdh_LONG_INTERACT | unfiltered | 248 | -0.194 | -0.107 | -0.256 | -0.184 | -3.22 | 0.0014 |
| non_deployed | MES | COMEX_SETTLE | O30 | 2.0 | long | ovn_took_pdh_TRUE | unfiltered | 248 | -0.194 | -0.107 | -0.256 | -0.184 | -3.22 | 0.0014 |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 2.0 | long | near_session_london_low | unfiltered | 192 | -0.158 | -0.120 | -0.333 | -0.460 | -3.21 | 0.0015 |
| non_deployed | MES | COMEX_SETTLE | O15 | 2.0 | long | ovn_took_pdh_TRUE | unfiltered | 254 | -0.272 | -0.107 | -0.265 | -0.084 | -3.19 | 0.0015 |
| non_deployed | MES | COMEX_SETTLE | O15 | 2.0 | long | ovn_took_pdh_LONG_INTERACT | unfiltered | 254 | -0.272 | -0.107 | -0.265 | -0.084 | -3.19 | 0.0015 |
| non_deployed | MNQ | NYSE_CLOSE | O5 | 1.0 | long | ovn_range_pct_GT80 | unfiltered | 160 | +0.219 | +0.152 | +0.203 | +0.434 | +3.19 | 0.0016 |
| non_deployed | MNQ | COMEX_SETTLE | O5 | 1.0 | long | garch_vol_pct_GT70 | unfiltered | 199 | +0.245 | +0.088 | +0.229 | +0.227 | +3.18 | 0.0016 |
| non_deployed | MES | EUROPE_FLOW | O5 | 1.0 | short | garch_vol_pct_GT70 | unfiltered | 184 | -0.003 | +0.090 | +0.217 | +0.158 | +3.14 | 0.0019 |
| non_deployed | MES | LONDON_METALS | O30 | 1.0 | long | ovn_range_pct_GT80 | unfiltered | 183 | +0.121 | +0.109 | +0.234 | +0.330 | +3.12 | 0.0020 |
| non_deployed | MES | EUROPE_FLOW | O5 | 2.0 | long | ovn_range_pct_GT80 | unfiltered | 180 | +0.079 | +0.101 | +0.322 | +0.134 | +3.12 | 0.0020 |
| non_deployed | MES | SINGAPORE_OPEN | O5 | 1.0 | long | atr_vel_HIGH | unfiltered | 288 | -0.094 | +0.092 | +0.172 | +0.580 | +3.11 | 0.0020 |
| non_deployed | MES | LONDON_METALS | O30 | 2.0 | long | ovn_range_pct_GT80 | unfiltered | 183 | +0.242 | +0.117 | +0.344 | +0.828 | +3.10 | 0.0021 |
| non_deployed | MGC | LONDON_METALS | O30 | 2.0 | long | ovn_range_pct_GT80 | unfiltered | 106 | +0.365 | +0.165 | +0.447 | +0.092 | +3.09 | 0.0023 |
| non_deployed | MGC | LONDON_METALS | O5 | 2.0 | short | ovn_range_pct_GT80 | unfiltered | 101 | +0.185 | +0.139 | +0.418 | +0.362 | +3.08 | 0.0025 |
| non_deployed | MGC | US_DATA_1000 | O15 | 2.0 | short | pre_velocity_HIGH | unfiltered | 129 | -0.215 | -0.157 | -0.339 | -0.134 | -3.07 | 0.0024 |
| twin | MES | EUROPE_FLOW | O5 | 1.5 | short | pit_range_atr_HIGH | unfiltered | 280 | -0.073 | +0.091 | +0.222 | +0.176 | +3.06 | 0.0023 |
| non_deployed | MES | NYSE_OPEN | O30 | 1.0 | long | is_friday_TRUE | unfiltered | 171 | +0.199 | +0.131 | +0.218 | +0.257 | +3.05 | 0.0025 |
| non_deployed | MNQ | CME_PRECLOSE | O15 | 2.0 | long | ovn_range_pct_GT80 | unfiltered | 118 | +0.260 | +0.133 | +0.201 | +0.202 | +3.04 | 0.0027 |
| non_deployed | MNQ | CME_REOPEN | O30 | 2.0 | long | atr_20_pct_GT80 | unfiltered | 121 | -0.243 | -0.133 | -0.254 | -0.641 | -3.04 | 0.0026 |
| non_deployed | MES | TOKYO_OPEN | O15 | 2.0 | long | garch_vol_pct_LT30 | unfiltered | 312 | -0.194 | -0.094 | -0.261 | -0.847 | -3.04 | 0.0025 |

## Flagged cells (excluded despite |t|>=3) — for transparency

Excluded because: tautology |corr|>0.7 (0), extreme fire <5% or >95% (533), arithmetic-only WR flat (260)

| Scope | Instr | Session | Feature | Pass | t | Fire% | T0 corr | Reason |
|-------|-------|---------|---------|------|---|-------|---------|--------|
| non_deployed | MNQ | EUROPE_FLOW | is_nfp_TRUE | unfiltered | -4.93 | 4.2% | 0.00 | FIRE(4.2%) |
| deployed | MNQ | EUROPE_FLOW | is_nfp_TRUE | filtered | -3.55 | 4.0% | 0.00 | FIRE(4.0%) |
| non_deployed | MNQ | COMEX_SETTLE | is_nfp_TRUE | unfiltered | -3.51 | 4.9% | 0.00 | FIRE(4.9%) |
| deployed | MNQ | EUROPE_FLOW | is_nfp_TRUE | unfiltered | -3.49 | 4.2% | 0.04 | FIRE(4.2%) |
| non_deployed | MGC | CME_REOPEN | pit_range_atr_LOW | unfiltered | -3.47 | 33.9% | 0.00 | ARITH(WR_Δ=-0.007) |
| non_deployed | MNQ | COMEX_SETTLE | is_nfp_TRUE | unfiltered | -3.35 | 4.9% | 0.00 | FIRE(4.9%) |
| non_deployed | MES | NYSE_CLOSE | near_session_london_high | unfiltered | -3.16 | 20.4% | 0.00 | ARITH(WR_Δ=-0.029) |
| non_deployed | MGC | CME_REOPEN | pit_range_atr_LOW | unfiltered | -3.14 | 32.6% | 0.00 | ARITH(WR_Δ=+0.001) |
| non_deployed | MNQ | COMEX_SETTLE | is_nfp_TRUE | unfiltered | -3.10 | 4.9% | 0.00 | FIRE(4.9%) |

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

## Verdict

Discovery scan (Pathway A). Output is research-survivor classification, not deploy-ready.
Strict survivors and BH_global / BH_family survivors are inputs to a separate Pathway B
K=1 confirmatory pre-reg per `docs/institutional/pre_registered_criteria.md` Amendment 3.0.
No survivor advances to deployment without C5 DSR + C6 WFE + C8 dir-match + C9 era-stability
+ C11 MC + C12 Shiryaev-Roberts checks. Promotion direct from this table to capital is forbidden.

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/comprehensive_deployed_lane_scan.py
```

- DB path: `pipeline.paths.GOLD_DB_PATH`
- Holdout: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` (2026-01-01)
- Scratch policy: scratch as 0R via `pnl_r IS NOT NULL` (rebuilt orb_outcomes Stage 5 populates ~99.7% active-instrument scratches)
- E2 break-bar look-ahead gate: per `trading_app/config.py:3857` + postmortem `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md`
- BH-FDR multi-K framings: K_global, K_family, K_lane, K_session, K_instrument, K_feature

## Caveats / limitations

- **No BH_global pass does NOT mean no signal.** K_family / K_lane is the legitimate cut for Pathway B with theory citation per Chordia HLZ-tier (t >= 3.00).
- **OOS power floor:** OOS sample is ~3 months; N_oos<50 → UNVERIFIED on dir-match, not failed.
- **Bucket-quantile leak:** `pre_velocity_*` and `atr_vel_*` use full-series 67/33 percentile (includes OOS). Hard-threshold features are unaffected.
- **Instrument-family discipline:** MGC ≠ MNQ/MES regime; per RULE 14 pooled findings need flip-rate disclosure.
- **Pathway A ≠ deployable.** Promotion requires Pathway B K=1 + DSR + WFE + lane-correlation + MC + Shiryaev-Roberts.

## Not done by this scan

- No DSR (C5 Bailey-LdP Eq.9), WFE (C6), lane-correlation matrix
- No 90-day account-death MC (C11), Shiryaev-Roberts monitor (C12)
- No vol-conditional-RR test (Carver Ch9-10)
- No direction-split per session
