# Garch W2e Prior-Session Carry Audit (V2 — broadened scope)

**Date:** 2026-04-16
**Revision:** V2 — scope broadened per user feedback (don't test one lane) + bootstrap null added per institutional rigor
**Boundary:** validated shelf only, prior trade must be fully resolved before target session start, descriptive-only (no deployment or allocator verdict)

## Scope

- Validated shelf rows considered: **26**
- Verified (filter SQL returns > 0 canonical rows): **25**
- Metadata-mismatch validated rows: **1**
- Prior × target × state × strategy cells tested: **262**
- Pooled (prior × target × state) groups: **36**
- Target sessions (validated W2 families): COMEX_SETTLE, EUROPE_FLOW, LONDON_METALS, SINGAPORE_OPEN, TOKYO_OPEN
- Prior candidates considered: BRISBANE_1025, CME_PRECLOSE, CME_REOPEN, COMEX_SETTLE, EUROPE_FLOW, LONDON_METALS, NYSE_CLOSE, NYSE_OPEN, SINGAPORE_OPEN, TOKYO_OPEN, US_DATA_1000, US_DATA_830
- Bootstrap iterations: 1000 (Phipson-Smyth, shuffled carry membership)
- Gates: `MIN_TOTAL=50`, `MIN_CONJ=30`, `GARCH_HIGH>=70.0`

### Structural limitation: validated shelf instrument coverage

The validated shelf in the 5 W2 target families is overwhelmingly MNQ.
T8 cross-instrument replication is blocked by this structural limitation,
NOT by a script bug. Any pooled handoff showing `supported_thin_single_instrument`
reflects this data reality. Until more MES/MGC strategies pass validation in
these sessions, single-instrument caveats are honest disclosures, not fixable.

## Metadata mismatches (validated-setups rows that did NOT match canonical data)

Per `integrity-guardian.md` RULE 7: metadata is not evidence. Any validated-setups row whose filter SQL returned zero rows in `orb_outcomes` is surfaced here.

| Strategy | Instrument | Target session | Filter | Reason |
|---|---|---|---|---|
| `GC_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_50` | GC | EUROPE_FLOW | OVNRNG_50 | filter_sql_returned_zero_canonical_rows |

## Supported pooled handoffs (conjunction beats all marginals AND bootstrap p ≤ 0.05 on at least one instrument)

| Prior → Target | State | Expected role | Cells | Instruments | Supporting | N total | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR | Δ conj-base | Δ conj-garch | Δ conj-carry | Verdict |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NYSE_OPEN → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | MNQ | 8999 | 872 | +0.108 | +0.250 | +0.210 | +0.378 | +0.270 | +0.128 | +0.168 | supported_thin_single_instrument |

## Unsupported pooled handoffs (enough data, but conjunction fails one or more marginals)

| Prior → Target | State | Expected role | Cells | Instruments | Supporting | N total | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR | Δ conj-base | Δ conj-garch | Δ conj-carry | Verdict |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BRISBANE_1025 → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 844 | +0.108 | +0.250 | +0.147 | +0.252 | +0.144 | +0.002 | +0.105 | unsupported |
| BRISBANE_1025 → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 921 | +0.108 | +0.250 | +0.112 | +0.209 | +0.101 | -0.041 | +0.097 | unsupported |
| CME_REOPEN → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 311 | +0.108 | +0.250 | +0.187 | +0.191 | +0.083 | -0.059 | +0.004 | unsupported |
| CME_REOPEN → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 309 | +0.108 | +0.250 | +0.157 | +0.360 | +0.252 | +0.110 | +0.203 | unsupported |
| EUROPE_FLOW → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 989 | +0.108 | +0.250 | +0.080 | +0.237 | +0.128 | -0.014 | +0.156 | unsupported |
| EUROPE_FLOW → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 929 | +0.108 | +0.250 | +0.115 | +0.280 | +0.172 | +0.030 | +0.165 | unsupported |
| LONDON_METALS → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 948 | +0.108 | +0.250 | +0.157 | +0.317 | +0.209 | +0.067 | +0.160 | unsupported |
| LONDON_METALS → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 797 | +0.108 | +0.250 | +0.100 | +0.229 | +0.121 | -0.021 | +0.129 | unsupported |
| NYSE_OPEN → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 851 | +0.108 | +0.250 | +0.065 | +0.233 | +0.125 | -0.017 | +0.168 | unsupported |
| SINGAPORE_OPEN → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 914 | +0.108 | +0.250 | +0.028 | +0.145 | +0.037 | -0.105 | +0.118 | unsupported |
| SINGAPORE_OPEN → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 950 | +0.108 | +0.250 | +0.164 | +0.325 | +0.217 | +0.075 | +0.161 | unsupported |
| TOKYO_OPEN → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 876 | +0.108 | +0.250 | +0.140 | +0.319 | +0.211 | +0.069 | +0.179 | unsupported |
| TOKYO_OPEN → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 1002 | +0.108 | +0.250 | +0.108 | +0.236 | +0.128 | -0.014 | +0.128 | unsupported |
| US_DATA_1000 → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 805 | +0.108 | +0.250 | +0.163 | +0.248 | +0.140 | -0.002 | +0.085 | unsupported |
| US_DATA_1000 → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 926 | +0.108 | +0.250 | +0.159 | +0.285 | +0.177 | +0.035 | +0.126 | unsupported |
| US_DATA_830 → COMEX_SETTLE | prior win align | take_pair | 9 | MNQ | — | 8999 | 904 | +0.108 | +0.250 | +0.170 | +0.278 | +0.169 | +0.027 | +0.108 | unsupported |
| US_DATA_830 → COMEX_SETTLE | prior win opposed | veto_pair | 9 | MNQ | — | 8999 | 798 | +0.108 | +0.250 | +0.083 | +0.333 | +0.225 | +0.083 | +0.250 | unsupported |
| BRISBANE_1025 → EUROPE_FLOW | prior win align | take_pair | 8 | MNQ | — | 8714 | 763 | +0.104 | +0.184 | +0.040 | +0.143 | +0.040 | -0.040 | +0.103 | unsupported |
| BRISBANE_1025 → EUROPE_FLOW | prior win opposed | veto_pair | 8 | MNQ | — | 8714 | 831 | +0.104 | +0.184 | +0.168 | +0.311 | +0.208 | +0.127 | +0.143 | unsupported |
| CME_REOPEN → EUROPE_FLOW | prior win align | take_pair | 8 | MNQ | — | 8714 | 240 | +0.104 | +0.184 | +0.072 | +0.028 | -0.076 | -0.156 | -0.044 | unsupported |
| CME_REOPEN → EUROPE_FLOW | prior win opposed | veto_pair | 8 | MNQ | — | 8714 | 297 | +0.104 | +0.184 | +0.118 | +0.354 | +0.250 | +0.170 | +0.235 | unsupported |
| LONDON_METALS → EUROPE_FLOW | prior win align | take_pair | 8 | MNQ | — | 8714 | 436 | +0.104 | +0.184 | +0.120 | +0.146 | +0.043 | -0.037 | +0.027 | unsupported |
| LONDON_METALS → EUROPE_FLOW | prior win opposed | veto_pair | 8 | MNQ | — | 8714 | 331 | +0.104 | +0.184 | +0.166 | +0.273 | +0.169 | +0.089 | +0.107 | unsupported |
| SINGAPORE_OPEN → EUROPE_FLOW | prior win align | take_pair | 8 | MNQ | — | 8714 | 786 | +0.104 | +0.184 | +0.033 | +0.120 | +0.017 | -0.063 | +0.087 | unsupported |
| SINGAPORE_OPEN → EUROPE_FLOW | prior win opposed | veto_pair | 8 | MNQ | — | 8714 | 912 | +0.104 | +0.184 | +0.102 | +0.168 | +0.064 | -0.016 | +0.066 | unsupported |
| TOKYO_OPEN → EUROPE_FLOW | prior win align | take_pair | 8 | MNQ | — | 8714 | 890 | +0.104 | +0.184 | +0.146 | +0.277 | +0.173 | +0.093 | +0.130 | unsupported |
| TOKYO_OPEN → EUROPE_FLOW | prior win opposed | veto_pair | 8 | MNQ | — | 8714 | 838 | +0.104 | +0.184 | +0.138 | +0.262 | +0.159 | +0.078 | +0.124 | unsupported |
| BRISBANE_1025 → SINGAPORE_OPEN | prior win align | take_pair | 2 | MNQ | — | 1655 | 164 | +0.118 | +0.129 | +0.004 | +0.006 | -0.111 | -0.122 | +0.003 | unsupported |
| BRISBANE_1025 → SINGAPORE_OPEN | prior win opposed | veto_pair | 2 | MNQ | — | 1655 | 190 | +0.118 | +0.129 | +0.126 | +0.150 | +0.032 | +0.021 | +0.023 | unsupported |
| CME_REOPEN → SINGAPORE_OPEN | prior win align | take_pair | 2 | MNQ | — | 1655 | 72 | +0.118 | +0.129 | +0.191 | -0.078 | -0.196 | -0.207 | -0.270 | unsupported |
| CME_REOPEN → SINGAPORE_OPEN | prior win opposed | veto_pair | 2 | MNQ | — | 1655 | 70 | +0.118 | +0.129 | +0.123 | +0.203 | +0.085 | +0.074 | +0.080 | unsupported |
| TOKYO_OPEN → SINGAPORE_OPEN | prior win align | take_pair | 2 | MNQ | — | 1655 | 196 | +0.118 | +0.129 | +0.159 | +0.143 | +0.026 | +0.014 | -0.016 | unsupported |
| TOKYO_OPEN → SINGAPORE_OPEN | prior win opposed | veto_pair | 2 | MNQ | — | 1655 | 209 | +0.118 | +0.129 | +0.210 | +0.240 | +0.123 | +0.111 | +0.030 | unsupported |

## Unclear pooled handoffs

| Prior → Target | State | Expected role | Cells | Instruments | Supporting | N total | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR | Δ conj-base | Δ conj-garch | Δ conj-carry | Verdict |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CME_REOPEN → TOKYO_OPEN | prior win align | take_pair | 4 | MNQ | — | 4756 | 70 | +0.087 | +0.135 | +0.162 | -0.008 | -0.096 | -0.143 | -0.171 | unclear |
| CME_REOPEN → TOKYO_OPEN | prior win opposed | veto_pair | 4 | MNQ | — | 4756 | 104 | +0.087 | +0.135 | -0.335 | -0.409 | -0.496 | -0.543 | -0.074 | unclear |

## Per-cell detail for supported handoffs

(Only pooled handoffs with at least one instrument-level supported verdict.)

| Prior → Target | Instrument | Strategy | Filter | RR | ORB | State | N | N resolved | N conj | Base ExpR | Garch ExpR | Carry ExpR | Conj ExpR | Δ conj-base | Boot p | dir_match | Cell verdict |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12` | COST_LT12 | 1.0 | 5 | prior win align | 1244 | 1215 | 111 | +0.102 | +0.207 | +0.193 | +0.327 | +0.224 | 0.057 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | ORB_G5 | 1.0 | 5 | prior win align | 1447 | 1406 | 116 | +0.079 | +0.210 | +0.165 | +0.328 | +0.249 | 0.056 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI` | ORB_G5_NOFRI | 1.0 | 5 | prior win align | 1161 | 1125 | 92 | +0.098 | +0.259 | +0.193 | +0.367 | +0.270 | 0.083 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` | OVNRNG_100 | 1.0 | 5 | prior win align | 526 | 514 | 68 | +0.148 | +0.284 | +0.217 | +0.394 | +0.246 | 0.095 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` | X_MES_ATR60 | 1.0 | 5 | prior win align | 633 | 621 | 102 | +0.144 | +0.235 | +0.231 | +0.328 | +0.184 | 0.098 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | ORB_G5 | 1.5 | 5 | prior win align | 1432 | 1391 | 111 | +0.100 | +0.254 | +0.219 | +0.416 | +0.316 | 0.039 | yes | supported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | OVNRNG_100 | 1.5 | 5 | prior win align | 518 | 506 | 65 | +0.189 | +0.365 | +0.313 | +0.532 | +0.343 | 0.072 | yes | unsupported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | X_MES_ATR60 | 1.5 | 5 | prior win align | 623 | 611 | 97 | +0.159 | +0.276 | +0.286 | +0.426 | +0.267 | 0.036 | yes | supported |
| NYSE_OPEN → COMEX_SETTLE | MNQ | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5` | ORB_G5 | 2.0 | 5 | prior win align | 1415 | 1375 | 110 | +0.078 | +0.216 | +0.192 | +0.357 | +0.279 | 0.114 | yes | unsupported |

## Guardrails and honest caveats

- **Chronology:** every row is kept only if the prior trade's `exit_ts` is strictly before the target session start timestamp from `pipeline.dst.orb_utc_window`. No static session-order shortcut.
- **Canonical sources:** all statistics are computed against `orb_outcomes` + `daily_features`. `validated_setups` is used ONLY to enumerate candidate target cells; each row is independently verified against canonical data, and mismatches are surfaced above.
- **Bootstrap null:** `BOOTSTRAP_ITERS=1000` shuffles of carry-state membership among the same population, Phipson-Smyth p. This is a descriptive null floor — it does NOT replace a cross-instrument or cross-era check.
- **Family gate:** a pooled handoff × state is `supported_multi_instrument` only if at least 2 instruments each had a cell-level supported verdict. Single-instrument support is flagged `supported_thin_single_instrument` and should not be treated as universal.
- **Holdout:** 2026-01-01 boundary is respected. This audit uses the full shelf including post-holdout rows — descriptive only, not promotion.
- **No deployment doctrine** is derivable from this audit. It is a state-family distinctness check, not a size/route/allocator claim.
