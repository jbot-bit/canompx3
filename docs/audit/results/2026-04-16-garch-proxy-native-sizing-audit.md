# Garch Proxy-Native Sizing Audit

**Date:** 2026-04-16
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-proxy-native-sizing-audit.yaml`
**Purpose:** compare garch to related vol proxies after letting each score earn its own session-family scaffold under the same locked family method.

## Native scaffolds

| Map | Native session scaffold |
|---|---|
| GARCH_SESSION_CLIPPED | BRISBANE_1025(H), CME_PRECLOSE(HL), COMEX_SETTLE(HLM), EUROPE_FLOW(HLM), LONDON_METALS(HLM), SINGAPORE_OPEN(HLM), TOKYO_OPEN(HLM), US_DATA_1000(H) |
| ATR_SESSION_CLIPPED | CME_PRECLOSE(M), COMEX_SETTLE(HLM), NYSE_CLOSE(M), SINGAPORE_OPEN(HL), TOKYO_OPEN(HLM) |
| OVN_SESSION_CLIPPED | BRISBANE_1025(HLM), CME_REOPEN(HLM), COMEX_SETTLE(HLM), EUROPE_FLOW(HLM), LONDON_METALS(LM), NYSE_CLOSE(M), NYSE_OPEN(H), SINGAPORE_OPEN(HLM), TOKYO_OPEN(HLM), US_DATA_1000(HM) |
| GARCH_ATR_MEAN_CLIPPED | CME_PRECLOSE(HLM), COMEX_SETTLE(HLM), LONDON_METALS(H), SINGAPORE_OPEN(H), TOKYO_OPEN(HLM) |
| GARCH_OVN_MEAN_CLIPPED | BRISBANE_1025(HL), CME_PRECLOSE(L), CME_REOPEN(LM), COMEX_SETTLE(HLM), EUROPE_FLOW(HLM), LONDON_METALS(HLM), SINGAPORE_OPEN(HLM), TOKYO_OPEN(HLM), US_DATA_1000(HM) |

## Broad scope

| Map | Full Δ$ | Full ΔR | Sharpe Δ | MaxDD ΔR | Worst day Δ$ | Worst 5d Δ$ | Max daily risk Δ$ | IS ExpR Δ | OOS ExpR Δ | OOS retention |
|---|---|---|---|---|---|---|---|---|---|---|
| OVN_SESSION_CLIPPED | +223166.4 | +3420.9 | +0.542 | +63.1 | -3385.3 | -4849.8 | +24509.3 | +0.0190 | +0.0164 | +0.86 |
| GARCH_OVN_MEAN_CLIPPED | +178571.9 | +2629.6 | +0.415 | +86.8 | -2209.5 | -3141.6 | +26646.1 | +0.0145 | +0.0141 | +0.98 |
| GARCH_SESSION_CLIPPED | +166854.2 | +2104.4 | +0.240 | +56.0 | -2828.6 | -2148.2 | +27249.3 | +0.0112 | +0.0166 | +1.48 |
| GARCH_ATR_MEAN_CLIPPED | +93873.5 | +1069.0 | +0.126 | +19.7 | -1951.9 | -3322.7 | +9407.4 | +0.0064 | -0.0021 | -0.33 |
| ATR_SESSION_CLIPPED | +78504.7 | +660.2 | +0.022 | -16.9 | -1051.8 | -1504.2 | +7204.9 | +0.0034 | +0.0064 | +1.86 |

### Broad best-map contributions: `OVN_SESSION_CLIPPED`

| Instrument | Session | Δ$ |
|---|---|---|
| MNQ | TOKYO_OPEN | +52549.4 |
| MNQ | COMEX_SETTLE | +31408.8 |
| MNQ | SINGAPORE_OPEN | +29506.8 |
| MNQ | EUROPE_FLOW | +24568.9 |
| MNQ | US_DATA_1000 | +20541.8 |
| MNQ | NYSE_OPEN | +18477.1 |
| MNQ | BRISBANE_1025 | +11318.8 |
| MNQ | CME_REOPEN | +7730.3 |
| MES | NYSE_OPEN | +7644.8 |
| MES | COMEX_SETTLE | +7233.9 |
| MES | TOKYO_OPEN | +4380.5 |
| MES | US_DATA_1000 | +3637.8 |
| MNQ | US_DATA_830 | +3179.0 |
| MES | EUROPE_FLOW | +3129.6 |
| MGC | CME_REOPEN | +2117.1 |

## Validated scope

| Map | Full Δ$ | Full ΔR | Sharpe Δ | MaxDD ΔR | Worst day Δ$ | Worst 5d Δ$ | Max daily risk Δ$ | IS ExpR Δ | OOS ExpR Δ | OOS retention |
|---|---|---|---|---|---|---|---|---|---|---|
| OVN_SESSION_CLIPPED | +56128.7 | +802.1 | +0.378 | +42.0 | -1917.2 | -1652.4 | +6982.2 | +0.0219 | +0.0113 | +0.52 |
| GARCH_SESSION_CLIPPED | +54293.4 | +672.4 | +0.190 | +50.3 | -2536.9 | -562.3 | +7986.1 | +0.0156 | +0.0583 | +3.75 |
| GARCH_OVN_MEAN_CLIPPED | +49074.5 | +676.9 | +0.253 | +43.2 | -2320.9 | -690.1 | +7657.3 | +0.0169 | +0.0367 | +2.17 |
| GARCH_ATR_MEAN_CLIPPED | +30508.0 | +355.3 | +0.079 | +16.0 | -1224.2 | -893.2 | +2141.5 | +0.0092 | +0.0134 | +1.45 |
| ATR_SESSION_CLIPPED | +25182.3 | +260.9 | -0.006 | -26.4 | -980.9 | -627.8 | +1732.9 | +0.0066 | +0.0136 | +2.08 |

### Validated best-map contributions: `OVN_SESSION_CLIPPED`

| Instrument | Session | Δ$ |
|---|---|---|
| MNQ | COMEX_SETTLE | +14089.3 |
| MNQ | TOKYO_OPEN | +13568.6 |
| MNQ | EUROPE_FLOW | +10709.5 |
| MNQ | SINGAPORE_OPEN | +6482.4 |
| MNQ | US_DATA_1000 | +6265.2 |
| MNQ | NYSE_OPEN | +5998.0 |
| MES | CME_PRECLOSE | -460.8 |
| MNQ | CME_PRECLOSE | -523.4 |

## Reading the audit

- This is the fair proxy-native counterpart to the common-scaffold additive audit.
- Positive `MaxDD ΔR` means drawdown became less severe.
- Positive `Worst day/5d Δ$` means the loss became smaller in magnitude.
- `Max daily risk Δ$` is a concentration proxy, not a breach simulation.
