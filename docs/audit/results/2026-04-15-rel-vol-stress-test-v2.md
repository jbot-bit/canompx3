# rel_vol_HIGH Stress Test v2 — Self-Audited

**Date:** 2026-04-15
**Purpose:** honest re-audit of v1 after fresh-perspective review revealed overweighting of explicitly-INFORMATIONAL DSR at inflated N_eff=14,261.

## Corrections applied

- **var_sr:** 0.012190 (empirical cross-lane per-trade SR variance across 64 lanes)
- **N_eff reported across 7 framings** — K=5 (lanes) to K=14,261 (raw scan). True effective N is unknown but is almost certainly NOT 14,261 (cells are highly correlated).
- **DSR labeled INFORMATIONAL per dsr.py line 35** — not a hard gate.
- **4 CORE hard gates** (bootstrap, temporal, exceeds_max_t, per_day) — these are the primary verdict drivers.
- Final verdict labels revised: REAL_EDGE / EDGE_WITH_CAVEAT / NOT_YET_VALIDATED.

**Expected max |t| from K=14,261 random trials (Gumbel):** 3.826
**Expected max |t| from K=300 (Bailey MinBTL):** 2.745
**Expected max |t| from K=72 (sess×instr×dir):** 2.243

## Per-lane result

### MES COMEX_SETTLE O5 RR1.0 short
- N_on=251, per-trade SR=+0.0993, Δ=+0.288
- Welch t=+4.480, p=0.000009

**CORE hard gates:**
- Block bootstrap p (autocorrelation-robust): 0.0005 PASS
- Temporal stability (IS split 50/50 sign+|t|≥2 both halves): PASS
  - H1 t=+3.82, H2 t=+2.64
- |t| > E[max_t from K=14,261 noise] (3.83): PASS
- Per-day aggregated t: +4.48 (PASS)

**INFORMATIONAL — DSR across N_eff framings:**

| Framing | K | SR0 (noise-max) | DSR | Pass@0.95 |
|---------|---|-----------------|-----|-----------|
| K=5 (survivor lanes) | 5 | +0.1317 | 0.3087 | no |
| K=12 (sessions) | 12 | +0.1838 | 0.0960 | no |
| K=36 (sessions×instr) | 36 | +0.2372 | 0.0167 | no |
| K=72 (sessions×instr×dir) | 72 | +0.2664 | 0.0049 | no |
| K=300 (Bailey MinBTL) | 300 | +0.3197 | 0.0003 | no |
| K=900 (volume family) | 900 | +0.3561 | 0.0000 | no |
| K=14261 (raw scan) | 14261 | +0.4357 | 0.0000 | no |

**Verdict:** **EDGE_WITH_CAVEAT — 4/4 core pass but DSR@realistic_K<0.5 (small effect size)**

---

### MGC LONDON_METALS O5 RR1.0 short
- N_on=147, per-trade SR=+0.1601, Δ=+0.349
- Welch t=+4.653, p=0.000005

**CORE hard gates:**
- Block bootstrap p (autocorrelation-robust): 0.0005 PASS
- Temporal stability (IS split 50/50 sign+|t|≥2 both halves): PASS
  - H1 t=+4.06, H2 t=+2.67
- |t| > E[max_t from K=14,261 noise] (3.83): PASS
- Per-day aggregated t: +4.65 (PASS)

**INFORMATIONAL — DSR across N_eff framings:**

| Framing | K | SR0 (noise-max) | DSR | Pass@0.95 |
|---------|---|-----------------|-----|-----------|
| K=5 (survivor lanes) | 5 | +0.1317 | 0.6265 | no |
| K=12 (sessions) | 12 | +0.1838 | 0.3934 | no |
| K=36 (sessions×instr) | 36 | +0.2372 | 0.1904 | no |
| K=72 (sessions×instr×dir) | 72 | +0.2664 | 0.1133 | no |
| K=300 (Bailey MinBTL) | 300 | +0.3197 | 0.0348 | no |
| K=900 (volume family) | 900 | +0.3561 | 0.0129 | no |
| K=14261 (raw scan) | 14261 | +0.4357 | 0.0009 | no |

**Verdict:** **EDGE_WITH_CAVEAT — 4/4 core pass but DSR@realistic_K<0.5 (small effect size)**

---

### MES TOKYO_OPEN O5 RR1.5 long
- N_on=278, per-trade SR=+0.0817, Δ=+0.321
- Welch t=+4.367, p=0.000015

**CORE hard gates:**
- Block bootstrap p (autocorrelation-robust): 0.0005 PASS
- Temporal stability (IS split 50/50 sign+|t|≥2 both halves): PASS
  - H1 t=+3.40, H2 t=+2.70
- |t| > E[max_t from K=14,261 noise] (3.83): PASS
- Per-day aggregated t: +4.37 (PASS)

**INFORMATIONAL — DSR across N_eff framings:**

| Framing | K | SR0 (noise-max) | DSR | Pass@0.95 |
|---------|---|-----------------|-----|-----------|
| K=5 (survivor lanes) | 5 | +0.1317 | 0.2033 | no |
| K=12 (sessions) | 12 | +0.1838 | 0.0450 | no |
| K=36 (sessions×instr) | 36 | +0.2372 | 0.0049 | no |
| K=72 (sessions×instr×dir) | 72 | +0.2664 | 0.0011 | no |
| K=300 (Bailey MinBTL) | 300 | +0.3197 | 0.0000 | no |
| K=900 (volume family) | 900 | +0.3561 | 0.0000 | no |
| K=14261 (raw scan) | 14261 | +0.4357 | 0.0000 | no |

**Verdict:** **EDGE_WITH_CAVEAT — 4/4 core pass but DSR@realistic_K<0.5 (small effect size)**

---

### MNQ SINGAPORE_OPEN O5 RR1.0 short
- N_on=276, per-trade SR=+0.1657, Δ=+0.266
- Welch t=+4.291, p=0.000021

**CORE hard gates:**
- Block bootstrap p (autocorrelation-robust): 0.0005 PASS
- Temporal stability (IS split 50/50 sign+|t|≥2 both halves): PASS
  - H1 t=+3.01, H2 t=+2.99
- |t| > E[max_t from K=14,261 noise] (3.83): PASS
- Per-day aggregated t: +4.29 (PASS)

**INFORMATIONAL — DSR across N_eff framings:**

| Framing | K | SR0 (noise-max) | DSR | Pass@0.95 |
|---------|---|-----------------|-----|-----------|
| K=5 (survivor lanes) | 5 | +0.1317 | 0.7045 | no |
| K=12 (sessions) | 12 | +0.1838 | 0.3869 | no |
| K=36 (sessions×instr) | 36 | +0.2372 | 0.1292 | no |
| K=72 (sessions×instr×dir) | 72 | +0.2664 | 0.0555 | no |
| K=300 (Bailey MinBTL) | 300 | +0.3197 | 0.0074 | no |
| K=900 (volume family) | 900 | +0.3561 | 0.0013 | no |
| K=14261 (raw scan) | 14261 | +0.4357 | 0.0000 | no |

**Verdict:** **EDGE_WITH_CAVEAT — 4/4 core pass but DSR@realistic_K<0.5 (small effect size)**

---

### MES COMEX_SETTLE O5 RR1.5 short
- N_on=249, per-trade SR=+0.0533, Δ=+0.300
- Welch t=+3.681, p=0.000259

**CORE hard gates:**
- Block bootstrap p (autocorrelation-robust): 0.0005 PASS
- Temporal stability (IS split 50/50 sign+|t|≥2 both halves): PASS
  - H1 t=+2.60, H2 t=+2.69
- |t| > E[max_t from K=14,261 noise] (3.83): FAIL
- Per-day aggregated t: +3.68 (PASS)

**INFORMATIONAL — DSR across N_eff framings:**

| Framing | K | SR0 (noise-max) | DSR | Pass@0.95 |
|---------|---|-----------------|-----|-----------|
| K=5 (survivor lanes) | 5 | +0.1317 | 0.1082 | no |
| K=12 (sessions) | 12 | +0.1838 | 0.0197 | no |
| K=36 (sessions×instr) | 36 | +0.2372 | 0.0019 | no |
| K=72 (sessions×instr×dir) | 72 | +0.2664 | 0.0004 | no |
| K=300 (Bailey MinBTL) | 300 | +0.3197 | 0.0000 | no |
| K=900 (volume family) | 900 | +0.3561 | 0.0000 | no |
| K=14261 (raw scan) | 14261 | +0.4357 | 0.0000 | no |

**Verdict:** **EDGE_WITH_CAVEAT — 3/4 core gates**

---


## Consolidated verdict

| Lane | Verdict | Core gates | DSR @ N_eff=36 | DSR @ N_eff=72 | DSR @ N_eff=300 |
|------|---------|------------|----------------|----------------|-----------------|
| MES COMEX_SETTLE O5 RR1.0 short | **EDGE_WITH_CAVEAT — 4/4 core pass but DSR@realistic_K<0.5 (small effect size)** | 4/4 | 0.017 | 0.005 | 0.000 |
| MGC LONDON_METALS O5 RR1.0 short | **EDGE_WITH_CAVEAT — 4/4 core pass but DSR@realistic_K<0.5 (small effect size)** | 4/4 | 0.190 | 0.113 | 0.035 |
| MES TOKYO_OPEN O5 RR1.5 long | **EDGE_WITH_CAVEAT — 4/4 core pass but DSR@realistic_K<0.5 (small effect size)** | 4/4 | 0.005 | 0.001 | 0.000 |
| MNQ SINGAPORE_OPEN O5 RR1.0 short | **EDGE_WITH_CAVEAT — 4/4 core pass but DSR@realistic_K<0.5 (small effect size)** | 4/4 | 0.129 | 0.056 | 0.007 |
| MES COMEX_SETTLE O5 RR1.5 short | **EDGE_WITH_CAVEAT — 3/4 core gates** | 3/4 | 0.002 | 0.000 | 0.000 |